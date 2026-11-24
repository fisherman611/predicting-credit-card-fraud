"""Compare spline likelihoods with/without digits and risk-velocity interactions.

Uses existing same-fold OOF baseline predictions and the previously inspected
holdout. Selection happens on development; holdout is exploratory, not pristine.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

CONTINUOUS = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km',
              'card_age_months', 'customer_age', 'account_balance_usd',
              'velocity_score', 'merchant_risk_score']
DIGITS = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km',
          'account_balance_usd', 'velocity_score', 'merchant_risk_score']


def columns(frame):
    cats = frame.drop(columns=CONTINUOUS).astype(str).copy()
    for name in DIGITS:
        # Competition numeric values are rounded to either 1 or 2 decimals.
        scale = 10 if name in ('velocity_score', 'merchant_risk_score') else 100
        value = np.rint(frame[name].to_numpy(dtype=float)*scale).astype(np.int64)
        cats[name+'__last_digit'] = (np.abs(value)%10).astype(str)
        cats[name+'__round'] = (np.abs(value)%10 == 0).astype(str)
    risk = np.floor(frame.merchant_risk_score.to_numpy()/10).clip(0, 10).astype(int)
    velocity = np.floor(frame.velocity_score.to_numpy()/10).clip(0, 10).astype(int)
    cats['risk_x_velocity'] = (risk*11+velocity).astype(str)
    cats['cvv_x_risk'] = (frame.cvv_retry_count.clip(0, 3).to_numpy().astype(int)*11+risk).astype(str)
    return cats


def matrices(train, valid, variant):
    tc, vc = columns(train), columns(valid)
    if variant == 'spline':
        tc, vc = tc.drop(columns=[c for c in tc if '__' in c or '_x_' in c]), vc.drop(columns=[c for c in vc if '__' in c or '_x_' in c])
    elif variant == 'digits':
        tc, vc = tc.drop(columns=['risk_x_velocity','cvv_x_risk']), vc.drop(columns=['risk_x_velocity','cvv_x_risk'])
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    train_c = encoder.fit_transform(tc)
    valid_c = encoder.transform(vc)
    spline = SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant')
    ts = spline.fit_transform(train[CONTINUOUS]).astype(np.float32)
    vs = spline.transform(valid[CONTINUOUS]).astype(np.float32)
    return hstack([train_c, csr_matrix(ts)], format='csr'), hstack([valid_c, csr_matrix(vs)], format='csr')


def fit_predict(x, y, val, variant):
    a, b = matrices(x, val, variant)
    model = LogisticRegression(solver='liblinear', C=.2, max_iter=200, random_state=42)
    model.fit(a, y)
    return model.decision_function(b)


def rank(x):
    return (rankdata(x)-.5)/len(x)


def metrics(y, p, cvv):
    hit = np.zeros(len(y),bool)
    hit[np.argsort(-p,kind='stable')[:len(y)//20]] = True
    return {'auc': float(roc_auc_score(y,p)), 'auc_cvv0':float(roc_auc_score(y[cvv],p[cvv])),
            'caught_top5':int(((y==1)&hit).sum()), 'missed_cvv0':int(((y==1)&~hit&cvv).sum())}


def main():
    ap = argparse.ArgumentParser()
    for name in ('train', 'test', 'development', 'holdout', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    start=time.monotonic()
    df=pd.read_csv(args.train).set_index('id',verify_integrity=True)
    test=pd.read_csv(args.test)
    dev=np.load(args.development); hold=np.load(args.holdout)
    assert np.array_equal(df.loc[dev['id'],'is_fraud'],dev['target'])
    assert np.array_equal(df.loc[hold['id'],'is_fraud'],hold['target'])
    x=df.loc[dev['id']].drop(columns='is_fraud').reset_index(drop=True)
    h=df.loc[hold['id']].drop(columns='is_fraud').reset_index(drop=True)
    y,hy=dev['target'],hold['target']; folds=dev['fold']; base=dev['baseline']; hb=hold['baseline']
    drift={}
    for c in DIGITS:
        scale=10 if c in ('velocity_score','merchant_risk_score') else 100
        def frac(values):
            d=np.abs(np.rint(values.to_numpy()*scale).astype('int64'))%10
            return np.bincount(d,minlength=10)/len(d)
        drift[c]={'train_test_digit_total_variation':float(np.abs(frac(x[c])-frac(test[c])).sum()/2)}
    output={'drift':drift,'development':{},'selection':None,'holdout':{},
        'limitations':['EDA and the holdout were previously inspected.',
                       'Development model/weight selection is exploratory.',
                       'Two models are cross-fit on the same existing base folds; this is not a repeated, nested evaluation.',
                       'OOF baseline is not the 0.95569 Step AB, whose OOF predictions are unavailable.']}
    candidates=[('baseline',0.0,float(np.mean([roc_auc_score(y[folds==f],base[folds==f]) for f in range(3)])))]
    for variant in ('spline','digits','digits_interactions'):
        predictions=np.empty(len(y)); fold_scores={}
        for f in range(3):
            tr=folds!=f; va=~tr
            predictions[va]=fit_predict(x.loc[tr],y[tr],x.loc[va],variant)
            fold_scores[str(f)]=float(roc_auc_score(y[va],predictions[va]))
            print(f'{variant} fold {f}: {fold_scores[str(f)]:.6f}; {time.monotonic()-start:.0f}s',flush=True)
        results={'model_fold_auc':fold_scores,'weights':{}}
        for weight in (.05,.10,.20):
            fs=[]
            for f in range(3):
                va=folds==f
                blended=(1-weight)*rank(base[va])+weight*rank(predictions[va])
                fs.append(float(roc_auc_score(y[va],blended)))
            avg=float(np.mean(fs)); results['weights'][str(weight)]={'fold_auc':fs,'mean_fold_auc':avg}
            candidates.append((variant,weight,avg))
        output['development'][variant]=results
    chosen=max(candidates,key=lambda z:z[2])
    output['selection']={'variant':chosen[0],'weight':chosen[1],'mean_fold_auc':chosen[2],
                         'baseline_mean_fold_auc':candidates[0][2]}
    output['holdout']['baseline']=metrics(hy,hb,h.cvv_retry_count.eq(0).to_numpy())
    if chosen[0]=='baseline':
        output['holdout']['selected']=output['holdout']['baseline']
    else:
        hp=fit_predict(x,y,h,chosen[0]); mix=(1-chosen[1])*rank(hb)+chosen[1]*rank(hp)
        output['holdout']['selected']=metrics(hy,mix,h.cvv_retry_count.eq(0).to_numpy())
    output['seconds']=round(time.monotonic()-start,1)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    main()
