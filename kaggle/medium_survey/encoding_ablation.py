"""Nested target-encoding ablations on three shared competition folds."""
import argparse
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from tokenizers import Tokenizer
from scipy.stats import rankdata

DIGIT_COLS=('amount_usd','hours_since_last_txn','distance_from_home_km',
            'account_balance_usd','velocity_score','merchant_risk_score')
TOKEN_COLS=('amount_usd','hours_since_last_txn','velocity_score','merchant_risk_score')


def main():
    p=argparse.ArgumentParser()
    for name in ('train','bundle','tokenizer','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=350)
    p.add_argument('--variants',nargs='+',choices=('raw','exact','digit','token','count'))
    p.add_argument('--wide',type=Path)
    args=p.parse_args(); start=time.monotonic()
    text=pd.read_csv(args.train,dtype=str); ids=text.pop('id').astype('int64').to_numpy()
    y=text.pop('is_fraud').astype('float64').astype('int8').to_numpy()
    z=np.load(args.bundle); folds=z['folds'];
    assert np.array_equal(ids,np.arange(len(ids))) and np.array_equal(y,z['y_train'])
    raw=text.copy()
    for c in raw:
        converted=pd.to_numeric(raw[c],errors='coerce')
        raw[c]=converted.astype('float32') if converted.notna().all() else pd.Categorical(raw[c])
    groups={'exact':{},'digit':{},'token':{}}
    for col in text:
        groups['exact']['e_'+col]=pd.factorize(text[col],sort=False)[0].astype('int32')
    for col in DIGIT_COLS:
        v=np.rint(pd.to_numeric(text[col]).to_numpy()*100).astype('int64')
        for suffix,value in [('whole',v//100),('tenth',v//10),('fraction',np.abs(v)%100),('tail',(np.abs(v)//100)%100)]:
            groups['digit'][f'd_{col}_{suffix}']=pd.factorize(value,sort=False)[0].astype('int32')
    tok=Tokenizer.from_file(str(args.tokenizer))
    for col in TOKEN_COLS:
        uniq, codes=np.unique(text[col].to_numpy(),return_inverse=True)
        enc=tok.encode_batch(uniq.tolist(),add_special_tokens=False)
        first=np.array([e.ids[0] if e.ids else -1 for e in enc],dtype='int32')
        last=np.array([e.ids[-1] if e.ids else -1 for e in enc],dtype='int32')
        count=np.array([len(e.ids) for e in enc],dtype='int32')
        for suffix,v in [('first',first),('last',last),('count',count)]:
            groups['token'][f't_{col}_{suffix}']=pd.factorize(v[codes],sort=False)[0].astype('int32')
    keys=pd.DataFrame({k:v for d in groups.values() for k,v in d.items()})
    groupcols={k:list(v) for k,v in groups.items()}
    config={'raw':[], 'exact':['exact'], 'digit':['exact','digit'],
            'token':['exact','digit','token'], 'count':['exact','digit','token']}
    if args.variants:
        config={k:v for k,v in config.items() if k in args.variants}
    wide=np.load(args.wide) if args.wide else None
    if wide is not None: assert len(wide)==len(y) and np.isfinite(wide).all()
    blends={k:{str(v):[] for v in (.05,.10,.20)} for k in config}
    scores={k:[] for k in config}; auc_fold=[]
    for fold in (0,1,2):
        tr=np.flatnonzero(folds!=fold); va=np.flatnonzero(folds==fold)
        encoder=TargetEncoder(target_type='binary',smooth=20,cv=3,shuffle=True,random_state=42)
        rates_tr=encoder.fit_transform(keys.iloc[tr],y[tr]).astype('float32')
        rates_va=encoder.transform(keys.iloc[va]).astype('float32')
        rate_tr=pd.DataFrame(rates_tr,columns=keys.columns); rate_va=pd.DataFrame(rates_va,columns=keys.columns)
        count_tr={}; count_va={}
        for c in keys:
            a=keys[c].to_numpy(); counts=np.bincount(a[tr]);
            count_tr['n_'+c]=np.log1p(counts[a[tr]]).astype('float32')
            count_va['n_'+c]=np.log1p(np.where(a[va]<len(counts),counts[np.minimum(a[va],len(counts)-1)],0)).astype('float32')
        raw_tr=raw.iloc[tr].reset_index(drop=True); raw_va=raw.iloc[va].reset_index(drop=True)
        for variant,wanted in config.items():
            names=[c for g in wanted for c in groupcols[g]]
            vt=pd.concat([raw_tr,rate_tr[names]],axis=1); vv=pd.concat([raw_va,rate_va[names]],axis=1)
            if variant=='count':
                vt=pd.concat([vt,pd.DataFrame(count_tr)],axis=1); vv=pd.concat([vv,pd.DataFrame(count_va)],axis=1)
            model=lgb.LGBMClassifier(n_estimators=args.rounds,learning_rate=.035,num_leaves=4,
                 min_child_samples=100,reg_lambda=1,n_jobs=3,random_state=42,
                 deterministic=True,force_col_wise=True,verbosity=-1)
            model.fit(vt,y[tr]); pred=model.predict_proba(vv)[:,1]
            score=float(roc_auc_score(y[va],pred)); scores[variant].append(score)
            if wide is not None:
                n=len(va); wr=(rankdata(wide[va])-.5)/n; pr=(rankdata(pred)-.5)/n
                for weight in (.05,.10,.20):
                    blends[variant][str(weight)].append(float(roc_auc_score(y[va],(1-weight)*wr+weight*pr)))
            print(f'fold={fold} {variant} auc={score:.7f} elapsed={time.monotonic()-start:.0f}s',flush=True)
        auc_fold.append({'fold':fold,'frauds':int(y[va].sum()),'rows':len(va)})
    report={'folds':auc_fold,'rounds':args.rounds,'fold_auc':scores,
            'mean_auc':{k:float(np.mean(v)) for k,v in scores.items()},
            'limitations':['Fold 3 and 4 were not evaluated.', 'Model hyperparameters were fixed, no early stopping.',
            'Screening is exploratory; no independent final holdout or Step AB OOF.',
            'Counts fitted on the outer train partition; token encoding reads the original CSV strings.',
            'Inner-fold encoded train features can share data with other inner folds; no outer labels enter encoding.'],
            'seconds':round(time.monotonic()-start,1)}
    if wide is not None:
        report['wide_fold_auc']=[float(roc_auc_score(y[folds==f],wide[folds==f])) for f in (0,1,2)]
        report['wide_blend_fold_auc']=blends
        report['limitations'].append('Count variant was selected on these same folds, so this blend comparison is exploratory.')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
