"""Refit the count ablation model and blend 5% by rank into pinned Step AB."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import TargetEncoder
from tokenizers import Tokenizer

DIGIT_COLS=('amount_usd','hours_since_last_txn','distance_from_home_km',
            'account_balance_usd','velocity_score','merchant_risk_score')
TOKEN_COLS=('amount_usd','hours_since_last_txn','velocity_score','merchant_risk_score')


def main():
    p=argparse.ArgumentParser()
    for name in ('train','test','step-ab','tokenizer','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); start=time.monotonic()
    train=pd.read_csv(args.train,dtype=str); test=pd.read_csv(args.test,dtype=str)
    n=len(train); ids=test['id'].astype('int64').to_numpy(); y=train.pop('is_fraud').astype('float64').astype('int8').to_numpy()
    train_ids=train.pop('id').astype('int64').to_numpy(); test.pop('id')
    if not np.array_equal(train_ids,np.arange(699960)) or not np.array_equal(ids,np.arange(699960,999944)):
        raise ValueError('Unexpected training or test IDs')
    if list(train.columns)!=list(test.columns): raise ValueError('Train/test features differ')
    if hashlib.sha256(args.step_ab.read_bytes()).hexdigest()!='3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038':
        raise ValueError('Step AB checksum mismatch')
    text=pd.concat([train,test],ignore_index=True); del train,test
    raw=text.copy()
    for c in raw:
        converted=pd.to_numeric(raw[c],errors='coerce')
        raw[c]=converted.astype('float32') if converted.notna().all() else pd.Categorical(raw[c])
    features={}
    for col in text:
        features['e_'+col]=pd.factorize(text[col],sort=False)[0].astype('int32')
    for col in DIGIT_COLS:
        v=np.rint(pd.to_numeric(text[col]).to_numpy()*100).astype('int64')
        for suffix,value in [('whole',v//100),('tenth',v//10),('fraction',np.abs(v)%100),('tail',(np.abs(v)//100)%100)]:
            features[f'd_{col}_{suffix}']=pd.factorize(value,sort=False)[0].astype('int32')
    tok=Tokenizer.from_file(str(args.tokenizer))
    for col in TOKEN_COLS:
        uniq,codes=np.unique(text[col].to_numpy(),return_inverse=True)
        enc=tok.encode_batch(uniq.tolist(),add_special_tokens=False)
        for suffix,v in [('first',np.array([e.ids[0] if e.ids else -1 for e in enc],dtype='int32')),
                         ('last',np.array([e.ids[-1] if e.ids else -1 for e in enc],dtype='int32')),
                         ('count',np.array([len(e.ids) for e in enc],dtype='int32'))]:
            features[f't_{col}_{suffix}']=pd.factorize(v[codes],sort=False)[0].astype('int32')
    keys=pd.DataFrame(features); del features,text
    encoder=TargetEncoder(target_type='binary',smooth=20,cv=3,shuffle=True,random_state=42)
    rate_tr=pd.DataFrame(encoder.fit_transform(keys.iloc[:n],y).astype('float32'),columns=keys.columns)
    rate_te=pd.DataFrame(encoder.transform(keys.iloc[n:]).astype('float32'),columns=keys.columns)
    count_tr={};count_te={}
    for c in keys:
        arr=keys[c].to_numpy();counts=np.bincount(arr[:n]); v=arr[n:]
        count_tr['n_'+c]=np.log1p(counts[arr[:n]]).astype('float32')
        count_te['n_'+c]=np.log1p(np.where(v<len(counts),counts[np.minimum(v,len(counts)-1)],0)).astype('float32')
    x=pd.concat([raw.iloc[:n].reset_index(drop=True),rate_tr,pd.DataFrame(count_tr)],axis=1)
    xt=pd.concat([raw.iloc[n:].reset_index(drop=True),rate_te,pd.DataFrame(count_te)],axis=1)
    del raw,keys,rate_tr,rate_te,count_tr,count_te
    model=lgb.LGBMClassifier(n_estimators=350,learning_rate=.035,num_leaves=4,
        min_child_samples=100,reg_lambda=1,n_jobs=3,random_state=42,
        deterministic=True,force_col_wise=True,verbosity=-1)
    model.fit(x,y); pred=model.predict_proba(xt)[:,1]
    ab=pd.read_csv(args.step_ab)
    if list(ab.columns)!=['id','is_fraud'] or not np.array_equal(ab.id,ids): raise ValueError('Step AB IDs differ')
    r=lambda v:(rankdata(v)-.5)/len(v)
    candidate=.95*r(ab.is_fraud)+.05*r(pred)
    if not np.isfinite(candidate).all() or not ((candidate>0)&(candidate<1)).all(): raise ValueError('Invalid predictions')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'submission.csv';pd.DataFrame({'id':ids,'is_fraud':candidate}).to_csv(path,index=False)
    pd.DataFrame({'id':ids,'is_fraud':pred}).to_csv(args.output/'count_model.csv',index=False)
    report={'rows':len(ids),'step_ab_weight':.95,'count_weight':.05,'model_rounds':350,
       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'elapsed_seconds':round(time.monotonic()-start,1),
       'caveat':'Exploratory rank blend. Existing three-fold wide-stack tests were inconsistent; Step AB lacks matched OOF.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
