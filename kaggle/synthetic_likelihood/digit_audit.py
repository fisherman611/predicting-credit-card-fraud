"""Describe last-digit distributions by target/fold and train/test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import DIGITS

ap=argparse.ArgumentParser()
for name in ('train','test','development','output'):
    ap.add_argument('--'+name,type=Path,required=True)
args=ap.parse_args()
data=pd.read_csv(args.train).set_index('id',verify_integrity=True)
dev=np.load(args.development)
df=data.loc[dev['id']].reset_index()
assert np.array_equal(df.is_fraud,dev['target'])
test=pd.read_csv(args.test)
out={}
for name in DIGITS:
    scale=10 if name in ('velocity_score','merchant_risk_score') else 100
    d=np.abs(np.rint(df[name].to_numpy()*scale).astype('int64'))%10
    t=np.abs(np.rint(test[name].to_numpy()*scale).astype('int64'))%10
    def dist(v): return np.bincount(v,minlength=10)/len(v)
    global_cv=dist(d)
    item={'train_test_tv':float(np.abs(global_cv-dist(t)).sum()/2),'folds':[]}
    for fold in range(3):
        mask=dev['fold']==fold
        item['folds'].append({})
        for label,group in [('all',mask),('cvv0',mask & df.cvv_retry_count.eq(0).to_numpy())]:
            pos=d[group & (dev['target']==1)]; neg=d[group & (dev['target']==0)]
            item['folds'][-1][label]={'frauds':len(pos),'legitimate':len(neg),
               'fraud_vs_legit_tv':float(np.abs(dist(pos)-dist(neg)).sum()/2)}
    out[name]=item
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
