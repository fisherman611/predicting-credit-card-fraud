"""Fixed-prediction paired bootstrap and sampled fraud-legitimate pair audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from evaluate import fit_predict, rank


def main():
    p=argparse.ArgumentParser()
    for name in ('train','development','holdout','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    data=pd.read_csv(args.train).set_index('id',verify_integrity=True)
    dev=np.load(args.development); hold=np.load(args.holdout)
    train=data.loc[dev['id']].drop(columns='is_fraud').reset_index(drop=True)
    valid=data.loc[hold['id']].drop(columns='is_fraud').reset_index(drop=True)
    y=hold['target']; base=hold['baseline']
    assert np.array_equal(data.loc[hold['id'],'is_fraud'],y)
    score=fit_predict(train,dev['target'],valid,'spline')
    mixed=.9*rank(base)+.1*rank(score)
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    rng=np.random.default_rng(422)
    deltas=[]
    for _ in range(500):
        idx=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        deltas.append(roc_auc_score(y[idx],mixed[idx])-roc_auc_score(y[idx],base[idx]))
    # Per-fraud sample of legitimate rows, uniformly sampled with replacement.
    pi=np.repeat(positive,1000); ni=rng.choice(negative,len(pi),replace=True)
    before=base[pi]>base[ni]; after=mixed[pi]>mixed[ni]
    report={'auc_delta':float(roc_auc_score(y,mixed)-roc_auc_score(y,base)),
            'bootstrap_ci95_fixed_predictions':np.quantile(deltas,[.025,.975]).tolist(),
            'sampled_pairs':len(pi), 'corrected_pairs':int((~before&after).sum()),
            'broken_pairs':int((before&~after).sum()),
            'caveats':'Paired bootstrap conditions on fixed predictions; both holdout and model choices have been inspected.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'holdout_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
