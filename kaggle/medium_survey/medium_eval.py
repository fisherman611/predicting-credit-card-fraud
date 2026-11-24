"""Check fold-0 medium LM contributions against matched public OOF streams."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def r(x): return (rankdata(x)-.5)/len(x)


def main():
    p=argparse.ArgumentParser()
    for x in ('bundle','medium','small','wide','output'): p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args()
    z=np.load(a.bundle); y=z['y_train']; fold=z['folds']; medium=np.load(a.medium); small=np.load(a.small); wide=np.load(a.wide)
    assert len(y)==len(fold)==len(medium)==len(small)==len(wide)==699960
    good=np.isfinite(medium)
    assert good.sum()==139992 and np.array_equal(good,fold==0)
    assert np.isfinite(small[good]).all() and np.isfinite(wide[good]).all()
    y=y[good]; m=medium[good]; s=small[good]; w=wide[good]
    fraud=np.flatnonzero(y==1); legit=np.flatnonzero(y==0); rng=np.random.default_rng(151)
    sampled_fraud=np.repeat(fraud,1000); sampled_legit=rng.choice(legit,len(sampled_fraud),replace=True)
    pairs_before=w[sampled_fraud]>w[sampled_legit]
    result={'rows':len(y),'frauds':len(fraud),'fold':0,
       'auc':{'medium':float(roc_auc_score(y,m)),'small':float(roc_auc_score(y,s)),
              'wide_nested':float(roc_auc_score(y,w))},
       'spearman_vs_wide':{'medium':float(spearmanr(m,w).statistic),'small':float(spearmanr(s,w).statistic)},
       'blend_vs_wide':{},'input_sha256':{},
       'limitations':['Only fold 0 medium OOF is available; weights are exploratory.',
         'Wide-stack nested OOF can share base-model training rows across meta folds.',
         'Step AB lacks matched OOF; this does not evaluate a blend with Step AB.']}
    for label,path in [('bundle',a.bundle),('medium',a.medium),('small',a.small),('wide',a.wide)]:
        result['input_sha256'][label]=hashlib.sha256(path.read_bytes()).hexdigest()
    for weight in (.05,.10,.20,.30):
        v=(1-weight)*r(w)+weight*r(m)
        delta=roc_auc_score(y,v)-roc_auc_score(y,w)
        after=v[sampled_fraud]>v[sampled_legit]
        result['blend_vs_wide'][str(weight)]={'auc':float(roc_auc_score(y,v)), 'delta_auc':float(delta),
            'sample_pairs':len(sampled_fraud),'corrected':int((~pairs_before&after).sum()),
            'broken':int((pairs_before&~after).sum())}
    # Bootstrap all fixed 10% predictions; not uncertainty from model fitting or selection.
    blend=.9*r(w)+.1*r(m); deltas=[]
    for _ in range(300):
        ids=np.r_[rng.choice(fraud,len(fraud),replace=True),rng.choice(legit,len(legit),replace=True)]
        deltas.append(roc_auc_score(y[ids],blend[ids])-roc_auc_score(y[ids],w[ids]))
    result['blend_vs_wide']['0.1']['bootstrap_ci_95']=np.quantile(deltas,[.025,.975]).tolist()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
