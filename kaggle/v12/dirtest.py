"""Does a test-only 'board direction' generalise? Fit it from features on test rows, apply to train OOF."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
S=sys.argv[1]; root=S+'/v11in'
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); te=pd.read_csv(find(root,'test.csv','playground-series-s6e11'))
y=tr.is_fraud.values
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values)
v8=(w+dv)/2; zt=zones(tr)
wt=lr(np.load(find(root,'wide_lm_stack_test.npy'))); dt=lr(pd.read_csv(find(root,'test_wall_stack.csv')).sort_values('id').stack_all.values)
v8t=(wt+dt)/2; ze=zones(te)
cols=[c for c in tr.columns if c not in('id','is_fraud')]
def X(f,v):
    d=f[cols].copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]): d[c]=d[c].astype(str).astype('category')
    d['v8']=v; return d
Xt,Xr=X(te,v8t),X(tr,v8)
for c in Xt.columns:
    if str(Xt[c].dtype)=='category': Xr[c]=pd.Categorical(Xr[c],categories=Xt[c].cat.categories)
g=lambda p:pd.read_csv(p).sort_values('id').is_fraud.values
pub=S+'/pub/'
dirs={'board_ag':np.load(S+'/pin/board_direction_ag.npy'),
      'board_ae':np.load(S+'/pin/board_direction_ae.npy'),
      'monya_minus_v8':lr(g(pub+'monya1_s6e11-zero-fraud-zones-and-rank-extrapolation/submission.csv'))-lr(g('/home/user/v8m/stack_mean.csv')),
      'AG_minus_AE':lr(g(pub+'goodpjw2008_s6e11-board-model-realmlp-lb-0-95687/submission.csv'))-lr(g(pub+'goodpjw2008_s6e11-board-model-realmlp-lb-0-95687/submission_step_ae.csv'))}
keep=~ze
base=roc_auc_score(y,v8-100*zt)
for k,d in dirs.items():
    d=np.asarray(d,float); d=d/np.std(d[keep])
    m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    idx=np.flatnonzero(keep); rng=np.random.default_rng(0); rng.shuffle(idx); a,b=idx[:len(idx)//2],idx[len(idx)//2:]
    m.fit(Xt.iloc[a],d[a]); p=m.predict(Xt.iloc[b]); r2=1-np.var(d[b]-p)/np.var(d[b])
    m.fit(Xt.iloc[idx],d[idx]); q=m.predict(Xr)
    # scale: in test, the move's size relative to v8 logit
    sd_move=np.std(np.asarray(dirs[k],float)[keep]) if k.endswith('v8') or k.startswith('AG') else None
    out=[f'{s:+.2f}:{roc_auc_score(y,v8+s*q-100*zt)-base:+.1e}' for s in (-0.3,-0.1,0.1,0.3,0.6)]
    print(f'{k:16s} R2 on held-out test rows {r2:.3f}  raw sd {np.std(np.asarray(dirs[k],float)[keep]):.4f}  OOF gain by scale(sd units): {out}',flush=True)
