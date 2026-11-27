"""Denoised ladder: learn (ladder - v8) from features on test rows (no labels), apply to train OOF; fold-honest scale."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, json
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
S=sys.argv[1]; root=S+'/v11in'; out={}
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); te=pd.read_csv(find(root,'test.csv','playground-series-s6e11'))
y=tr.is_fraud.values
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values)
wt=lr(np.load(find(root,'wide_lm_stack_test.npy'))); dt=lr(pd.read_csv(find(root,'test_wall_stack.csv')).sort_values('id').stack_all.values)
v8,v8t=(w+dv)/2,(wt+dt)/2; zt,ze=zones(tr),zones(te)
cols=[c for c in tr.columns if c not in('id','is_fraud')]
def X(f,a,b):
    d=f[cols].copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]): d[c]=d[c].astype(str).astype('category')
    d['wide'],d['davut']=a,b; return d
Xt,Xr=X(te,wt,dt),X(tr,w,dv)
for c in Xt.columns:
    if str(Xt[c].dtype)=='category': Xr[c]=pd.Categorical(Xr[c].astype(str),categories=Xt[c].cat.categories)
g=lambda p:pd.read_csv(p).sort_values('id').is_fraud.values
pub=S+'/pub/'
L={'monya':pub+'monya1_s6e11-zero-fraud-zones-and-rank-extrapolation/submission.csv',
   'AG':pub+'goodpjw2008_s6e11-board-model-realmlp-lb-0-95687/submission.csv',
   'cv_led':pub+'goodpjw2008_s6e11-board-model-realmlp-lb-0-95687/submission_cv_led.csv'}
keep=np.flatnonzero(~ze)
folds=list(StratifiedKFold(5,shuffle=True,random_state=2027).split(np.zeros(len(y)),y))
SC=np.round(np.arange(0,1.01,0.05),2)
for k,p in L.items():
    d=lr(g(p))-v8t
    # cross-fit on test (2 halves) for honest R2 and test-side prediction; full fit for train
    prm=dict(n_estimators=1500,learning_rate=0.03,num_leaves=127,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    ft=np.zeros(len(te))
    for a,b in KFold(2,shuffle=True,random_state=0).split(keep):
        m=lgb.LGBMRegressor(**prm).fit(Xt.iloc[keep[a]],d[keep[a]]); ft[keep[b]]=m.predict(Xt.iloc[keep[b]])
    r2=1-np.var(d[keep]-ft[keep])/np.var(d[keep])
    m=lgb.LGBMRegressor(**prm).fit(Xt.iloc[keep],d[keep]); fr=m.predict(Xr)
    base=[roc_auc_score(y[b],(v8-100*zt)[b]) for _,b in folds]
    G=np.array([[roc_auc_score(y[b],(v8+s*fr-100*zt)[b])-base[i] for s in SC] for i,(_,b) in enumerate(folds)])
    hon=[]
    for i in range(5):
        s=SC[int(np.argmax(G[np.arange(5)!=i].mean(0)))]; hon.append((float(s),float(G[i,list(SC).index(s)])))
    full=G.mean(0)
    print(f'{k:7s} R2 {r2:.3f} | mean gain by scale: '+' '.join(f'{s}:{g_:+.1e}' for s,g_ in zip(SC[::2],full[::2])),flush=True)
    print(f'        honest (scale, fold gain): {hon}  folds up {sum(h[1]>0 for h in hon)}/5  mean {np.mean([h[1] for h in hon]):+.2e}',flush=True)
    np.save(f'{S}/v12/denoise_{k}_test.npy',ft); np.save(f'{S}/v12/denoise_{k}_testfull.npy',m.predict(Xt)); np.save(f'{S}/v12/denoise_{k}_train.npy',fr)
    out[k]={'r2':r2,'honest':hon,'gain_by_scale':dict(zip(map(float,SC),map(float,full)))}
json.dump(out,open(f'{S}/v12/denoise.json','w'),indent=1)
