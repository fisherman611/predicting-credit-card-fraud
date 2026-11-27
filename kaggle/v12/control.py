"""Leak control: a full-train LightGBM (no new signal) put through the same distill-on-test procedure."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
S=sys.argv[1]; root=S+'/v11in'
exec(open(S+'/v12/denoise.py').read().split("g=lambda")[0].split("S=sys.argv[1]")[1].replace("; root=S+'/v11in'; out={}","",1)) if False else None
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
R=Xr[cols]; T=Xt[cols]
folds=list(StratifiedKFold(5,shuffle=True,random_state=2027).split(np.zeros(len(y)),y))
cp=dict(n_estimators=800,learning_rate=0.05,num_leaves=255,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
oof=np.zeros(len(y))
for a,b in folds: oof[b]=lgb.LGBMClassifier(**cp).fit(R.iloc[a],y[a]).predict_proba(R.iloc[b])[:,1]
full=lgb.LGBMClassifier(**cp).fit(R,y); pt=full.predict_proba(T)[:,1]
print('control lgb OOF auc',roc_auc_score(y,oof),flush=True)
base=[roc_auc_score(y[b],(v8-100*zt)[b]) for _,b in folds]
SC=[0.1,0.3,0.5,0.7,1.0]
# honest use of its OOF
for s in SC: print('honest OOF blend',s,np.mean([roc_auc_score(y[b],(v8+s*(lr(oof)-v8)-100*zt)[b])-base[i] for i,(_,b) in enumerate(folds)]))
# distilled
keep=np.flatnonzero(~ze); d=lr(pt)-v8t
prm=dict(n_estimators=1500,learning_rate=0.03,num_leaves=127,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
fr=lgb.LGBMRegressor(**prm).fit(Xt.iloc[keep],d[keep]).predict(Xr)
for s in SC: print('distilled',s,np.mean([roc_auc_score(y[b],(v8+s*fr-100*zt)[b])-base[i] for i,(_,b) in enumerate(folds)]),flush=True)
