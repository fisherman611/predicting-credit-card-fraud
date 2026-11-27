"""Nearest original rows as features: fold-honest residual on top of v8 (the original 20k rows are external, labels known)."""
import sys,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
S=sys.argv[1]; root=S+'/v11in'
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); y=tr.is_fraud.values; z=zones(tr)
og=pd.read_csv(find(root,'credit_card_fraud_2026.csv'))
lab=[c for c in og.columns if 'fraud' in c.lower()][0]; yo=og[lab].astype(float).values
num=[c for c in tr.columns if c not in('id','is_fraud') and pd.api.types.is_numeric_dtype(tr[c]) and not pd.api.types.is_bool_dtype(tr[c])]
mu,sd=og[num].mean(),og[num].std()
A=((og[num]-mu)/sd).values; B=((tr[num]-mu)/sd).values
nn=NearestNeighbors(n_neighbors=20).fit(A); dist,idx=nn.kneighbors(B)
F=pd.DataFrame({'d1':dist[:,0],'y1':yo[idx[:,0]],'y5':yo[idx[:,:5]].mean(1),'y20':yo[idx].mean(1),'d5':dist[:,4]})
exact=(dist[:,0]<1e-9); print('exact matches',exact.sum(),'fraud rate there',y[exact].mean() if exact.any() else None,flush=True)
print('AUC of y20 alone',roc_auc_score(y,F.y20),' y5',roc_auc_score(y,F.y5),flush=True)
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values); v8=(w+dv)/2
F['v8']=v8
folds=list(StratifiedKFold(5,shuffle=True,random_state=2027).split(np.zeros(len(y)),y))
res=[]
for i,(a,b) in enumerate(folds):
    t=LogisticRegression(C=1e6,max_iter=1000).fit(v8[a,None],y[a]).decision_function(v8[:,None])
    ka=a[~z[a]]; inner=np.random.default_rng(i).random(len(ka))<0.2
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,verbose=-1)
    m.fit(F.iloc[ka[~inner]],y[ka[~inner]],init_score=t[ka[~inner]],eval_set=[(F.iloc[ka[inner]],y[ka[inner]])],eval_init_score=[t[ka[inner]]],eval_metric='auc',callbacks=[lgb.early_stopping(200,verbose=False)])
    s=m.predict(F.iloc[b],raw_score=True); base=roc_auc_score(y[b],t[b]-100*z[b])
    g={beta:roc_auc_score(y[b],t[b]+beta*s-100*z[b])-base for beta in (0.5,1.0)}; res.append(g); print(i,m.best_iteration_,{k:f'{v:+.1e}' for k,v in g.items()},flush=True)
print('mean',{k:f'{np.mean([r[k] for r in res]):+.2e}' for k in res[0]})
