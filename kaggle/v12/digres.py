"""Fold-honest residual on digit features: LightGBM with init_score = Platt(v8) on last-digit / round-number features."""
import sys,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
S=sys.argv[1]; root=S+'/v11in'
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); y=tr.is_fraud.values; z=zones(tr)
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values); v8=(w+dv)/2
F=pd.DataFrame(index=tr.index)
dec={'amount_usd':2,'hours_since_last_txn':2,'distance_from_home_km':2,'account_balance_usd':2,'velocity_score':1,'merchant_risk_score':1}
for c,p in dec.items():
    iv=np.round(tr[c].values*10**p).astype(np.int64)
    F[c+'_d0']=iv%10; F[c+'_d1']=(iv//10)%10; F[c+'_d2']=(iv//100)%10; F[c+'_mod100']=iv%100
    F[c+'_trail0']=(iv%10==0).astype(int)+(iv%100==0).astype(int)+(iv%1000==0).astype(int)
    F[c+'_len']=np.char.str_len(iv.astype(str))
for c in ['txn_count_last_24h','card_age_months','customer_age','time_of_day_hour','cvv_retry_count','prior_disputes']: F[c]=tr[c]
F['v8']=v8
folds=list(StratifiedKFold(5,shuffle=True,random_state=2027).split(np.zeros(len(y)),y))
res=[]
for i,(a,b) in enumerate(folds):
    pl=LogisticRegression(C=1e6,max_iter=1000).fit(v8[a,None],y[a]); t=pl.decision_function(v8[:,None])
    ka=a[~z[a]]
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,verbose=-1)
    # inner early stopping on a slice of fit rows
    rng=np.random.default_rng(i); inner=rng.random(len(ka))<0.2
    m.fit(F.iloc[ka[~inner]],y[ka[~inner]],init_score=t[ka[~inner]],eval_set=[(F.iloc[ka[inner]],y[ka[inner]])],eval_init_score=[t[ka[inner]]],eval_metric='auc',callbacks=[lgb.early_stopping(200,verbose=False)])
    s=m.predict(F.iloc[b],raw_score=True)
    base=roc_auc_score(y[b],t[b]-100*z[b])
    g={beta:roc_auc_score(y[b],t[b]+beta*s-100*z[b])-base for beta in (0.25,0.5,1.0)}
    res.append(g); print(i,'iters',m.best_iteration_,{k:f'{v:+.1e}' for k,v in g.items()},flush=True)
print('mean',{k:f'{np.mean([r[k] for r in res]):+.2e}' for k in res[0]})
imp=pd.Series(m.feature_importances_,F.columns).sort_values(ascending=False); print(imp.head(12).to_string())
