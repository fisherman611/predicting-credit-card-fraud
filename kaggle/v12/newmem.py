import sys, numpy as np, pandas as pd, glob, os
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
S=sys.argv[1]; root=S+'/v11in'; P=S+'/pin/'
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); y=tr.is_fraud.values; zt=zones(tr)
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values)
M={}
for f in sorted(glob.glob(P+'oof_weni_*.npy')): M[os.path.basename(f)[4:-4]]=np.load(f)
o=pd.read_csv(P+'davutkursun_xgb_glm_bag_oof.csv'); print(o.columns.tolist()); 
o=o.set_index('id').loc[tr.id]; M['davut_xgbglm']=o.iloc[:,-1].values
o=pd.read_csv(P+'yekenot_tabm_pytabkit_oof_preds.csv'); print(o.columns.tolist()); M['yek_tabm']=o.set_index('id').loc[tr.id].iloc[:,-1].values
for k,v in M.items(): print(f'{k:22s} auc {roc_auc_score(y,v):.5f}')
folds=list(StratifiedKFold(5,shuffle=True,random_state=2027).split(np.zeros(len(y)),y))
def nested(X):
    o=np.zeros(len(y))
    for a,b in folds: o[b]=LogisticRegression(C=0.05,max_iter=3000).fit(X[a],y[a]).decision_function(X[b])
    return o
v8=(w+dv)/2
fa=lambda s:[roc_auc_score(y[b],(s-100*zt)[b]) for _,b in folds]
b0=fa(v8); print('v8',np.mean(b0))
base=nested(np.column_stack([w,dv])); 
NEW=np.column_stack([lr(v) for v in M.values()])
for name,X in [('lr[w,dv]',np.column_stack([w,dv])),('lr[w,dv,+all new]',np.column_stack([w,dv,NEW]))]+[(f'lr[w,dv,+{k}]',np.column_stack([w,dv,lr(M[k])])) for k in M]:
    s=nested(X); f=fa(s); print(f'{name:34s} {np.mean(f):.6f} gain {np.mean(f)-np.mean(b0):+.2e} folds up {sum(a>b for a,b in zip(f,b0))}/5',flush=True)
