import sys, numpy as np, pandas as pd
sys.path.insert(0,'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
S=sys.argv[1]; root=S+'/v11in'
tr=pd.read_csv(find(root,'train.csv','playground-series-s6e11')); y=tr.is_fraud.values
w=lr(np.load(find(root,'wide_lm_stack_nested.npy'))); dv=lr(pd.read_csv(find(root,'oof_wall_stack.csv')).sort_values('id').stack_all.values)
s=(w+dv)/2-100*zones(tr)
order=np.argsort(s); rk=np.empty(len(s)); rk[order]=np.arange(len(s))
nl,nf=(y==0).sum(),(y==1).sum()
# for each fraud: number of legits scored above it (misordered pairs)
legit_rank=np.sort(s[y==0])
mis_f=nl-np.searchsorted(legit_rank,s[y==1],side='right')
fraud_rank=np.sort(s[y==1]); mis_l=np.searchsorted(fraud_rank,s[y==0],side='left')  # frauds below each legit
tot=mis_f.sum(); print('AUC',1-tot/(nl*nf), 'loss',tot/(nl*nf))
f=tr[y==1].copy(); f['mis']=mis_f/(nl*nf)
l=tr[y==0].copy(); l['mis']=mis_l/(nl*nf)
# loss concentration
for q in (0.01,0.05,0.1,0.2,0.5):
    k=int(q*nf); print(f'worst {q:.0%} frauds carry {np.sort(f.mis.values)[::-1][:k].sum()/f.mis.sum():.1%} of loss')
for q in (0.001,0.01,0.05):
    k=int(q*nl); print(f'worst {q:.1%} legits carry {np.sort(l.mis.values)[::-1][:k].sum()/l.mis.sum():.1%} of loss')
cols=[c for c in tr.columns if c not in('id','is_fraud')]
print(cols)
def seg(d,c):
    v=d[c]
    if pd.api.types.is_numeric_dtype(v) and v.nunique()>12: return pd.qcut(tr[c].rank(method='first'),8,labels=False).loc[d.index]
    return v
rows=[]
for c in cols:
    gf=f.groupby(seg(f,c)).mis.sum(); gl=l.groupby(seg(l,c)).mis.sum(); n=tr.groupby(seg(tr,c)).size(); fr=tr.groupby(seg(tr,c)).is_fraud.mean()
    t=pd.DataFrame({'loss_f':gf,'loss_l':gl,'share':n/len(tr),'rate':fr}).fillna(0); t['loss']=t.loss_f+t.loss_l
    t['excess']=t.loss/ (2*tot/(nl*nf)) - t.share
    for k,r in t.iterrows(): rows.append((c,k,r.share,r.rate,r.loss/(2*tot/(nl*nf)),r.excess))
R=pd.DataFrame(rows,columns=['col','val','share','fraud_rate','loss_share','excess']).sort_values('excess',ascending=False)
print(R.head(20).to_string(index=False))
# hard frauds profile: worst 10% frauds vs all frauds
hf=f.nlargest(int(0.1*nf),'mis')
num=[c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]
print(pd.DataFrame({'hard_fraud':hf[num].mean(),'all_fraud':f[num].mean(),'legit':l[num].mean()}).round(3).to_string())
print('hard frauds median pct rank',np.median(rk[y==1][np.argsort(-mis_f)[:int(0.1*nf)]]/len(s)))
