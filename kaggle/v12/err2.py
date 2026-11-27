"""Error analysis, part 2: are v8's hard frauds hard for every model, and how noisy are the labels?"""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, 'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
S = sys.argv[1]; root = S + '/v11in'
tr = pd.read_csv(find(root, 'train.csv', 'playground-series-s6e11')); y = tr.is_fraud.values.astype(int); z = zones(tr)
w = lr(np.load(find(root, 'wide_lm_stack_nested.npy'))); dv = lr(pd.read_csv(find(root, 'oof_wall_stack.csv')).sort_values('id').stack_all.values)
v8 = (w + dv) / 2 - 100 * z
pct = lambda v: (pd.Series(v).rank().values - 0.5) / len(v)
M = {}
for f in sorted(glob.glob('/home/user/ext/s6e11-fraud-series-artifacts/*_oof.npy')):
    a = np.load(f)
    if a.ndim == 1 and len(a) == len(y): M[os.path.basename(f)[:-8]] = a
for f in sorted(glob.glob(S + '/pin/oof_weni_*.npy')): M[os.path.basename(f)[4:-4]] = np.load(f)
M['yek_tabm'] = pd.read_csv(S + '/pin/yekenot_tabm_pytabkit_oof_preds.csv').set_index('id').loc[tr.id].iloc[:, -1].values
M['davut_stack'] = dv; M['wide_stack'] = w
P = {k: pct(v) for k, v in M.items()}; pv = pct(v8)
fr = np.flatnonzero(y == 1); hard = fr[np.argsort(pv[fr])[:len(fr) // 10]]
print(f'{len(M)} members; hard frauds = worst 10% of frauds under v8 ({len(hard)}), their v8 pct rank median {np.median(pv[hard]):.3f}')
rows = [(k, roc_auc_score(y, M[k]), np.median(P[k][hard]), (P[k][hard] > 0.9).mean()) for k in M]
R = pd.DataFrame(rows, columns=['member', 'auc', 'hard_median_pct', 'hard_in_top10%']).sort_values('hard_median_pct', ascending=False)
print(R.round(4).head(12).to_string(index=False))
best_any = np.max(np.column_stack([P[k][hard] for k in M]), 1)
print(f'hard frauds that at least one member puts in its top 10%: {(best_any > 0.9).mean():.3f} (top 5%: {(best_any > 0.95).mean():.3f})')
# same statistic for random legit rows, as a null for "some member likes it"
leg = np.random.default_rng(0).choice(np.flatnonzero((y == 0) & (np.abs(pv - np.median(pv[hard])) < 0.02)), len(hard), replace=False)
bl = np.max(np.column_stack([P[k][leg] for k in M]), 1)
print(f'null: legit rows at the same v8 rank, share with some member top 10%: {(bl > 0.9).mean():.3f} (top 5%: {(bl > 0.95).mean():.3f})')
# label-noise evidence: calibration at the top and exact duplicates with conflicting labels
o = np.argsort(-v8)
for q in (0.001, 0.005, 0.01, 0.02, 0.05):
    k = int(q * len(y)); print(f'top {q:.1%} by v8: fraud rate {y[o[:k]].mean():.3f}')
feat = [c for c in tr.columns if c not in ('id', 'is_fraud')]
g = tr.groupby(feat, dropna=False).is_fraud.agg(['size', 'mean'])
dup = g[g['size'] > 1]
print(f'exact duplicate feature vectors: {len(dup)} groups, {int(dup["size"].sum())} rows, groups with both labels: {int(((dup["mean"] > 0) & (dup["mean"] < 1)).sum())}')
# top false positives profile: highest-scored legits vs frauds at the same rank
fp = np.flatnonzero(y == 0)[np.argsort(-v8[y == 0])[:2000]]
tp = fr[np.argsort(-v8[fr])[:2000]]
num = [c for c in feat if pd.api.types.is_numeric_dtype(tr[c])]
print(pd.DataFrame({'top_legit': tr.iloc[fp][num].mean(), 'top_fraud': tr.iloc[tp][num].mean()}).round(3).to_string())
