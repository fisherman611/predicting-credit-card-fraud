"""Local label consistency: do v8's hard frauds have more fraud neighbours than legit rows at the same v8 rank?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.neighbors import NearestNeighbors
S = sys.argv[1]; root = S + '/v11in'
tr = pd.read_csv(find(root, 'train.csv', 'playground-series-s6e11')); y = tr.is_fraud.values.astype(int); z = zones(tr)
w = lr(np.load(find(root, 'wide_lm_stack_nested.npy'))); dv = lr(pd.read_csv(find(root, 'oof_wall_stack.csv')).sort_values('id').stack_all.values)
v8 = (w + dv) / 2 - 100 * z
q = (pd.Series(v8).rank().values - 0.5) / len(v8)
feat = [c for c in tr.columns if c not in ('id', 'is_fraud')]
X = pd.get_dummies(tr[feat].astype({c: float for c in feat if pd.api.types.is_bool_dtype(tr[c])}), dtype=float)
X = ((X - X.mean()) / X.std()).fillna(0).values.astype(np.float32)
rng = np.random.default_rng(0)
fr = np.flatnonzero(y == 1)
hard = fr[(q[fr] > 0.6) & (q[fr] < 0.85)]
hard = rng.choice(hard, min(1500, len(hard)), replace=False)
band = np.flatnonzero((y == 0) & (q > 0.6) & (q < 0.85))
legs = rng.choice(band, 1500, replace=False)
nn = NearestNeighbors(n_neighbors=51).fit(X)
for name, rows in (('hard frauds (v8 rank 0.60-0.85)', hard), ('legits, same v8 band', legs)):
    d, i = nn.kneighbors(X[rows]); i = i[:, 1:]
    for k in (10, 50):
        print(f'{name:34s} k={k}: neighbour fraud rate {y[i[:, :k]].mean():.4f}, neighbour mean v8 rank {q[i[:, :k]].mean():.3f}')
print('base fraud rate in band', y[(q > 0.6) & (q < 0.85)].mean().round(4))
