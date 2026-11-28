"""Local board step on the real bank (the method sim v4 ranks first): kernel ridge on files within rms maxd of the best,
leave-one-out check on those files, then best + rms * g/|g| for the given step sizes."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, 'kaggle/v11'); sys.path.insert(0, 'kaggle/v13')
from residual_lm import find, lr, zones
from bank import BANK, S
maxd, tau, lam = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); steps = [float(x) for x in sys.argv[4].split(',')]
R = np.load(S + '/v12/bank_R.npy').astype(np.float64); s = np.load(S + '/v12/bank_s.npy'); names = list(BANK)
ref = int(np.argmax(s)); d = np.sqrt(((R - R[ref]) ** 2).mean(1)); near = np.flatnonzero(d <= maxd)
print('files used:', [(names[i].split('/')[-2][:22], s[i], round(d[i], 3)) for i in near])
def grad(idx):
    X = R[idx] - R[ref]; t = s[idx] - s[ref]; w = np.exp(-d[idx] / tau); K = X @ X.T / X.shape[1]
    a = np.linalg.solve(np.diag(w) @ K + lam * np.eye(len(idx)), np.diag(w) @ t); return X.T @ a / X.shape[1]
for i in near:
    if i == ref: continue
    g = grad(np.array([j for j in near if j != i]))
    print(f'  LOO {names[i].split("/")[-2][:22]:22s} true {s[i]:.5f} predicted {s[ref] + g @ (R[i] - R[ref]):.5f}')
g = grad(near); u = g / np.sqrt(np.mean(g ** 2))
te = pd.read_csv(find(S + '/v11in', 'test.csv', 'playground-series-s6e11')).sort_values('id'); z = zones(te)
base = pd.read_csv(names[ref]).sort_values('id'); r0 = lr(base.is_fraud.values)
for st in steps:
    r = r0.copy(); r[~z] += st * u; r[z] -= 100
    out = f'{S}/v12/local_board_rms{st:g}.csv'
    pd.DataFrame({'id': base.id.values, 'is_fraud': (rankdata(r) - 0.5) / len(r)}).to_csv(out, index=False)
    print(f'rms {st}: predicted {s[ref] + st * (g @ u):.5f}, spearman with best {spearmanr(r, r0).correlation:.7f} -> {out}')
