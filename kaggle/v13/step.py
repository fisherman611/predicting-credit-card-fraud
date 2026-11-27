"""Build candidate files: reference + t * g, where g is the board-model gradient; t set by the predicted gain."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, 'kaggle/v11'); sys.path.insert(0, 'kaggle/v13')
from residual_lm import find, lr, zones
from bank import BANK, S
lam, tau = float(sys.argv[1]), float(sys.argv[2]); gains = [float(x) for x in sys.argv[3].split(',')]
R = np.load(S + '/v12/bank_R.npy').astype(np.float64); s = np.load(S + '/v12/bank_s.npy')
ref = int(np.argmax(s)); dist = np.sqrt(((R - R[ref]) ** 2).mean(1))
w = np.exp(-dist / tau); X = R - R[ref]; t = s - s[ref]; K = X @ X.T / X.shape[1]
a = np.linalg.solve(np.diag(w) @ K + lam * np.eye(len(s)), np.diag(w) @ t); g = X.T @ a / X.shape[1]
te = pd.read_csv(find(S + '/v11in', 'test.csv', 'playground-series-s6e11')).sort_values('id'); z = zones(te)
refp = list(BANK)[ref]; base = pd.read_csv(refp).sort_values('id'); r0 = lr(base.is_fraud.values)
g2 = g @ g
for gain in gains:
    step = gain / g2                                    # predicted gain of r0 + step * g is step * g.g
    r = r0.copy(); r[~z] += step * g; r[z] -= 100
    p = (rankdata(r) - 0.5) / len(r)
    out = f'{S}/v12/board_gain{gain:g}.csv'
    pd.DataFrame({'id': base.id.values, 'is_fraud': p}).to_csv(out, index=False)
    print(f'predicted +{gain:g}: rms move {np.sqrt(np.mean((step * g) ** 2)):.4f}, spearman with reference {spearmanr(r, r0).correlation:.6f} -> {out}')
