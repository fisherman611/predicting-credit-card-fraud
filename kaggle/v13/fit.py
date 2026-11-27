"""Kernel ridge (linear kernel) from logit-rank vectors to public score; leave-one-out error and the step it implies."""
import sys, numpy as np
S = sys.argv[1]
R = np.load(S + '/v12/bank_R.npy').astype(np.float64); s = np.load(S + '/v12/bank_s.npy')
ref = int(np.argmax(s)); n = len(s)
dist = np.sqrt(((R - R[ref]) ** 2).mean(1))
def fit(idx, lam, tau):
    w = np.exp(-dist[idx] / tau)                       # nearby files count more (linear only holds locally)
    X = R[idx] - R[ref]; t = s[idx] - s[ref]
    K = X @ X.T / X.shape[1]
    W = np.diag(w)
    a = np.linalg.solve(W @ K + lam * np.eye(len(idx)), W @ t)
    return X.T @ a / X.shape[1]                         # gradient over rows: score ~ s_ref + g . (r - r_ref)
for tau in (0.05, 0.1, 0.2, 1.0):
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        err, err0 = [], []
        for i in range(n):
            if i == ref: continue
            idx = np.array([j for j in range(n) if j != i])
            g = fit(idx, lam, tau)
            pred = s[ref] + g @ (R[i] - R[ref])
            err.append((pred - s[i]) * np.exp(-dist[i] / 0.1)); err0.append((s[ref] - s[i]) * np.exp(-dist[i] / 0.1))
        e, e0 = np.sqrt(np.mean(np.square(err))), np.sqrt(np.mean(np.square(err0)))
        print(f'tau {tau:<5} lam {lam:<7} weighted LOO rmse {e:.2e} (constant model {e0:.2e}) ratio {e/e0:.2f}', flush=True)
