"""Board-ordered block probe on the real board (method B, which sim v6 ranks first).

The local board gradient (files within rms 0.1 of the best, tau 0.2, lam 1e-4) is fitted ONCE on the bank as it
is now; rows in the 0.2-0.8 rank band of the best file are sorted by it and cut into blocks of 1000. Probe j
raises block j by +0.5 logit-rank on top of the CURRENT best file; keep it only if public beats the current best.

usage: python kaggle/v13/probe_board.py order          (fit and save the block order)
       python kaggle/v13/probe_board.py CURRENT.csv j OUT.csv
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'kaggle/v11'); sys.path.insert(0, 'kaggle/v13')
from residual_lm import find, lr, zones
from bank import BANK, S
ORDER = S + '/v12/probe_board_order.npz'
te = pd.read_csv(find(S + '/v11in', 'test.csv', 'playground-series-s6e11')).sort_values('id'); z = zones(te); ids = te.id.values
if sys.argv[1] == 'order':
    R = np.load(S + '/v12/bank_R.npy').astype(np.float64); s = np.load(S + '/v12/bank_s.npy'); names = list(BANK)
    ref = int(np.argmax(s)); d = np.sqrt(((R - R[ref]) ** 2).mean(1)); near = np.flatnonzero(d <= 0.1)
    X = R[near] - R[ref]; t = s[near] - s[ref]; w = np.exp(-d[near] / 0.2)
    a = np.linalg.solve(np.diag(w) @ (X @ X.T / X.shape[1]) + 1e-4 * np.eye(len(near)), np.diag(w) @ t)
    g = np.zeros(len(ids)); g[~z] = X.T @ a / X.shape[1]
    best = pd.read_csv(names[ref]).set_index('id').loc[ids].is_fraud.values
    fr = (rankdata(best) - 0.5) / len(best); band = np.flatnonzero((fr >= 0.2) & (fr < 0.8) & ~z)
    order = band[np.argsort(-g[band])]
    np.savez(ORDER, order=order, start=names[ref])
    print('start', names[ref], s[ref], 'files used', len(near), 'band rows', len(band),
          'g share >0 in first 5 blocks', [round(float((g[order[i*1000:(i+1)*1000]] > 0).mean()), 2) for i in range(5)])
else:
    cur_p, j, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    order = np.load(ORDER)['order']; block = order[j * 1000:(j + 1) * 1000]
    cur = pd.read_csv(cur_p).set_index('id').loc[ids].is_fraud.values
    r = lr(cur); r[block] += 0.5
    pd.DataFrame({'id': ids, 'is_fraud': (rankdata(r) - 0.5) / len(r)}).to_csv(out, index=False)
    print(out, 'block', j, 'zone rows max rank', round(float(((rankdata(r) - 0.5) / len(r))[z].max()), 4))
