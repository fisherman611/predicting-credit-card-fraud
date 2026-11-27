"""Simulated leaderboard on labelled train rows: does a board model / extrapolation gain public, and does it carry to private?

A fake test set is 300k train rows; 20% of it is the fake public split, the rest fake private. Files are built from real
OOF members (logit ranks). A greedy public ladder (like the real one) produces a bank of scored files; then each method
steps from the bank's best file, and its gain is measured on fake public and fake private.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--art', required=True); ap.add_argument('--pin', required=True); ap.add_argument('--train', required=True)
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--out', default='sim_results.json')
args = ap.parse_args()
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
y_all = pd.read_csv(args.train, usecols=['is_fraud']).is_fraud.values.astype(int)
M = []
for f in sorted(glob.glob(args.art + '/*_oof.npy')) + sorted(glob.glob(args.pin + '/oof_weni_*.npy')):
    a = np.load(f)
    if a.ndim == 1 and len(a) == len(y_all): M.append(a)
print(len(M), 'members', flush=True)

def wfit(X, t, w, lam):
    K = X @ X.T / X.shape[1]
    a = np.linalg.solve(np.diag(w) @ K + lam * np.eye(len(t)), np.diag(w) @ t)
    return X.T @ a / X.shape[1]

res = []
for seed in range(args.seeds):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(y_all), 300000, replace=False)
    y = y_all[rows]; pub = rng.random(len(rows)) < 0.2
    L = np.array([lr(m[rows]) for m in M])                    # members as logit ranks on the fake test
    auc = lambda r, m: roc_auc_score(y[m], r[m])
    base = L.mean(0)
    # greedy public ladder: random small member moves, keep the ones that raise fake public; every try is scored
    bank_r, bank_s = [base], [auc(base, pub)]
    cur = base.copy()
    for it in range(60):
        d = L[rng.integers(len(L))] - L[rng.integers(len(L))] + 0.3 * rng.standard_normal(len(y))
        cand = cur + rng.choice([0.03, 0.06, 0.1]) * d / d.std()
        sc = auc(cand, pub); bank_r.append(cand); bank_s.append(sc)
        if sc > auc(cur, pub): cur = cand
    R, s = np.array(bank_r), np.array(bank_s)
    ref = int(np.argmax(s)); r0 = R[ref]
    out = {'seed': seed, 'ladder_public_gain': float(s[ref] - s[0]), 'ladder_private_gain': float(auc(r0, ~pub) - auc(base, ~pub))}
    dist = np.sqrt(((R - r0) ** 2).mean(1))
    for tau in (0.05, 0.2):
        for lam in (1e-4, 1e-2):
            g = wfit(R - r0, s - s[ref], np.exp(-dist / tau), lam)
            for rms in (0.01, 0.03, 0.06):
                r = r0 + rms * g / np.sqrt(np.mean(g ** 2))
                out[f'board tau{tau} lam{lam} rms{rms}'] = (float(auc(r, pub) - s[ref]), float(auc(r, ~pub) - auc(r0, ~pub)))
    for wb in (0.25, 0.5, 0.75):                              # merge the ladder top with the robust (unclimbed) base
        r = wb * r0 + (1 - wb) * base
        out[f'merge {wb:g} top + {1 - wb:g} base'] = (float(auc(r, pub) - s[ref]), float(auc(r, ~pub) - auc(r0, ~pub)))
    top3 = R[np.argsort(s)[-3:]].mean(0)                     # fusion of the three best scored files
    out['fusion top3'] = (float(auc(top3, pub) - s[ref]), float(auc(top3, ~pub) - auc(r0, ~pub)))
    out['base vs top (private)'] = float(auc(base, ~pub) - auc(r0, ~pub))
    sec = int(np.argsort(s)[-2])
    for k in (0.5, 1.0):
        r = r0 + k * (r0 - R[sec])
        out[f'extrapolate {k}'] = (float(auc(r, pub) - s[ref]), float(auc(r, ~pub) - auc(r0, ~pub)))
    res.append(out); print(json.dumps(out), flush=True)

keys = [k for k in res[0] if k not in ('seed',)]
summary = {}
for k in keys:
    v = np.array([r[k] for r in res])
    summary[k] = v.mean(0).tolist() if v.ndim > 1 else float(v.mean())
print('\nmean over seeds (public gain, private gain):')
for k, v in summary.items(): print(f'  {k:32s} {v}')
json.dump({'runs': res, 'summary': summary}, open(args.out, 'w'), indent=1)
