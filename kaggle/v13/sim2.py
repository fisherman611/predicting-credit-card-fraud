"""Simulated leaderboard v2: a ladder that mixes real member moves and public-fitted board steps, then candidate finals.

Fake test = 300k labelled train rows, 20% fake public. The honest base is v8's OOF (mean of the wide LM stack and Davut's
stack_all, rank-logit). The ladder starts at the base and makes 90 tries, keeping a try only when fake public rises:
  member move  a small step along (member a - member b): moves that carry real model signal
  board step   every 10th try, our kernel-ridge board model is fitted on all scored tries and the file is moved
               along its gradient (row-specific, fitted to public only)
Candidate finals are scored on fake private relative to the base:
  top, merges of top with base, fusion of the 3 best, and 'projection': base + the part of (top - base) that a ridge
  regression on the 33 members' test predictions can explain (no labels used), plus merges of it.
"""
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ('art', 'pin', 'train', 'davut'): ap.add_argument('--' + k, required=True)
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--tries', type=int, default=90)
ap.add_argument('--out', default='sim2_results.json')
ap.add_argument('--scenario', default='v8', choices=['v8', 'wide'],
                help='v8: base is v8, the ladder has no extra signal; wide: base is the wide stack only and Davut\'s stack is a member the ladder can find')
ap.add_argument('--noise', type=float, default=0.5, help='row-specific noise in each ladder move (public-only fitting)')
args = ap.parse_args()
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
tr = pd.read_csv(args.train, usecols=['id', 'is_fraud']); y_all = tr.is_fraud.values.astype(int)
dv = pd.read_csv(args.davut).set_index('id').loc[tr.id].stack_all.values
wide = np.load(glob.glob(args.art + '/wide_lm_stack_nested.npy')[0])
M = []
for f in sorted(glob.glob(args.art + '/*_oof.npy')) + sorted(glob.glob(args.pin + '/oof_weni_*.npy')):
    a = np.load(f)
    if a.ndim == 1 and len(a) == len(y_all) and np.isfinite(a).all(): M.append(a)
if args.scenario == 'wide': M.append(dv)
print(len(M), 'members', args.scenario, flush=True)

def board(R, s, ref, tau=0.2, lam=1e-4):
    X = R - R[ref]; t = s - s[ref]; w = np.exp(-np.sqrt((X ** 2).mean(1)) / tau)
    K = X @ X.T / X.shape[1]
    a = np.linalg.solve(np.diag(w) @ K + lam * np.eye(len(t)), np.diag(w) @ t)
    g = X.T @ a / X.shape[1]
    return g / np.sqrt(np.mean(g ** 2) + 1e-18)

res = []
for seed in range(args.seeds):
    rng = np.random.default_rng(100 + seed)
    rows = rng.choice(len(y_all), 300000, replace=False); y = y_all[rows]; pub = rng.random(len(rows)) < 0.2
    L = np.array([lr(m[rows]) for m in M])
    base = (lr(wide[rows]) + lr(dv[rows])) / 2 if args.scenario == 'v8' else lr(wide[rows])
    auc = lambda r, m: roc_auc_score(y[m], r[m])
    R, s, cur = [base], [auc(base, pub)], base.copy()
    for it in range(1, args.tries + 1):
        if it % 10 == 0:
            ref = int(np.argmax(s)); cand = R[ref] + 0.03 * board(np.array(R), np.array(s), ref)
        else:
            i, j = rng.choice(len(L), 2, replace=False); d = (L[i] - L[j]) / (L[i] - L[j]).std() + args.noise * rng.standard_normal(len(y))
            cand = cur + rng.choice([0.03, 0.06, 0.1]) * d / d.std()
        sc = auc(cand, pub); R.append(cand); s.append(sc)
        if sc > auc(cur, pub): cur = cand
    R, s = np.array(R), np.array(s); top = R[int(np.argmax(s))]
    Lc = L - L.mean(1, keepdims=True); D = top - base
    cands = {'top': top}
    for w in (0.25, 0.5, 0.75): cands[f'merge {w:g} top'] = w * top + (1 - w) * base
    cands['fusion top3'] = R[np.argsort(s)[-3:]].mean(0)
    for alpha in (1e-2, 1.0, 100.0):
        beta = np.linalg.solve(Lc @ Lc.T / Lc.shape[1] + alpha / Lc.shape[1] * np.eye(len(L)), Lc @ D / Lc.shape[1])
        P = Lc.T @ beta
        cands[f'projection a{alpha:g}'] = base + P
        cands[f'projection a{alpha:g} (R2 {1 - np.var(D - P) / np.var(D):.2f}) merge 0.5'] = base + 0.5 * P
    b_pub, b_pri = auc(base, pub), auc(base, ~pub)
    out = {'seed': seed, 'scenario': args.scenario, 'kept_steps': int(sum(1 for i in range(1, len(s)) if s[i] > max(s[:i]))),
           **{k: (float(auc(r, pub) - b_pub), float(auc(r, ~pub) - b_pri)) for k, r in cands.items()}}
    res.append(out); print(json.dumps(out), flush=True)
summ = {}
for k in res[0]:
    if k in ('seed', 'scenario'): continue
    try: summ[k] = np.mean([r[k] for r in res], 0).tolist()
    except Exception: pass
print('\nmean over seeds, gain over the honest base (fake public, fake private):')
for k, v in summ.items(): print(f'  {k:44s} {v}')
json.dump({'runs': res, 'summary': summ}, open(args.out, 'w'), indent=1)
