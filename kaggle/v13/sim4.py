"""Simulated leaderboard v4: which next step from the top of a dense ladder raises public, and what does it cost private?

Same fake board as sim2 (300k labelled train rows, 20% fake public, base = v8 OOF, 90-try greedy ladder with member moves,
row noise and board steps). The bank also gets 20 far files (merges of the base with single members), like our real bank.
From the bank's best file each method makes one candidate; we report its fake public and fake private change versus the
best file, averaged over seeds, and the share of seeds where public rises.
  fusion k          logit-rank mean of the k best scored files
  extrap k          best + k * (best - second best)
  local board       kernel ridge fitted only on files within an rms distance of the best, stepped by rms s
"""
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
for k in ('art', 'pin', 'train', 'davut'): ap.add_argument('--' + k, required=True)
ap.add_argument('--seeds', type=int, default=6); ap.add_argument('--tries', type=int, default=90)
ap.add_argument('--noise', type=float, default=0.5); ap.add_argument('--out', default='sim4_results.json')
args = ap.parse_args()
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
tr = pd.read_csv(args.train, usecols=['id', 'is_fraud']); y_all = tr.is_fraud.values.astype(int)
dv = pd.read_csv(args.davut).set_index('id').loc[tr.id].stack_all.values
wide = np.load(glob.glob(args.art + '/wide_lm_stack_nested.npy')[0])
M = [a for f in sorted(glob.glob(args.art + '/*_oof.npy')) + sorted(glob.glob(args.pin + '/oof_weni_*.npy'))
     for a in [np.load(f)] if a.ndim == 1 and len(a) == len(y_all) and np.isfinite(a).all()]

def board(R, s, ref, tau, lam, maxd=None):
    X = R - R[ref]; t = s - s[ref]; d = np.sqrt((X ** 2).mean(1))
    m = np.ones(len(s), bool) if maxd is None else d <= maxd
    X, t, d = X[m], t[m], d[m]; w = np.exp(-d / tau)
    K = X @ X.T / X.shape[1]
    a = np.linalg.solve(np.diag(w) @ K + lam * np.eye(len(t)), np.diag(w) @ t)
    g = X.T @ a / X.shape[1]
    return g / np.sqrt(np.mean(g ** 2) + 1e-18)

res = []
for seed in range(args.seeds):
    rng = np.random.default_rng(500 + seed)
    rows = rng.choice(len(y_all), 300000, replace=False); y = y_all[rows]; pub = rng.random(len(rows)) < 0.2
    L = np.array([lr(m[rows]) for m in M]); base = (lr(wide[rows]) + lr(dv[rows])) / 2
    auc = lambda r, m: roc_auc_score(y[m], r[m])
    R, s, cur = [base], [auc(base, pub)], base.copy()
    for it in range(1, args.tries + 1):
        if it % 10 == 0:
            ref = int(np.argmax(s)); cand = R[ref] + 0.03 * board(np.array(R), np.array(s), ref, 0.2, 1e-4)
        else:
            i, j = rng.choice(len(L), 2, replace=False); d = (L[i] - L[j]) / (L[i] - L[j]).std() + args.noise * rng.standard_normal(len(y))
            cand = cur + rng.choice([0.03, 0.06, 0.1]) * d / d.std()
        sc = auc(cand, pub); R.append(cand); s.append(sc)
        if sc > auc(cur, pub): cur = cand
    for i in rng.choice(len(L), 20, replace=False):           # far files, as in the real bank
        f = 0.5 * base + 0.5 * L[i]; R.append(f); s.append(auc(f, pub))
    R, s = np.array(R), np.array(s); order = np.argsort(s)[::-1]; ref = order[0]; top = R[ref]
    tp, tq = s[ref], auc(top, ~pub)
    C = {}
    for k in (2, 3, 5): C[f'fusion {k}'] = R[order[:k]].mean(0)
    for k in (0.25, 0.5): C[f'extrap {k}'] = top + k * (top - R[order[1]])
    for maxd in (None, 0.1, 0.05):
        for tau in (0.02, 0.05, 0.2):
            for lam in (1e-4, 1e-3):
                try: g = board(R, s, ref, tau, lam, maxd)
                except np.linalg.LinAlgError: continue
                for st in (0.005, 0.01, 0.02):
                    C[f'board maxd{maxd} tau{tau} lam{lam} rms{st}'] = top + st * g
    out = {k: (float(auc(r, pub) - tp), float(auc(r, ~pub) - tq)) for k, r in C.items()}
    res.append(out); print(seed, json.dumps({k: v for k, v in out.items() if not k.startswith('board')}), flush=True)
summ = {}
for k in res[0]:
    v = np.array([r[k] for r in res if k in r])
    summ[k] = {'public': float(v[:, 0].mean()), 'private': float(v[:, 1].mean()), 'public_up_share': float((v[:, 0] > 0).mean())}
best = sorted(summ.items(), key=lambda kv: -kv[1]['public'])
print('\nranked by mean fake-public change (x1e-4): public / private / share of seeds with public up')
for k, v in best[:20]: print(f"  {k:44s} {v['public']*1e4:+.2f} / {v['private']*1e4:+.2f} / {v['public_up_share']:.2f}")
for k in [k for k in summ if not k.startswith('board')]: print(f"  {k:44s} {summ[k]['public']*1e4:+.2f} / {summ[k]['private']*1e4:+.2f} / {summ[k]['public_up_share']:.2f}")
json.dump({'runs': res, 'summary': summ}, open(args.out, 'w'), indent=1)
