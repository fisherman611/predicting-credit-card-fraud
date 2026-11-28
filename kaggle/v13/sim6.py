"""Sim v6: does the board model pick better blocks to probe than random blocks?

Fake board as in sim v4: 300k labelled train rows, 20% fake public / 80% fake private, public scores rounded
to 5 decimals. A 90-try greedy ladder plus 20 far files form the bank; its best file is the start (our 0.95765).
Each method then probes blocks greedily (raise a block by +0.5 logit-rank, keep it only if the rounded public
score rises) and we record the change versus the start after 4, 14 and 24 submissions.
  A random      random blocks of 1000 from the start file's 0.2-0.8 rank band (the current plan)
  B board       band rows sorted by the local board gradient, cut into blocks of 1000, best first
  C board-sum   the random blocks of A, probed in order of their summed board gradient
  D adaptive    like B, but every probe is added to the bank and the board is refitted before the next one
  E board k500  like B with blocks of 500
"""
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit

ap = argparse.ArgumentParser()
for k in ('art', 'pin', 'train', 'davut'): ap.add_argument('--' + k, required=True)
ap.add_argument('--seeds', type=int, default=6); ap.add_argument('--tries', type=int, default=90)
ap.add_argument('--noise', type=float, default=0.5); ap.add_argument('--out', default='sim6.json')
args = ap.parse_args()
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
tr = pd.read_csv(args.train, usecols=['id', 'is_fraud']); y_all = tr.is_fraud.values.astype(int)
dv = pd.read_csv(args.davut).set_index('id').loc[tr.id].stack_all.values
wide = np.load((glob.glob(args.art + '/wide_lm_stack_nested.npy') + glob.glob(args.pin + '/wide_lm_stack_nested.npy'))[0])
M = [a for f in sorted(glob.glob(args.art + '/*_oof.npy')) + sorted(glob.glob(args.pin + '/oof_weni_*.npy'))
     for a in [np.load(f)] if a.ndim == 1 and len(a) == len(y_all) and np.isfinite(a).all()]
K_BLOCK, D, BUDGETS = 1000, 0.5, (4, 14, 24)

def board(R, s, ref, tau=0.2, lam=1e-4, maxd=0.1):
    X = R - R[ref]; t = s - s[ref]; d = np.sqrt((X ** 2).mean(1))
    m = d <= maxd
    if m.sum() < 4: m = np.ones(len(s), bool)
    X, t, d = X[m], t[m], d[m]; w = np.exp(-d / tau)
    a = np.linalg.solve(np.diag(w) @ (X @ X.T / X.shape[1]) + lam * np.eye(len(t)), np.diag(w) @ t)
    return X.T @ a / X.shape[1]

summary = {}
for seed in range(args.seeds):
    rng = np.random.default_rng(700 + seed)
    rows = rng.choice(len(y_all), 300000, replace=False); y = y_all[rows]; pub = rng.random(len(rows)) < 0.2
    yp, yq = y[pub], y[~pub]; pp, pq = yp.sum(), yq.sum()
    def auc_m(r, m, yy, p):
        rk = rankdata(r[m]); return (rk[yy == 1].sum() - p * (p + 1) / 2) / (p * (len(yy) - p))
    P = lambda r: auc_m(r, pub, yp, pp); Q = lambda r: auc_m(r, ~pub, yq, pq)
    L = np.array([lr(m[rows]) for m in M]); base = (lr(wide[rows]) + lr(dv[rows])) / 2
    R, s, cur = [base], [round(P(base), 5)], base.copy()
    for it in range(1, args.tries + 1):
        if it % 10 == 0:
            ref = int(np.argmax(s)); g = board(np.array(R), np.array(s), ref)
            cand = R[ref] + 0.03 * g / np.sqrt(np.mean(g ** 2) + 1e-18)
        else:
            i, j = rng.choice(len(L), 2, replace=False); dd = (L[i] - L[j]) / (L[i] - L[j]).std() + args.noise * rng.standard_normal(len(y))
            cand = cur + rng.choice([0.03, 0.06, 0.1]) * dd / dd.std()
        sc = round(P(cand), 5); R.append(cand); s.append(sc)
        if sc > round(P(cur), 5): cur = cand
    for i in rng.choice(len(L), 20, replace=False):
        f = 0.5 * base + 0.5 * L[i]; R.append(f); s.append(round(P(f), 5))
    R, s = np.array(R), np.array(s); ref = int(np.argmax(s)); start = lr(R[ref]); s0, q0 = P(start), Q(start)
    fr = (rankdata(start) - 0.5) / len(start); band = np.flatnonzero((fr >= 0.2) & (fr < 0.8))
    g0 = board(R, s, ref)
    perm = band.copy(); rng.shuffle(perm)
    rand_blocks = [perm[i:i + K_BLOCK] for i in range(0, len(perm) - K_BLOCK + 1, K_BLOCK)]
    sorted_band = band[np.argsort(-g0[band])]
    plans = {
        'A random': rand_blocks[:max(BUDGETS)],
        'B board': [sorted_band[i * K_BLOCK:(i + 1) * K_BLOCK] for i in range(max(BUDGETS))],
        'C board-sum': sorted(rand_blocks, key=lambda b: -g0[b].sum())[:max(BUDGETS)],
        'E board k500': [sorted_band[i * 500:(i + 1) * 500] for i in range(max(BUDGETS))],
    }
    out = {}
    for name, blocks in list(plans.items()) + [('D adaptive', None), ('F board+reboost', 'F'), ('G board+halves', 'G')]:
        x, cur_s = start.copy(), round(s0, 5); used = np.zeros(len(y), bool)
        Rb, sb = list(R), list(s); queue, nxt, boosted = [], 0, set()
        for n in range(1, max(BUDGETS) + 1):
            if isinstance(blocks, str):
                if queue: b = queue.pop(0)
                else: b = plans['B board'][nxt] if nxt < max(BUDGETS) else sorted_band[nxt * K_BLOCK:(nxt + 1) * K_BLOCK]; nxt += 1
            elif blocks is None:
                g = board(np.array(Rb), np.array(sb), int(np.argmax(sb))) if n > 1 else g0
                c = band[~used[band]]; b = c[np.argsort(-g[c])][:K_BLOCK]
            else:
                b = blocks[n - 1]
            used[b] = True
            t = x.copy(); t[b] += D; sc = round(P(t), 5)
            Rb.append(t); sb.append(sc)
            if sc > cur_s:
                x, cur_s = t, sc
                key = (int(b[0]), len(b))
                if blocks == 'F' and key not in boosted: boosted.add(key); queue.append(b)
                if blocks == 'G' and len(b) >= 500: queue += [b[:len(b) // 2], b[len(b) // 2:]]
            if n in BUDGETS: out[f'{name} @{n}'] = (P(x) - s0, Q(x) - q0)
    print(seed, {k: (round(v[0] * 1e4, 2), round(v[1] * 1e4, 2)) for k, v in out.items() if k.endswith('@24')}, flush=True)
    for k, v in out.items(): summary.setdefault(k, []).append(v)
res = {}
print('\nmethod @ submissions: fake public / fake private change vs start (x1e-4), share of seeds with public up')
for k, v in summary.items():
    v = np.array(v); res[k] = {'public': float(v[:, 0].mean()), 'private': float(v[:, 1].mean()),
                               'public_up_share': float((v[:, 0] > 0).mean()), 'public_std': float(v[:, 0].std())}
    print(f"  {k:20s} {res[k]['public']*1e4:+6.2f} ± {res[k]['public_std']*1e4:.2f} / {res[k]['private']*1e4:+6.2f} / {res[k]['public_up_share']:.2f}")
json.dump({'summary': res, 'runs': summary}, open(args.out, 'w'), indent=1)
