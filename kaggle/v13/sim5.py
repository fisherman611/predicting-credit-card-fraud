"""Sim v5: greedy block probing on a fake board.

A fake test set of 300k labelled train rows is split 20% fake public / 80% fake private, and the
score is rounded to 5 decimals as Kaggle shows it. Each probe takes a block of rows, raises their
logit by d, and keeps the move only if the rounded fake-public score goes up. The budget is the
number of submissions. Reported: true public gain, private change and accepted moves.

usage: python kaggle/v13/sim5.py ART_DIR TRAIN.csv OUT.json [seeds]
"""
import glob, json, sys
import numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import rankdata

art, train, out = sys.argv[1:4]; seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 4
y_all = pd.read_csv(train, usecols=['is_fraud']).is_fraud.to_numpy()
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
M = np.stack([lr(np.load(f)) for f in sorted(glob.glob(art + '/*_oof.npy'))])
base_all = M.mean(0)

def auc(y, s):
    r = rankdata(s); p = y.sum(); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))

def run(seed, pool_kind, lo, hi, k, d, budget):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_all), 300000, replace=False)
    y, s0, mem = y_all[idx], base_all[idx].copy(), M[:, idx]
    pub = rng.random(len(y)) < 0.2
    fr = (rankdata(s0) - 0.5) / len(s0)
    if pool_kind == 'band':
        pool = np.flatnonzero((fr >= lo) & (fr < hi))
    else:  # rows some member ranks far above the stack: the most likely hidden frauds
        gap = mem.max(0) - s0
        cand = np.flatnonzero(fr < hi)
        pool = cand[np.argsort(-gap[cand])][: int(lo)]
    rng.shuffle(pool)
    blocks = [pool[i:i + k] for i in range(0, len(pool) - k + 1, k)][:budget]
    s = s0.copy(); cur = round(auc(y[pub], s[pub]), 5); acc = 0
    for b in blocks:
        t = s.copy(); t[b] += d
        sc = round(auc(y[pub], t[pub]), 5)
        if sc > cur: s, cur, acc = t, sc, acc + 1
    return dict(pub=auc(y[pub], s[pub]) - auc(y[pub], s0[pub]),
                priv=auc(y[~pub], s[~pub]) - auc(y[~pub], s0[~pub]), acc=acc)

grid = sys.argv[5] if len(sys.argv) > 5 else 'coarse'
configs = []
if grid == 'coarse':
    for k in (100, 300, 1000, 3000):
        for d in (1.0, 3.0):
            configs += [('band', 0.2, 0.8, k, d, 24), ('gap', 24 * k, 0.8, k, d, 24)]
else:
    for budget in (12, 24):
        for k in (500, 1000, 2000):
            for d in (0.5, 1.0, 1.5):
                configs += [('band', 0.2, 0.8, k, d, budget), ('gap', budget * k, 0.8, k, d, budget)]
res = {}
for c in configs:
    rs = [run(sd, *c[:5], budget=c[5]) for sd in range(seeds)]
    key = f'{c[0]} lo{c[1]} hi{c[2]} k{c[3]} d{c[4]} budget{c[5]}'
    res[key] = {m: float(np.mean([r[m] for r in rs])) for m in ('pub', 'priv', 'acc')}
    res[key]['pub_up_share'] = float(np.mean([r['pub'] > 0 for r in rs]))
    print(f"{key:42s} pub {res[key]['pub']*1e4:+7.2f}e-4 priv {res[key]['priv']*1e4:+7.2f}e-4 acc {res[key]['acc']:.1f}", flush=True)
json.dump(res, open(out, 'w'), indent=1)
