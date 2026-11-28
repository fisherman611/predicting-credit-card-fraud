"""Block probe on the real board (the method sim v5 ranked best).

Pool: rows whose rank fraction in the BASE file is in [0.2, 0.8), shuffled once with a fixed seed and cut
into blocks of 1000. Probe j raises block j by +0.5 logit-rank on top of the CURRENT best file; keep it
only if its public score beats the current best.

usage: python kaggle/v13/probe.py BASE.csv CURRENT.csv j OUT.csv
"""
import sys
import numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import rankdata

base_p, cur_p, j, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
K, D, LO, HI, SEED = 1000, 0.5, 0.2, 0.8, 611
base = pd.read_csv(base_p).sort_values('id').reset_index(drop=True)
cur = pd.read_csv(cur_p).set_index('id').loc[base.id].is_fraud.to_numpy()
fr = (rankdata(base.is_fraud) - 0.5) / len(base)
pool = np.flatnonzero((fr >= LO) & (fr < HI))
np.random.default_rng(SEED).shuffle(pool)
block = pool[j * K:(j + 1) * K]
r = logit((rankdata(cur) - 0.5) / len(cur))
r[block] += D
pd.DataFrame({'id': base.id, 'is_fraud': (rankdata(r) - 0.5) / len(r)}).to_csv(out, index=False)
print(out, 'block', j, 'rows', len(block), 'pool', len(pool))
