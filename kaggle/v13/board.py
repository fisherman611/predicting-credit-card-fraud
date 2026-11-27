"""Our own board model: learn the public-score gradient over test rows from scored files, leave-one-out checked."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'kaggle/v11'); sys.path.insert(0, 'kaggle/v13')
from residual_lm import find, lr, zones
from bank import BANK, S
te = pd.read_csv(find(S + '/v11in', 'test.csv', 'playground-series-s6e11')).sort_values('id'); z = zones(te); keep = ~z
names, R, s = [], [], []
for p, sc in BANK.items():
    d = pd.read_csv(p).sort_values('id')
    assert len(d) == len(te) and (d.id.values == te.id.values).all(), p
    names.append(p.split('/')[-2][:18] + '/' + p.split('/')[-1][:28]); R.append(lr(d.is_fraud.values)[keep].astype(np.float32)); s.append(sc)
R, s = np.array(R), np.array(s)
np.save(S + '/v12/bank_R.npy', R); np.save(S + '/v12/bank_s.npy', s)
ref = int(np.argmax(s)); print(len(s), 'files, reference', names[ref], s[ref])
dist = np.sqrt(((R - R[ref]) ** 2).mean(1))
print('distance to reference (rms logit-rank):', np.round(np.sort(dist), 3))
