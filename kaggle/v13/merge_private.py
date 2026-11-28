"""Private-first finals: w * lr(top ladder file) + (1 - w) * lr(v8), zero-fraud zones at the bottom.

usage: python kaggle/v13/merge_private.py TEST.csv TOP.csv V8.csv OUTDIR w1,w2,...
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v12'))
from final import zones  # noqa: E402

test_p, top_p, v8_p, out, ws = sys.argv[1:6]
lr = lambda v: logit((rankdata(v) - 0.5) / len(v))
test = pd.read_csv(test_p)
ids, z = test.id.to_numpy(), zones(test)
v8 = pd.read_csv(v8_p).set_index('id').loc[ids].is_fraud.to_numpy()
top = pd.read_csv(top_p).set_index('id').loc[ids].is_fraud.to_numpy()
for w in map(float, ws.split(',')):
    m = w * lr(top) + (1 - w) * lr(v8)
    m[z] -= 100
    p = (rankdata(m) - 0.5) / len(m)
    path = Path(out) / f'priv{int(round(w * 100))}_step1_v8.csv'
    pd.DataFrame({'id': ids, 'is_fraud': p}).to_csv(path, index=False)
    print(path.name, 'spearman top', round(spearmanr(p, top).correlation, 5), 'v8', round(spearmanr(p, v8).correlation, 5),
          'zones max rank', round(p[z].max(), 4), 'zone rows', int(z.sum()))
