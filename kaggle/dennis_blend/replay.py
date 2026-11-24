"""Reproduce Dennis's 12% clipped rank residual over public Step AB."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(root, name, expected):
    found = [p for p in root.rglob(name) if p.is_file() and sha(p) == expected]
    if not found:
        raise ValueError(f'No checksum-matched {name} under {root}')
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    sources = {
        'step_ab.csv': '3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038',
        'clipped.csv': '3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c'}
    a, b = [pd.read_csv(locate(args.input, name, digest)) for name, digest in sources.items()]
    ids = np.arange(699960, 999944)
    for frame in (a, b):
        if list(frame.columns) != ['id', 'is_fraud'] or not np.array_equal(frame.id, ids):
            raise ValueError('Incorrect columns or test IDs')
        if not np.isfinite(frame.is_fraud).all() or not frame.is_fraud.between(0, 1).all():
            raise ValueError('Incorrect prediction values')
    n = len(a)
    l = lambda v: logit((rankdata(v)-.5)/n)
    left, right = l(a.is_fraud), l(b.is_fraud)
    gap = right-left
    cap = float(np.quantile(np.abs(gap), .985))
    ranking = rankdata(left + .12*np.clip(gap, -cap, cap))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'submission.csv'
    a.assign(is_fraud=(ranking-.5)/n).to_csv(path, index=False)
    actual = sha(path)
    # The author has since updated their notebook to a different 9% blend;
    # this is our frozen checksum for the earlier 12% recipe.
    expected = '7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c'
    if actual != expected:
        raise ValueError(f'Output checksum mismatch: {actual}')
    displacement = np.abs(ranking-rankdata(a.is_fraud))
    report = {'rows': n, 'sha256': actual, 'matches_author': None, 'cap': cap,
              'median_rank_displacement': float(np.median(displacement)),
              'top5_new_rows': int(((ranking > .95*n) & (rankdata(a.is_fraud) <= .95*n)).sum())}
    (args.output/'replay.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
