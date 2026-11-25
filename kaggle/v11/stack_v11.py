"""v11 pipeline step: add a residual language-model score to v8, fold-honestly.

Reads the per-fold outputs of residual_lm.py (v11/{model}_f{k}_val.npy, _test.npy, fold{k}.npz) for
one model, from any number of kernel output directories under --input. With all five folds present:

  t_k   v8 rank-logit, Platt-scaled on the fit rows of fold k (the offset the model was trained on)
  s_k   residual score of the model trained on fold k's fit rows, on fold k's val rows
  final score on fold k = t_k + beta_k * s_k - 100 * zone, beta_k picked on the other four folds

Test: mean over folds of t_k(test) + beta * s_k(test), beta = median of the beta_k. The output replaces
v8 only if the residual wins on at least 4 of 5 folds; otherwise stack_v11.csv is plain v8.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from residual_lm import BETAS, find, v8_scores, zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--model', default='laya')
    args = ap.parse_args()
    train = pd.read_csv(find(args.input, 'train.csv', 'playground-series-s6e11'))
    test = pd.read_csv(find(args.input, 'test.csv', 'playground-series-s6e11'))
    y = train.is_fraud.to_numpy().astype(int)
    z_tr, z_te = zones(train), zones(test)
    v8_oof, v8_test = v8_scores(args.input, train, test)

    folds = {}
    for k in range(5):
        try:
            fd = np.load(find(args.input, f'fold{k}.npz'))
            s_val = np.nan_to_num(np.load(find(args.input, f'{args.model}_f{k}_val.npy')))
            s_test = np.nan_to_num(np.load(find(args.input, f'{args.model}_f{k}_test.npy')))
        except FileNotFoundError:
            continue
        folds[k] = dict(val=fd['val'], t=fd['t_val'], s=s_val, t_test=fd['t_test'], s_test=s_test)
    print('folds found', sorted(folds), flush=True)

    grid = (0.0,) + BETAS

    def gains(k):
        f = folds[k]
        yv, zv = y[f['val']], z_tr[f['val']]
        base = roc_auc_score(yv, f['t'] - 100 * zv)
        return np.array([roc_auc_score(yv, f['t'] + b * f['s'] - 100 * zv) - base for b in grid])

    table = {k: gains(k) for k in folds}
    report = {'model': args.model, 'folds': sorted(folds), 'betas': list(grid), 'per_fold': {}}
    picked = []
    for k in folds:
        others = [table[j] for j in folds if j != k]
        b_i = int(np.argmax(np.mean(others, 0))) if others else 0
        picked.append(grid[b_i])
        report['per_fold'][k] = {'beta': grid[b_i], 'gain': float(table[k][b_i]),
                                 'gain_by_beta': [float(g) for g in table[k]]}
    wins = sum(v['gain'] > 0 for v in report['per_fold'].values())
    beta = float(np.median(picked)) if picked else 0.0
    use = len(folds) == 5 and wins >= 4
    report.update({'wins': int(wins), 'beta': beta, 'mean_gain': float(np.mean(
        [v['gain'] for v in report['per_fold'].values()])) if folds else 0.0, 'replaces_v8': bool(use)})

    if use:
        final = np.mean([f['t_test'] + beta * f['s_test'] for f in folds.values()], 0)
    else:
        final = v8_test.copy()
    final = final - 100 * z_te
    pd.DataFrame({'id': test.id, 'is_fraud': (np.argsort(np.argsort(final)) + 0.5) / len(final)}).to_csv(
        args.output / 'stack_v11.csv', index=False)
    (args.output / 'stack_v11_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
