"""Screen a hard-example residual using existing out-of-fold predictions.

The development baseline predictions were produced by another three-fold fit;
the meta folds here are therefore not a fully nested cross-validation.
"""
import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def views(train, valid, base_train, base_valid):
    a, b = train.copy(), valid.copy()
    for col in a.select_dtypes(exclude=['number', 'bool']).columns:
        dtype = pd.CategoricalDtype(categories=pd.Index(a[col].dropna().unique()))
        a[col], b[col] = a[col].astype(dtype), b[col].astype(dtype)
    for frame, base in ((a, base_train), (b, base_valid)):
        frame['baseline_logit'] = logit(np.clip(base, 1e-8, 1 - 1e-8))
    return a, b


def correction(train, valid, y, base_train, base_valid):
    a, b = views(train, valid, base_train, base_valid)
    # Emphasize the two sides of hard ranking pairs without constructing all
    # positive-negative combinations. This is a heuristic, not a pairwise loss.
    weights = np.ones(len(y))
    weights[(y == 1) & train.cvv_retry_count.eq(0).to_numpy()] = 2.0
    threshold = np.quantile(base_train[y == 0], .90)
    weights[(y == 0) & (base_train >= threshold)] = 2.0
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=.03, num_leaves=4,
        min_child_samples=250, reg_lambda=20, n_jobs=3, deterministic=True,
        force_col_wise=True, verbosity=-1, random_state=42, boost_from_average=False)
    model.fit(a, y, sample_weight=weights, init_score=a['baseline_logit'].to_numpy())
    return model.predict(b, raw_score=True)


def score(y, p, cvv):
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-p, kind='stable')[:len(y)//20]] = True
    return {'auc': float(roc_auc_score(y, p)),
            'cvv0_auc': float(roc_auc_score(y[cvv], p[cvv])),
            'caught_top5': int(((y == 1) & flagged).sum()),
            'missed_cvv0_top5': int(((y == 1) & cvv & ~flagged).sum())}


def run(args):
    started = time.monotonic()
    data = pd.read_csv(args.train).set_index('id', verify_integrity=True)
    dev = np.load(args.development)
    hold = np.load(args.holdout)
    x = data.loc[dev['id']].drop(columns='is_fraud').reset_index(drop=True)
    h = data.loc[hold['id']].drop(columns='is_fraud').reset_index(drop=True)
    y, hy = dev['target'], hold['target']
    assert np.array_equal(data.loc[dev['id'], 'is_fraud'], y)
    assert np.array_equal(data.loc[hold['id'], 'is_fraud'], hy)
    base, hb = dev['baseline'], hold['baseline']
    logbase = logit(np.clip(base, 1e-8, 1-1e-8))
    weights = [0.05, 0.10, 0.20]
    results = {}
    for seed in (113, 937):
        split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        pred = np.empty(len(y))
        folds = np.empty(len(y), dtype=int)
        for fold, (tr, va) in enumerate(split.split(x, y)):
            pred[va] = correction(x.iloc[tr], x.iloc[va], y[tr], base[tr], base[va])
            folds[va] = fold
            print(f'seed={seed} fold={fold} elapsed={time.monotonic()-started:.0f}s', flush=True)
        results[str(seed)] = {}
        for weight in [0.0] + weights:
            p = expit(logbase + weight * pred)
            results[str(seed)][str(weight)] = {'overall': score(y, p, x.cvv_retry_count.eq(0).to_numpy()),
                'fold_auc': [float(roc_auc_score(y[folds == f], p[folds == f])) for f in range(3)]}
    # Selection uses only development. Zero is eligible, so a model with no
    # positive average development gain will not be proposed as an improvement.
    means = {str(w): np.mean([results[str(s)][str(w)]['overall']['auc'] for s in (113, 937)]) for w in [0.0] + weights}
    selected = max(means, key=means.get)
    hp = expit(logit(np.clip(hb, 1e-8, 1-1e-8)) + float(selected) * correction(x, h, y, base, hb)) if float(selected) else hb
    report = {'development': results, 'development_mean_auc': means, 'selected_weight': float(selected),
              'holdout_baseline': score(hy, hb, h.cvv_retry_count.eq(0).to_numpy()),
              'holdout_selected': score(hy, hp, h.cvv_retry_count.eq(0).to_numpy()),
              'limitations': ['Development meta folds are not nested with base-model OOF fits.',
                              'Holdout was previously inspected and cannot serve as a pristine final test.',
                              'Only the residual baseline is evaluated; the 0.95568 frontier blend lacks matched holdout predictions.'],
              'seconds': round(time.monotonic()-started, 1)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'holdout_selected.npz', id=hold['id'], target=hy, baseline=hb, selected=hp)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('train', 'development', 'holdout', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
