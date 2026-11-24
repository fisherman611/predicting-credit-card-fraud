"""v8: does extra training data help? Fold-clean test of two augmentations.

base      4-leaf LightGBM on raw columns, key counts and in-fold target encodings.
orig      base + the 20,000 labelled rows of the original dataset
          (uditjain13/credit-card-fraud-detection-2026) in the fit set, with an is_orig flag.
pseudo    knowledge distillation / soft pseudo-labels: the base model of this fold
          (never trained on the scored fold) predicts the 299,984 test rows; a student
          is trained on the real fit rows plus those test rows with the teacher's
          probabilities as soft targets (objective cross_entropy), weight --pseudo-weight.

Only rows of the competition train set are scored, on StratifiedKFold(5, seed 2027).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v6'))
from base import build_keys, find, raw_numeric  # noqa: E402

SEED = 2027
PARAMS = dict(objective='cross_entropy', learning_rate=0.05, num_leaves=4, min_child_samples=200,
              feature_fraction=0.4, bagging_fraction=0.85, bagging_freq=1, lambda_l2=5.0,
              num_threads=4, verbose=-1, seed=SEED, deterministic=True, force_col_wise=True)


def fit_predict(X, y, w, X_apply, seed, rounds=None):
    """Early-stop on a 10% split of the real (unweighted, hard-label) rows only."""
    real = np.flatnonzero(w == 1.0) if w is not None else np.arange(len(y))
    rng = np.random.RandomState(seed)
    es = rng.choice(real, len(real) // 10, replace=False)
    mask = np.ones(len(y), bool)
    mask[es] = False
    dtr = lgb.Dataset(X[mask], y[mask], weight=None if w is None else w[mask])
    des = lgb.Dataset(X[es], y[es])
    p = dict(PARAMS, seed=seed)
    model = lgb.train(p, dtr, rounds or 4000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(150, verbose=False)] if rounds is None else [])
    return [model.predict(A) for A in X_apply], model.best_iteration or rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--pseudo-weight', type=float, default=1.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    orig = pd.read_csv(find(args.input, 'credit_card_fraud_2026.csv'))
    y = train.pop('is_fraud').to_numpy().astype(float)
    y_orig = orig.pop('is_fraud').to_numpy().astype(float)
    cols = [c for c in train.columns if c != 'id']
    n, m, k = len(train), len(test), len(orig)
    full = pd.concat([train[cols], test[cols], orig[cols]], ignore_index=True)
    keys, counts = build_keys(full)
    dense = np.hstack([raw_numeric(full).values, counts.values]).astype(np.float32)
    is_orig = np.r_[np.zeros(n + m), np.ones(k)].astype(np.float32)[:, None]
    idx_test = np.arange(n, n + m)
    idx_orig = np.arange(n + m, n + m + k)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(n), y))
    cvv0 = train.cvv_retry_count.to_numpy() == 0
    print(f'ready {time.time() - t0:.0f}s', flush=True)

    def encoded(fit_rows, y_fit, apply_rows):
        te = TargetEncoder(target_type='binary', smooth=20.0, cv=3, shuffle=True, random_state=SEED)
        a = te.fit_transform(keys.iloc[fit_rows], (y_fit > 0.5).astype(int)).astype(np.float32)
        return a, [te.transform(keys.iloc[r]).astype(np.float32) for r in apply_rows]

    rows = []
    for f in args.folds:
        fit, val = folds[f]
        ts = time.time()
        res = {'fold': f}
        # base
        te_fit, (te_val, te_test) = encoded(fit, y[fit], [val, idx_test])
        X_fit = np.hstack([dense[fit], is_orig[fit], te_fit])
        X_val = np.hstack([dense[val], is_orig[val], te_val])
        X_test = np.hstack([dense[idx_test], is_orig[idx_test], te_test])
        (p_val, p_test), it = fit_predict(X_fit, y[fit], None, [X_val, X_test], SEED + f)
        res['base'], res['base_iters'] = roc_auc_score(y[val], p_val), it
        res['base_cvv0'] = roc_auc_score(y[val][cvv0[val]], p_val[cvv0[val]])
        # orig augmentation: original rows join the fit set, including the encoder fit
        rows_fo = np.r_[fit, idx_orig]
        y_fo = np.r_[y[fit], y_orig]
        te_fo, (te_val2,) = encoded(rows_fo, y_fo, [val])
        X_fo = np.hstack([dense[rows_fo], is_orig[rows_fo], te_fo])
        X_val2 = np.hstack([dense[val], is_orig[val], te_val2])
        (p_val2,), it2 = fit_predict(X_fo, y_fo, np.r_[np.ones(len(fit)), np.ones(k)], [X_val2], SEED + 100 + f)
        res['orig'], res['orig_iters'] = roc_auc_score(y[val], p_val2), it2
        res['orig_cvv0'] = roc_auc_score(y[val][cvv0[val]], p_val2[cvv0[val]])
        # soft pseudo-labels on test rows from this fold's teacher (encoder unchanged)
        X_ps = np.vstack([X_fit, X_test])
        y_ps = np.r_[y[fit], p_test]
        w_ps = np.r_[np.ones(len(fit)), np.full(m, args.pseudo_weight)]
        (p_val3,), it3 = fit_predict(X_ps, y_ps, w_ps, [X_val], SEED + 200 + f)
        res['pseudo'], res['pseudo_iters'] = roc_auc_score(y[val], p_val3), it3
        res['pseudo_cvv0'] = roc_auc_score(y[val][cvv0[val]], p_val3[cvv0[val]])
        res['seconds'] = round(time.time() - ts)
        rows.append(res)
        print(json.dumps(res), flush=True)

    summary = {'folds': rows}
    for v in ('orig', 'pseudo'):
        d = [r[v] - r['base'] for r in rows]
        summary[v + '_minus_base'] = {'mean': float(np.mean(d)), 'per_fold': d, 'wins': int(sum(x > 0 for x in d))}
    (args.output / 'augment_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != 'folds'}, indent=2))


if __name__ == '__main__':
    main()
