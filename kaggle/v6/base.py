"""v6 base members trained on the stack folds (StratifiedKFold(5, seed 2027)).

B1  glmboost   4-leaf LightGBM started from an inner-cross-fitted GLM logit,
               on target encodings at smoothing 5/20/100 plus joint keys and key counts.
B2  cvv0       same design, rows with cvv_retry_count == 0 weighted x3 so the
               trees spend capacity where 83% of top-5% misses sit.
B3  aggregate  B1 plus target-free group aggregates (count / mean / std / z-score
               of amount and balance within categorical combinations).

Each member writes oof_<name>.npy (699,960) and test_<name>.npy (299,984) in
competition id order. results.json records overall and CVV=0 AUC per fold.
"""
import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

SEED = 2027
THREADS = min(4, os.cpu_count() or 1)
NUMERIC_DIGITS = ('amount_usd', 'hours_since_last_txn', 'distance_from_home_km',
                  'account_balance_usd', 'velocity_score', 'merchant_risk_score')
CATEGORICAL = ('merchant_category', 'card_type', 'auth_method', 'channel', 'device_type')
FLAGS = ('is_foreign_transaction', 'is_new_merchant', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt')
JOINT = (('merchant_category', 'channel'), ('auth_method', 'device_type'), ('card_type', 'channel'),
         ('merchant_category', 'auth_method'), ('cvv_retry_count', 'prior_disputes'),
         ('cvv_retry_count', 'auth_method'), ('channel', 'device_type'), ('time_of_day_hour', 'day_of_week'))
GROUPS = (('merchant_category', 'channel', 'auth_method'), ('card_type', 'device_type'),
          ('merchant_category', 'time_of_day_hour'), ('cvv_retry_count', 'prior_disputes', 'channel'))
SMOOTHING = (5.0, 20.0, 100.0)


def find(root, name):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def digit_keys(frame):
    keys = {}
    for name in NUMERIC_DIGITS:
        cents = np.rint(100 * frame[name].to_numpy(np.float64)).astype(np.int64)
        mag = np.abs(cents)
        whole = mag // 100
        length = np.searchsorted(10 ** np.arange(1, 13), whole) + 1
        lead = whole // (10 ** np.maximum(0, length - 2))
        keys.update({f'{name}__exact': cents, f'{name}__whole': cents // 100, f'{name}__tenth': cents // 10,
                     f'{name}__fraction': mag % 100, f'{name}__whole_tail': whole % 100,
                     f'{name}__scale_lead': length * 100 + lead,
                     f'{name}__frac_scale': 100 * length + mag % 100})
    return pd.DataFrame(keys, index=frame.index)


def build_keys(full):
    """Integer-coded categorical keys over train+test (no target involved)."""
    cols = {}
    for name in full.columns:
        cols[name] = full[name]
    keys = pd.concat([pd.DataFrame(cols), digit_keys(full)], axis=1)
    for a, b in JOINT:
        keys[f'{a}*{b}'] = full[a].astype(str) + '|' + full[b].astype(str)
    coded = pd.DataFrame({c: pd.factorize(keys[c])[0].astype(np.int32) for c in keys.columns}, index=full.index)
    counts = pd.DataFrame({f'cnt__{c}': np.log1p(coded[c].map(coded[c].value_counts()).to_numpy(np.float32))
                           for c in coded.columns}, index=full.index)
    return coded, counts


def build_aggregates(full):
    out = {}
    for g in GROUPS:
        key = full[list(g)].astype(str).agg('|'.join, axis=1)
        tag = '+'.join(g)
        out[f'agg_n__{tag}'] = np.log1p(key.map(key.value_counts()).to_numpy(np.float32))
        for col in ('amount_usd', 'account_balance_usd', 'velocity_score', 'merchant_risk_score'):
            grp = full[col].groupby(key)
            mean, std = key.map(grp.mean()), key.map(grp.std()).fillna(0) + 1e-6
            out[f'agg_z__{tag}__{col}'] = ((full[col] - mean) / std).to_numpy(np.float32)
            out[f'agg_rk__{tag}__{col}'] = grp.rank(pct=True).to_numpy(np.float32)
    return pd.DataFrame(out, index=full.index)


def raw_numeric(full):
    raw = full.copy()
    for c in CATEGORICAL:
        raw[c] = pd.factorize(raw[c])[0]
    for c in FLAGS:
        raw[c] = raw[c].astype(np.float32)
    return raw.astype(np.float32)


def glm_logit(keys_fit, y_fit, keys_apply):
    enc = OneHotEncoder(min_frequency=20, handle_unknown='infrequent_if_exist', dtype=np.float32)
    m = enc.fit_transform(keys_fit)
    glm = LogisticRegression(C=0.1, max_iter=300, tol=1e-5)
    glm.fit(m, y_fit)
    return [glm.decision_function(enc.transform(k)) for k in keys_apply]


def fold_glm(keys, y, fit, apply_sets, seed):
    """Inner 3-fold cross-fitted GLM logit for fit rows; full-fit logit for the others."""
    init_fit = np.zeros(len(fit))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(fit, y[fit]):
        init_fit[b] = glm_logit(keys.iloc[fit[a]], y[fit[a]], [keys.iloc[fit[b]]])[0]
    others = glm_logit(keys.iloc[fit], y[fit], [keys.iloc[s] for s in apply_sets])
    return init_fit, others


def target_encode(keys, y, fit, apply_sets, seed):
    fit_parts, other_parts = [], [[] for _ in apply_sets]
    for m in SMOOTHING:
        te = TargetEncoder(target_type='binary', smooth=m, cv=3, shuffle=True, random_state=seed)
        fit_parts.append(te.fit_transform(keys.iloc[fit], y[fit]).astype(np.float32))
        for i, s in enumerate(apply_sets):
            other_parts[i].append(te.transform(keys.iloc[s]).astype(np.float32))
    return np.hstack(fit_parts), [np.hstack(p) for p in other_parts]


PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=4, min_child_samples=200,
              feature_fraction=0.4, bagging_fraction=0.85, bagging_freq=1, lambda_l2=5.0,
              num_threads=THREADS, verbose=-1, seed=SEED, deterministic=True, force_col_wise=True)


def boost(Xf, yf, init_f, wf, X_apply, init_apply, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(yf))
    k = len(idx) // 10
    tr_i, es_i = idx[k:], idx[:k]
    dtr = lgb.Dataset(Xf[tr_i], yf[tr_i], init_score=init_f[tr_i], weight=None if wf is None else wf[tr_i])
    des = lgb.Dataset(Xf[es_i], yf[es_i], init_score=init_f[es_i], weight=None if wf is None else wf[es_i])
    model = lgb.train(PARAMS, dtr, 4000, valid_sets=[des], callbacks=[lgb.early_stopping(200, verbose=False)])
    return [model.predict(X, raw_score=True) + i0 for X, i0 in zip(X_apply, init_apply)], model.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--folds', type=int, nargs='*', default=None, help='subset of folds (debug)')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    y = train.pop('is_fraud').to_numpy().astype(int)
    n = len(train)
    full = pd.concat([train.drop(columns='id'), test.drop(columns='id')], ignore_index=True)
    keys, counts = build_keys(full)
    aggs = build_aggregates(full)
    raw = raw_numeric(full)
    print(f'features ready {time.time()-t0:.0f}s keys={keys.shape[1]} aggs={aggs.shape[1]}', flush=True)
    dense_tr = np.hstack([raw.values[:n], counts.values[:n]])
    dense_te = np.hstack([raw.values[n:], counts.values[n:]])
    agg_tr, agg_te = aggs.values[:n], aggs.values[n:]
    cvv0 = (train.cvv_retry_count.to_numpy() == 0)

    names = ('glmboost', 'cvv0', 'aggregate')
    oof = {k: np.full(n, np.nan) for k in names}
    tst = {k: np.zeros(len(test)) for k in names}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(n), y))
    use = range(5) if args.folds is None else args.folds
    log = []
    for f in use:
        fit, val = folds[f]
        ts = time.time()
        init_fit, others = fold_glm(keys, y, fit, [val, np.arange(n, n + len(test))], SEED + f)
        init_val, init_te = others
        te_fit, (te_val, te_te) = target_encode(keys, np.concatenate([y, np.zeros(len(test), int)]),
                                                fit, [val, np.arange(n, n + len(test))], SEED + f)
        X_fit = np.hstack([dense_tr[fit], te_fit]).astype(np.float32)
        X_val = np.hstack([dense_tr[val], te_val]).astype(np.float32)
        X_te = np.hstack([dense_te, te_te]).astype(np.float32)
        yf = y[fit]
        (pv, pt), it1 = boost(X_fit, yf, init_fit, None, [X_val, X_te], [init_val, init_te], SEED + f)
        oof['glmboost'][val], tst['glmboost'] = pv, tst['glmboost'] + pt / 5
        w = np.where(cvv0[fit], 3.0, 1.0)
        (pv, pt), it2 = boost(X_fit, yf, init_fit, w, [X_val, X_te], [init_val, init_te], SEED + 100 + f)
        oof['cvv0'][val], tst['cvv0'] = pv, tst['cvv0'] + pt / 5
        Xa_fit, Xa_val, Xa_te = (np.hstack([X_fit, agg_tr[fit]]), np.hstack([X_val, agg_tr[val]]),
                                 np.hstack([X_te, agg_te]))
        (pv, pt), it3 = boost(Xa_fit, yf, init_fit, None, [Xa_val, Xa_te], [init_val, init_te], SEED + 200 + f)
        oof['aggregate'][val], tst['aggregate'] = pv, tst['aggregate'] + pt / 5
        row = {'fold': f, 'seconds': round(time.time() - ts), 'iters': [it1, it2, it3],
               'glm_auc': roc_auc_score(y[val], init_val)}
        for k in names:
            row[k] = roc_auc_score(y[val], oof[k][val])
            row[k + '_cvv0'] = roc_auc_score(y[val][cvv0[val]], oof[k][val][cvv0[val]])
        log.append(row)
        print(json.dumps(row), flush=True)

    done = ~np.isnan(oof['glmboost'])
    summary = {'folds': log, 'seconds': round(time.time() - t0)}
    for k in names:
        if done.all():
            np.save(args.output / f'oof_v6_{k}.npy', oof[k].astype(np.float32))
            np.save(args.output / f'test_v6_{k}.npy', tst[k].astype(np.float32))
        summary[k] = roc_auc_score(y[done], oof[k][done])
        summary[k + '_cvv0'] = roc_auc_score(y[done & cvv0], oof[k][done & cvv0])
    (args.output / 'base_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'folds'}, indent=2))


if __name__ == '__main__':
    main()
