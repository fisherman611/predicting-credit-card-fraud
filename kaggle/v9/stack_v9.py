"""v9: the v5 logistic stack with three member sources it never saw, then the v8 rank mean.

New sources (all real out-of-fold predictions):
  goodpjw2008/s6e11-oof-bank        70 models (EBM, GLM, LightGBM/XGBoost on digit keys, RealMLP)
  busyaprime/s6e11-33-own-members   52 models in oof.npy / test.npy (names in manifest.csv) + TabM
  yekenot RealMLP PyTabKit / PyTorch oof_preds.csv + submission.csv

Variants, each scored on StratifiedKFold(5, seed 2027) with the zero-fraud zones applied:
  v8       mean(lr(wide), lr(davut))                       the current CV-best submission
  mix9     nested logistic stack on all members, mixed with wide (fold-honest weight)
  mean9    mean(lr(mix9), lr(davut))
  mean9w   mean(lr(mix9), lr(wide), lr(davut))
The variant with the best OOF AUC is written to stack_v9.csv; no weight is tuned on the full OOF.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mega_stack'))
from members import N_TEST, N_TRAIN, _add, collect  # noqa: E402

SEED = 2027
GRID = np.linspace(0, 1, 11)
FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']


def find(root, name, inside=None):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file() and (inside is None or inside in str(p)))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def zones(frame):
    clean = frame.cvv_retry_count.eq(0) & frame.prior_disputes.eq(0) & ~frame[FLAGS].astype(bool).any(axis=1)
    r, v = frame.merchant_risk_score, frame.velocity_score
    return ((r < 13.1) | ((v <= 4.1) & (r <= 20)) |
            (clean & (((v <= 4.1) & (r <= 33.8)) | ((v <= 6) & (r <= 25))))).to_numpy()


def lr(v):
    return logit((rankdata(v) - 0.5) / len(v))


def extra_members(root, train_ids, test_ids):
    """Members collect() cannot pair on its own."""
    out = {}
    for man in Path(root).rglob('manifest.csv'):
        oof, tst = man.with_name('oof.npy'), man.with_name('test.npy')
        if not (oof.is_file() and tst.is_file()):
            continue
        names = pd.read_csv(man).name.tolist()
        a, b = np.load(oof, mmap_mode='r'), np.load(tst, mmap_mode='r')
        if a.ndim == 2 and a.shape == (N_TRAIN, len(names)) and b.shape == (N_TEST, len(names)):
            for j, nm in enumerate(names):
                _add(out, 'busy:' + nm, np.array(a[:, j]), np.array(b[:, j]))
    for sub in Path(root).rglob('oof_preds.csv'):
        twin = sub.with_name('submission.csv')
        if twin.is_file():
            a = pd.read_csv(sub).set_index('id').loc[train_ids].iloc[:, -1].to_numpy()
            b = pd.read_csv(twin).set_index('id').loc[test_ids].iloc[:, -1].to_numpy()
            _add(out, sub.parent.name + ':realmlp', a, b)
    return out


def nested(X, T, y, wide_oof, wide_test, folds):
    oof, test = np.zeros(len(y)), np.zeros(T.shape[0])
    for a, b in folds:
        model = LogisticRegression(C=0.05, max_iter=3000).fit(X[a], y[a])
        oof[b] = model.decision_function(X[b])
        test += model.decision_function(T) / len(folds)
    ol = lr(oof)
    mixed, weights = np.zeros(len(y)), []
    for a, b in folds:
        w = GRID[int(np.argmax([roc_auc_score(y[a], g * ol[a] + (1 - g) * wide_oof[a]) for g in GRID]))]
        weights.append(float(w))
        mixed[b] = w * ol[b] + (1 - w) * wide_oof[b]
    w = float(np.median(weights))
    return mixed, w * lr(test) + (1 - w) * wide_test, weights, roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(find(args.input, 'train.csv', 'playground-series-s6e11'))
    test = pd.read_csv(find(args.input, 'test.csv', 'playground-series-s6e11'))
    y = train.is_fraud.to_numpy().astype(int)
    pool = collect(args.input, y, verbose=False)
    pool.update(extra_members(args.input, train.id.to_numpy(), test.id.to_numpy()))
    pool = {k: v for k, v in pool.items() if roc_auc_score(y, v[0]) >= 0.90}
    print('members', len(pool), flush=True)

    wide_oof = lr(np.load(find(args.input, 'wide_lm_stack_nested.npy')))
    wide_test = lr(np.load(find(args.input, 'wide_lm_stack_test.npy')))
    d_oof = pd.read_csv(find(args.input, 'oof_wall_stack.csv')).sort_values('id')
    d_test = pd.read_csv(find(args.input, 'test_wall_stack.csv')).sort_values('id')
    assert np.array_equal(d_oof.id, train.id) and np.array_equal(d_test.id, test.id)
    dv_oof, dv_test = lr(d_oof.stack_all.to_numpy()), lr(d_test.stack_all.to_numpy())

    names = sorted(pool)
    X = np.column_stack([lr(pool[k][0]) for k in names]).astype(np.float32)
    T = np.column_stack([lr(pool[k][1]) for k in names]).astype(np.float32)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    mix_oof, mix_test, weights, lr_auc = nested(X, T, y, wide_oof, wide_test, folds)

    variants = {
        'v8': ((wide_oof + dv_oof) / 2, (wide_test + dv_test) / 2),
        'mix9': (mix_oof, mix_test),
        'mean9': ((lr(mix_oof) + dv_oof) / 2, (lr(mix_test) + dv_test) / 2),
        'mean9w': ((lr(mix_oof) + wide_oof + dv_oof) / 3, (lr(mix_test) + wide_test + dv_test) / 3),
    }
    z_tr, z_te = zones(train), zones(test)
    report = {'n_members': len(names), 'logistic_stack_auc': lr_auc, 'mix_weights': weights,
              'new_members': sorted(k for k in names if k.startswith(('busy:', 'oof-bank:', 's6e11-oof-bank:'))
                                    or 'realmlp' in k or 'tabm' in k),
              'variants': {}}
    base_folds = None
    for name, (o, _) in variants.items():
        s = o - 100 * z_tr
        fa = [roc_auc_score(y[b], s[b]) for _, b in folds]
        if name == 'v8':
            base_folds = fa
        report['variants'][name] = {'auc_zones': roc_auc_score(y, s), 'fold_auc': fa,
                                    'folds_over_v8': int(sum(a > b for a, b in zip(fa, base_folds)))}
    best = max(variants, key=lambda k: report['variants'][k]['auc_zones'])
    report['best'] = best
    final = variants[best][1].copy()
    final[z_te] -= 100
    pd.DataFrame({'id': test.id, 'is_fraud': (rankdata(final) - 0.5) / len(final)}).to_csv(
        args.output / 'stack_v9.csv', index=False)
    for name, (_, t) in variants.items():
        t = t.copy()
        t[z_te] -= 100
        pd.DataFrame({'id': test.id, 'is_fraud': (rankdata(t) - 0.5) / len(t)}).to_csv(
            args.output / f'stack_v9_{name}.csv', index=False)
    (args.output / 'stack_v9_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'new_members'}, indent=2))


if __name__ == '__main__':
    main()
