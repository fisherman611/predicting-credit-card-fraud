"""v8 submission: rank mean of two independently built nested stacks + zero-fraud zones.

  goodpjw2008  wide LM stack, nested OOF   (goodpjw2008/s6e11-fraud-series-artifacts)
  DavutKursun  "The 0.9526 Wall" stack_all (davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal)

No weight is fitted, so the OOF AUC of the mean is an honest estimate. The idea of
averaging independent stacks follows MathsOnChain's S6E11 progress report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']


def find(root, name):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    y = train.is_fraud.to_numpy().astype(int)
    wide_oof = np.load(find(args.input, 'wide_lm_stack_nested.npy'))
    wide_test = np.load(find(args.input, 'wide_lm_stack_test.npy'))
    d_oof = pd.read_csv(find(args.input, 'oof_wall_stack.csv')).sort_values('id')
    d_test = pd.read_csv(find(args.input, 'test_wall_stack.csv')).sort_values('id')
    assert np.array_equal(d_oof.id, train.id) and np.array_equal(d_test.id, test.id)
    oof = (lr(wide_oof) + lr(d_oof.stack_all.to_numpy())) / 2
    tst = (lr(wide_test) + lr(d_test.stack_all.to_numpy())) / 2
    z_tr, z_te = zones(train), zones(test)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=2027).split(np.zeros(len(y)), y))
    scored = oof - 100 * z_tr
    res = {'wide_auc': roc_auc_score(y, wide_oof), 'davut_all_auc': roc_auc_score(y, d_oof.stack_all),
           'mean_auc': roc_auc_score(y, oof), 'mean_plus_zones_auc': roc_auc_score(y, scored),
           'mean_plus_zones_fold_auc': [roc_auc_score(y[b], scored[b]) for _, b in folds],
           'zones_train_fraud': int(y[z_tr].sum()), 'zones_test_rows': int(z_te.sum())}
    final = tst.copy()
    final[z_te] -= 100
    pred = (rankdata(final) - 0.5) / len(final)
    pd.DataFrame({'id': test.id, 'is_fraud': pred}).to_csv(args.output / 'stack_mean.csv', index=False)
    (args.output / 'stack_mean_results.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
