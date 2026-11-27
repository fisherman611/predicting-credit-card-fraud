"""v12 final files, rebuilt from public inputs only (no training).

  v8                  mean(lr(goodpjw2008 wide LM stack), lr(DavutKursun stack_all)) + zero-fraud zones
  cv_led_zones        goodpjw2008 step A (submission_cv_led.csv, before the public-fitted board steps) + zones
  cvled50_v8_zones    rank mean 50/50 of CV-led and v8, zones at the bottom  -> submission.csv (robust final)
  ladder              monya1 zero-fraud zones + rank extrapolation (public 0.95700), copied as is

Inputs: the competition data, goodpjw2008/s6e11-fraud-series-artifacts, and the outputs of
davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal, goodpjw2008/s6e11-board-model-realmlp-lb-0-95687
and monya1/s6e11-zero-fraud-zones-and-rank-extrapolation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr

FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']


def find(root, name, inside=None):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file() and (inside is None or inside in str(p)))
    if not hits:
        raise FileNotFoundError(f'{name} ({inside})')
    return hits[0]


def zones(frame):
    clean = frame.cvv_retry_count.eq(0) & frame.prior_disputes.eq(0) & ~frame[FLAGS].astype(bool).any(axis=1)
    r, v = frame.merchant_risk_score, frame.velocity_score
    return ((r < 13.1) | ((v <= 4.1) & (r <= 20)) |
            (clean & (((v <= 4.1) & (r <= 33.8)) | ((v <= 6) & (r <= 25))))).to_numpy()


def lr(v):
    return logit((rankdata(v) - 0.5) / len(v))


def frac(v):
    return (rankdata(v) - 0.5) / len(v)


def by_id(path, ids):
    d = pd.read_csv(path).set_index('id')
    return d.loc[ids].is_fraud.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(find(args.input, 'test.csv', 'playground-series-s6e11'))
    ids, z = test.id.to_numpy(), zones(test)

    d_test = pd.read_csv(find(args.input, 'test_wall_stack.csv')).set_index('id').loc[ids]
    v8 = (lr(np.load(find(args.input, 'wide_lm_stack_test.npy'))) + lr(d_test.stack_all.to_numpy())) / 2
    v8[z] -= 100
    cv_led = by_id(find(args.input, 'submission_cv_led.csv', 'board-model-realmlp'), ids)
    ladder = by_id(find(args.input, 'submission.csv', 'rank-extrapolation'), ids)

    cz = frac(cv_led)
    cz[z] -= 10
    mix = 0.5 * frac(cv_led) + 0.5 * frac(v8)
    mix[z] -= 10
    files = {'v8.csv': frac(v8), 'cv_led_zones.csv': frac(cz), 'cvled50_v8_zones.csv': frac(mix),
             'ladder_monya1.csv': ladder}
    report = {'zone_rows': int(z.sum())}
    for name, p in files.items():
        pd.DataFrame({'id': ids, 'is_fraud': p}).to_csv(args.output / name, index=False)
        report[name] = {'sha256': hashlib.sha256((args.output / name).read_bytes()).hexdigest(),
                        'spearman_with_ladder': float(spearmanr(p, ladder).correlation),
                        'zone_rows_max_rank': float(frac(p)[z].max())}
    pd.DataFrame({'id': ids, 'is_fraud': files['cvled50_v8_zones.csv']}).to_csv(args.output / 'submission.csv', index=False)
    (args.output / 'final_v12.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
