"""Rebuild Areeb Ali's clipped correction and verify Goodpjw2008's Step AB.

Sources and five-decimal scores are public-leaderboard tuned; they do not
constitute independent holdout validation. See README for attribution.
"""
import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find(root, name, sha=None):
    paths = sorted(root.rglob(name))
    paths = [p for p in paths if p.is_file() and (sha is None or digest(p) == sha)]
    if not paths:
        raise FileNotFoundError(f'{name} with expected checksum {sha} not found in {root}')
    return paths[0]


def ranks(values):
    return (rankdata(values) - .5) / len(values)


def logit_ranks(values):
    return logit(ranks(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    root, output = args.input, args.output
    output.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(find(root, 'sample_submission.csv'))
    test = pd.read_csv(find(root, 'test.csv'))
    if not sample.id.equals(test.id) or not sample.id.is_unique or len(sample) != 299984:
        raise ValueError('Invalid competition test IDs')
    def csv(values):
        return sample.assign(is_fraud=values).to_csv(index=False, lineterminator='\n')
    def verify(frame):
        if list(frame.columns) != ['id', 'is_fraud'] or not frame.id.equals(sample.id):
            raise ValueError('Invalid submission schema or IDs')
        if not np.isfinite(frame.is_fraud).all() or not frame.is_fraud.between(0, 1).all():
            raise ValueError('Invalid probabilities')
    with np.load(find(root, 'scored_rankings.npz'), allow_pickle=False) as z:
        assert np.array_equal(z['id'], sample.id)
        step_z = z['Z'].copy() if 'Z' in z.files else None
    with np.load(find(root, 'additional_scored_rankings.npz'), allow_pickle=False) as z:
        assert np.array_equal(z['id'], sample.id)
        v42 = z['own_v42'].copy()
        if step_z is None:
            step_z = z['Z'].copy()
    flags = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
             'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']
    clean = test.cvv_retry_count.eq(0) & test.prior_disputes.eq(0) & ~test[flags].astype(bool).any(axis=1)
    risk, velocity = test.merchant_risk_score, test.velocity_score
    zones = ((risk < 13.1) | ((velocity <= 4.1) & (risk <= 20)) |
             (clean & (((velocity <= 4.1) & (risk <= 33.8)) | ((velocity <= 6) & (risk <= 25))))).to_numpy()
    v42l = logit_ranks(v42)
    v43 = ranks(ranks(v42l - (logit_ranks(step_z) - v42l)) - 2 * zones)
    v43_csv = csv(v43)
    assert hashlib.sha256(v43_csv.encode()).hexdigest() == '5689d07cc601becb3b2530b87fa5c03241a79d54fb2aba10fe2773d6eaf0464c'
    base = pd.read_csv(StringIO(v43_csv)).is_fraud.to_numpy()
    base_logit = logit_ranks(base)
    score = base_logit.copy()
    score[~zones] += .25 * (base_logit - v42l)[~zones]
    score[zones] -= 100
    quarter_csv = csv(ranks(score))
    assert hashlib.sha256(quarter_csv.encode()).hexdigest() == '60ec5bd11eca0445c3b0b222a2715d3ffeee9594968cbce56061db16d87f5c62'
    anchor = logit_ranks(pd.read_csv(StringIO(quarter_csv)).is_fraud)
    aa = pd.read_csv(find(root, 'submission_step_aa.csv',
        '51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c'))
    verify(aa)
    residual = logit_ranks(aa.is_fraud) - anchor
    cap = np.quantile(np.abs(residual[~zones]), .95)
    score = anchor.copy()
    score[~zones] += .70 * np.clip(residual[~zones], -cap, cap)
    score[zones] -= 100
    assert score[zones].max() < score[~zones].min()
    np.testing.assert_array_equal(rankdata(score[zones]), rankdata(anchor[zones]))
    clipped = csv(ranks(score)).encode()
    assert hashlib.sha256(clipped).hexdigest().startswith('3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792')
    (output/'clipped.csv').write_bytes(clipped)
    step_ab = find(root, 'submission.csv', '3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038')
    ab = pd.read_csv(step_ab)
    verify(ab)
    (output/'step_ab.csv').write_bytes(step_ab.read_bytes())
    candidate = pd.read_csv(find(root, 'residual_025.csv'))
    verify(candidate)
    incumbent = .95 * ranks(base) + .05 * ranks(candidate.is_fraud)
    own = csv(incumbent).encode()
    assert hashlib.sha256(own).hexdigest() == '52ec1113780c6245d9b28c63cc410db185d0e5b6719c3a257adde2ebb4fb3620'
    result = {'rows': len(sample), 'zones': int(zones.sum()), 'clip_cap': float(cap),
              'clipped_sha256': hashlib.sha256(clipped).hexdigest(), 'step_ab_sha256': digest(step_ab),
              'incumbent_sha256': hashlib.sha256(own).hexdigest(), 'comparisons': {}}
    for name, values in [('clipped', pd.read_csv(StringIO(clipped.decode())).is_fraud), ('step_ab', ab.is_fraud)]:
        old_rank = rankdata(incumbent)
        new_rank = rankdata(values)
        movement = np.abs(new_rank-old_rank)
        top_old = old_rank > len(old_rank)*.95
        top_new = new_rank > len(new_rank)*.95
        result['comparisons'][name] = {'spearman': float(spearmanr(incumbent, values).statistic),
            'median_rank_movement': float(np.median(movement)), 'p95_rank_movement': float(np.quantile(movement,.95)),
            'top5_new_rows': int((top_new & ~top_old).sum())}
    (output/'replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
