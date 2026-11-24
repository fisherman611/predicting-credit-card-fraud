"""v5 mega stack: CV-validated stacking of public OOF members.

Outputs (in --output):
  stack.csv        rank-scaled prediction of the CV-selected stack
  blend.csv        Step AB logit-rank plus a clipped share of the stack (if Step AB is found)
  results.json     member list, CV scores and the selected configuration
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from members import collect, logit_rank  # noqa: E402

SEED = 2027
WIDE = 's6e11-fraud-series-artifacts:wide_lm_stack_nested'
STEP_AB_SHA = '3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038'
FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']


def zones(frame):
    """Low-risk regions with zero fraud in all 699,960 train rows."""
    clean = frame.cvv_retry_count.eq(0) & frame.prior_disputes.eq(0) & ~frame[FLAGS].astype(bool).any(axis=1)
    r, v = frame.merchant_risk_score, frame.velocity_score
    return ((r < 13.1) | ((v <= 4.1) & (r <= 20)) |
            (clean & (((v <= 4.1) & (r <= 33.8)) | ((v <= 6) & (r <= 25))))).to_numpy()


def find(root, name, sha=None):
    for p in sorted(Path(root).rglob(name)):
        if p.is_file() and (sha is None or hashlib.sha256(p.read_bytes()).hexdigest() == sha):
            return p
    return None


def pct(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--blend-weight', type=float, default=0.15)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    y = train.is_fraud.to_numpy().astype(int)
    members = collect(args.input, y)
    names = sorted(members)
    X = np.column_stack([logit_rank(members[n][0]) for n in names]).astype(np.float32)
    T = np.column_stack([logit_rank(members[n][1]) for n in names]).astype(np.float32)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    oof = np.zeros(len(y))
    test_lr = np.zeros(len(test))
    for a, b in folds:
        model = LogisticRegression(C=0.05, max_iter=3000).fit(X[a], y[a])
        oof[b] = model.decision_function(X[b])
        test_lr += model.decision_function(T) / len(folds)
    results = {'members': names, 'n_members': len(names),
               'lr_auc': roc_auc_score(y, oof),
               'lr_fold_auc': [roc_auc_score(y[b], oof[b]) for _, b in folds]}

    # Mix the LR stack with the strongest public nested stack; weight chosen on
    # held-out folds (choose on 4 folds, score the 5th) to keep the estimate honest.
    wide_i = names.index(WIDE) if WIDE in names else int(np.argmax([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])]))
    oof_l, wide = logit_rank(oof), X[:, wide_i]
    grid = np.linspace(0, 1, 11)
    honest = np.zeros(len(y))
    chosen = []
    for a, b in folds:
        w = grid[int(np.argmax([roc_auc_score(y[a], g * oof_l[a] + (1 - g) * wide[a]) for g in grid]))]
        chosen.append(float(w))
        honest[b] = w * oof_l[b] + (1 - w) * wide[b]
    w_final = float(np.median(chosen))
    results.update({'wide_member': names[wide_i], 'wide_auc': roc_auc_score(y, wide),
                    'mix_weights_per_fold': chosen, 'mix_weight': w_final,
                    'mix_auc_nested': roc_auc_score(y, honest)})

    z_tr, z_te = zones(train), zones(test)
    mixed_oof = pct(w_final * oof_l + (1 - w_final) * wide)
    results['zones_train_fraud'] = int(y[z_tr].sum())
    results['mix_plus_zones_auc'] = roc_auc_score(y, mixed_oof - 2 * z_tr)

    stack_test = pct(w_final * logit_rank(test_lr) + (1 - w_final) * T[:, wide_i])
    stack_test = pct(stack_test - 2 * z_te)
    out = pd.DataFrame({'id': test.id, 'is_fraud': stack_test})
    out.to_csv(args.output / 'stack.csv', index=False)

    step_ab = find(args.input, 'step_ab.csv', STEP_AB_SHA)
    if step_ab is not None:
        ab = pd.read_csv(step_ab)
        assert ab.id.equals(test.id)
        left, right = logit_rank(ab.is_fraud.to_numpy()), logit_rank(stack_test)
        gap = right - left
        cap = float(np.quantile(np.abs(gap), 0.95))
        score = left + args.blend_weight * np.clip(gap, -cap, cap)
        score[z_te] -= 100
        pd.DataFrame({'id': test.id, 'is_fraud': pct(score)}).to_csv(args.output / 'blend.csv', index=False)
        results.update({'blend_weight': args.blend_weight, 'blend_cap': cap,
                        'blend_top5_changed': int(((rankdata(score) > .95 * len(score)) &
                                                   (rankdata(left) <= .95 * len(left))).sum())})
    results['zones_test_rows'] = int(z_te.sum())
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: v for k, v in results.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    main()
