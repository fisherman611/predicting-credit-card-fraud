"""v6 stack: v5 mega stack plus gated repo members (oof_v6_*/test_v6_*).

Gate: a v6 member is kept only if adding it to the logistic stack raises the
nested mix AUC on at least 4 of 5 folds and does not lower CVV=0 AUC.
Members are tried greedily, best single-fold-sum gain first.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mega_stack'))
from members import collect, logit_rank  # noqa: E402
from stack import WIDE, find, pct, zones  # noqa: E402

SEED = 2027
GRID = np.linspace(0, 1, 11)


def nested(X, T, y, wide_oof, wide_test, folds, cvv0):
    oof, test = np.zeros(len(y)), np.zeros(T.shape[0])
    for a, b in folds:
        model = LogisticRegression(C=0.05, max_iter=3000).fit(X[a], y[a])
        oof[b] = model.decision_function(X[b])
        test += model.decision_function(T) / len(folds)
    ol = logit_rank(oof)
    mixed, weights = np.zeros(len(y)), []
    for a, b in folds:
        w = GRID[int(np.argmax([roc_auc_score(y[a], g * ol[a] + (1 - g) * wide_oof[a]) for g in GRID]))]
        weights.append(float(w))
        mixed[b] = w * ol[b] + (1 - w) * wide_oof[b]
    w = float(np.median(weights))
    fold_auc = [roc_auc_score(y[b], mixed[b]) for _, b in folds]
    return {'auc': roc_auc_score(y, mixed), 'fold_auc': fold_auc, 'weight': w,
            'cvv0_auc': roc_auc_score(y[cvv0], mixed[cvv0]),
            'test': w * logit_rank(test) + (1 - w) * wide_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    y = train.is_fraud.to_numpy().astype(int)
    cvv0 = train.cvv_retry_count.to_numpy() == 0
    pool = collect(args.input, y, verbose=False)
    v6 = sorted(k for k in pool if ':v6_' in k)
    base = sorted(k for k in pool if k not in v6)
    print('base members', len(base), 'v6 members', v6, flush=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    L = {k: (logit_rank(pool[k][0]).astype(np.float32), logit_rank(pool[k][1]).astype(np.float32)) for k in pool}
    wide = WIDE if WIDE in L else max(base, key=lambda k: roc_auc_score(y, L[k][0]))

    def run(names):
        X = np.column_stack([L[k][0] for k in names])
        T = np.column_stack([L[k][1] for k in names])
        return nested(X, T, y, L[wide][0], L[wide][1], folds, cvv0)

    report = {'v6_single': {k: {'auc': roc_auc_score(y, pool[k][0]),
                                'cvv0_auc': roc_auc_score(y[cvv0], pool[k][0][cvv0])} for k in v6}}
    current = run(base)
    report['base'] = {k: v for k, v in current.items() if k != 'test'}
    print('base', json.dumps(report['base']), flush=True)
    chosen = list(base)
    trials = []
    remaining = list(v6)
    while remaining:
        cands = []
        for k in remaining:
            r = run(chosen + [k])
            wins = sum(n > o for n, o in zip(r['fold_auc'], current['fold_auc']))
            cands.append((r['auc'] - current['auc'], wins, k, r))
            trials.append({'member': k, 'gain': r['auc'] - current['auc'], 'fold_wins': wins,
                           'cvv0_gain': r['cvv0_auc'] - current['cvv0_auc'], 'n_selected_before': len(chosen) - len(base)})
            print('try', json.dumps(trials[-1]), flush=True)
        cands.sort(key=lambda c: -c[0])
        gain, wins, k, r = cands[0]
        if gain <= 0 or wins < 4 or r['cvv0_auc'] < current['cvv0_auc']:
            break
        chosen.append(k)
        remaining.remove(k)
        current = r
    report['trials'] = trials
    report['selected_v6'] = [k for k in chosen if k in v6]
    report['final'] = {k: v for k, v in current.items() if k != 'test'}

    z_tr, z_te = zones(train), zones(test)
    pred = pct(pct(current['test']) - 2 * z_te)
    pd.DataFrame({'id': test.id, 'is_fraud': pred}).to_csv(args.output / 'stack_v6.csv', index=False)
    report['zones_test_rows'] = int(z_te.sum())
    report['zones_train_fraud'] = int(y[z_tr].sum())
    (args.output / 'stack_v6_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('base', 'selected_v6', 'final')}, indent=2))


if __name__ == '__main__':
    main()
