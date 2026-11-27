"""Nested-CV search for more zero-fraud zones on top of v8 + the four known zones.

In each outer fold, decision trees fitted on the fit rows (raw features, outside the known zones) give leaves; a leaf with
at least MIN rows and no fraud in the fit rows becomes a candidate zone, and is kept only if it also has no fraud in an
inner holdout of the fit rows. Kept zones are pushed to the bottom on the outer validation fold.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'kaggle/v11'); from residual_lm import find, lr, zones
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
S = sys.argv[1]; root = S + '/v11in'
tr = pd.read_csv(find(root, 'train.csv', 'playground-series-s6e11')); y = tr.is_fraud.values.astype(int); z = zones(tr)
w = lr(np.load(find(root, 'wide_lm_stack_nested.npy'))); dv = lr(pd.read_csv(find(root, 'oof_wall_stack.csv')).sort_values('id').stack_all.values)
v8 = (w + dv) / 2 - 100 * z
feat = [c for c in tr.columns if c not in ('id', 'is_fraud')]
X = pd.get_dummies(tr[feat], dtype=float).values
folds = list(StratifiedKFold(5, shuffle=True, random_state=2027).split(X, y))
for MIN in (300, 1000, 3000):
    gains, sizes = [], []
    for k, (a, b) in enumerate(folds):
        rng = np.random.default_rng(k); a = a[~z[a]]
        inner = rng.random(len(a)) < 0.3; A, H = a[~inner], a[inner]
        new = np.zeros(len(y), bool)
        for depth, seed in ((6, 0), (8, 1), (10, 2)):
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN, random_state=seed, class_weight={0: 1, 1: 50}).fit(X[A], y[A])
            la, lh, lall = t.apply(X[A]), t.apply(X[H]), t.apply(X)
            for leaf in np.unique(la):
                if y[A][la == leaf].sum() == 0 and y[H][lh == leaf].sum() == 0 and (lh == leaf).sum() >= MIN * 0.3 / 0.7:
                    new |= lall == leaf
        new &= ~z
        s = v8.copy(); s[new] -= 50
        g = roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], v8[b]); gains.append(g); sizes.append(int(new[b].sum()))
        print(f'MIN {MIN} fold {k}: new zone rows in val {new[b].sum()}, frauds there {y[b][new[b]].sum()}, gain {g:+.2e}', flush=True)
    print(f'MIN {MIN}: mean gain {np.mean(gains):+.2e}, folds up {sum(g > 0 for g in gains)}/5', flush=True)
