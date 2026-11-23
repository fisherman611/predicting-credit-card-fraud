"""Reproduce a 20-member public OOF stack with a GPT-2 member."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="Directory containing the public Kaggle datasets")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
bank_paths = list(args.input.rglob("folds.csv"))
bank_paths = [p for p in bank_paths if "s6e11-oof-bank" in str(p)]
bundle_paths = list(args.input.rglob("rank1_q17_streams.npz"))
if len(bank_paths) != 1 or len(bundle_paths) != 1:
    raise ValueError("Expected one OOF bank and one rank fusion bundle.")
bank = bank_paths[0].parent
bundle = np.load(bundle_paths[0], allow_pickle=False)
labels, folds = bundle["y_train"], bundle["folds"]
assert np.array_equal(pd.read_csv(bank / "folds.csv")["fold"], folds)
manifest = pd.read_csv(bank / "manifest.csv")
names = manifest.loc[manifest.core, "name"].tolist()

def features(split, use_lm):
    arrays = [np.load(bank / f"{split}_{name}.npy") for name in names]
    if use_lm:
        arrays.append(bundle[f"{split}_megayak_5f_lm"])
    n = len(arrays[0])
    assert all(len(a) == n and np.isfinite(a).all() for a in arrays)
    ranks = np.column_stack([(rankdata(a) - 0.5) / n for a in arrays])
    return logit(ranks)

args.output.mkdir(parents=True, exist_ok=True)
metrics = {}
for use_lm, name in [(False, "core"), (True, "core_plus_lm")]:
    x, xt = features("oof", use_lm), features("test", use_lm)
    oof = np.zeros(len(labels))
    for f in sorted(np.unique(folds)):
        fit, val = folds != f, folds == f
        model = LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky")
        model.fit(x[fit], labels[fit])
        p = model.decision_function(x[val])
        oof[val] = (rankdata(p) - 0.5) / len(p)
    model = LogisticRegression(C=0.01, tol=1e-8, max_iter=200, solver="newton-cholesky")
    model.fit(x, labels)
    pred = model.decision_function(xt)
    np.save(args.output / f"{name}_oof.npy", oof)
    pd.DataFrame({"id": np.arange(699960, 999944), "is_fraud": (rankdata(pred)-0.5)/len(pred)}).to_csv(
        args.output / f"{name}_submission.csv", index=False
    )
    metrics[name] = {
        "auc": roc_auc_score(labels, oof),
        "fold_auc": [roc_auc_score(labels[folds == f], oof[folds == f]) for f in sorted(np.unique(folds))],
        "members": x.shape[1],
    }
    print(name, metrics[name], flush=True)
(args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
