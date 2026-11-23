"""Compare and blend aligned OOF predictions without tuning the blend weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def aligned_scores(directory, name, labels=None):
    with np.load(directory / "oof.npz", allow_pickle=False) as artifact:
        columns = {key: artifact[key] for key in ("id", "fold", "is_fraud", "cvv_zero")}
        if labels is not None:
            for key, values in labels.items():
                if not np.array_equal(columns[key], values):
                    raise ValueError(f"{directory}: mismatched {key}")
        score = artifact[name]
    sample = pd.read_csv(directory / "submission.csv")
    if not np.isfinite(score).all() or not sample["is_fraud"].between(0, 1).all():
        raise ValueError(f"{directory}: invalid predictions")
    return columns, score, sample


def percentile(scores):
    return (rankdata(scores) - 0.5) / len(scores)


def summarize(labels, scores):
    y, cvv_zero, folds = (labels[key] for key in ("is_fraud", "cvv_zero", "fold"))
    top = np.argsort(-scores, kind="stable")[:len(scores) // 20]
    checked = np.zeros(len(scores), dtype=bool)
    checked[top] = True
    masked = (cvv_zero & (y == 1), ~cvv_zero & (y == 0))
    cross_difficult = masked[0] | masked[1]
    return {
        "auc": roc_auc_score(y, scores),
        "fold_auc": [roc_auc_score(y[folds == fold], scores[folds == fold]) for fold in np.unique(folds)],
        "auc_cvv_zero": roc_auc_score(y[cvv_zero], scores[cvv_zero]),
        "auc_cvv_zero_fraud_vs_cvv_other_legit": roc_auc_score(y[cross_difficult], scores[cross_difficult]),
        "top5_caught": int(y[top].sum()),
        "top5_missed_cvv_zero": int((cvv_zero & (y == 1) & ~checked).sum()),
    }


def run(args):
    labels, v1_oof, v1_test = aligned_scores(args.v1, "blend_80_20")
    _, real_oof, real_test = aligned_scores(args.realmlp, "realmlp", labels=labels)
    _, tabm_oof, tabm_test = aligned_scores(args.tabm, "tabm", labels=labels)
    if not v1_test["id"].is_unique or not all(
        frame["id"].equals(v1_test["id"]) for frame in (real_test, tabm_test)
    ):
        raise ValueError("Submission IDs do not align.")

    oof = {"v1": v1_oof, "realmlp": real_oof, "tabm": tabm_oof}
    test = {"v1": v1_test["is_fraud"], "realmlp": real_test["is_fraud"], "tabm": tabm_test["is_fraud"]}
    rank_oof = {key: percentile(value) for key, value in oof.items()}
    rank_test = {key: percentile(value) for key, value in test.items()}
    blend_weights = {
        "v1_realmlp": {"v1": 0.75, "realmlp": 0.25},
        "v1_tabm": {"v1": 0.75, "tabm": 0.25},
        "v1_both": {"v1": 0.70, "realmlp": 0.15, "tabm": 0.15},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, weights in blend_weights.items():
        oof[name] = sum(weight * rank_oof[key] for key, weight in weights.items())
        test[name] = sum(weight * rank_test[key] for key, weight in weights.items())
        v1_test.assign(is_fraud=test[name]).to_csv(args.output / f"submission_{name}.csv", index=False)
    results = {}
    for name, values in oof.items():
        result = summarize(labels, values)
        result["rank_correlation_with_v1"] = float(spearmanr(v1_oof, values).statistic)
        results[name] = result
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--realmlp", type=Path, required=True)
    parser.add_argument("--tabm", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/neural-comparison"))
    run(parser.parse_args())
