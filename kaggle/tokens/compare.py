"""Compare development predictions and freeze a candidate before using the holdout."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokens.train import score_set

STACK_MEMBERS = ("v1", "transformer", "original_glm")

BLENDS = {
    "v1_plus_transformer_05": {"v1": 0.95, "transformer": 0.05},
    "v1_plus_transformer_15": {"v1": 0.85, "transformer": 0.15},
    "v1_plus_original_05": {"v1": 0.95, "original_glm": 0.05},
    "v1_plus_original_15": {"v1": 0.85, "original_glm": 0.15},
    "v1_plus_transformer": {"v1": 0.75, "transformer": 0.25},
    "v1_plus_original": {"v1": 0.75, "original_glm": 0.25},
    "v1_plus_both": {"v1": 0.70, "transformer": 0.15, "original_glm": 0.15},
}


def ensure_same_rows(reference, values, fields):
    for name in fields:
        if not np.array_equal(reference[name], values[name]):
            raise ValueError(f"Prediction files differ in {name}.")


def rank_blend(scores, weights):
    return sum(weight * (rankdata(scores[name]) - 0.5) / len(scores[name]) for name, weight in weights.items())


def stack_matrix(predictions):
    return np.column_stack([logit(np.clip(predictions[name], 1e-6, 1 - 1e-6))
                            for name in STACK_MEMBERS])


def fit_stack(x, y):
    scaler = StandardScaler()
    model = LogisticRegression(C=0.01, max_iter=250)
    model.fit(scaler.fit_transform(x), y)
    coefficients = model.coef_[0] / scaler.scale_
    intercept = model.intercept_[0] - float(coefficients @ scaler.mean_)
    return {"coefficients": coefficients.tolist(), "intercept": float(intercept)}


def predict_stack(x, config):
    return expit(x @ np.asarray(config["coefficients"]) + config["intercept"])


def stack_development(predictions, y, folds):
    """Cross-fit each stack inside one held-out base-model fold."""
    x = stack_matrix(predictions)
    result = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        split = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 + int(fold))
        for fit, val in split.split(rows, y[rows]):
            config = fit_stack(x[rows[fit]], y[rows[fit]])
            result[rows[val]] = predict_stack(x[rows[val]], config)
    # This final combiner sees only development OOF scores and their labels.
    return result, fit_stack(x, y)


def development(args):
    with np.load(args.tree / "dev-oof.npz", allow_pickle=False) as tree, np.load(
        args.transformer / "dev-oof.npz", allow_pickle=False
    ) as transformer, np.load(args.transfer / "dev-oof.npz", allow_pickle=False) as transfer:
        for source in (transformer, transfer):
            ensure_same_rows(tree, source, ("id", "fold", "is_fraud", "cvv_zero"))
        y, group, folds = (tree[name] for name in ("is_fraud", "cvv_zero", "fold"))
        scores = {name: tree[name] for name in ("v1", "token_frequency", "token_target")}
        scores.update(transformer=transformer["transformer"], original_glm=transfer["original_glm"])

    scores["stack_logistic"], stack = stack_development(scores, y, folds)
    for name, weights in BLENDS.items():
        scores[name] = rank_blend(scores, weights)
    report = {}
    for name, prediction in scores.items():
        result = score_set(y, prediction, group)
        result["fold_auc"] = [
            float(roc_auc_score(y[folds == f], prediction[folds == f])) for f in np.unique(folds)
        ]
        report[name] = result

    v1 = report["v1"]
    eligible = []
    for name, result in report.items():
        result["auc_gain"] = result["auc"] - v1["auc"]
        result["folds_up"] = sum(a > b for a, b in zip(result["fold_auc"], v1["fold_auc"], strict=True))
        if result["auc_gain"] > 0 and result["folds_up"] >= 2:
            eligible.append(name)
    selected = max(eligible, key=lambda name: report[name]["auc"]) if eligible else "v1"
    return {"development_rows": len(y), "selected": selected, "stack_config": stack, "results": report}


def holdout(args):
    if args.development_report is None:
        raise ValueError("Provide the frozen --development-report before evaluating holdout.")
    report = json.loads(args.development_report.read_text())
    if report["selected"] != args.selected:
        raise ValueError("The development report did not select this challenger.")
    is_stack = args.selected == "stack_logistic"
    weights = BLENDS.get(args.selected, {args.selected: 1.0})
    required = set(STACK_MEMBERS if is_stack else weights)
    if args.selected == "v1":
        raise ValueError("No challenger was selected on development data.")
    with np.load(args.baseline / "holdout.npz", allow_pickle=False) as v1:
        reference = {name: v1[name] for name in ("id", "is_fraud", "cvv_zero")}
        scores = {"v1": v1["blend_80_20"]}
    for model, source in (("tree", args.tree), ("transformer", args.transformer), ("original_glm", args.transfer)):
        members = {"token_frequency", "token_target"} if model == "tree" else {model}
        for member in required & members:
            if source is None:
                raise ValueError(f"Missing holdout predictions for {member}.")
            with np.load(source / "holdout.npz", allow_pickle=False) as data:
                ensure_same_rows(reference, data, ("id", "is_fraud", "cvv_zero"))
                scores[member] = data[member]
    y, group = reference["is_fraud"], reference["cvv_zero"]
    if is_stack:
        prediction = predict_stack(stack_matrix(scores), report["stack_config"])
    else:
        prediction = rank_blend(scores, weights) if len(weights) > 1 else scores[args.selected]
    return {
        "selected": args.selected, "holdout_rows": len(y),
        "v1": score_set(y, scores["v1"], group), "challenger": score_set(y, prediction, group),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--selected", action="store", choices=(
        "v1", "token_frequency", "token_target", "transformer", "original_glm", "stack_logistic", *BLENDS,
    ))
    parser.add_argument("--output", type=Path, default=Path("submissions/token-comparison.json"))
    args = parser.parse_args()
    if args.selected is not None:
        if args.baseline is None:
            parser.error("--baseline is required when a holdout challenger is specified.")
        result = holdout(args)
    else:
        if not all((args.tree, args.transformer, args.transfer)):
            parser.error("--tree, --transformer, and --transfer are required to select a challenger.")
        result = development(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
