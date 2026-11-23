"""Evaluate a calibrated CVV-zero specialist against the v1 baseline."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# The Kaggle bundle prepends the existing v1 implementation.
spec = importlib.util.spec_from_file_location("fraud_baseline", Path(__file__).resolve().parents[1] / "train.py")
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
TARGET = baseline_module.TARGET
blend_probabilities = baseline_module.blend_probabilities
competition_data = baseline_module.competition_data
cvv_auc_summary = baseline_module.cvv_auc_summary
decimal_keys = baseline_module.decimal_keys
fold_features = baseline_module.fold_features
predict_glm = baseline_module.predict_glm
predict_tree = baseline_module.predict_tree
top5_summary = baseline_module.top5_summary

WEIGHTS = (0.0, 0.10, 0.25, 0.50)
HOLDOUT_SEED = 73191
DEV_SEED = 42


def fit_pair(frame, digits, y, fit_idx, score_idx, rounds):
    fit = frame.iloc[fit_idx]
    score = frame.iloc[score_idx]
    # Pass score twice because the shared feature builder supports validation and test.
    raw, keys, trees = fold_features(
        (fit, score, score),
        (digits.iloc[fit_idx], digits.iloc[score_idx], digits.iloc[score_idx]), y[fit_idx],
    )
    base_tree = predict_tree(trees, y[fit_idx], rounds)[0]
    masks = [part["cvv_retry_count"].eq(0).to_numpy() for part in raw]
    expert = predict_tree(trees, y[fit_idx], min(rounds, 900), masks)[0]
    glm = predict_glm(keys, y[fit_idx])[0]
    base = blend_probabilities(base_tree, glm, 0.8)
    specialist = np.full(len(score_idx), np.nan)
    specialist[masks[1]] = expert
    return base, specialist


def calibration(p, y):
    model = LogisticRegression(C=1.0, solver="lbfgs")
    model.fit(logit(np.clip(p, 1e-8, 1-1e-8)).reshape(-1, 1), y)
    return model


def transform(model, p):
    return model.predict_proba(logit(np.clip(p, 1e-8, 1-1e-8)).reshape(-1, 1))[:, 1]


def mix(base, expert, mask, weight):
    scores = base.copy()
    if weight:
        scores[mask] = blend_probabilities(base[mask], expert[mask], 1-weight)
    return scores


def metrics(y, score, mask):
    return {"auc": float(roc_auc_score(y, score)), "cvv": cvv_auc_summary(y, score, mask),
            "top_5pct": top5_summary(y, score, mask)}


def run(args):
    start = time.monotonic()
    frame = pd.read_csv(args.data / "train.csv")
    if args.smoke:
        frame = frame.sample(n=min(len(frame), 20000), random_state=19).sort_index().reset_index(drop=True)
    ids = frame.pop("id").to_numpy()
    y = frame.pop(TARGET).to_numpy()
    digits = decimal_keys(frame)
    dev, hold = next(StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=HOLDOUT_SEED).split(frame, y))
    group = frame.cvv_retry_count.eq(0).to_numpy()
    baseline = np.full(len(dev), np.nan)
    expert = np.full(len(dev), np.nan)
    fold_ids = np.full(len(dev), -1, dtype=int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=DEV_SEED)
    for fold, (a, b) in enumerate(cv.split(dev, y[dev])):
        baseline[b], expert[b] = fit_pair(frame, digits, y, dev[a], dev[b], args.rounds)
        fold_ids[b] = fold
        print(f"dev fold {fold+1} completed in {time.monotonic()-start:.0f}s", flush=True)
    dev_y, dev_mask = y[dev], group[dev]
    # Each scored fold gets a calibrator fitted on the other folds' OOF predictions.
    calibrated = np.full(len(dev), np.nan)
    for fold in range(3):
        fit = (fold_ids != fold) & dev_mask
        val = (fold_ids == fold) & dev_mask
        calibrated[val] = transform(calibration(expert[fit], dev_y[fit]), expert[val])
    results = {}
    for weight in WEIGHTS:
        score = mix(baseline, calibrated, dev_mask, weight)
        results[str(weight)] = {
            "overall": metrics(dev_y, score, dev_mask),
            "fold_auc": [float(roc_auc_score(dev_y[fold_ids == f], score[fold_ids == f])) for f in range(3)],
        }
    selected = max(WEIGHTS, key=lambda weight: results[str(weight)]["overall"]["auc"])
    print("development selection", selected, json.dumps(results), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "development_oof.npz", id=ids[dev], target=dev_y,
                        fold=fold_ids, cvv_zero=dev_mask, baseline=baseline,
                        expert=expert, expert_calibrated=calibrated)
    # Selection is frozen above. The outer holdout was excluded from all base fits
    # and from the specialist calibrator.
    base_hold, expert_hold = fit_pair(frame, digits, y, dev, hold, args.rounds)
    hold_mask = group[hold]
    cal = calibration(expert[dev_mask], dev_y[dev_mask])
    calibrated_hold = np.full(len(hold), np.nan)
    calibrated_hold[hold_mask] = transform(cal, expert_hold[hold_mask])
    candidate = mix(base_hold, calibrated_hold, hold_mask, selected)
    out = {"holdout_seed": HOLDOUT_SEED, "development_seed": DEV_SEED,
           "development_rows": len(dev), "holdout_rows": len(hold),
           "weights": list(WEIGHTS), "selected_weight": selected,
           "development": results, "holdout_baseline": metrics(y[hold], base_hold, hold_mask),
           "holdout_candidate": metrics(y[hold], candidate, hold_mask),
           "rounds": args.rounds, "elapsed_seconds": round(time.monotonic()-start, 1),
           "calibration": "cross-fitted dev OOF; holdout calibrator fits all dev OOF",
           "warning": "Development base-model OOF used for calibration can share fit rows across folds."}
    np.savez_compressed(args.output / "holdout.npz", id=ids[hold], target=y[hold],
                        cvv_zero=hold_mask, baseline=base_hold, candidate=candidate)
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
