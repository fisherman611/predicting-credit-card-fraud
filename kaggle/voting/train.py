"""Evaluate rank voting and stacking of three tabular fraud models."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# The Kaggle bundle prepends the existing baseline implementation.
spec = importlib.util.spec_from_file_location("fraud_baseline", Path(__file__).resolve().parents[1] / "train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
competition_data = module.competition_data
decimal_keys = module.decimal_keys
fold_features = module.fold_features
predict_tree = module.predict_tree
predict_glm = module.predict_glm
blend_probabilities = module.blend_probabilities
top5_summary = module.top5_summary
cvv_auc_summary = module.cvv_auc_summary

SPLIT_SEED = 24681357
FOLD_SEED = 42
RANK_WEIGHTS = {
    "rank_equal": [1/3, 1/3, 1/3],
    "rank_tree_glm_cat_50_20_30": [.5, .2, .3],
    "rank_tree_glm_cat_35_15_50": [.35, .15, .5],
}


def fit_models(fit, valid, target, rounds, cat_rounds):
    fit_digits, valid_digits = decimal_keys(fit), decimal_keys(valid)
    raw, keys, trees = fold_features((fit, valid, valid), (fit_digits, valid_digits, valid_digits), target)
    tree = predict_tree(trees, target, rounds)[0]
    glm = predict_glm(keys, target)[0]
    del raw, keys, trees
    xfit, xval = fit.copy(), valid.copy()
    categorical = list(xfit.select_dtypes(include=["object", "category", "bool"]).columns)
    for name in categorical:
        xfit[name] = xfit[name].fillna("__missing__").astype(str)
        xval[name] = xval[name].fillna("__missing__").astype(str)
    cat = CatBoostClassifier(iterations=cat_rounds, depth=6, learning_rate=.04,
                             l2_leaf_reg=6, random_seed=42, thread_count=4,
                             verbose=False, allow_writing_files=False)
    cat.fit(xfit, target, cat_features=categorical)
    return np.column_stack((tree, glm, cat.predict_proba(xval)[:, 1]))


def score_members(pred):
    n = len(pred)
    ranks = np.column_stack([(rankdata(pred[:, j])-.5)/n for j in range(pred.shape[1])])
    return {name: ranks @ np.asarray(weights) for name, weights in RANK_WEIGHTS.items()}


def logits(pred):
    return logit(np.clip(pred, 1e-7, 1-1e-7))


def stack_model(x, y):
    return LogisticRegression(C=.1, max_iter=500, solver="lbfgs").fit(logits(x), y)


def measures(y, p, group):
    return {"auc": float(roc_auc_score(y, p)), "cvv": cvv_auc_summary(y, p, group),
            "top_5pct": top5_summary(y, p, group)}


def run(args):
    start = time.monotonic()
    data = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    sample = pd.read_csv(args.data / "sample_submission.csv")
    if args.smoke:
        data = data.sample(n=min(20000, len(data)), random_state=19).sort_index().reset_index(drop=True)
        test = test.iloc[:1000].copy()
        sample = sample.iloc[:1000].copy()
    ids = data.pop("id").to_numpy()
    y = data.pop("is_fraud").to_numpy()
    test_ids = test.pop("id")
    if set(data.columns) != set(test.columns) or not sample.id.equals(test_ids):
        raise ValueError("Input features or test IDs do not match.")
    test = test[data.columns]
    dev, hold = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=SPLIT_SEED).split(data, y))
    dev_y, dev_group = y[dev], data.iloc[dev].cvv_retry_count.eq(0).to_numpy()
    predictions = np.full((len(dev), 3), np.nan)
    fold_ids = np.full(len(dev), -1, dtype=np.int8)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=FOLD_SEED)
    for fold, (fit, val) in enumerate(cv.split(dev, dev_y)):
        predictions[val] = fit_models(data.iloc[dev[fit]], data.iloc[dev[val]], y[dev[fit]], args.rounds, args.cat_rounds)
        fold_ids[val] = fold
        print(f"development fold {fold+1} finished after {time.monotonic()-start:.0f}s", flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError("Missing development OOF predictions")
    meta = np.full(len(dev), np.nan)
    for fold in range(3):
        fit, val = fold_ids != fold, fold_ids == fold
        meta[val] = stack_model(predictions[fit], dev_y[fit]).predict_proba(logits(predictions[val]))[:, 1]
    choices = {"baseline_v1": blend_probabilities(predictions[:, 0], predictions[:, 1], .8),
               **score_members(predictions), "stack": meta}
    scores = {name: {"overall": measures(dev_y, p, dev_group),
                     "fold_auc": [float(roc_auc_score(dev_y[fold_ids==i], p[fold_ids==i])) for i in range(3)]}
              for name, p in choices.items()}
    selected = max(choices, key=lambda name: scores[name]["overall"]["auc"])
    print("selected by development", selected, json.dumps({k:v["overall"]["auc"] for k,v in scores.items()}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "development_oof.npz", id=ids[dev], target=dev_y,
                        cvv_zero=dev_group, folds=fold_ids, members=predictions)
    stack = stack_model(predictions, dev_y) if selected == "stack" else None

    def evaluate(pred):
        if selected == "baseline_v1":
            return blend_probabilities(pred[:, 0], pred[:, 1], .8)
        if selected == "stack":
            return stack.predict_proba(logits(pred))[:, 1]
        return score_members(pred)[selected]

    hold_pred = fit_models(data.iloc[dev], data.iloc[hold], y[dev], args.rounds, args.cat_rounds)
    hold_base = blend_probabilities(hold_pred[:, 0], hold_pred[:, 1], .8)
    hold_selected = evaluate(hold_pred)
    hold_group = data.iloc[hold].cvv_retry_count.eq(0).to_numpy()
    outcome = {"seed": SPLIT_SEED, "development_rows": len(dev), "holdout_rows": len(hold),
               "development": scores, "selected": selected,
               "holdout_baseline": measures(y[hold], hold_base, hold_group),
               "holdout_candidate": measures(y[hold], hold_selected, hold_group),
               "base_models": ["LightGBM", "Digit GLM", "CatBoost"],
               "notice": "Meta-model OOF can share base training rows across folds; holdout is excluded from fitting and selection.",
               "rounds": args.rounds, "cat_rounds": args.cat_rounds}
    (args.output / "metrics.json").write_text(json.dumps(outcome, indent=2)+"\n")
    np.savez_compressed(args.output / "holdout.npz", id=ids[hold], target=y[hold],
                        cvv_zero=hold_group, baseline=hold_base, candidate=hold_selected, members=hold_pred)
    print("holdout metrics",json.dumps(outcome),flush=True)
    final_pred = fit_models(data, test, y, args.rounds, args.cat_rounds)
    final = evaluate(final_pred)
    # All candidates are bounded probabilities or rank fractions.
    if not np.isfinite(final).all() or not ((final >= 0) & (final <= 1)).all():
        raise ValueError("Invalid test predictions")
    sample.assign(is_fraud=final).to_csv(args.output / "submission.csv", index=False)
    np.savez_compressed(args.output / "test_members.npz", id=test_ids.to_numpy(), members=final_pred)
    outcome["elapsed_seconds"] = round(time.monotonic()-start, 1)
    (args.output / "metrics.json").write_text(json.dumps(outcome, indent=2)+"\n")
    print("Finished", outcome["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=competition_data())
    p.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--rounds", type=int, default=1500)
    p.add_argument("--cat-rounds", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
