"""Train and compare fraud models on the official S6E11 data."""

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from threadpoolctl import threadpool_limits

COMPETITION = "playground-series-s6e11"
TARGET = "is_fraud"
FILES = ("train.csv", "test.csv", "sample_submission.csv")
DIGIT_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
)
SEED = 42
FOLDS = 5
TREE_WEIGHT = 0.8
THREADS = min(4, os.cpu_count() or 1)


def competition_data():
    kaggle_input = Path("/kaggle/input")
    for sample in kaggle_input.rglob(FILES[-1]):
        if COMPETITION in sample.parts and all((sample.parent / name).is_file() for name in FILES):
            return sample.parent
    return Path("data")


def decimal_keys(frame):
    """Group amounts by decimal representation as well as numeric magnitude."""
    keys = {}
    for name in DIGIT_COLUMNS:
        if name not in frame:
            continue
        raw = frame[name].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw)
        cents = np.zeros(len(raw), dtype=np.int64)
        cents[valid] = np.rint(100 * raw[valid]).astype(np.int64)
        magnitude = np.abs(cents)
        whole = magnitude // 100
        length = np.searchsorted(10 ** np.arange(1, 13), whole) + 1
        leading = whole // (10 ** np.maximum(0, length - 2))
        groups = {
            "exact": cents,
            "whole": cents // 100,
            "tenth": cents // 10,
            "fraction": magnitude % 100,
            "whole_tail": whole % 100,
            "leading": leading,
            "scale_leading": length * 100 + leading,
        }
        for suffix, array in groups.items():
            keys[f"{name}__{suffix}"] = np.where(valid, array, -2_000_000_000).astype(np.int32)
    return pd.DataFrame(keys, index=frame.index)


def raw_views(train, valid, test):
    fit = train.copy()
    validation = valid.copy()
    inference = test.copy()
    categorical = fit.select_dtypes(exclude=["number", "bool"]).columns
    for name in categorical:
        dtype = pd.CategoricalDtype(categories=pd.Index(fit[name].dropna().unique()))
        for view in (fit, validation, inference):
            view[name] = view[name].astype(dtype)
    return fit, validation, inference


def interaction_keys(frames):
    """Fit quantile cutoffs on the outer training partition."""
    risk = "merchant_risk_score"
    velocity = "velocity_score"
    cutoffs = {
        name: np.unique(frames[0][name].dropna().quantile(np.arange(1, 12) / 12))
        for name in (risk, velocity)
    }
    keys = []
    for frame in frames:
        bins = {
            name: np.where(frame[name].isna(), 0, 1 + np.searchsorted(cuts, frame[name]))
            for name, cuts in cutoffs.items()
        }
        cvv = frame["cvv_retry_count"].clip(lower=0, upper=2).fillna(3).astype(np.int32)
        keys.append(pd.DataFrame({
            "cvv_risk": 16 * cvv + bins[risk],
            "cvv_velocity": 16 * cvv + bins[velocity],
            "risk_velocity": 16 * bins[risk] + bins[velocity],
        }, index=frame.index))
    return keys


def predict_tree(views, target, rounds, slice_masks=None):
    """Use the same shallow-tree configuration for each ablation."""
    tree = lgb.LGBMClassifier(
        n_estimators=rounds, learning_rate=0.035, num_leaves=4,
        min_child_samples=100, colsample_bytree=0.55, subsample=0.85,
        subsample_freq=1, reg_lambda=1.0, n_jobs=THREADS,
        random_state=SEED, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    if slice_masks is None:
        tree.fit(views[0], target)
        return [tree.predict_proba(frame)[:, 1] for frame in views[1:]]
    tree.fit(views[0].loc[slice_masks[0]], target[slice_masks[0]])
    return [
        tree.predict_proba(frame.loc[mask])[:, 1] if mask.any() else np.empty(0)
        for frame, mask in zip(views[1:], slice_masks[1:], strict=True)
    ]


def blend_probabilities(a, b, weight):
    return expit(
        weight * logit(np.clip(a, 1e-9, 1 - 1e-9))
        + (1 - weight) * logit(np.clip(b, 1e-9, 1 - 1e-9))
    )


def fit_fold(raw, digits, target, rounds, *, experiment, compare_baseline):
    fit, valid, test = raw_views(*raw)
    key_views = []
    for frame, decimal in zip((fit, valid, test), digits, strict=True):
        basic = pd.DataFrame(index=frame.index)
        for name in frame:
            if name in DIGIT_COLUMNS:
                continue
            col = frame[name]
            if isinstance(col.dtype, pd.CategoricalDtype):
                basic[name] = col.cat.codes
            else:
                basic[name] = col.astype(np.float32).fillna(-2_000_000_000)
        key_views.append(pd.concat([basic, decimal], axis=1))

    encoder = TargetEncoder(target_type="binary", smooth=20.0, cv=3, random_state=SEED)
    encoded = [
        encoder.fit_transform(key_views[0], target),
        encoder.transform(key_views[1]),
        encoder.transform(key_views[2]),
    ]
    tree_views = []
    for frame, decimal, rates in zip((fit, valid, test), digits, encoded, strict=True):
        rates = pd.DataFrame(
            rates.astype(np.float32), index=frame.index,
            columns=[f"te__{name}" for name in key_views[0]],
        )
        tree_views.append(pd.concat([frame, decimal.filter(like="__fraction"), rates], axis=1))

    del encoded
    baseline = None
    if experiment == "v1" or compare_baseline:
        baseline = predict_tree(tree_views, target, rounds)
    if experiment == "v2":
        extra_keys = interaction_keys((fit, valid, test))
        interaction_encoder = TargetEncoder(target_type="binary", smooth=50.0, cv=3, random_state=SEED)
        interaction_rates = [
            interaction_encoder.fit_transform(extra_keys[0], target),
            interaction_encoder.transform(extra_keys[1]),
            interaction_encoder.transform(extra_keys[2]),
        ]
        for view, rates in zip(tree_views, interaction_rates, strict=True):
            for name, rate in zip(extra_keys[0], rates.T, strict=True):
                view[f"te__{name}"] = rate.astype(np.float32)
        tree_predictions = predict_tree(tree_views, target, rounds)
        masks = [frame["cvv_retry_count"].eq(0).to_numpy() for frame in (fit, valid, test)]
        specialist = predict_tree(tree_views, target, min(rounds, 900), masks)
    del tree_views

    onehot = OneHotEncoder(
        min_frequency=20, handle_unknown="infrequent_if_exist", dtype=np.float32,
    )
    matrix = onehot.fit_transform(key_views[0])
    glm = LogisticRegression(C=0.1, max_iter=250, tol=1e-5, solver="lbfgs")
    with threadpool_limits(limits=THREADS):
        glm.fit(matrix, target)
    del matrix
    glm_predictions = [glm.predict_proba(onehot.transform(keys))[:, 1] for keys in key_views[1:]]

    results = []
    for j, glm_pred in enumerate(glm_predictions):
        if experiment == "v1":
            tree_pred = baseline[j]
            blend = blend_probabilities(tree_pred, glm_pred, TREE_WEIGHT)
            results.append(np.column_stack([tree_pred, glm_pred, blend]))
            continue
        interaction_blend = blend_probabilities(tree_predictions[j], glm_pred, TREE_WEIGHT)
        specialist_blend = interaction_blend.copy()
        mask = masks[j + 1]
        specialist_blend[mask] = blend_probabilities(interaction_blend[mask], specialist[j], 0.75)
        columns = [interaction_blend, specialist_blend]
        if baseline is not None:
            columns.insert(0, blend_probabilities(baseline[j], glm_pred, TREE_WEIGHT))
        results.append(np.column_stack(columns))
    return results


def top5_summary(y, scores, no_cvv_retry):
    reviewed = np.argsort(-scores, kind="stable")[:len(y) // 20]
    flagged = np.zeros(len(y), dtype=bool)
    flagged[reviewed] = True
    fraud = y == 1
    return {
        "caught": int((flagged & fraud).sum()),
        "false_alarms": int((flagged & ~fraud).sum()),
        "missed_cvv_zero": int((~flagged & fraud & no_cvv_retry).sum()),
    }


def run(data_dir, output_dir, rounds, *, experiment="v1", seed=SEED, compare_baseline=False):
    start = time.monotonic()
    train, test, sample = (pd.read_csv(data_dir / name) for name in FILES)
    train_ids = train.pop("id")
    test_ids = test.pop("id")
    target = train.pop(TARGET).to_numpy()
    if set(train.columns) != set(test.columns):
        raise ValueError("Train and test feature columns differ")
    if not train_ids.is_unique or not test_ids.is_unique:
        raise ValueError("Expected unique train and test IDs")
    if list(sample.columns) != ["id", TARGET] or not sample["id"].equals(test_ids):
        raise ValueError("Sample submission does not match the order of test IDs")
    test = test[train.columns]
    train_digits = decimal_keys(train)
    test_digits = decimal_keys(test)
    if experiment == "v1":
        members = ("lightgbm", "digit_glm", "blend_80_20")
    else:
        members = ("interactions", "cvv_specialist")
        if compare_baseline:
            members = ("baseline", *members)
    oof = np.full((len(train), len(members)), np.nan)
    predictions = np.zeros((len(test), len(members)))
    fold_scores = []
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for fold, (fit, val) in enumerate(splitter.split(train, target), start=1):
        valid, test_pred = fit_fold(
            (train.iloc[fit], train.iloc[val], test),
            (train_digits.iloc[fit], train_digits.iloc[val], test_digits),
            target[fit], rounds, experiment=experiment, compare_baseline=compare_baseline,
        )
        oof[val] = valid
        predictions += test_pred / FOLDS
        scores = {name: roc_auc_score(target[val], col)
                  for name, col in zip(members, valid.T, strict=True)}
        fold_scores.append(scores)
        print(f"Fold {fold}/{FOLDS}: {scores}; elapsed {time.monotonic() - start:.0f}s", flush=True)

    if not np.isfinite(oof).all() or not np.isfinite(predictions).all():
        raise ValueError("Missing or non-finite predictions")
    scores = {name: roc_auc_score(target, oof[:, j]) for j, name in enumerate(members)}
    selected = max(scores, key=scores.get)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample[TARGET] = predictions[:, members.index(selected)]
    if not sample[TARGET].between(0, 1).all():
        raise ValueError("Submission scores must be within [0, 1]")
    sample.to_csv(output_dir / "submission.csv", index=False)
    if experiment == "v2":
        for j, name in enumerate(members):
            if name != "baseline":
                sample.assign(**{TARGET: predictions[:, j]}).to_csv(
                    output_dir / f"submission_{name}.csv", index=False,
                )
    no_cvv_retry = train["cvv_retry_count"].eq(0).to_numpy() if "cvv_retry_count" in train else np.zeros(len(train), dtype=bool)
    metrics = {
        "experiment": experiment,
        "split_seed": seed,
        "top_5pct": {name: top5_summary(target, oof[:, j], no_cvv_retry)
                     for j, name in enumerate(members)},
        "fold_scores": fold_scores,
        "oof_auc": scores,
        "selected": selected,
        "rows": {"train": len(train), "test": len(test)},
        "fraud_rate": float(target.mean()),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "tree_rounds": rounds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    destination = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("submissions")
    parser.add_argument("--output", type=Path, default=destination)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--experiment", choices=("v1", "v2"), default="v2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.rounds, experiment=args.experiment,
        seed=args.seed, compare_baseline=args.compare_baseline)
