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
POSITIONAL_DIGIT_COLUMNS = ("merchant_risk_score", "velocity_score", "account_balance_usd")
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


def positional_digit_keys(decimal):
    """Pool equal decimal chunks at the same integer scale."""
    keys = {}
    for name in POSITIONAL_DIGIT_COLUMNS:
        exact = decimal[f"{name}__exact"].to_numpy(dtype=np.int64)
        observed = exact != -2_000_000_000
        magnitude = np.abs(exact)
        whole = magnitude // 100
        length = np.searchsorted(10 ** np.arange(1, 13), whole) + 1
        prefix = magnitude // 10 ** (length - 1)
        chunks = {
            "prefix3": 1000 * length + prefix,
            "boundary2": 100 * length + (magnitude // 10) % 100,
            "boundary3": 1000 * length + magnitude % 1000,
            "fraction_by_scale": 100 * length + magnitude % 100,
        }
        for chunk, value in chunks.items():
            keys[f"{name}__{chunk}"] = np.where(observed, value, -1).astype(np.int32)
    return pd.DataFrame(keys, index=decimal.index)


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
    if experiment in ("v2", "v3"):
        if experiment == "v2":
            extra_keys = interaction_keys((fit, valid, test))
        else:
            extra_keys = [positional_digit_keys(decimal) for decimal in digits]
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
        if experiment == "v2":
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
        blend = blend_probabilities(tree_predictions[j], glm_pred, TREE_WEIGHT)
        columns = [blend]
        if experiment == "v2":
            specialist_blend = blend.copy()
            mask = masks[j + 1]
            specialist_blend[mask] = blend_probabilities(blend[mask], specialist[j], 0.75)
            columns.append(specialist_blend)
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


def auc_across_groups(positive_scores, negative_scores):
    """Return AUC and the number of compared pairs, counting ties as one half."""
    pairs = len(positive_scores) * len(negative_scores)
    if not pairs:
        return None, 0
    ordered = np.sort(negative_scores)
    lower = np.searchsorted(ordered, positive_scores, side="left")
    upper = np.searchsorted(ordered, positive_scores, side="right")
    return float((lower + upper).sum() / (2 * pairs)), pairs


def cvv_auc_summary(y, scores, no_cvv_retry):
    """Decompose AUC into within-group and cross-group fraud/legitimate pairs."""
    groups = {
        "cvv_zero": no_cvv_retry,
        "cvv_other": ~no_cvv_retry,
    }
    components = {
        f"{p}_fraud_vs_{n}_legit": auc_across_groups(
            scores[(y == 1) & positive_group], scores[(y == 0) & negative_group],
        )
        for p, positive_group in groups.items()
        for n, negative_group in groups.items()
    }
    cross_keys = ("cvv_zero_fraud_vs_cvv_other_legit", "cvv_other_fraud_vs_cvv_zero_legit")
    cross = [(components[key][0], components[key][1]) for key in cross_keys if components[key][1]]
    return {
        "within_cvv_zero": components["cvv_zero_fraud_vs_cvv_zero_legit"][0],
        "within_cvv_other": components["cvv_other_fraud_vs_cvv_other_legit"][0],
        "cross_cvv_groups": sum(auc * pairs for auc, pairs in cross) / sum(pairs for _, pairs in cross) if cross else None,
        "directional_cross_auc": {key: components[key][0] for key in cross_keys},
    }


def run(data_dir, output_dir, rounds, *, experiment="v1", seed=SEED,
        compare_baseline=False, save_oof=False):
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
        members = ("interactions", "cvv_specialist") if experiment == "v2" else ("positional_digits",)
        if compare_baseline:
            members = ("baseline", *members)
    oof = np.full((len(train), len(members)), np.nan)
    predictions = np.zeros((len(test), len(members)))
    fold_scores = []
    fold_ids = np.zeros(len(train), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for fold, (fit, val) in enumerate(splitter.split(train, target), start=1):
        valid, test_pred = fit_fold(
            (train.iloc[fit], train.iloc[val], test),
            (train_digits.iloc[fit], train_digits.iloc[val], test_digits),
            target[fit], rounds, experiment=experiment, compare_baseline=compare_baseline,
        )
        oof[val] = valid
        fold_ids[val] = fold
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
    if save_oof:
        if experiment == "v1" and selected != "blend_80_20":
            sample.assign(**{TARGET: predictions[:, members.index("blend_80_20")]}).to_csv(
                output_dir / "submission_blend_80_20.csv", index=False,
            )
        np.savez_compressed(
            output_dir / "oof.npz",
            id=train_ids.to_numpy(), fold=fold_ids, is_fraud=target,
            cvv_zero=no_cvv_retry, **dict(zip(members, oof.T, strict=True)),
        )
    metrics = {
        "experiment": experiment,
        "cvv_auc": {name: cvv_auc_summary(target, oof[:, j], no_cvv_retry)
                    for j, name in enumerate(members)},
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


def run_residual(data_dir, control_dir, output_dir, seed):
    """Fit each correction on rows the corresponding v1 model never trained on."""
    start = time.monotonic()
    if control_dir is None:
        files = list(Path("/kaggle/input").rglob("oof.npz"))
        if len(files) != 1:
            raise ValueError("Attach one private baseline run, or set --control.")
        control_dir = files[0].parent
    metrics = json.loads((control_dir / "metrics.json").read_text())
    if metrics["experiment"] != "v1":
        raise ValueError("Expected the v1 control artifacts.")
    if metrics["split_seed"] != seed:
        raise ValueError("The control run must use the requested seed.")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    control_file = "submission.csv" if metrics["selected"] == "blend_80_20" else "submission_blend_80_20.csv"
    baseline_test = pd.read_csv(control_dir / control_file)
    with np.load(control_dir / "oof.npz", allow_pickle=False) as artifact:
        if not np.array_equal(train["id"], artifact["id"]) or not test["id"].equals(baseline_test["id"]):
            raise ValueError("Control predictions do not match competition IDs.")
        if not np.array_equal(train[TARGET], artifact[TARGET]):
            raise ValueError("Control OOF labels do not match the training data.")
        folds = artifact["fold"]
        baseline = artifact["blend_80_20"]
    if not np.array_equal(np.unique(folds), np.arange(1, FOLDS + 1)):
        raise ValueError("Expected five completed control folds.")
    if not np.isfinite(baseline).all() or not baseline_test[TARGET].between(0, 1).all():
        raise ValueError("Control predictions contain invalid scores.")

    target = train.pop(TARGET).to_numpy()
    cvv_zero = train["cvv_retry_count"].eq(0).to_numpy()
    train_ids = train.pop("id")
    test.pop("id")
    train["baseline_logit"] = logit(np.clip(baseline, 1e-9, 1 - 1e-9))
    baseline_logit = logit(np.clip(baseline_test[TARGET].to_numpy(), 1e-9, 1 - 1e-9))
    test["baseline_logit"] = baseline_logit
    test = test[train.columns]

    corrected_oof = np.full(len(train), np.nan)
    test_correction = np.zeros(len(test))
    weight = 0.5
    fold_scores = []
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        halves = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + int(fold))
        for learn_idx, check_idx in halves.split(rows, target[rows]):
            learn, check = rows[learn_idx], rows[check_idx]
            fit, valid, infer = raw_views(train.iloc[learn], train.iloc[check], test)
            booster = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.03, num_leaves=4, min_child_samples=200,
                reg_lambda=20, n_jobs=THREADS, deterministic=True, force_col_wise=True,
                boost_from_average=False, verbosity=-1, random_state=SEED,
            )
            booster.fit(fit, target[learn], init_score=fit["baseline_logit"].to_numpy())
            corrected_oof[check] = expit(
                valid["baseline_logit"].to_numpy() + weight * booster.predict(valid, raw_score=True)
            )
            test_correction += booster.predict(infer, raw_score=True) / (2 * FOLDS)
        scores = {
            "baseline": roc_auc_score(target[rows], baseline[rows]),
            "residual": roc_auc_score(target[rows], corrected_oof[rows]),
        }
        fold_scores.append(scores)
        print(f"Fold {fold}/{FOLDS}: {scores}", flush=True)
    if not np.isfinite(corrected_oof).all() or not np.isfinite(test_correction).all():
        raise ValueError("Residual scores are incomplete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    submission = baseline_test.assign(**{TARGET: expit(baseline_logit + weight * test_correction)})
    submission.to_csv(output_dir / "submission.csv", index=False)
    np.savez_compressed(output_dir / "oof.npz", id=train_ids.to_numpy(), fold=folds,
                        is_fraud=target, cvv_zero=cvv_zero, baseline=baseline, residual=corrected_oof)
    members = {"baseline": baseline, "residual": corrected_oof}
    report = {
        "experiment": "residual",
        "split_seed": seed,
        "oof_auc": {name: roc_auc_score(target, pred) for name, pred in members.items()},
        "cvv_auc": {name: cvv_auc_summary(target, pred, cvv_zero) for name, pred in members.items()},
        "top_5pct": {name: top5_summary(target, pred, cvv_zero) for name, pred in members.items()},
        "fold_scores": fold_scores,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    destination = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("submissions")
    parser.add_argument("--output", type=Path, default=destination)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--experiment", choices=("v1", "v2", "v3", "residual"), default="v3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--save-oof", action="store_true")
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    if args.experiment == "residual":
        run_residual(args.data, args.control, args.output, args.seed)
    else:
        run(args.data, args.output, args.rounds, experiment=args.experiment,
            seed=args.seed, compare_baseline=args.compare_baseline,
            save_oof=args.save_oof or args.experiment == "v3")
