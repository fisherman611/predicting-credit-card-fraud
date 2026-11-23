"""Evaluate interaction features and an OOF residual correction."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# The Kaggle bundle prepends the existing baseline.
spec = importlib.util.spec_from_file_location("fraud_baseline", Path(__file__).resolve().parents[1] / "train.py")
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
competition_data = baseline_module.competition_data
decimal_keys = baseline_module.decimal_keys
fit_fold = baseline_module.fit_fold
raw_views = baseline_module.raw_views
cvv_auc_summary = baseline_module.cvv_auc_summary
top5_summary = baseline_module.top5_summary

SPLIT_SEED = 718231
FOLD_SEED = 42
CANDIDATES = ("interactions", "residual_025", "residual_050")


def meta_views(fit, valid, inference, base_fit, base_valid, base_inference):
    views = raw_views(fit, valid, inference)
    for frame, base in zip(views, (base_fit, base_valid, base_inference), strict=True):
        frame["baseline_logit"] = logit(np.clip(base, 1e-9, 1-1e-9))
    return views


def residual_fit(fit, y):
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=.03, num_leaves=4,
                               min_child_samples=200, reg_lambda=20, n_jobs=4,
                               deterministic=True, force_col_wise=True, boost_from_average=False,
                               verbosity=-1, random_state=42)
    model.fit(fit, y, init_score=fit["baseline_logit"].to_numpy())
    return model


def meta_score(model, frame, weight):
    return expit(frame["baseline_logit"].to_numpy() + weight * model.predict(frame, raw_score=True))


def stats(y, p, group):
    return {"auc": float(roc_auc_score(y, p)), "cvv": cvv_auc_summary(y, p, group),
            "top_5pct": top5_summary(y, p, group)}


def run(args):
    started = time.monotonic()
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
        raise ValueError("Features or test IDs differ")
    test = test[data.columns]
    digits = decimal_keys(data)
    dev, hold = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=SPLIT_SEED).split(data, y))
    dev_y = y[dev]
    group = data.cvv_retry_count.eq(0).to_numpy()
    predictions = np.full((len(dev), 2), np.nan)
    folds = np.full(len(dev), -1, dtype=int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=FOLD_SEED)
    for fold, (fit, val) in enumerate(cv.split(dev, dev_y)):
        a, b = dev[fit], dev[val]
        valid, _ = fit_fold((data.iloc[a], data.iloc[b], data.iloc[b]),
                            (digits.iloc[a], digits.iloc[b], digits.iloc[b]),
                            y[a], args.rounds, experiment="v2", compare_baseline=True)
        predictions[val] = valid[:, :2]
        folds[val] = fold
        print(f"development fold {fold+1} finished after {time.monotonic()-started:.0f}s", flush=True)
    base = predictions[:, 0]
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete development predictions.")
    corrections = np.full((len(dev), 2), np.nan)
    for fold in range(3):
        fit, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        views = meta_views(data.iloc[dev[fit]], data.iloc[dev[val]], data.iloc[dev[val]],
                           base[fit], base[val], base[val])
        model = residual_fit(views[0], dev_y[fit])
        corrections[val, 0] = meta_score(model, views[1], .25)
        corrections[val, 1] = meta_score(model, views[1], .50)
    versions = {"baseline": base, "interactions": predictions[:, 1],
                "residual_025": corrections[:, 0], "residual_050": corrections[:, 1]}
    dev_result = {name: {"overall": stats(dev_y, p, group[dev]),
                         "fold_auc": [float(roc_auc_score(dev_y[folds == f], p[folds == f])) for f in range(3)]}
                  for name,p in versions.items()}
    selected = max(CANDIDATES, key=lambda n: dev_result[n]["overall"]["auc"])
    print("Selected challenger",selected,json.dumps({n:r["overall"]["auc"] for n,r in dev_result.items()}),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output / "development_versions.npz", id=ids[dev], target=dev_y, fold=folds,
                        **versions)
    valid, _ = fit_fold((data.iloc[dev], data.iloc[hold], data.iloc[hold]),
                        (digits.iloc[dev], digits.iloc[hold], digits.iloc[hold]),
                        y[dev], args.rounds, experiment="v2", compare_baseline=True)
    meta = meta_views(data.iloc[dev], data.iloc[hold], data.iloc[hold], base, valid[:,0], valid[:,0])
    correction = residual_fit(meta[0], dev_y)
    hold_versions = {"baseline": valid[:,0], "interactions": valid[:,1],
                     "residual_025": meta_score(correction,meta[1],.25),
                     "residual_050": meta_score(correction,meta[1],.50)}
    # Only the selected challenger and its baseline are evaluated on holdout.
    report = {"seed": SPLIT_SEED, "development_rows":len(dev), "holdout_rows":len(hold),
              "selected_challenger":selected, "development":dev_result,
              "holdout_baseline":stats(y[hold],hold_versions["baseline"],group[hold]),
              "holdout_challenger":stats(y[hold],hold_versions[selected],group[hold]),
              "rounds":args.rounds,
              "limitation":"Prior EDA explored the full train. Meta OOF predictions share base-training rows across folds."}
    np.savez_compressed(args.output/"holdout.npz",id=ids[hold],target=y[hold],cvv_zero=group[hold],
                        baseline=hold_versions["baseline"],candidate=hold_versions[selected])
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Holdout",json.dumps({"selected":selected,"baseline":report["holdout_baseline"],
                                "challenger":report["holdout_challenger"]}),flush=True)
    test_digits = decimal_keys(test)
    valid, test_preds = fit_fold((data.iloc[dev],data.iloc[hold],test),
                                 (digits.iloc[dev],digits.iloc[hold],test_digits),
                                 y[dev],args.rounds,experiment="v2",compare_baseline=True)
    # Development-only training is preserved for production so the correction
    # uses OOF scores from the same development population.
    baseline_test, interaction_test = test_preds[:,0], test_preds[:,1]
    views = meta_views(data.iloc[dev],data.iloc[hold],test,base,valid[:,0],baseline_test)
    residual = residual_fit(views[0],dev_y)
    test_versions={"baseline":baseline_test,"interactions":interaction_test,
                   "residual_025":meta_score(residual,views[2],.25),
                   "residual_050":meta_score(residual,views[2],.50)}
    for name,values in test_versions.items():
        if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
            raise ValueError(f"Invalid predictions for {name}")
        sample.assign(is_fraud=values).to_csv(args.output/f"{name}.csv",index=False)
    sample.assign(is_fraud=test_versions[selected]).to_csv(args.output/"submission.csv",index=False)
    report["elapsed_seconds"]=round(time.monotonic()-started,1)
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Finished",report["elapsed_seconds"],"seconds",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=competition_data())
    p.add_argument("--output",type=Path,default=Path("/kaggle/working"))
    p.add_argument("--rounds",type=int,default=1500)
    p.add_argument("--smoke",action="store_true")
    run(p.parse_args())
