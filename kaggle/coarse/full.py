"""Fit the CVV specialist on all training rows and produce competition predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

# The bundle prepends the v1 training functions.
spec = importlib.util.spec_from_file_location("fraud_baseline", Path(__file__).resolve().parents[1] / "train.py")
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
competition_data = baseline_module.competition_data
decimal_keys = baseline_module.decimal_keys
fold_features = baseline_module.fold_features
predict_tree = baseline_module.predict_tree
predict_glm = baseline_module.predict_glm
blend_probabilities = baseline_module.blend_probabilities

DEV_OOF_SHA256 = "42d425e05ff093a469f1542c2a8d75aa2a6fcbd8e839ede38b6b6527d80a06e6"
EXPERT_WEIGHT = 0.10


def run(args):
    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    sample = pd.read_csv(args.data / "sample_submission.csv")
    train_ids = train.pop("id")
    test_ids = test.pop("id")
    target = train.pop("is_fraud").to_numpy()
    if not train_ids.is_unique or not test_ids.is_unique or not sample.id.equals(test_ids):
        raise ValueError("The train/test IDs or sample submission do not match.")
    if set(train.columns) != set(test.columns):
        raise ValueError("Feature columns differ between train and test.")
    test = test[train.columns]
    digest = hashlib.sha256(args.dev_oof.read_bytes()).hexdigest()
    if not args.smoke and digest != DEV_OOF_SHA256:
        raise ValueError("Development OOF file does not match the selected experiment.")
    with np.load(args.dev_oof, allow_pickle=False) as data:
        id_index = pd.Index(train_ids)
        rows = id_index.get_indexer(data["id"])
        if (rows < 0).any() or len(np.unique(rows)) != len(rows):
            raise ValueError("OOF IDs are missing or duplicated.")
        if not np.array_equal(target[rows], data["target"]):
            raise ValueError("OOF target does not match train.")
        if not np.array_equal(train.cvv_retry_count.iloc[rows].eq(0), data["cvv_zero"]):
            raise ValueError("OOF CVV groups do not match train.")
        group = data["cvv_zero"]
        expert_oof = data["expert"][group]
        if not np.isfinite(expert_oof).all():
            raise ValueError("Specialist OOF predictions are missing.")
        x = logit(np.clip(expert_oof, 1e-8, 1 - 1e-8)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs").fit(x, data["target"][group])
    print("Calibrator fitted using development OOF predictions", flush=True)

    raw, keys, trees = fold_features(
        (train, test, test),
        (decimal_keys(train), decimal_keys(test), decimal_keys(test)),
        target,
    )
    tree = predict_tree(trees, target, args.rounds)[0]
    mask = test.cvv_retry_count.eq(0).to_numpy()
    masks = [train.cvv_retry_count.eq(0).to_numpy(), mask, mask]
    expert = predict_tree(trees, target, min(args.rounds, 900), masks)[0]
    glm = predict_glm(keys, target)[0]
    baseline = blend_probabilities(tree, glm, 0.8)
    x = logit(np.clip(expert, 1e-8, 1 - 1e-8)).reshape(-1, 1)
    calibrated = calibrator.predict_proba(x)[:, 1]
    candidate = baseline.copy()
    candidate[mask] = blend_probabilities(baseline[mask], calibrated, 1 - EXPERT_WEIGHT)
    if not np.isfinite(candidate).all() or not ((candidate >= 0) & (candidate <= 1)).all():
        raise ValueError("Invalid probabilities.")
    args.output.mkdir(parents=True, exist_ok=True)
    sample.assign(is_fraud=candidate).to_csv(args.output / "submission.csv", index=False)
    sample.assign(is_fraud=baseline).to_csv(args.output / "baseline.csv", index=False)
    result = {"training_rows": len(train), "test_rows": len(test), "specialist_weight": EXPERT_WEIGHT,
              "rounds": args.rounds, "dev_oof_sha256": digest,
              "calibrator_rows": int(group.sum()),
              "submission_sha256": hashlib.sha256((args.output / "submission.csv").read_bytes()).hexdigest()}
    (args.output / "full_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--dev-oof", type=Path)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dev_oof is None:
        files = list(Path("/kaggle/input").rglob("development_oof.npz"))
        if len(files) != 1:
            raise ValueError("Attach exactly one development OOF output.")
        args.dev_oof = files[0]
    run(args)
