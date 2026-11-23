"""Score the competition with a GLM trained only on the linked original dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

# Local imports are omitted when the sources are bundled for Kaggle.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokens.train import LOCKED_SEED, score_set
from train import DIGIT_COLUMNS, SEED, TARGET, competition_data


SOURCE = "uditjain13/credit-card-fraud-detection-2026"
SOURCE_FILE = "credit_card_fraud_2026.csv"


def original_data():
    return next(Path("/kaggle/input").rglob(SOURCE_FILE), Path("data") / SOURCE_FILE)


def transfer_features(frame):
    x = frame.copy()
    risk = x["merchant_risk_score"]
    velocity = x["velocity_score"]
    x["risk_velocity"] = risk * velocity
    x["amount_relative_balance"] = np.log1p(x["amount_usd"].clip(lower=0)) - np.log1p(x["account_balance_usd"].abs())
    return x


def run_transfer(args):
    train = pd.read_csv(args.data / "train.csv")
    ids = train.pop("id").to_numpy()
    y = train.pop(TARGET).to_numpy()
    original = pd.read_csv(args.original)
    original_y = original.pop(TARGET).to_numpy()
    if not set(train).issubset(original):
        raise ValueError("The original dataset is missing competition features.")
    original = original[train.columns]

    # Detect accidentally repeated features without using any competition labels.
    combined = pd.concat([original, train], ignore_index=True)
    hashes = pd.util.hash_pandas_object(combined, index=False).to_numpy()
    keep = ~np.isin(hashes[:len(original)], hashes[len(original):])
    del combined, hashes

    raw = transfer_features(original.loc[keep])
    continuous = [*DIGIT_COLUMNS, "risk_velocity", "amount_relative_balance"]
    categorical = [name for name in raw if name not in continuous]
    pipeline = make_pipeline(
        ColumnTransformer([
            ("smooth", SplineTransformer(n_knots=5, degree=2, knots="quantile"), continuous),
            ("categories", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
        ]),
        LogisticRegression(C=0.5, max_iter=500, random_state=SEED),
    )
    pipeline.fit(raw, original_y[keep])
    dev, holdout = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=LOCKED_SEED,
    ).split(train, y))
    holdout.sort()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": args.stage, "locked_seed": LOCKED_SEED, "source": SOURCE,
        "source_rows": int(keep.sum()), "removed_exact_overlaps": int((~keep).sum()),
    }
    if args.stage == "full":
        test = pd.read_csv(args.data / "test.csv")
        test_ids = test.pop("id")
        sample = pd.read_csv(args.data / "sample_submission.csv")
        if not sample["id"].equals(test_ids) or set(test) != set(train):
            raise ValueError("Unexpected competition test rows or columns.")
        sample[TARGET] = pipeline.predict_proba(transfer_features(test[train.columns]))[:, 1]
        sample.to_csv(args.output / "submission.csv", index=False)
    else:
        rows = dev if args.stage == "development" else holdout
        p = pipeline.predict_proba(transfer_features(train.iloc[rows]))[:, 1]
        cvv_zero = train["cvv_retry_count"].eq(0).to_numpy()[rows]
        data = {"id": ids[rows], "is_fraud": y[rows], "cvv_zero": cvv_zero, "original_glm": p}
        report["metrics"] = score_set(y[rows], p, cvv_zero)
        if args.stage == "development":
            folds = np.zeros(len(dev), dtype=np.int8)
            report["fold_auc"] = []
            for fold, (_, val_idx) in enumerate(
                StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(dev, y[dev]), start=1,
            ):
                folds[val_idx] = fold
                report["fold_auc"].append(float(roc_auc_score(y[dev][val_idx], p[val_idx])))
            data["fold"] = folds
        name = "dev-oof.npz" if args.stage == "development" else "holdout.npz"
        np.savez_compressed(args.output / name, **data)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    parser.add_argument("--original", type=Path, default=original_data())
    on_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working" if on_kaggle else "submissions/transfer"))
    parser.add_argument("--stage", choices=("development", "holdout", "full"), default="development")
    run_transfer(parser.parse_args())
