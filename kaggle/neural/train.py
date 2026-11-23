"""Produce out-of-fold and test predictions for a neural fraud model."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

COMPETITION = "playground-series-s6e11"
TARGET = "is_fraud"
DIGIT_COLS = (
    "amount_usd", "account_balance_usd", "merchant_risk_score",
    "velocity_score", "distance_from_home_km", "hours_since_last_txn",
)


def get_data():
    for path in Path("/kaggle/input").rglob("train.csv"):
        if COMPETITION in path.parts and (path.parent / "sample_submission.csv").is_file():
            return path.parent
    return Path("data")


def input_views(train, valid, test, learn_rows):
    """Reuse the same raw and label-free decimal features for both networks."""
    frames = [train.copy(), valid.copy(), test.copy()]
    for name in train:
        if pd.api.types.is_bool_dtype(train[name]):
            for frame in frames:
                frame[name] = frame[name].astype(np.float32)
        elif pd.api.types.is_numeric_dtype(train[name]):
            for frame in frames:
                frame[name] = frame[name].astype(np.float32)
        else:
            categories = pd.Index(train[name].iloc[learn_rows].dropna().unique())
            for frame in frames:
                frame[name] = pd.Categorical(frame[name], categories=categories)

    for name in DIGIT_COLS:
        keys = [np.rint(frame[name].astype(np.float64) * 100) for frame in (train, valid, test)]
        frequency = keys[0].iloc[learn_rows].value_counts()
        for frame, cents in zip(frames, keys, strict=True):
            frame[f"{name}__fraction"] = (cents.abs() % 100).astype(np.float32)
            frame[f"{name}__digit_support"] = np.log1p(cents.map(frequency).fillna(0)).astype(np.float32)
    return frames


def estimator(name, seed, device, epochs):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    common = dict(
        device=device, random_state=seed, n_threads=4, n_cv=1, n_refit=0,
        n_epochs=epochs, val_metric_name="1-auc_ovr", verbosity=1,
    )
    if name == "realmlp":
        return RealMLP_TD_Classifier(
            **common, batch_size=1024, hidden_width=192, predict_batch_size=4096,
            use_early_stopping=True, early_stopping_additive_patience=5,
        )
    return TabM_D_Classifier(
        **common, arch_type="tabm-mini", tabm_k=8, num_emb_type="pwl",
        num_emb_n_bins=32, d_embedding=8, d_block=128, n_blocks=3,
        batch_size=1024, patience=5, compile_model=False, allow_amp=True,
    )


def run(args):
    if args.install:
        if importlib.util.find_spec("pytabkit") is None or importlib.util.find_spec("rtdl_num_embeddings") is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "pytabkit==1.8.0", "rtdl-num-embeddings==0.0.12"],
                check=True,
            )
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no Kaggle GPU is available.")
    start = time.monotonic()
    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    sample = pd.read_csv(args.data / "sample_submission.csv")
    train_ids = train.pop("id")
    test_ids = test.pop("id")
    y = train.pop(TARGET).to_numpy()
    cvv_zero = train["cvv_retry_count"].eq(0).to_numpy()
    if set(train.columns) != set(test.columns) or not sample["id"].equals(test_ids):
        raise ValueError("Unexpected feature schema or test IDs.")
    test = test[train.columns]
    oof = np.full(len(train), np.nan)
    prediction = np.zeros(len(test))
    assigned = np.zeros(len(train), dtype=np.int8)
    scores = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit, val) in enumerate(splitter.split(train, y), start=1):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed + fold)
        learn, stop = next(inner.split(fit, y[fit]))
        train_view, valid_view, test_view = input_views(train.iloc[fit], train.iloc[val], test, learn)
        model = estimator(args.model, args.seed + fold, args.device, args.epochs)
        model.fit(train_view, y[fit], val_idxs=stop)
        oof[val] = model.predict_proba(valid_view)[:, 1]
        prediction += model.predict_proba(test_view)[:, 1] / 5
        assigned[val] = fold
        scores.append(float(roc_auc_score(y[val], oof[val])))
        print(f"{args.model} fold {fold}/5 AUC {scores[-1]:.8f}, elapsed {time.monotonic()-start:.0f}s", flush=True)
        del model, train_view, valid_view, test_view
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not np.isfinite(oof).all() or not np.isfinite(prediction).all():
        raise ValueError("Missing or non-finite model output.")
    if not np.all((prediction >= 0) & (prediction <= 1)):
        raise ValueError("Test probabilities must be within [0, 1].")
    args.output.mkdir(parents=True, exist_ok=True)
    sample.assign(**{TARGET: prediction}).to_csv(args.output / "submission.csv", index=False)
    np.savez_compressed(
        args.output / "oof.npz", id=train_ids.to_numpy(), fold=assigned, is_fraud=y,
        cvv_zero=cvv_zero, **{args.model: oof},
    )
    metrics = {
        "model": args.model, "seed": args.seed, "fold_auc": scores,
        "oof_auc": float(roc_auc_score(y, oof)), "rows": {"train": len(train), "test": len(test)},
        "oof_auc_cvv_zero": float(roc_auc_score(y[cvv_zero], oof[cvv_zero])),
        "epochs_limit": args.epochs, "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("realmlp", "tabm"), default="realmlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=get_data())
    in_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working" if in_kaggle else "submissions/neural"))
    parser.add_argument("--install", action=argparse.BooleanOptionalAction, default=in_kaggle)
    run(parser.parse_args())
