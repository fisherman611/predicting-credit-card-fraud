"""Compare GPT-2 token features with v1 using the locked S6E11 split."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import TargetEncoder

# Local imports are omitted when the two sources are bundled for Kaggle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (
    DIGIT_COLUMNS, SEED, TARGET, TREE_WEIGHT, blend_probabilities, competition_data,
    cvv_auc_summary, decimal_keys, fold_features, predict_glm, predict_tree, top5_summary,
)


VARIANTS = ("v1", "token_frequency", "token_target")
LOCKED_SEED = 8128


def token_keys(frame, encoding):
    keys = {}
    for name in DIGIT_COLUMNS:
        raw = frame[name].to_numpy(dtype=float)
        finite = np.isfinite(raw)
        cents = np.zeros(len(raw), dtype=np.int64)
        cents[finite] = np.rint(raw[finite] * 100).astype(np.int64)
        values, inverse = np.unique(cents, return_inverse=True)
        phrases = [f" {value/100:.2f}".rstrip("0").rstrip(".") for value in values]
        tokens = encoding.encode_ordinary_batch(phrases, num_threads=4)
        groups = {
            "head": [seq[0] for seq in tokens],
            "second": [seq[1] if len(seq) > 1 else -1 for seq in tokens],
            "last": [seq[-1] for seq in tokens],
            "last_pair": [
                (seq[-2] + 1) * encoding.n_vocab + seq[-1] if len(seq) > 1 else -1
                for seq in tokens
            ],
        }
        for kind, values in groups.items():
            keys[f"{name}__bpe_{kind}"] = np.where(finite, np.asarray(values)[inverse], -2)
    return pd.DataFrame(keys, index=frame.index)


def fit_token_models(raw, digits, keys, y, rounds, requested):
    _, glm_keys, views = fold_features(raw, digits, y)
    results = {}
    if "v1" in requested:
        results["v1"] = predict_tree(views, y, rounds)
    if "token_frequency" in requested or "token_target" in requested:
        for name in keys[0]:
            counts = keys[0][name].value_counts()
            for view, key in zip(views, keys, strict=True):
                view[f"freq__{name}"] = np.log1p(key[name].map(counts).fillna(0)).astype(np.float32)
        if "token_frequency" in requested:
            results["token_frequency"] = predict_tree(views, y, rounds)
    if "token_target" in requested:
        encoder = TargetEncoder(target_type="binary", smooth=50.0, cv=3, random_state=SEED)
        encoded = [
            encoder.fit_transform(keys[0], y),
            encoder.transform(keys[1]),
            encoder.transform(keys[2]),
        ]
        for view, rates in zip(views, encoded, strict=True):
            view[[f"bpe_te__{name}" for name in keys[0]]] = rates.astype(np.float32)
        results["token_target"] = predict_tree(views, y, rounds)

    glm = predict_glm(glm_keys, y)
    return {name: tuple(blend_probabilities(a, b, TREE_WEIGHT) for a, b in zip(scores, glm, strict=True))
            for name, scores in results.items()}


def score_set(y, scores, group):
    return {
        "auc": float(roc_auc_score(y, scores)),
        "cvv": cvv_auc_summary(y, scores, group),
        "top_5pct": top5_summary(y, scores, group),
    }


def run_tokens(args):
    if args.install_tokenizer:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tiktoken==0.12.0"], check=True)
    import tiktoken

    start = time.monotonic()
    encoding = tiktoken.get_encoding("gpt2")
    train = pd.read_csv(args.data / "train.csv")
    ids = train.pop("id").to_numpy()
    target = train.pop(TARGET).to_numpy()
    group = train["cvv_retry_count"].eq(0).to_numpy()
    raw_digits = decimal_keys(train)
    keys = token_keys(train, encoding)
    dev, holdout = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=LOCKED_SEED,
    ).split(train, target))
    holdout.sort()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"stage": args.stage, "locked_seed": LOCKED_SEED, "rounds": args.rounds}

    if args.stage == "development":
        oof = {name: np.full(len(dev), np.nan) for name in VARIANTS}
        folds = np.zeros(len(dev), dtype=np.int8)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        per_fold = []
        for fold, (fit_idx, val_idx) in enumerate(cv.split(dev, target[dev]), start=1):
            fit, val = dev[fit_idx], dev[val_idx]
            scores = fit_token_models(
                (train.iloc[fit], train.iloc[val], train.iloc[val[:1]]),
                (raw_digits.iloc[fit], raw_digits.iloc[val], raw_digits.iloc[val[:1]]),
                (keys.iloc[fit], keys.iloc[val], keys.iloc[val[:1]]),
                target[fit], args.rounds, VARIANTS,
            )
            for name, (p, _) in scores.items():
                oof[name][val_idx] = p
            folds[val_idx] = fold
            auc = {name: float(roc_auc_score(target[val], p)) for name, (p, _) in scores.items()}
            per_fold.append(auc)
            print(f"Development fold {fold}/3: {auc}, elapsed {time.monotonic()-start:.0f}s", flush=True)
        report["fold_auc"] = per_fold
        report["development_rows"] = len(dev)
        report["metrics"] = {name: score_set(target[dev], p, group[dev]) for name, p in oof.items()}
        np.savez_compressed(args.output / "dev-oof.npz", id=ids[dev], fold=folds, is_fraud=target[dev],
                            cvv_zero=group[dev], **oof)
    elif args.stage == "holdout":
        scores = fit_token_models(
            (train.iloc[dev], train.iloc[holdout], train.iloc[holdout[:1]]),
            (raw_digits.iloc[dev], raw_digits.iloc[holdout], raw_digits.iloc[holdout[:1]]),
            (keys.iloc[dev], keys.iloc[holdout], keys.iloc[holdout[:1]]),
            target[dev], args.rounds, ("v1", args.variant),
        )
        pred = {name: p for name, (p, _) in scores.items()}
        report["metrics"] = {name: score_set(target[holdout], p, group[holdout]) for name, p in pred.items()}
        report["holdout_rows"] = len(holdout)
        np.savez_compressed(args.output / "holdout.npz", id=ids[holdout], is_fraud=target[holdout],
                            cvv_zero=group[holdout], **pred)
    else:
        test = pd.read_csv(args.data / "test.csv")
        test_ids = test.pop("id")
        sample = pd.read_csv(args.data / "sample_submission.csv")
        if not test_ids.equals(sample["id"]) or not test_ids.is_unique or set(test) != set(train):
            raise ValueError("Unexpected competition test rows or columns.")
        test = test[train.columns]
        results = fit_token_models(
            (train, test.iloc[:1], test), (raw_digits, decimal_keys(test.iloc[:1]), decimal_keys(test)),
            (keys, token_keys(test.iloc[:1], encoding), token_keys(test, encoding)),
            target, args.rounds, (args.variant,),
        )
        sample[TARGET] = results[args.variant][1]
        if not sample[TARGET].between(0, 1).all():
            raise ValueError("Invalid test scores.")
        sample.to_csv(args.output / "submission.csv", index=False)
        report.update(variant=args.variant, train_rows=len(train), test_rows=len(test))
    report["elapsed_seconds"] = round(time.monotonic() - start, 1)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    on_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working" if on_kaggle else "submissions/tokens"))
    parser.add_argument("--stage", choices=("development", "holdout", "full"), default="development")
    parser.add_argument("--variant", choices=VARIANTS, default="token_target")
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--install-tokenizer", action=argparse.BooleanOptionalAction, default=on_kaggle)
    run_tokens(parser.parse_args())
