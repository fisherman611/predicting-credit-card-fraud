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


def tabm_inputs(raw, val, fit_rows, predict_rows, seed, device):
    """Fit numeric transforms and categorical vocabularies on fit_rows."""
    import torch
    from sklearn.preprocessing import QuantileTransformer

    train_view, val_view, _ = input_views(raw, val, val.iloc[:1], fit_rows)
    fit = train_view.iloc[fit_rows]
    views = [fit, train_view.iloc[predict_rows], val_view] if len(predict_rows) else [fit, val_view]
    numerical = fit.select_dtypes(include="number").columns
    numerical = [name for name in numerical if fit[name].fillna(0).nunique() > 1]
    categorical = fit.select_dtypes(include="category").columns
    normalizer = QuantileTransformer(
        n_quantiles=min(512, len(fit)), subsample=50_000,
        output_distribution="normal", random_state=seed,
    )
    normalizer.fit(fit[numerical].fillna(0))
    x_num = [
        torch.as_tensor(normalizer.transform(frame[numerical].fillna(0)), dtype=torch.float32, device=device)
        for frame in views
    ]
    x_cat = [
        torch.as_tensor(
            np.column_stack([frame[name].cat.codes.to_numpy().astype(np.int64) + 1 for name in categorical]),
            dtype=torch.long, device=device,
        )
        for frame in views
    ]
    cardinalities = [len(fit[name].cat.categories) + 1 for name in categorical]
    return list(zip(x_num, x_cat, strict=True)), cardinalities


def fit_tabm_phase(raw, val, y, fit_rows, select_rows, seed, epochs, device, choose_epoch):
    """Run a selection pass or a fixed-length refit without a validation set."""
    import torch
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM

    torch.manual_seed(seed)
    views, categories = tabm_inputs(raw, val, fit_rows, select_rows, seed, device)
    fit_num, fit_cat = views[0]
    sample_rows = np.random.default_rng(seed).choice(len(fit_num), min(50_000, len(fit_num)), replace=False)
    bins = compute_bins(fit_num[sample_rows].cpu(), n_bins=min(32, len(sample_rows) - 1))
    model = TabM.make(
        n_num_features=fit_num.shape[1], cat_cardinalities=categories, d_out=2,
        arch_type="tabm-mini", k=8, d_block=128, n_blocks=3, dropout=0.1,
        num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=8, activation=False, version="B"),
    ).to(device)
    target = torch.as_tensor(y[fit_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
    use_amp = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    device_type = torch.device(device).type

    def predict(view):
        model.eval()
        num, cat = view
        with torch.inference_mode():
            return torch.cat([
                model(a, b).float().softmax(dim=-1)[:, :, 1].mean(dim=1).cpu()
                for a, b in zip(num.split(2048), cat.split(2048), strict=True)
            ]).numpy()

    best_auc, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(target), device=device).split(1024):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(fit_num[indices], fit_cat[indices])
                labels = target[indices, None].expand(-1, logits.shape[1])
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            optimizer.step()
        if choose_epoch:
            auc = float(roc_auc_score(y[select_rows], predict(views[1])))
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            print(f"TabM epoch {epoch}/{epochs}: inner AUC {auc:.6f}", flush=True)
            if epoch - best_epoch >= 5:
                break
    if choose_epoch:
        model.load_state_dict(best_state)
    return predict(views[-1]), best_epoch if choose_epoch else epochs


def run_locked_holdout(args):
    """Compare inner-selected and full-refit neural scores on a fixed outer holdout."""
    if args.install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "pytabkit==1.8.0", "rtdl-num-embeddings==0.0.12", "tabm==0.0.3"],
            check=True,
        )
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no Kaggle GPU is available.")
    start = time.monotonic()
    full = pd.read_csv(args.data / "train.csv")
    ids = full.pop("id").to_numpy()
    target = full.pop(TARGET).to_numpy()
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=8128)
    fit, val = next(outer.split(full, target))
    val.sort()
    train, validation, y = full.iloc[fit], full.iloc[val], target[fit]
    learn, stop = next(StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed + 1)
                       .split(train, y))
    cvv_zero = validation["cvv_retry_count"].eq(0).to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    models = ("realmlp", "tabm") if args.model == "both" else (args.model,)
    for name in models:
        if name == "realmlp":
            from pytabkit import RealMLP_TD_Classifier

            x, validation_view, _ = input_views(train, validation, validation.iloc[:1], learn)
            model = RealMLP_TD_Classifier(
                device=args.device, random_state=args.seed + 1, n_threads=4,
                n_cv=1, n_refit=1, n_epochs=args.epochs, val_metric_name="1-auc_ovr",
                batch_size=1024, hidden_width=192, predict_batch_size=4096,
                use_early_stopping=True, early_stopping_additive_patience=5, verbosity=1,
            )
            model.fit(x, y, val_idxs=stop)
            refit = model.predict_proba(validation_view)[:, 1]
            fitted_interface = model.alg_interface_
            if fitted_interface is not model.refit_alg_interface_:
                raise RuntimeError("RealMLP did not select the refitted model.")
            model.alg_interface_ = model.cv_alg_interface_
            before = model.predict_proba(validation_view)[:, 1]
            model.alg_interface_ = fitted_interface
            del model
        else:
            before, best_epoch = fit_tabm_phase(
                train, validation, y, learn, stop, args.seed + 1, args.epochs, args.device, True,
            )
            print(f"TabM refit on {len(fit)} rows for {best_epoch} epochs", flush=True)
            refit, _ = fit_tabm_phase(
                train, validation, y, np.arange(len(fit)), np.empty(0, dtype=int),
                args.seed + 1, best_epoch, args.device, False,
            )

        predictions[f"{name}_before_refit"] = before
        predictions[f"{name}_refit"] = refit
        for stage, values in (("before refit", before), ("refit", refit)):
            print(f"{name} {stage}: holdout AUC {roc_auc_score(target[val], values):.8f}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        np.savez_compressed(args.output / "holdout.npz", id=ids[val], is_fraud=target[val],
                            cvv_zero=cvv_zero, **predictions)

    report = {
        "split_seed": 8128,
        "inner_train_rows": len(learn),
        "inner_early_stopping_rows": len(stop),
        "refit_rows": len(fit),
        "holdout_rows": len(val),
        "holdout_fraud": int(target[val].sum()),
        "holdout_auc": {name: float(roc_auc_score(target[val], score)) for name, score in predictions.items()},
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("realmlp", "tabm", "both"), default="realmlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", type=Path, default=get_data())
    in_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working" if in_kaggle else "submissions/neural"))
    parser.add_argument("--install", action=argparse.BooleanOptionalAction, default=in_kaggle)
    parser.add_argument("--locked-holdout", action="store_true")
    args = parser.parse_args()
    if args.locked_holdout:
        run_locked_holdout(args)
    else:
        if args.model == "both":
            parser.error("--model both currently requires --locked-holdout")
        run(args)
