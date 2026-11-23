"""Fit a compact field Transformer on the three development folds."""

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

# Local imports are omitted when the sources are bundled for Kaggle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokens.train import LOCKED_SEED, score_set, token_keys
from train import DIGIT_COLUMNS, SEED, TARGET, competition_data


def encode_views(fit, valid, fit_keys, valid_keys, vocab):
    arrays = [np.zeros((len(frame), len(fit.columns), 3), dtype=np.int32) for frame in (fit, valid)]
    next_id = vocab + 2
    for j, name in enumerate(fit):
        if name in DIGIT_COLUMNS:
            cols = [f"{name}__bpe_{part}" for part in ("head", "second", "last")]
            for dest, keys in zip(arrays, (fit_keys, valid_keys), strict=True):
                dest[:, j] = keys[cols].to_numpy() + 2
        else:
            levels = pd.Index(fit[name].dropna().unique())
            for dest, frame in zip(arrays, (fit, valid), strict=True):
                codes = levels.get_indexer(frame[name])
                dest[:, j, 0] = np.where(codes >= 0, codes + next_id, 1)
            next_id += len(levels)
    return arrays, next_id


def train_sequence(fit, valid, fit_keys, valid_keys, y, vocab, seed, epochs, device):
    import torch
    from torch import nn

    np.random.seed(seed)
    torch.manual_seed(seed)
    encoded, size = encode_views(fit, valid, fit_keys, valid_keys, vocab)
    x, check = [torch.as_tensor(a, device=device) for a in encoded]
    target = torch.as_tensor(y, dtype=torch.float32, device=device)

    class FieldTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            width = 48
            self.embedding = nn.Embedding(size, 16, padding_idx=0)
            self.fields = nn.Parameter(torch.randn(1, len(fit.columns), width) * 0.02)
            layer = nn.TransformerEncoderLayer(width, 4, 96, dropout=0.10, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            with torch.no_grad():
                self.readout[-1].bias.fill_(np.log(float(y.mean()) / (1 - float(y.mean()))))

        def forward(self, tokens):
            data = self.embedding(tokens).flatten(2)
            return self.readout(self.encoder(data + self.fields).mean(dim=1)).squeeze(1)

    model = FieldTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    amp = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    dev_type = torch.device(device).type
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(target), device=device).split(1024):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=amp):
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]), target[batch])
            loss.backward()
            optimizer.step()
        print(f"Transformer epoch {epoch}/{epochs}, last batch loss {loss.item():.5f}", flush=True)
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).float().sigmoid().cpu() for batch in check.split(2048)]).numpy()


def run_sequence(args):
    if args.install_tokenizer:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tiktoken==0.12.0"], check=True)
    import tiktoken
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The sequence experiment requires a GPU.")
    if args.device == "cpu":
        torch.set_num_threads(4)
    start = time.monotonic()
    encoding = tiktoken.get_encoding("gpt2")
    frame = pd.read_csv(args.data / "train.csv")
    ids = frame.pop("id").to_numpy()
    target = frame.pop(TARGET).to_numpy()
    group = frame["cvv_retry_count"].eq(0).to_numpy()
    keys = token_keys(frame, encoding)
    dev, holdout = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=LOCKED_SEED,
    ).split(frame, target))
    holdout.sort()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"stage": args.stage, "locked_seed": LOCKED_SEED, "epochs": args.epochs}
    if args.stage == "development":
        oof = np.full(len(dev), np.nan)
        folds = np.zeros(len(dev), dtype=np.int8)
        report["fold_auc"] = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for fold, (fit_idx, val_idx) in enumerate(cv.split(dev, target[dev]), start=1):
            fit, val = dev[fit_idx], dev[val_idx]
            p = train_sequence(frame.iloc[fit], frame.iloc[val], keys.iloc[fit], keys.iloc[val],
                               target[fit], encoding.n_vocab, SEED+fold, args.epochs, args.device)
            oof[val_idx], folds[val_idx] = p, fold
            auc = float(roc_auc_score(target[val], p))
            report["fold_auc"].append(auc)
            print(f"Transformer development fold {fold}/3 AUC {auc:.8f}, elapsed {time.monotonic()-start:.0f}s", flush=True)
        report["metrics"] = score_set(target[dev], oof, group[dev])
        np.savez_compressed(args.output / "dev-oof.npz", id=ids[dev], fold=folds,
                            is_fraud=target[dev], cvv_zero=group[dev], transformer=oof)
    elif args.stage == "holdout":
        p = train_sequence(frame.iloc[dev], frame.iloc[holdout], keys.iloc[dev], keys.iloc[holdout],
                           target[dev], encoding.n_vocab, SEED+1, args.epochs, args.device)
        report["metrics"] = score_set(target[holdout], p, group[holdout])
        np.savez_compressed(args.output / "holdout.npz", id=ids[holdout], is_fraud=target[holdout],
                            cvv_zero=group[holdout], transformer=p)
    else:
        test = pd.read_csv(args.data / "test.csv")
        test_ids = test.pop("id")
        sample = pd.read_csv(args.data / "sample_submission.csv")
        if not sample["id"].equals(test_ids) or set(test) != set(frame):
            raise ValueError("Unexpected competition test rows or columns.")
        test = test[frame.columns]
        p = train_sequence(frame, test, keys, token_keys(test, encoding), target,
                           encoding.n_vocab, SEED+1, args.epochs, args.device)
        sample[TARGET] = p
        if not sample[TARGET].between(0, 1).all():
            raise ValueError("Invalid test probabilities.")
        sample.to_csv(args.output / "submission.csv", index=False)
    report["elapsed_seconds"] = round(time.monotonic() - start, 1)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=competition_data())
    on_kaggle = Path("/kaggle/working").is_dir()
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working" if on_kaggle else "submissions/token-transformer"))
    parser.add_argument("--stage", choices=("development", "holdout", "full"), default="development")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--install-tokenizer", action=argparse.BooleanOptionalAction, default=on_kaggle)
    run_sequence(parser.parse_args())
