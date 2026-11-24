"""v7 character-level digit Transformer.

Every numeric value is written as a fixed-width string (integer digits padded,
two decimals) and each character becomes its own token with a learned
(column, position) embedding. Integer, boolean and categorical columns are one
token each. No target encoding is used, so the signal is learned only from the
raw digit strings. Trained on StratifiedKFold(5, seed 2027), the stack folds.

Writes oof_v7_digit.npy / test_v7_digit.npy and digit_results.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 2027
DECIMAL = ('amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd',
           'velocity_score', 'merchant_risk_score')


def find(root, name):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def tokenize(full):
    """Return int32 tokens (rows, length) and the vocabulary size.

    Token ids are unique per (column, position, character) so a single
    embedding table carries both the symbol and where it sits.
    """
    blocks, offset = [], 1  # 0 is unused
    for name in full.columns:
        if name in DECIMAL:
            cents = np.rint(np.abs(full[name].to_numpy(np.float64)) * 100).astype(np.int64)
            sign = (full[name].to_numpy() < 0).astype(np.int64)
            width = max(3, len(str(int(cents.max()))))
            digits = np.stack([(cents // 10 ** p) % 10 for p in range(width - 1, -1, -1)], axis=1)
            # Leading zeros get their own symbol (10) so magnitude is explicit.
            lead = np.cumsum(digits, axis=1) == 0
            lead[:, -3:] = False
            digits = np.where(lead, 10, digits)
            for p in range(width):
                blocks.append(digits[:, p] + offset)
                offset += 11
            blocks.append(sign + offset)
            offset += 2
        else:
            codes, uniq = pd.factorize(full[name].astype(str))
            blocks.append(codes + offset)
            offset += len(uniq)
    return np.stack(blocks, axis=1).astype(np.int32), offset


def train_fold(tok_fit, y_fit, tok_apply, vocab, seed, args, device):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.width
            self.emb = nn.Embedding(vocab, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, args.heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, args.layers, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            prior = float(y_fit.mean())
            with torch.no_grad():
                self.out[-1].bias.fill_(np.log(prior / (1 - prior)))

        def forward(self, t):
            h = self.emb(t)
            h = torch.cat([self.cls.expand(len(t), -1, -1), h], 1)
            return self.out(self.enc(h)[:, 0]).squeeze(-1)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_fit))
    k = len(perm) // 20
    es, tr = perm[:k], perm[k:]
    X = torch.as_tensor(tok_fit, device=device, dtype=torch.long)
    Y = torch.as_tensor(y_fit, device=device, dtype=torch.float32)
    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def predict(T):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
            for i in range(0, len(T), 16384):
                out.append(model(torch.as_tensor(T[i:i + 16384], device=device, dtype=torch.long)).float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    best, best_state, history = -1.0, None, []
    tr_t = torch.as_tensor(tr, device=device)
    for epoch in range(args.epochs):
        order = tr_t[torch.randperm(len(tr_t), device=device)]
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(X[b]), Y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        score = roc_auc_score(y_fit[es], predict(tok_fit[es]))
        history.append(round(score, 6))
        if score > best:
            best = score
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return [predict(T) for T in tok_apply], history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--folds', type=int, nargs='*', default=None)
    ap.add_argument('--subsample', type=int, default=0, help='debug: rows per fold')
    args = ap.parse_args()
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = pd.read_csv(find(args.input, 'train.csv'))
    test = pd.read_csv(find(args.input, 'test.csv'))
    y = train.pop('is_fraud').to_numpy().astype(int)
    n = len(train)
    full = pd.concat([train.drop(columns='id'), test.drop(columns='id')], ignore_index=True)
    tokens, vocab = tokenize(full)
    tok_tr, tok_te = tokens[:n], tokens[n:]
    print(f'device={device} tokens={tokens.shape} vocab={vocab}', flush=True)
    cvv0 = train.cvv_retry_count.to_numpy() == 0
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(n), y))
    oof, tst, log = np.full(n, np.nan), np.zeros(len(test)), []
    for f in (range(5) if args.folds is None else args.folds):
        fit, val = folds[f]
        if args.subsample:
            fit = np.random.RandomState(f).choice(fit, args.subsample, replace=False)
        ts = time.time()
        (pv, pt), hist = train_fold(tok_tr[fit], y[fit], [tok_tr[val], tok_te], vocab, SEED + f, args, device)
        oof[val], tst = pv, tst + pt / 5
        row = {'fold': f, 'seconds': round(time.time() - ts), 'es_auc_by_epoch': hist,
               'auc': roc_auc_score(y[val], pv), 'cvv0_auc': roc_auc_score(y[val][cvv0[val]], pv[cvv0[val]])}
        log.append(row)
        print(json.dumps(row), flush=True)
    done = ~np.isnan(oof)
    summary = {'folds': log, 'seconds': round(time.time() - t0), 'args': {k: str(v) for k, v in vars(args).items()},
               'auc': roc_auc_score(y[done], oof[done]), 'cvv0_auc': roc_auc_score(y[done & cvv0], oof[done & cvv0])}
    if done.all():
        np.save(args.output / 'oof_v7_digit.npy', oof.astype(np.float32))
        np.save(args.output / 'test_v7_digit.npy', tst.astype(np.float32))
    (args.output / 'digit_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('folds', 'args')}, indent=2))


if __name__ == '__main__':
    main()
