"""v10 pilot (gate 1): fine-tune Laya (RLCD) on serialized transactions, with soft labels from a teacher.

Fold 0 of StratifiedKFold(5, seed 2027). Laya is trained on a stratified sample of the fit rows and
scored on all 139,992 rows of the held-out fold, plus the test set. Two runs share the machine,
one per T4:
  a0    hard labels (Laya's default label smoothing 0.05)
  a30   soft gold = 0.7 * y + 0.3 * teacher, teacher = goodpjw2008's wide LM stack (nested OOF),
        Platt-scaled on the fit rows only. Held-out rows never see the teacher.

Each row becomes a Laya state (JSON of column -> value) with one noul question. The score is the
logit gap true - false; temperature calibration is skipped because it cannot change AUC.

Gate 1 passes when the held-out AUC is >= 0.948 and the Spearman correlation with the wide stack
on the same rows is < 0.97.

  python laya_pilot.py                 (parent: data, tokenization, two workers, evaluation)
  python laya_pilot.py --worker a30    (one training run, launched by the parent)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2027
FOLD = 0
BASE_MODEL, SUBFOLDER = 'convaiinnovations/laya', 'multilingual'
QUESTION = {'t': 'noul', 'ins': 'Is this card transaction fraudulent?', 'crit': {}}
RUNS = {'a0': 0.0, 'a30': 0.3}
FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']
LABEL_SMOOTH = 0.05


def find(root, name, inside=None):
    hits = sorted(p for p in Path(root).rglob(name) if p.is_file() and (inside is None or inside in str(p)))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def zones(frame):
    clean = frame.cvv_retry_count.eq(0) & frame.prior_disputes.eq(0) & ~frame[FLAGS].astype(bool).any(axis=1)
    r, v = frame.merchant_risk_score, frame.velocity_score
    return ((r < 13.1) | ((v <= 4.1) & (r <= 20)) |
            (clean & (((v <= 4.1) & (r <= 33.8)) | ((v <= 6) & (r <= 25))))).to_numpy()


def lr(v):
    from scipy.special import logit
    from scipy.stats import rankdata
    return logit((rankdata(v) - 0.5) / len(v))


def model_dir():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    pre = f'{SUBFOLDER}/'
    repo = snapshot_download(BASE_MODEL, allow_patterns=[pre + n for n in (
        'rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*')])
    d = os.path.join(repo, SUBFOLDER)
    _fix_tokenizer_config(d)
    return d


# ---------------------------------------------------------------- parent
def tokenize(tok, cfg, frame, cols):
    """Padded int32 id matrix: fixed question head + serialized state, clamped to max_len."""
    from laya.common import build_head, serialize_state, uses_parallel_layout
    assert not uses_parallel_layout(cfg), 'pilot assumes the sequential option layout'
    head, markers, _ = build_head(tok, QUESTION, cfg['head_max_len'])
    room = cfg['max_len'] - len(head) - 1
    texts = [serialize_state({c: str(v) for c, v in zip(cols, row)}).replace(tok.mask_token, ' ')
             for row in frame[cols].itertuples(index=False, name=None)]
    out = []
    for i in range(0, len(texts), 20000):
        enc = tok(texts[i:i + 20000], add_special_tokens=False)['input_ids']
        out.extend(head + e[:room] + [tok.sep_token_id] for e in enc)
    L = max(map(len, out))
    ids = np.full((len(out), L), tok.pad_token_id, dtype=np.int32)
    lens = np.array([len(s) for s in out], dtype=np.int32)
    for i, s in enumerate(out):
        ids[i, :len(s)] = s
    return ids, lens, markers


def parent(args):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoTokenizer

    t0 = time.time()
    work = args.output / 'pilot'
    work.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(find(args.input, 'train.csv', 'playground-series-s6e11'))
    test = pd.read_csv(find(args.input, 'test.csv', 'playground-series-s6e11'))
    y = train.is_fraud.to_numpy().astype(int)
    cols = [c for c in train.columns if c not in ('id', 'is_fraud')]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    fit, val = folds[FOLD]
    rng = np.random.RandomState(SEED)
    pos, neg = fit[y[fit] == 1], fit[y[fit] == 0]
    share = args.train_rows / len(fit)
    sub = np.sort(np.r_[rng.choice(pos, int(round(len(pos) * share)), replace=False),
                        rng.choice(neg, args.train_rows - int(round(len(pos) * share)), replace=False)])

    wide_oof = np.load(find(args.input, 'wide_lm_stack_nested.npy'))
    wide_test = np.load(find(args.input, 'wide_lm_stack_test.npy'))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lr(wide_oof)[fit, None], y[fit])
    teacher = platt.predict_proba(lr(wide_oof)[sub, None])[:, 1]

    mdir = model_dir()
    cfg = json.load(open(os.path.join(mdir, 'rl_agent_config.json')))
    tok = AutoTokenizer.from_pretrained(os.path.join(mdir, 'tokenizer'))
    for name, rows, frame in (('train', sub, train.iloc[sub]), ('val', val, train.iloc[val]), ('test', None, test)):
        ids, lens, markers = tokenize(tok, cfg, frame, cols)
        np.save(work / f'{name}_ids.npy', ids)
        np.save(work / f'{name}_lens.npy', lens)
        print(f'{name}: {ids.shape} tokens, mean length {lens.mean():.0f}, {time.time() - t0:.0f}s', flush=True)
    print('example:', tok.decode(np.load(work / 'val_ids.npy')[0][:np.load(work / 'val_lens.npy')[0]]), flush=True)
    np.save(work / 'train_y.npy', y[sub].astype(np.float32))
    np.save(work / 'train_teacher.npy', teacher.astype(np.float32))
    json.dump({'markers': markers, 'model_dir': mdir, 'pad_id': tok.pad_token_id},
              open(work / 'meta.json', 'w'))

    import torch
    n_gpu = torch.cuda.device_count()
    print('GPUs', n_gpu, flush=True)
    procs = []
    for i, run in enumerate(RUNS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, n_gpu)))
        cmd = [sys.executable, __file__, '--worker', run, '--output', str(args.output),
               '--epochs', str(args.epochs), '--micro-batch', str(args.micro_batch),
               '--max-train-minutes', str(args.max_train_minutes)]
        log = open(work / f'{run}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((run, p, log))
        if n_gpu < 2:
            p.wait()
    for run, p, log in procs:
        p.wait()
        log.close()
        print(f'--- {run} exit {p.returncode}\n' + (work / f'{run}.log').read_text()[-3000:], flush=True)

    # ---------------- evaluation
    d_oof = pd.read_csv(find(args.input, 'oof_wall_stack.csv')).sort_values('id')
    d_test = pd.read_csv(find(args.input, 'test_wall_stack.csv')).sort_values('id')
    assert np.array_equal(d_oof.id, train.id) and np.array_equal(d_test.id, test.id)
    v8_oof = (lr(wide_oof) + lr(d_oof.stack_all.to_numpy())) / 2
    v8_test = (lr(wide_test) + lr(d_test.stack_all.to_numpy())) / 2
    z_tr, z_te = zones(train), zones(test)
    yv, zv = y[val], z_tr[val]
    cvv0 = train.cvv_retry_count.to_numpy()[val] == 0
    v8_val = lr(v8_oof[val]) - 100 * zv
    res = {'fold': FOLD, 'train_rows': int(len(sub)), 'val_rows': int(len(val)),
           'v8_val_auc_zones': roc_auc_score(yv, v8_val), 'wide_val_auc': roc_auc_score(yv, wide_oof[val]),
           'runs': {}}
    best, best_auc = None, -1
    for run in RUNS:
        f_val, f_test = work / f'{run}_val.npy', work / f'{run}_test.npy'
        if not (f_val.is_file() and f_test.is_file()):
            res['runs'][run] = {'error': 'no predictions'}
            continue
        s_val, s_test = np.load(f_val), np.load(f_test)
        r = {'auc': roc_auc_score(yv, s_val), 'auc_cvv0': roc_auc_score(yv[cvv0], s_val[cvv0]),
             'spearman_wide': float(spearmanr(s_val, wide_oof[val]).correlation),
             'train_info': json.load(open(work / f'{run}_info.json'))}
        r['gate1'] = bool(r['auc'] >= 0.948 and r['spearman_wide'] < 0.97)
        # A fixed, pre-registered small weight next to v8 on the held-out fold (no weight is tuned).
        for w in (0.05, 0.1, 0.2):
            r[f'v8_plus_{w}'] = roc_auc_score(yv, lr(v8_oof[val]) + w * lr(s_val) - 100 * zv) - res['v8_val_auc_zones']
        res['runs'][run] = r
        pred = lr(s_test) - 100 * z_te
        pd.DataFrame({'id': test.id, 'is_fraud': (np.argsort(np.argsort(pred)) + 0.5) / len(pred)}).to_csv(
            args.output / f'laya_{run}.csv', index=False)
        if r['auc'] > best_auc:
            best, best_auc = run, r['auc']
    if best is not None:
        s_test = np.load(work / f'{best}_test.npy')
        combo = lr(v8_test) + 0.1 * lr(s_test) - 100 * z_te
        pd.DataFrame({'id': test.id, 'is_fraud': (np.argsort(np.argsort(combo)) + 0.5) / len(combo)}).to_csv(
            args.output / 'v8_plus_laya.csv', index=False)
        res['combo_run'] = best
    res['minutes'] = round((time.time() - t0) / 60, 1)
    (args.output / 'laya_pilot_results.json').write_text(json.dumps(res, indent=2) + '\n')
    for f in work.glob('*_ids.npy'):
        f.unlink()
    print(json.dumps(res, indent=2), flush=True)


# ---------------------------------------------------------------- worker
def worker(args):
    import torch
    from safetensors.torch import load_file
    from laya.common import QTYPES, build_model, proper_reward

    run, alpha = args.worker, RUNS[args.worker]
    work = args.output / 'pilot'
    meta = json.load(open(work / 'meta.json'))
    markers, pad_id = meta['markers'], meta['pad_id']
    cfg = json.load(open(os.path.join(meta['model_dir'], 'rl_agent_config.json')))
    torch.manual_seed(SEED)
    dev = torch.device('cuda')
    model = build_model(cfg, encoder_dir=os.path.join(meta['model_dir'], 'encoder'))
    model.load_state_dict(load_file(os.path.join(meta['model_dir'], 'model.safetensors')), strict=True)
    if hasattr(model.encoder.config, 'reference_compile'):
        model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(dev).train()

    ids, lens = np.load(work / 'train_ids.npy'), np.load(work / 'train_lens.npy')
    y, teacher = np.load(work / 'train_y.npy'), np.load(work / 'train_teacher.npy')
    hard = y * (1 - LABEL_SMOOTH) + (1 - y) * LABEL_SMOOTH
    p_true = (1 - alpha) * hard + alpha * teacher
    target_all = np.stack([1 - p_true, p_true], 1).astype(np.float32)
    K = len(markers)
    mpos = torch.tensor(markers, device=dev)[None].expand(args.micro_batch, K)
    qt = QTYPES['noul']

    def batch(rows, ids_m, lens_m):
        L = int(lens_m[rows].max())
        x = torch.from_numpy(ids_m[rows, :L].astype(np.int64)).to(dev)
        att = (torch.arange(L, device=dev)[None] < torch.from_numpy(lens_m[rows]).to(dev)[:, None]).long()
        n = len(rows)
        return x, att, mpos[:1].expand(n, K), torch.ones((n, K), dtype=torch.bool, device=dev), \
            torch.full((n,), qt, dtype=torch.long, device=dev)

    enc_p = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    head_p = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': 2.5e-5}, {'params': head_p, 'lr': 1e-4}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(y) / args.micro_batch)
    warm = min(200, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))) + 1e-3)
    scaler = torch.amp.GradScaler('cuda')
    G, t0, step, stopped = 4, time.time(), 0, False
    rng = np.random.RandomState(SEED + int(alpha * 100))
    for ep in range(args.epochs):
        order = rng.permutation(len(y))
        for b0 in range(0, len(order), args.micro_batch):
            rows = order[b0:b0 + args.micro_batch]
            sigma = 0.4 + (0.1 - 0.4) * step / max(1, steps - 1)
            x, att, mp, mm, qty = batch(rows, ids, lens)
            target = torch.from_numpy(target_all[rows]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, act = model(x, att, mp, mm, qty)
            logits = logits.float()
            eps = torch.randn((G,) + logits.shape, device=dev) * sigma
            eps = eps - eps.mean(-1, keepdim=True)
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z, -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qty, mm, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -((z - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
            loss = -(adv * logp).mean() - (target * torch.log_softmax(logits, -1)).sum(-1).mean() + 0.0 * act.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f'{run} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
            if (time.time() - t0) / 60 > args.max_train_minutes:
                stopped = True
                break
        if stopped:
            break
    train_min = (time.time() - t0) / 60
    print(f'{run} trained {step} steps in {train_min:.1f} min (stopped early: {stopped})', flush=True)

    model.eval()

    @torch.no_grad()
    def score(name):
        ids_m, lens_m = np.load(work / f'{name}_ids.npy'), np.load(work / f'{name}_lens.npy')
        out = np.zeros(len(lens_m), dtype=np.float32)
        order = np.argsort(lens_m)
        for b0 in range(0, len(order), 256):
            rows = order[b0:b0 + 256]
            x, att, mp, mm, qty = batch(rows, ids_m, lens_m)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, _ = model(x, att, mp, mm, qty)
            out[rows] = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
        return out

    t1 = time.time()
    np.save(work / f'{run}_val.npy', score('val'))
    np.save(work / f'{run}_test.npy', score('test'))
    json.dump({'alpha': alpha, 'steps': step, 'planned_steps': steps, 'stopped_early': stopped,
               'train_minutes': round(train_min, 1), 'score_minutes': round((time.time() - t1) / 60, 1)},
              open(work / f'{run}_info.json', 'w'))
    print(f'{run} scored in {(time.time() - t1) / 60:.1f} min', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--worker', choices=sorted(RUNS))
    ap.add_argument('--train-rows', type=int, default=100000)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--micro-batch', type=int, default=32)
    ap.add_argument('--max-train-minutes', type=float, default=150)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.5.3'], check=True)
    parent(args)


if __name__ == '__main__':
    main()
