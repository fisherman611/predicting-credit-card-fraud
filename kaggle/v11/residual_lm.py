"""v11: language models trained on what v8 gets wrong (residual / offset learning).

v10b showed that a text model distilled from the stack mostly re-learns the stack (Spearman 0.94 with
the wide stack) and adds nothing next to v8. v11 changes the objective instead of the weights: the
teacher is no longer a target but a fixed offset.

    p(fraud | row) = sigmoid(t(row) + s(row))

t is v8 (rank-logit, Platt-scaled on the fit rows of the fold; out-of-fold for every row) and s is
the language model's score. Only s is trained, on hard labels, so s can only gain by moving rows
whose order v8 gets wrong. At blend time the score is t + beta * s; beta = 0 is v8 exactly.

Two scorers, each producing two logits [no, yes] so the same loss (Laya's RLCD: GRPO on noisy logits
with a proper reward, plus cross-entropy) is used for both:
  laya   convaiinnovations/laya multilingual (mmBERT, 322M), noul question, compact row text
  qwen   Qwen2.5-0.5B-Instruct with LoRA; the score is the log-prob gap of " Yes" vs " No" after
         "Is this transaction fraudulent? Answer:" (an LLM with log-probs). Its zero-shot AUC on
         a val sample is reported before training.

Rows inside the zero-fraud zones are skipped for training and scoring (they go to the bottom anyway).
Fold k of StratifiedKFold(5, seed 2027); outputs per job in /kaggle/working/v11/:
  {model}_f{k}_val.npy, {model}_f{k}_test.npy (residual s on the fold's val rows / all test rows,
  NaN on zone rows), fold{k}_val_idx.npy, {model}_f{k}_info.json
The parent evaluates each job on the held-out fold: AUC of s, of t + s, Spearman with v8, the gain
of t + beta * s over v8 for fixed beta, and an honest split-half gain (beta picked on one half of the
fold, scored on the other, both ways). Gate: honest gain > 0 on both halves.

  python residual_lm.py --jobs laya:0 qwen:0     (parent: one worker per GPU, then evaluation)
  python residual_lm.py --worker laya:0          (one job, launched by the parent)
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2027
LAYA, LAYA_SUB = 'convaiinnovations/laya', 'multilingual'
QWEN = 'Qwen/Qwen2.5-0.5B-Instruct'
QUESTION = {'t': 'noul', 'ins': 'Is this card transaction fraudulent?', 'crit': {}}
PROMPT = 'Card transaction: {}\nIs this transaction fraudulent? Answer:'
FLAGS = ['is_new_merchant', 'is_foreign_transaction', 'used_vpn', 'ip_country_mismatch',
         'billing_shipping_mismatch', 'is_ai_generated_scam_attempt']
LABEL_SMOOTH = 0.005
BETAS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


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


def state_text(cols, row):
    parts, flags = [], []
    for c, v in zip(cols, row):
        if isinstance(v, (bool, np.bool_)):
            if v:
                flags.append(c)
        else:
            parts.append(f'{c}={v}')
    return '; '.join(parts) + '; flags: ' + (', '.join(flags) if flags else 'none')


def v8_scores(root, train, test):
    wide_oof = lr(np.load(find(root, 'wide_lm_stack_nested.npy')))
    wide_test = lr(np.load(find(root, 'wide_lm_stack_test.npy')))
    d_oof = pd.read_csv(find(root, 'oof_wall_stack.csv')).sort_values('id')
    d_test = pd.read_csv(find(root, 'test_wall_stack.csv')).sort_values('id')
    assert np.array_equal(d_oof.id, train.id) and np.array_equal(d_test.id, test.id)
    return (wide_oof + lr(d_oof.stack_all.to_numpy())) / 2, (wide_test + lr(d_test.stack_all.to_numpy())) / 2


# ---------------------------------------------------------------- parent
def parent(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    work = args.output / 'v11'
    work.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(find(args.input, 'train.csv', 'playground-series-s6e11'))
    test = pd.read_csv(find(args.input, 'test.csv', 'playground-series-s6e11'))
    y = train.is_fraud.to_numpy().astype(int)
    cols = [c for c in train.columns if c not in ('id', 'is_fraud')]
    z_tr, z_te = zones(train), zones(test)
    v8_oof, v8_test = v8_scores(args.input, train, test)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))

    texts = {'train': [state_text(cols, r) for r in train[cols].itertuples(index=False, name=None)],
             'test': [state_text(cols, r) for r in test[cols].itertuples(index=False, name=None)]}
    pickle.dump(texts, open(work / 'texts.pkl', 'wb'))
    print('example:', texts['train'][0], f'({time.time() - t0:.0f}s)', flush=True)
    np.save(work / 'test_keep.npy', np.flatnonzero(~z_te))
    np.save(work / 'y_all.npy', y)
    np.save(work / 'n_test.npy', np.array(len(test)))

    for k in sorted({int(j.split(':')[1]) for j in args.jobs}):
        fit, val = folds[k]
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(v8_oof[fit, None], y[fit])
        t_all = platt.decision_function(v8_oof[:, None])
        t_test = platt.decision_function(v8_test[:, None])
        rng = np.random.RandomState(SEED + k)
        pool = fit[~z_tr[fit]]
        pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
        n_pos = int(round(len(pos) * args.train_rows / len(pool)))
        sub = np.sort(np.r_[rng.choice(pos, n_pos, replace=False),
                            rng.choice(neg, args.train_rows - n_pos, replace=False)])
        np.savez(work / f'fold{k}.npz', sub=sub, y_sub=y[sub].astype(np.float32),
                 t_sub=t_all[sub].astype(np.float32), val=val, val_keep=val[~z_tr[val]],
                 t_val=t_all[val], t_test=t_test)
        np.save(work / f'fold{k}_val_idx.npy', val)

    import torch
    n_gpu = max(1, torch.cuda.device_count())
    print('GPUs', n_gpu, flush=True)
    lanes = [[j for i, j in enumerate(args.jobs) if i % n_gpu == g] for g in range(n_gpu)]

    def lane(g, jobs):
        for job in jobs:
            name = job.replace(':', '_f')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [sys.executable, __file__, '--worker', job, '--output', str(args.output),
                   '--epochs', str(args.epochs), '--micro-batch', str(args.micro_batch),
                   '--qwen-rows', str(args.qwen_rows), '--qwen-epochs', str(args.qwen_epochs),
                   '--max-train-minutes', str(args.max_train_minutes)]
            with open(work / f'{name}.log', 'w') as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f'--- {job} exit {rc}\n' + (work / f'{name}.log').read_text()[-2500:], flush=True)

    threads = [threading.Thread(target=lane, args=(g, js)) for g, js in enumerate(lanes) if js]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    evaluate(args, work, train, test, y, z_tr, z_te, v8_oof, folds)
    print(f'total {(time.time() - t0) / 60:.1f} min', flush=True)


def honest_gain(yv, t, s, zv, rng):
    """Pick beta on one stratified half of the fold, score it on the other half, both ways."""
    from sklearn.metrics import roc_auc_score
    idx = np.arange(len(yv))
    a = np.r_[rng.permutation(idx[yv == 1])[:int((yv == 1).sum() // 2)],
              rng.permutation(idx[yv == 0])[:int((yv == 0).sum() // 2)]]
    half = np.zeros(len(yv), bool)
    half[a] = True
    out = []
    for pick, score in ((half, ~half), (~half, half)):
        grid = (0.0,) + BETAS
        aucs = [roc_auc_score(yv[pick], (t + b * s - 100 * zv)[pick]) for b in grid]
        b = grid[int(np.argmax(aucs))]
        out.append({'beta': b, 'gain': roc_auc_score(yv[score], (t + b * s - 100 * zv)[score]) -
                    roc_auc_score(yv[score], (t - 100 * zv)[score])})
    return out


def evaluate(args, work, train, test, y, z_tr, z_te, v8_oof, folds):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    res = {'jobs': {}, 'train_rows': args.train_rows, 'epochs': args.epochs}
    best = None
    for job in args.jobs:
        name = job.replace(':', '_f')
        k = int(job.split(':')[1])
        f_val, f_test = work / f'{name}_val.npy', work / f'{name}_test.npy'
        if not (f_val.is_file() and f_test.is_file()):
            res['jobs'][job] = {'error': 'no predictions'}
            continue
        fd = np.load(work / f'fold{k}.npz')
        val, t = fd['val'], fd['t_val']
        yv, zv = y[val], z_tr[val]
        s = np.nan_to_num(np.load(f_val))
        keep = ~zv
        base = roc_auc_score(yv, t - 100 * zv)
        r = {'v8_auc': base, 'auc_s': roc_auc_score(yv[keep], s[keep]),
             'auc_t_plus_s': roc_auc_score(yv, t + s - 100 * zv),
             'spearman_s_v8': float(spearmanr(s[keep], t[keep]).correlation),
             'spearman_full_v8': float(spearmanr((t + s)[keep], t[keep]).correlation),
             'info': json.load(open(work / f'{name}_info.json'))}
        for b in BETAS:
            r[f'gain_beta_{b}'] = roc_auc_score(yv, t + b * s - 100 * zv) - base
        r['honest'] = honest_gain(yv, t, s, zv, np.random.RandomState(SEED))
        r['gate'] = bool(all(h['gain'] > 0 for h in r['honest']))
        res['jobs'][job] = r
        # Exploratory submission: v8 + beta * s with beta chosen on this fold (not honest across folds).
        b = max(BETAS, key=lambda b: r[f'gain_beta_{b}'])
        s_test = np.nan_to_num(np.load(f_test))
        pred = fd['t_test'] + b * s_test - 100 * z_te
        pd.DataFrame({'id': test.id, 'is_fraud': (np.argsort(np.argsort(pred)) + 0.5) / len(pred)}).to_csv(
            args.output / f'v8_plus_{name}.csv', index=False)
        r['submission_beta'] = b
        if best is None or r[f'gain_beta_{b}'] > res['jobs'][best][f'gain_beta_{res["jobs"][best]["submission_beta"]}']:
            best = job
    res['best_job'] = best
    (args.output / 'v11_results.json').write_text(json.dumps(res, indent=2) + '\n')
    (work / 'texts.pkl').unlink(missing_ok=True)
    print(json.dumps(res, indent=2), flush=True)


# ---------------------------------------------------------------- workers
def pad_matrix(seqs, pad_id):
    L = max(map(len, seqs))
    ids = np.full((len(seqs), L), pad_id, dtype=np.int32)
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    return ids, lens


class LayaScorer:
    def __init__(self, dev):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from laya.agent import _fix_tokenizer_config
        from laya.common import QTYPES, build_head, build_model, uses_parallel_layout
        repo = snapshot_download(LAYA, allow_patterns=[f'{LAYA_SUB}/{n}' for n in (
            'rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*')])
        d = os.path.join(repo, LAYA_SUB)
        _fix_tokenizer_config(d)
        cfg = json.load(open(os.path.join(d, 'rl_agent_config.json')))
        assert not uses_parallel_layout(cfg)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(d, 'tokenizer'))
        self.head, self.markers, _ = build_head(self.tok, QUESTION, cfg['head_max_len'])
        self.room = cfg['max_len'] - len(self.head) - 1
        self.model = build_model(cfg, encoder_dir=os.path.join(d, 'encoder'))
        self.model.load_state_dict(load_file(os.path.join(d, 'model.safetensors')), strict=True)
        if hasattr(self.model.encoder.config, 'reference_compile'):
            self.model.encoder.config.reference_compile = False
        self.model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.to(dev)
        self.dev, self.qt, self.torch = dev, QTYPES['noul'], torch
        self.pad_id = self.tok.pad_token_id

    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 20000):
            enc = self.tok([t.replace(self.tok.mask_token, ' ') for t in texts[i:i + 20000]],
                           add_special_tokens=False)['input_ids']
            out.extend(self.head + e[:self.room] + [self.tok.sep_token_id] for e in enc)
        return pad_matrix(out, self.pad_id)

    def param_groups(self):
        enc = [p for n, p in self.model.named_parameters() if n.startswith('encoder.')]
        head = [p for n, p in self.model.named_parameters() if not n.startswith('encoder.')]
        return [{'params': enc, 'lr': 2.5e-5}, {'params': head, 'lr': 1e-4}]

    def logits(self, ids, lens):
        torch, n, K = self.torch, len(lens), len(self.markers)
        L = int(lens.max())
        x = torch.from_numpy(ids[:, :L].astype(np.int64)).to(self.dev)
        att = (torch.arange(L, device=self.dev)[None] < torch.from_numpy(lens).to(self.dev)[:, None]).long()
        mp = torch.tensor(self.markers, device=self.dev)[None].expand(n, K)
        mm = torch.ones((n, K), dtype=torch.bool, device=self.dev)
        qt = torch.full((n,), self.qt, dtype=torch.long, device=self.dev)
        out, act = self.model(x, att, mp, mm, qt)
        return out + 0.0 * act.sum()


class QwenScorer:
    def __init__(self, dev):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        base = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float32)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        self.model.to(dev)
        self.yes = self.tok.encode(' Yes', add_special_tokens=False)[0]
        self.no = self.tok.encode(' No', add_special_tokens=False)[0]
        self.dev, self.torch = dev, torch
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 20000):
            out.extend(self.tok([PROMPT.format(t) for t in texts[i:i + 20000]], add_special_tokens=False)['input_ids'])
        return pad_matrix(out, self.pad_id)

    def param_groups(self):
        return [{'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': 2e-4}]

    def logits(self, ids, lens):
        """Left-pad so the answer position is last; return the [No, Yes] next-token logits."""
        torch, n = self.torch, len(lens)
        L = int(lens.max())
        x = np.full((n, L), self.pad_id, dtype=np.int64)
        att = np.zeros((n, L), dtype=np.int64)
        for i in range(n):
            x[i, L - lens[i]:] = ids[i, :lens[i]]
            att[i, L - lens[i]:] = 1
        x, att = torch.from_numpy(x).to(self.dev), torch.from_numpy(att).to(self.dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        try:
            out = self.model(input_ids=x, attention_mask=att, position_ids=pos, logits_to_keep=1).logits
        except TypeError:
            out = self.model(input_ids=x, attention_mask=att, position_ids=pos, num_logits_to_keep=1).logits
        return out[:, -1, [self.no, self.yes]]


def worker(args):
    import torch
    from laya.common import QTYPES, proper_reward
    from sklearn.metrics import roc_auc_score

    model_name, k = args.worker.split(':')
    k = int(k)
    name = f'{model_name}_f{k}'
    work = args.output / 'v11'
    fd = np.load(work / f'fold{k}.npz')
    texts = pickle.load(open(work / 'texts.pkl', 'rb'))
    torch.manual_seed(SEED + k)
    dev = torch.device('cuda')
    sc = (LayaScorer if model_name == 'laya' else QwenScorer)(dev)
    t0 = time.time()
    sub, val_keep, test_keep = fd['sub'], fd['val_keep'], np.load(work / 'test_keep.npy')
    y, offset_all, epochs = fd['y_sub'], fd['t_sub'].astype(np.float32), args.epochs
    if model_name == 'qwen':  # about 3x slower per row than Laya: fewer rows, one epoch
        pick = np.sort(np.random.RandomState(SEED + k).choice(len(sub), min(args.qwen_rows, len(sub)), replace=False))
        sub, y, offset_all, epochs = sub[pick], y[pick], offset_all[pick], args.qwen_epochs
    tr_ids, tr_lens = sc.encode([texts['train'][i] for i in sub])
    va_ids, va_lens = sc.encode([texts['train'][i] for i in val_keep])
    te_ids, te_lens = sc.encode([texts['test'][i] for i in test_keep])
    del texts
    print(f'{name}: tokens train {tr_ids.shape} val {va_ids.shape} test {te_ids.shape}, '
          f'mean length {tr_lens.mean():.0f}, {time.time() - t0:.0f}s', flush=True)

    @torch.no_grad()
    def score(ids, lens, bs, limit=None):
        sc.model.eval()
        order = np.argsort(lens) if limit is None else np.random.RandomState(SEED).choice(
            len(lens), limit, replace=False)
        out = np.full(len(lens), np.nan, dtype=np.float32)
        for b0 in range(0, len(order), bs):
            rows = order[b0:b0 + bs]
            with torch.autocast('cuda', dtype=torch.float16):
                lg = sc.logits(ids[rows], lens[rows]).float()
            out[rows] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        sc.model.train()
        return out

    info = {'job': args.worker}
    if model_name == 'qwen':
        # zero-shot log-prob gap on 20k held-out rows: how much does the raw LLM know?
        z0 = score(va_ids, va_lens, 64, limit=20000)
        m = ~np.isnan(z0)
        yv = np.load(work / 'y_all.npy')[val_keep]
        info['zero_shot_auc'] = float(roc_auc_score(yv[m], z0[m]))
        print(f'{name} zero-shot AUC {info["zero_shot_auc"]:.4f}', flush=True)

    hard = y * (1 - LABEL_SMOOTH) + (1 - y) * LABEL_SMOOTH
    target_all = np.stack([1 - hard, hard], 1).astype(np.float32)
    qt = QTYPES['noul']
    opt = torch.optim.AdamW(sc.param_groups(), weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / args.micro_batch)
    warm = max(1, min(200, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))) + 1e-3)
    scaler = torch.amp.GradScaler('cuda')
    G, t1, step, stopped = 4, time.time(), 0, False
    rng = np.random.RandomState(SEED + k)
    sc.model.train()
    for ep in range(epochs):
        order = rng.permutation(len(y))
        for b0 in range(0, len(order), args.micro_batch):
            rows = order[b0:b0 + args.micro_batch]
            n = len(rows)
            sigma = 0.4 + (0.1 - 0.4) * step / max(1, steps - 1)
            target = torch.from_numpy(target_all[rows]).to(dev)
            offset = torch.from_numpy(offset_all[rows]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                lg = sc.logits(tr_ids[rows], tr_lens[rows])
            # v8 is a fixed offset on the "yes" logit: only the residual is learned.
            lg = lg.float() + torch.stack([torch.zeros_like(offset), offset], 1)
            mm = torch.ones((n, 2), dtype=torch.bool, device=dev)
            qty = torch.full((n,), qt, dtype=torch.long, device=dev)
            eps = torch.randn((G,) + lg.shape, device=dev) * sigma
            eps = eps - eps.mean(-1, keepdim=True)
            zz = lg.detach().unsqueeze(0) + eps
            with torch.no_grad():
                r = proper_reward(torch.softmax(zz, -1), target.unsqueeze(0), qty, mm, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
            logp = -((zz - lg.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
            loss = -(adv * logp).mean() - (target * torch.log_softmax(lg, -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in sc.param_groups() for p in g['params']], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f'{name} step {step}/{steps} loss {loss.item():.4f} {time.time() - t1:.0f}s', flush=True)
            if (time.time() - t1) / 60 > args.max_train_minutes:
                stopped = True
                break
        if stopped:
            break
    info.update({'train_rows': int(len(y)), 'epochs': epochs, 'steps': step, 'planned_steps': steps, 'stopped_early': stopped,
                 'train_minutes': round((time.time() - t1) / 60, 1)})
    print(f'{name} trained {step} steps in {info["train_minutes"]} min (stopped early: {stopped})', flush=True)

    if model_name == 'qwen':
        sc.model = sc.model.merge_and_unload()
    t2 = time.time()
    bs = 256 if model_name == 'laya' else 128
    val_full = np.full(len(fd['val']), np.nan, dtype=np.float32)
    val_full[np.searchsorted(fd['val'], val_keep)] = score(va_ids, va_lens, bs)
    test_full = np.full(int(np.load(work / 'n_test.npy')), np.nan, dtype=np.float32)
    test_full[test_keep] = score(te_ids, te_lens, bs)
    np.save(work / f'{name}_val.npy', val_full)
    np.save(work / f'{name}_test.npy', test_full)
    info['score_minutes'] = round((time.time() - t2) / 60, 1)
    json.dump(info, open(work / f'{name}_info.json', 'w'))
    print(f'{name} scored in {info["score_minutes"]} min', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--worker')
    ap.add_argument('--jobs', nargs='*', default=['laya:0', 'qwen:0'])
    ap.add_argument('--train-rows', type=int, default=100000)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--qwen-rows', type=int, default=60000)
    ap.add_argument('--qwen-epochs', type=int, default=1)
    ap.add_argument('--micro-batch', type=int, default=32)
    ap.add_argument('--max-train-minutes', type=float, default=110)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'laya==0.5.3', 'peft'], check=True)
    parent(args)


if __name__ == '__main__':
    main()
