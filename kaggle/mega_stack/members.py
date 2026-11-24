"""Collect public out-of-fold / test prediction pairs for S6E11.

Every member must provide an out-of-fold prediction for all 699,960 train rows
and a test prediction for all 299,984 test rows, both in competition id order.
Near-duplicate members (Spearman > 0.99995 to an already kept member) are
dropped so the stacker does not see the same column twice.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata

N_TRAIN, N_TEST = 699960, 299984
# Stacks of other members; they are kept as members but reported separately.
STACK_HINTS = ('stack', 'anchor')


def _ok(a, n):
    return a is not None and a.ndim == 1 and len(a) == n and np.isfinite(a).all()


def _add(out, name, oof, test):
    oof = np.asarray(oof, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    if _ok(oof, N_TRAIN) and _ok(test, N_TEST) and oof.std() > 0 and test.std() > 0:
        out[name] = (oof, test)


def _npy_pairs(root, out):
    files = {p.name: p for p in root.rglob('*.npy') if p.is_file()}
    for name, path in files.items():
        stem = name[:-4]
        cands = []
        if stem.startswith('oof_'):
            cands = ['test_' + stem[4:] + '.npy']
        elif stem.endswith('_oof'):
            base = stem[:-4]
            cands = [base + '_test.npy', base + '_full.npy']
        elif stem.endswith('_nested'):
            cands = [stem[:-7] + '_test.npy']
        elif stem == 'oof':
            cands = ['test.npy']
        for cand in cands:
            match = path.with_name(cand)
            if match.is_file():
                tag = path.parent.name + ':' + (stem if stem != 'oof' else 'oof')
                try:
                    _add(out, tag, np.load(path), np.load(match))
                except ValueError:
                    pass
                break


def _npz_pairs(root, out):
    for path in root.rglob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key.startswith('oof_') and 'test_' + key[4:] in z.files:
                    _add(out, path.stem + ':' + key[4:], z[key], z['test_' + key[4:]])


def _parquet_pairs(root, out):
    for path in root.rglob('oof*.parquet'):
        twin = path.with_name(path.name.replace('oof', 'test', 1))
        if not twin.is_file():
            continue
        a, b = pd.read_parquet(path), pd.read_parquet(twin)
        if 'id' in a:
            a = a.sort_values('id')
        if 'id' in b:
            b = b.sort_values('id')
        for col in a.columns:
            if col.startswith('oof_') and 'test_' + col[4:] in b.columns:
                _add(out, path.parent.name + ':pq_' + col[4:], a[col].to_numpy(), b['test_' + col[4:]].to_numpy())


def _csv_pairs(root, out):
    for path in root.rglob('*oof*.csv'):
        twin = path.with_name(path.name.replace('oof', 'test', 1))
        if not twin.is_file():
            continue
        try:
            a, b = pd.read_csv(path), pd.read_csv(twin)
        except Exception:
            continue
        if not {'id', 'is_fraud'} <= set(a.columns) or not {'id', 'is_fraud'} <= set(b.columns):
            continue
        _add(out, path.parent.name + ':' + path.stem,
             a.sort_values('id').is_fraud.to_numpy(), b.sort_values('id').is_fraud.to_numpy())


def logit_rank(v):
    return logit((rankdata(v) - 0.5) / len(v))


def collect(root, y, verbose=True):
    """Return {name: (oof, test)} after AUC sanity and near-duplicate pruning."""
    from sklearn.metrics import roc_auc_score

    raw = {}
    for fn in (_npy_pairs, _npz_pairs, _parquet_pairs, _csv_pairs):
        fn(Path(root), raw)
    scored = []
    for name, (oof, test) in raw.items():
        auc = roc_auc_score(y, oof)
        if auc < 0.5:
            continue  # inverted or broken
        if auc < 0.90:
            continue  # zero-shot or toy members only add noise
        scored.append((auc, name))
    scored.sort(reverse=True)
    kept, ranks = {}, []
    for auc, name in scored:
        r = rankdata(raw[name][0]).astype(np.float32)
        r = (r - r.mean()) / r.std()
        if any(float(np.dot(r, q)) / len(r) > 0.99995 for q in ranks):
            continue
        ranks.append(r)
        kept[name] = raw[name]
        if verbose:
            print(f'{auc:.6f} {name}')
    if verbose:
        print(f'{len(raw)} pairs found, {len(kept)} kept')
    return kept


def is_stack(name):
    return any(h in name.lower() for h in STACK_HINTS) or re.search(r'part\d', name.lower()) is not None
