"""Blend the selected challenger with the pinned public frontier."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--frontier", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--metrics", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
digest = hashlib.sha256(args.frontier.read_bytes()).hexdigest()
if digest != "5689d07cc601becb3b2530b87fa5c03241a79d54fb2aba10fe2773d6eaf0464c":
    raise ValueError("Public frontier differs from 0.95567 submission.")
metrics = json.loads(args.metrics.read_text())
if metrics["selected_challenger"] not in ("interactions", "residual_025", "residual_050"):
    raise ValueError("Unknown challenger")
frontier = pd.read_csv(args.frontier)
candidate = pd.read_csv(args.candidate)
ids = np.arange(699960, 999944)
for df in (frontier, candidate):
    if list(df.columns) != ["id", "is_fraud"] or not np.array_equal(df.id, ids):
        raise ValueError("Unexpected column names or test IDs.")
    if not np.isfinite(df.is_fraud).all() or not df.is_fraud.between(0, 1).all():
        raise ValueError("Invalid probability.")
n = len(frontier)
rank = lambda v: (rankdata(v)-.5)/n
scores = .95*rank(frontier.is_fraud) + .05*rank(candidate.is_fraud)
args.output.parent.mkdir(parents=True, exist_ok=True)
frontier.assign(is_fraud=scores).to_csv(args.output, index=False)
report = {"frontier_weight":.95, "challenger_weight":.05,
          "selected_challenger":metrics["selected_challenger"], "rows":n,
          "frontier_sha256":digest,
          "candidate_sha256":hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
          "output_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),
          "note":"The final blend with the public frontier has no independent OOF validation."}
args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
