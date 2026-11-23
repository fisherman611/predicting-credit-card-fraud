"""Export the fixed four-model exploratory rank blend with the 0.95567 frontier."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--frontier", type=Path, required=True)
p.add_argument("--members", type=Path, required=True)
p.add_argument("--public-bundle", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
frontier_digest = hashlib.sha256(args.frontier.read_bytes()).hexdigest()
expected = "5689d07cc601becb3b2530b87fa5c03241a79d54fb2aba10fe2773d6eaf0464c"
if frontier_digest != expected:
    raise ValueError("Frontier CSV does not match the 0.95567 submission.")
base = pd.read_csv(args.frontier)
expected_ids = np.arange(699960, 999944)
if list(base.columns) != ["id", "is_fraud"] or not np.array_equal(base.id, expected_ids):
    raise ValueError("Unexpected frontier IDs.")
with np.load(args.members, allow_pickle=False) as own, np.load(args.public_bundle, allow_pickle=False) as public:
    if not np.array_equal(own["id"], expected_ids):
        raise ValueError("Own prediction IDs do not match.")
    # This public dataset documents competition test row order.
    predictions = [own["members"][:, 0], own["members"][:, 1],
                   public["test_sadam_v4_cat"], public["test_megayak_5f_lm"]]
    if any(len(x) != len(base) or not np.isfinite(x).all() for x in predictions):
        raise ValueError("Invalid member predictions.")
    n = len(base)
    ranks = [(rankdata(x)-.5)/n for x in predictions]
    exploratory = sum(w*r for w, r in zip((.4, .1, .25, .25), ranks, strict=True))
rank_base = (rankdata(base.is_fraud)-.5)/n
combined = .95*rank_base + .05*exploratory
if not np.isfinite(combined).all() or not ((combined >= 0) & (combined <= 1)).all():
    raise ValueError("Invalid final scores.")
args.output.parent.mkdir(parents=True, exist_ok=True)
base.assign(is_fraud=combined).to_csv(args.output, index=False)
metadata = {"rows": n, "frontier_weight": .95, "exploratory_weight": .05,
            "member_weights": {"lightgbm": .4, "digit_glm": .1, "public_catboost": .25, "public_gpt2": .25},
            "frontier_sha256": frontier_digest,
            "members_sha256": hashlib.sha256(args.members.read_bytes()).hexdigest(),
            "bundle_sha256": hashlib.sha256(args.public_bundle.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "validation": "The 4-member OOF was exploratory and the blend with frontier has no independent OOF."}
args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
print(json.dumps(metadata, indent=2))
