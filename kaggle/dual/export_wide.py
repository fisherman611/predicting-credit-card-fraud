"""Export goodpjw2008's public nested-CV wide LM stack."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--sample", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sample = pd.read_csv(args.sample)
scores = np.load(args.artifacts / "wide_lm_stack_test.npy", allow_pickle=False)
if not np.array_equal(sample["id"], np.arange(699960, 999944)):
    raise ValueError("Unexpected competition sample IDs.")
if len(scores) != len(sample) or not np.isfinite(scores).all():
    raise ValueError("Invalid stack predictions.")
sample["is_fraud"] = (rankdata(scores) - 0.5) / len(scores)
args.output.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(args.output, index=False)
