"""Compare fixed OOF rank votes with published CatBoost and GPT-2 predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--baseline-oof", type=Path, required=True)
p.add_argument("--public-bundle", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
v = np.load(args.baseline_oof, allow_pickle=False)
z = np.load(args.public_bundle, allow_pickle=False)
if not np.array_equal(v["is_fraud"], z["y_train"]) or not np.array_equal(v["fold"]-1, z["folds"]):
    raise ValueError("The labels or folds do not match.")
y = v["is_fraud"]
dev, _ = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=24681357).split(np.zeros(len(y)), y))
members = {"lightgbm": v["lightgbm"][dev], "glm": v["digit_glm"][dev],
           "catboost": z["oof_sadam_v4_cat"][dev], "gpt2": z["oof_megayak_5f_lm"][dev]}
ranks = {name: (rankdata(pred)-.5)/len(dev) for name, pred in members.items()}
configs = {"tree_glm_rank": {"lightgbm": .8, "glm": .2},
           "tree_glm_cat_rank": {"lightgbm": .5, "glm": .2, "catboost": .3},
           "tree_glm_cat_gpt_rank": {"lightgbm": .4, "glm": .1, "catboost": .25, "gpt2": .25}}
report = {"note": "Exploratory public OOF; base models can have trained on the outer holdout.",
          "rows": len(dev),
          "single_model_auc": {name: roc_auc_score(y[dev], pred) for name, pred in members.items()},
          "voting_auc": {name: roc_auc_score(y[dev], sum(ranks[k]*w for k, w in weights.items()))
                         for name, weights in configs.items()}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
