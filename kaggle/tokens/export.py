"""Export the frozen candidate after it clears the locked holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare import BLENDS, STACK_MEMBERS, predict_stack, rank_blend, stack_matrix


def export(args):
    development = json.loads(args.development_report.read_text())
    holdout = json.loads(args.holdout_report.read_text())
    selected = development["selected"]
    if selected == "v1" or selected != holdout["selected"]:
        raise ValueError("Development and holdout reports must identify the same challenger.")
    if holdout["challenger"]["auc"] <= holdout["v1"]["auc"]:
        raise ValueError("The challenger did not improve on the locked holdout.")

    paths = {
        "v1": args.v1, "token_frequency": args.tree, "token_target": args.tree,
        "transformer": args.transformer, "original_glm": args.transfer,
    }
    weights = BLENDS.get(selected, {selected: 1.0})
    required = STACK_MEMBERS if selected == "stack_logistic" else tuple(weights)
    scores = {}
    ids = None
    for name in required:
        path = paths[name]
        if path is None:
            raise ValueError(f"Missing full-training submission for {name}.")
        data = pd.read_csv(path)
        if list(data.columns) != ["id", "is_fraud"] or not data.id.is_unique:
            raise ValueError(f"Invalid submission shape for {name}.")
        if ids is None:
            ids = data.id
        elif not ids.equals(data.id):
            raise ValueError(f"Test IDs do not align for {name}.")
        if not data["is_fraud"].between(0, 1).all():
            raise ValueError(f"Invalid predictions for {name}.")
        scores[name] = data["is_fraud"].to_numpy()
    if selected == "stack_logistic":
        prediction = predict_stack(stack_matrix(scores), development["stack_config"])
    else:
        prediction = rank_blend(scores, weights) if len(weights) > 1 else scores[selected]
    if not np.isfinite(prediction).all():
        raise ValueError("The final submission has invalid predictions.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "is_fraud": prediction}).to_csv(args.output, index=False)
    print(f"{selected}: {len(prediction)} test rows saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--v1", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submissions/token-v4.csv"))
    export(parser.parse_args())
