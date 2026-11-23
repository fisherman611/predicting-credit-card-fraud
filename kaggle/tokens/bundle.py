"""Bundle the shared training code and token experiment as one Kaggle script."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--owner", default="huthnhlng")
parser.add_argument("--model", choices=("tree", "transformer", "transfer"), default="tree")
parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
parser.add_argument("--stage", choices=("development", "holdout", "full"), default="development")
parser.add_argument("--variant", choices=("v1", "token_frequency", "token_target"), default="token_target")
args = parser.parse_args()
here = Path(__file__).resolve().parent
v1 = (here.parent / "train.py").read_text().split('if __name__ == "__main__":')[0]
tokens = (here / "train.py").read_text()
local_import = tokens.index("# Local imports")
tokens = tokens[:local_import] + tokens[tokens.index("\n\nVARIANTS", local_import):]
tokens = tokens.replace('default="development")', f'default="{args.stage}")')
tokens = tokens.replace('default="token_target")', f'default="{args.variant}")')
args.output.mkdir(parents=True, exist_ok=True)
if args.model != "tree":
    tokens = tokens.split('if __name__ == "__main__":')[0]
    filename, first_definition = {
        "transformer": ("sequence.py", "def encode_views"),
        "transfer": ("transfer.py", "SOURCE ="),
    }[args.model]
    extension = (here / filename).read_text()
    local = extension.index("# Local imports")
    extension = extension[:local] + extension[extension.index(first_definition, local):]
    extension = extension.replace('default="development")', f'default="{args.stage}")')
    if args.model == "transformer":
        extension = extension.replace('default="cuda")', f'default="{args.device}")')
    source = v1 + tokens + extension
else:
    source = v1 + tokens
(args.output / "train.py").write_text(source)
model_title = {"tree": "Tokens", "transformer": "Token Transformer", "transfer": "Original GLM"}[args.model]
title = f"S6E11 Fraud {model_title} {args.stage}"
if args.model == "transformer" and args.device == "cpu":
    title += " CPU"
slug = title.lower().replace(" ", "-")
metadata = {
    "id": f"{args.owner}/{slug}", "title": title, "code_file": "train.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": args.model == "transformer" and args.device == "cuda", "enable_internet": args.model != "transfer",
    "dataset_sources": ["uditjain13/credit-card-fraud-detection-2026"] if args.model == "transfer" else [],
    "competition_sources": ["playground-series-s6e11"], "kernel_sources": [],
}
(args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
