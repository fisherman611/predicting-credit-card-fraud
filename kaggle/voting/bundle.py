"""Build the voting experiment for Kaggle."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--owner", default="huthnhlng")
args = p.parse_args()
here = Path(__file__).resolve().parent
base = (here.parent / "train.py").read_text().split('if __name__ == "__main__":')[0]
code = (here / "train.py").read_text()
start = code.index("# The Kaggle bundle prepends")
end = code.index("\nSPLIT_SEED =", start)
code = code[:start] + code[end:]
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "train.py").write_text(base + "\n" + code)
metadata = {"id": f"{args.owner}/s6e11-rank-voting-holdout", "title": "S6E11 Rank Voting Holdout",
            "code_file": "train.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": ["playground-series-s6e11"], "kernel_sources": []}
(args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
