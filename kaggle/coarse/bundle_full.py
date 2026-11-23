"""Bundle the full-data specialist refit for Kaggle."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--owner", default="huthnhlng")
args = parser.parse_args()
here = Path(__file__).resolve().parent
baseline = (here.parent / "train.py").read_text().split('if __name__ == "__main__":')[0]
full = (here / "full.py").read_text()
start = full.index("# The bundle prepends")
end = full.index("\nDEV_OOF_SHA256 =", start)
full = full[:start] + full[end:]
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "train.py").write_text(baseline + "\n" + full)
metadata = {
    "id": f"{args.owner}/s6e11-cvv-specialist-full",
    "title": "S6E11 CVV Specialist Full",
    "code_file": "train.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["playground-series-s6e11"],
    "kernel_sources": [f"{args.owner}/s6e11-cvv-specialist-holdout"],
}
(args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
