"""Package the public Board Model + RealMLP v39 Step Z replay for Kaggle."""

import argparse
import hashlib
import json
from pathlib import Path


NOTEBOOK_SHA256 = "5b5fd6284d844eee0b1555b9f0a9a6b731f7ea5588a88e07aeefca806ad287d7"
DATASETS = [
    "goodpjw2008/s6e11-board-model-pinned-inputs",
    "goodpjw2008/s6e11-fraud-series-artifacts",
    "goodpjw2008/s6e11-oof-bank",
    "busyaprime/s6e11-33-own-members",
    "asteriospanoras/s6e11-public-board-files",
    "amanatar/s6e11-rank1-q17-meta-bundle",
]
HEADER = """# Step Z replay of Board Model + RealMLP v39 by goodpjw2008 and contributors.
# https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95562?scriptVersionId=372884035
import os
from pathlib import Path

Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working")).mkdir(parents=True, exist_ok=True)
"""
EXPORT = """
# Export and verify Step Z against the author's published v39 output.
submission = sample[["id"]].assign(is_fraud=U(FILE_Z))
expected_ids = np.arange(699960, 999944)
if not np.array_equal(submission["id"], expected_ids):
    raise ValueError("Step Z IDs differ from the competition test IDs.")
scores = submission["is_fraud"].to_numpy()
if not np.isfinite(scores).all() or not ((scores > 0) & (scores < 1)).all():
    raise ValueError("Step Z predictions must be finite probabilities in (0, 1).")

path = WORK / "submission.csv"
submission.to_csv(path, index=False)
exported = pd.read_csv(path, float_precision="round_trip")
rank_bytes = rankdata(exported["is_fraud"]).astype("<f8").tobytes()
rank_digest = hashlib.sha256(rank_bytes).hexdigest()
csv_digest = sha256(path)
metrics = {
    "source_version": 39,
    "rows": len(exported),
    "csv_sha256": csv_digest,
    "rank_sha256": rank_digest,
    "matches_source_csv": csv_digest == "ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11",
    "matches_source_ranks": rank_digest == "461f3e6f040bd7b90fde882c6014d0724f6e56313910e95b88a22a830df5c43a",
}
report = json.dumps(metrics, indent=2)
(WORK / "replay.json").write_text(report + "\\n")
print(report)
if not metrics["matches_source_ranks"]:
    raise ValueError("The exported ranking differs from the author's v39 submission.")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--owner", default="huthnhlng")
    args = parser.parse_args()

    notebook_bytes = args.notebook.read_bytes()
    if hashlib.sha256(notebook_bytes).hexdigest() != NOTEBOOK_SHA256:
        parser.error("Pass the pinned version 39 notebook linked in the README.")
    cells = json.loads(notebook_bytes)["cells"]
    setup = "".join(cells[2]["source"])
    replay, _ = "".join(cells[3]["source"]).split("for name, t in (", 1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay.py").write_text(
        "\n\n".join((HEADER, setup, replay, EXPORT)), encoding="utf-8"
    )
    metadata = {
        "id": f"{args.owner}/s6e11-board-model-v39-replay",
        "title": "S6E11 Board Model V39 Replay",
        "code_file": "replay.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": ["playground-series-s6e11"],
        "kernel_sources": [],
    }
    (args.output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
