"""Submit the frozen residual blend once when Kaggle quota is available."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

COMPETITION = "playground-series-s6e11"
MESSAGE = "Residual025 5pct rank blend frontier 95 seed718231"
EXPECTED = "52ec1113780c6245d9b28c63cc410db185d0e5b6719c3a257adde2ebb4fb3620"


def command(cli, env, *args):
    result = subprocess.run([str(cli), *args], env=env, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout


def submissions(cli, env):
    text = command(cli, env, "competitions", "submissions", "-c", COMPETITION, "-v")
    return [r for r in csv.DictReader(io.StringIO(text)) if r.get("description") == MESSAGE]


def run(args):
    if hashlib.sha256(args.file.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Frozen submission file SHA-256 does not match.")
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(args.credentials_dir)
    env["KAGGLE_API_TOKEN"] = (args.credentials_dir / "token").read_text().strip()
    args.record.parent.mkdir(parents=True, exist_ok=True)
    # Hold the lock across submission and scoring so a concurrent retry cannot duplicate it.
    with args.record.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = submissions(args.kaggle, env)
        if len(existing) > 1:
            raise ValueError("Multiple matching submissions found; inspect the leaderboard.")
        if not existing:
            limits = json.loads(command(args.kaggle, env, "competitions", "submission-limits",
                                        COMPETITION, "--json"))
            if int(limits.get("numAllowedNow", 0)) < 1:
                print(json.dumps({"status": "waiting_for_quota", "limits": limits}))
                return
            command(args.kaggle, env, "competitions", "submit", "-c", COMPETITION,
                    "-f", str(args.file), "-m", MESSAGE)
        for _ in range(30):
            existing = submissions(args.kaggle, env)
            if existing and (existing[0].get("publicScore") or
                             any(x in existing[0].get("status", "") for x in ("ERROR", "FAILED", "CANCEL"))):
                args.record.write_text(json.dumps(existing[0], indent=2)+"\n")
                print(json.dumps(existing[0], ensure_ascii=False))
                return
            time.sleep(10)
        raise RuntimeError("Submission sent but scoring is pending; check by its description before retrying.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--credentials-dir", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--kaggle", type=Path, default=Path("/tmp/s6e11-cli/bin/kaggle"))
    run(p.parse_args())
