"""Package sim.py as a Kaggle CPU kernel: python kaggle/v13/bundle_sim.py OUT [--seeds 5]"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path); ap.add_argument('--seeds', type=int, default=5)
a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
src = (Path(__file__).resolve().parent / 'sim.py').read_text()
launcher = (
    'import subprocess, sys\nfrom pathlib import Path\n'
    f'SRC = {src!r}\n'
    "p = Path('/kaggle/working/sim.py'); p.write_text(SRC)\n"
    "root = Path('/kaggle/input')\n"
    "art = next(q.parent for q in root.rglob('gb_s0_oof.npy'))\n"
    "pin = next(q.parent for q in root.rglob('oof_weni_bpe_glm.npy'))\n"
    "train = next(q for q in root.rglob('train.csv') if 'playground-series-s6e11' in str(q))\n"
    f"subprocess.run([sys.executable, str(p), '--art', str(art), '--pin', str(pin), '--train', str(train), '--seeds', '{a.seeds}', '--out', '/kaggle/working/sim_results.json'], check=True)\n")
(a.output / 'launcher.py').write_text(launcher)
meta = {'id': 'huthnhlng/s6e11-v13-simulated-leaderboard', 'title': 'S6E11 V13 Simulated Leaderboard', 'code_file': 'launcher.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts', 'goodpjw2008/s6e11-board-model-pinned-inputs']}
(a.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
