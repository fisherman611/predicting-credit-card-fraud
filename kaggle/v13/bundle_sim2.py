"""Package sim2.py as a Kaggle CPU kernel running both scenarios: python kaggle/v13/bundle_sim2.py OUT"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path); ap.add_argument('--seeds', type=int, default=3)
a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
src = (Path(__file__).resolve().parent / 'sim2.py').read_text()
launcher = (
    'import subprocess, sys\nfrom pathlib import Path\n'
    f'SRC = {src!r}\n'
    "p = Path('/kaggle/working/sim2.py'); p.write_text(SRC)\n"
    "root = Path('/kaggle/input')\n"
    "art = next(q.parent for q in root.rglob('gb_s0_oof.npy'))\n"
    "pin = next(q.parent for q in root.rglob('oof_weni_bpe_glm.npy'))\n"
    "train = next(q for q in root.rglob('train.csv') if 'playground-series-s6e11' in str(q))\n"
    "davut = next(root.rglob('oof_wall_stack.csv'))\n"
    "for sc in ('v8', 'wide'):\n"
    "  for nz in ('0.25', '0.5', '1.0'):\n"
    f"    subprocess.run([sys.executable, str(p), '--art', str(art), '--pin', str(pin), '--train', str(train), '--davut', str(davut), '--seeds', '{a.seeds}', '--scenario', sc, '--noise', nz, '--out', f'/kaggle/working/sim2_{{sc}}_noise{{nz}}.json'], check=True)\n")
(a.output / 'launcher.py').write_text(launcher)
meta = {'id': 'huthnhlng/s6e11-v13-simulated-leaderboard-v2', 'title': 'S6E11 V13 Simulated Leaderboard V3', 'code_file': 'launcher.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts', 'goodpjw2008/s6e11-board-model-pinned-inputs'],
        'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal']}
(a.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
