"""Package sim4.py as a Kaggle CPU kernel (noise 0.25, 0.5, 1.0): python kaggle/v13/bundle_sim4.py OUT [--seeds 6]"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path); ap.add_argument('--seeds', type=int, default=6)
a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
src = (Path(__file__).resolve().parent / 'sim4.py').read_text()
launcher = (
    'import subprocess, sys\nfrom pathlib import Path\n'
    f'SRC = {src!r}\n'
    "p = Path('/kaggle/working/sim4.py'); p.write_text(SRC)\n"
    "root = Path('/kaggle/input')\n"
    "art = next(q.parent for q in root.rglob('gb_s0_oof.npy'))\n"
    "pin = next(q.parent for q in root.rglob('oof_weni_bpe_glm.npy'))\n"
    "train = next(q for q in root.rglob('train.csv') if 'playground-series-s6e11' in str(q))\n"
    "davut = next(root.rglob('oof_wall_stack.csv'))\n"
    "for nz in ('0.25', '0.5', '1.0'):\n"
    f"    subprocess.run([sys.executable, str(p), '--art', str(art), '--pin', str(pin), '--train', str(train), '--davut', str(davut), '--seeds', '{a.seeds}', '--noise', nz, '--out', f'/kaggle/working/sim4_noise{{nz}}.json'], check=True)\n")
(a.output / 'launcher.py').write_text(launcher)
meta = {'id': 'huthnhlng/s6e11-v13-simulated-leaderboard-v4', 'title': 'S6E11 V13 Simulated Leaderboard V4', 'code_file': 'launcher.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts', 'goodpjw2008/s6e11-board-model-pinned-inputs'],
        'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal']}
(a.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
