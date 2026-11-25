"""Package residual_lm.py as a Kaggle GPU kernel: python kaggle/v11/bundle.py OUT --args "--jobs laya:0 qwen:0".

Kaggle runs script kernels through a notebook, where __file__ is not set, so the kernel is a small
launcher that writes residual_lm.py to /kaggle/working and runs it as a file (its workers re-launch
the same file, one per GPU).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
ap.add_argument('--slug', default='s6e11-v11-residual-lm')
ap.add_argument('--title', default='S6E11 V11 Residual LM')
ap.add_argument('--args', default='', help='extra arguments for residual_lm.py')
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
src = (Path(__file__).resolve().parent / 'residual_lm.py').read_text()
launcher = (
    'import subprocess, sys\n'
    'from pathlib import Path\n'
    f'SRC = {src!r}\n'
    "path = Path('/kaggle/working/residual_lm.py')\n"
    'path.write_text(SRC)\n'
    f'subprocess.run([sys.executable, str(path)] + {args.args.split()!r}, check=True)\n'
)
(args.output / 'launcher.py').write_text(launcher)
meta = {'id': 'huthnhlng/' + args.slug, 'title': args.title, 'code_file': 'launcher.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True,
        'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
        'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts'],
        'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal']}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
