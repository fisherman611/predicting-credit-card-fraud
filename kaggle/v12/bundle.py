"""Package final.py as a Kaggle CPU script kernel: python kaggle/v12/bundle.py OUT"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
ap.add_argument('--slug', default='s6e11-v12-final-files')
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
shutil.copy(Path(__file__).resolve().parent / 'final.py', args.output / 'final.py')
meta = {'id': 'huthnhlng/' + args.slug, 'title': 'S6E11 V12 Final Files', 'code_file': 'final.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts'],
        'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal',
                           'goodpjw2008/s6e11-board-model-realmlp-lb-0-95687',
                           'monya1/s6e11-zero-fraud-zones-and-rank-extrapolation']}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
