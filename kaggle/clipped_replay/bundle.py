"""Build the private Kaggle replay using public, checksum-pinned inputs."""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(__file__).with_name('replay.py'), args.output/'replay.py')
metadata = {'id': 'huthnhlng/s6e11-clipped-and-step-ab-replay',
    'title': 'S6E11 Clipped and Step AB Replay', 'code_file': 'replay.py',
    'language': 'python', 'kernel_type': 'script', 'is_private': True,
    'enable_gpu': False, 'enable_internet': False,
    'competition_sources': ['playground-series-s6e11'],
    'dataset_sources': ['malyshevdanil/s6e11-nov21-expanded-public-frontier',
                        'malyshevdanil/s6e11-nov23-round2-public-frontier'],
    'kernel_sources': ['goodpjw2008/s6e11-board-model-realmlp-lb-0-95569',
                       'huthnhlng/s6e11-fraud-residual-interactions']}
(args.output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
