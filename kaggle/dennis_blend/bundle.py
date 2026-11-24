import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(__file__).with_name('replay.py'), args.output/'replay.py')
meta = {'id': 'huthnhlng/s6e11-dennis-12pct-replay', 'title': 'S6E11 Dennis 12pct Replay',
        'code_file': 'replay.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['playground-series-s6e11'], 'dataset_sources': [],
        'kernel_sources': ['huthnhlng/s6e11-clipped-and-step-ab-replay']}
(args.output/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
