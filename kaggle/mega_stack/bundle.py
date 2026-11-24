"""Package the v5 mega stack as a single-file Kaggle script kernel."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
members = (here / 'members.py').read_text()
stack = (here / 'stack.py').read_text()
marker = 'sys.path.insert(0, str(Path(__file__).parent))\nfrom members import collect, logit_rank  # noqa: E402\n'
assert marker in stack
stack = stack.replace(marker, '')
(args.output / 'stack.py').write_text(members + '\n\n# ---- stack.py ----\n' + stack)
metadata = {
    'id': 'huthnhlng/s6e11-v5-mega-stack',
    'title': 'S6E11 V5 Mega Stack',
    'code_file': 'stack.py',
    'language': 'python',
    'kernel_type': 'script',
    'is_private': True,
    'enable_gpu': False,
    'enable_internet': False,
    'competition_sources': ['playground-series-s6e11'],
    'dataset_sources': [
        'dariushafshar/s6e11-golem-oof-library',
        'mathsonchain/s6e11-fraud-oof-test-17-single-models',
        'sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs',
        'goodpjw2008/s6e11-fraud-series-artifacts',
        'amanatar/s6e11-rank1-q17-meta-bundle',
        'masayakawamata/s6e11-gpt2-target-first-oof',
        'dariushafshar/s6e11-gpu-oof-members',
    ],
    'kernel_sources': ['huthnhlng/s6e11-clipped-and-step-ab-replay'],
}
(args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
