"""Package the v9 stack as one Kaggle script: python kaggle/v9/bundle.py OUT -> huthnhlng/s6e11-v9-stack."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
members = (here.parent / 'mega_stack' / 'members.py').read_text()
stack = (here / 'stack_v9.py').read_text()
marker = ("sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mega_stack'))\n"
          "from members import N_TEST, N_TRAIN, _add, collect  # noqa: E402\n")
assert marker in stack
(args.output / 'stack_v9.py').write_text(members + '\n\n# ---- stack_v9.py ----\n' + stack.replace(marker, ''))
meta = {'id': 'huthnhlng/s6e11-v9-stack', 'title': 'S6E11 V9 Stack', 'code_file': 'stack_v9.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
        'enable_internet': False, 'competition_sources': ['playground-series-s6e11'],
        'dataset_sources': ['dariushafshar/s6e11-golem-oof-library',
                            'mathsonchain/s6e11-fraud-oof-test-17-single-models',
                            'sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs',
                            'goodpjw2008/s6e11-fraud-series-artifacts',
                            'amanatar/s6e11-rank1-q17-meta-bundle',
                            'masayakawamata/s6e11-gpt2-target-first-oof',
                            'dariushafshar/s6e11-gpu-oof-members',
                            'goodpjw2008/s6e11-oof-bank',
                            'busyaprime/s6e11-33-own-members'],
        'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal',
                           'yekenot/ps-s6-e11-realmlp-pytabkit',
                           'yekenot/ps-s6-e11-realmlp-pytorch']}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
