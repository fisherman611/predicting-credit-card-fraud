"""Package v8 kernels.

  python kaggle/v8/bundle.py mean    OUT -> huthnhlng/s6e11-v8-stack-mean
  python kaggle/v8/bundle.py augment OUT -> huthnhlng/s6e11-v8-augment-test
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('kind', choices=('mean', 'augment'))
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
common = {'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
          'enable_internet': False, 'competition_sources': ['playground-series-s6e11']}
if args.kind == 'mean':
    (args.output / 'stack_mean.py').write_text((here / 'stack_mean.py').read_text())
    meta = {'id': 'huthnhlng/s6e11-v8-stack-mean', 'title': 'S6E11 V8 Stack Mean', 'code_file': 'stack_mean.py',
            'dataset_sources': ['goodpjw2008/s6e11-fraud-series-artifacts'],
            'kernel_sources': ['davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal'], **common}
else:
    base = (here.parent / 'v6' / 'base.py').read_text()
    entry = "\nif __name__ == '__main__':\n    main()\n"
    assert base.endswith(entry)
    aug = (here / 'augment.py').read_text()
    marker = ("sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v6'))\n"
              "from base import build_keys, find, raw_numeric  # noqa: E402\n")
    assert marker in aug
    # Inline v6/base.py helpers (renaming its main) ahead of augment.py.
    script = base[:-len(entry)].replace('def main():', 'def _base_main():') + '\n\n' + aug.replace(marker, '')
    (args.output / 'augment.py').write_text(script)
    meta = {'id': 'huthnhlng/s6e11-v8-augment-test', 'title': 'S6E11 V8 Augment Test', 'code_file': 'augment.py',
            'dataset_sources': ['uditjain13/credit-card-fraud-detection-2026'], 'kernel_sources': [], **common}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
