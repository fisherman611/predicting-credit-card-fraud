"""Package v6 as two Kaggle script kernels.

  python kaggle/v6/bundle.py base  OUT   -> huthnhlng/s6e11-v6-base-members
  python kaggle/v6/bundle.py stack OUT   -> huthnhlng/s6e11-v6-gated-stack
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('kind', choices=('base', 'stack'))
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
common = {'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
          'enable_internet': False, 'competition_sources': ['playground-series-s6e11']}
if args.kind == 'base':
    (args.output / 'base.py').write_text((here / 'base.py').read_text())
    meta = {'id': 'huthnhlng/s6e11-v6-base-members', 'title': 'S6E11 V6 Base Members',
            'code_file': 'base.py', 'dataset_sources': [], 'kernel_sources': [], **common}
else:
    # Ship the three modules as files written at start-up, then run stack_v6.main().
    mods = {name: (here.parent / 'mega_stack' / name).read_text() for name in ('members.py', 'stack.py')}
    mods['stack_v6.py'] = (here / 'stack_v6.py').read_text().replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mega_stack'))\n", '')
    lines = ['import sys', 'from pathlib import Path', "lib = Path('/kaggle/working/lib')",
             'lib.mkdir(parents=True, exist_ok=True)']
    for name, src in mods.items():
        lines.append(f"(lib / {name!r}).write_text({src!r})")
    lines += ['sys.path.insert(0, str(lib))', 'import stack_v6', 'stack_v6.main()']
    (args.output / 'run.py').write_text('\n'.join(lines) + '\n')
    meta = {'id': 'huthnhlng/s6e11-v6-gated-stack', 'title': 'S6E11 V6 Gated Stack', 'code_file': 'run.py',
            'dataset_sources': [
                'dariushafshar/s6e11-golem-oof-library',
                'mathsonchain/s6e11-fraud-oof-test-17-single-models',
                'sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs',
                'goodpjw2008/s6e11-fraud-series-artifacts',
                'amanatar/s6e11-rank1-q17-meta-bundle',
                'masayakawamata/s6e11-gpt2-target-first-oof',
                'dariushafshar/s6e11-gpu-oof-members'],
            'kernel_sources': ['huthnhlng/s6e11-v6-base-members'], **common}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
