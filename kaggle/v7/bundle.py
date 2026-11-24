"""Package v7 kernels.

  python kaggle/v7/bundle.py digits OUT  -> huthnhlng/s6e11-v7-digit-transformer (GPU)
  python kaggle/v7/bundle.py digits_cpu OUT -> huthnhlng/s6e11-v7-digit-transformer-cpu (smaller, CPU)
  python kaggle/v7/bundle.py stack  OUT  -> huthnhlng/s6e11-v7-gated-stack
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('kind', choices=('digits', 'digits_cpu', 'stack'))
ap.add_argument('output', type=Path)
args = ap.parse_args()
here = Path(__file__).resolve().parent
if args.kind == 'stack':
    subprocess.run([sys.executable, str(here.parent / 'v6' / 'bundle.py'), 'stack', str(args.output)], check=True)
    meta = json.loads((args.output / 'kernel-metadata.json').read_text())
    meta.update({'id': 'huthnhlng/s6e11-v7-gated-stack', 'title': 'S6E11 V7 Gated Stack',
                 'kernel_sources': ['huthnhlng/s6e11-v6-base-members', 'huthnhlng/s6e11-v7-digit-transformer']})
else:
    args.output.mkdir(parents=True, exist_ok=True)
    src = (here / 'digits.py').read_text()
    cpu = args.kind == 'digits_cpu'
    if cpu:
        entry = "\nif __name__ == '__main__':\n    main()\n"
        assert src.endswith(entry)
        src = src[:-len(entry)] + ("\nif __name__ == '__main__':\n    import sys, torch\n    torch.set_num_threads(4)\n"
                                   "    sys.argv = ['digits', '--width', '32', '--layers', '2', '--epochs', '6', '--batch', '4096']\n"
                                   "    main()\n")
    (args.output / 'digits.py').write_text(src)
    suffix = '-cpu' if cpu else ''
    meta = {'id': 'huthnhlng/s6e11-v7-digit-transformer' + suffix, 'title': 'S6E11 V7 Digit Transformer' + (' CPU' if cpu else ''),
            'code_file': 'digits.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': not cpu, 'enable_internet': False, 'competition_sources': ['playground-series-s6e11'],
            'dataset_sources': [], 'kernel_sources': []}
(args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
