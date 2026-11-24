"""Package the likelihood screening job with the previous OOF artifacts."""
import argparse
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('output',type=Path)
args=ap.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name('evaluate.py').read_text()
entry='''\nif __name__=='__main__':\n    main()\n'''
assert source.endswith(entry)
driver='''
if __name__ == '__main__':
    import sys
    root = Path('/kaggle/input')
    def find(name):
        paths = sorted(p for p in root.rglob(name) if p.is_file())
        if len(paths) != 1:
            raise ValueError(f'Expected one {name}, found {paths}')
        return str(paths[0])
    sys.argv = ['evaluate.py', '--train', find('train.csv'), '--test', find('test.csv'),
                '--development', find('development_versions.npz'), '--holdout', find('holdout.npz'),
                '--output', '/kaggle/working']
    main()
'''
(args.output/'evaluate.py').write_text(source[:-len(entry)]+driver)
meta={'id':'huthnhlng/s6e11-synthetic-likelihood-audit',
      'title':'S6E11 Synthetic Likelihood Audit','code_file':'evaluate.py','language':'python',
      'kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,
      'competition_sources':['playground-series-s6e11'], 'dataset_sources':[],
      'kernel_sources':['huthnhlng/s6e11-fraud-residual-interactions']}
(args.output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
