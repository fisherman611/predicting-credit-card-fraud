"""Package a read-only Kaggle audit with reproducible public prediction inputs."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--mode',choices=('medium','encoding','submit'),required=True)
p.add_argument('--tokenizer',type=Path)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name({'medium':'medium_eval.py','encoding':'encoding_ablation.py','submit':'count_submit.py'}[a.mode]).read_text()
entry="\nif __name__=='__main__': main()\n"
assert source.endswith(entry)
bootstrap='''\nimport os, sys\nos.environ['TOKENIZERS_PARALLELISM']='false'\nroot=Path('/kaggle/input')\ndef find(name):\n    hits=sorted(p for p in root.rglob(name) if p.is_file())\n    if len(hits)!=1: raise ValueError(f'Expected one {name}, got {hits}')\n    return str(hits[0])\n'''
if a.mode in ('encoding','submit'):
    if not a.tokenizer: p.error('--tokenizer required for encoding')
    if hashlib.sha256(a.tokenizer.read_bytes()).hexdigest() != '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6':
        p.error('tokenizer checksum differs from the GPT-2 asset used in this audit')
    packed=base64.b64encode(gzip.compress(a.tokenizer.read_bytes(),mtime=0)).decode()
    bootstrap+=f"import base64, gzip\nPath('/tmp/s6e11-tokenizer.json').write_bytes(gzip.decompress(base64.b64decode({packed!r})))\n"
    if a.mode=='encoding':
        driver="sys.argv=['audit','--train',find('train.csv'),'--bundle',find('rank1_q17_streams.npz'),'--tokenizer','/tmp/s6e11-tokenizer.json','--output','/kaggle/working/ablation.json']; main()\n"
        ident='s6e11-encoding-ablation-audit'; title='S6E11 Encoding Ablation Audit'
        datasets=['amanatar/s6e11-rank1-q17-meta-bundle']; kernels=[]
    else:
        driver="sys.argv=['submit','--train',find('train.csv'),'--test',find('test.csv'),'--step-ab',find('step_ab.csv'),'--tokenizer','/tmp/s6e11-tokenizer.json','--output','/kaggle/working']; main()\n"
        ident='s6e11-count-step-ab-blend'; title='S6E11 Count Step AB Blend'
        datasets=[]; kernels=['huthnhlng/s6e11-clipped-and-step-ab-replay']
else:
    driver="sys.argv=['audit','--bundle',find('rank1_q17_streams.npz'),'--medium',find('lm_oof_medium_fold0.npy'),'--small',find('lm_oof.npy'),'--wide',find('wide_lm_stack_nested.npy'),'--output','/kaggle/working/medium.json']; main()\n"
    ident='s6e11-gpt2-medium-oof-audit'; title='S6E11 GPT2 Medium OOF Audit'
    datasets=['megayak/s6e11-target-first-lm','amanatar/s6e11-rank1-q17-meta-bundle','goodpjw2008/s6e11-fraud-series-artifacts']
    kernels=[]
(a.output/'audit.py').write_text(source[:-len(entry)]+bootstrap+driver)
metadata={'id':'huthnhlng/'+ident,'title':title,'code_file':'audit.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':False,'enable_internet':False,'competition_sources':['playground-series-s6e11'],
          'dataset_sources':datasets,'kernel_sources':kernels}
(a.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
