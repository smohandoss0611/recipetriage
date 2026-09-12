import argparse
import json
import shutil
from pathlib import Path
from .pipeline import build_dataset, load_raw

parser = argparse.ArgumentParser(description="Build RecipeTriage Dataset v1; no training")
parser.add_argument('input',type=Path)
parser.add_argument('--format',choices=['json','jsonl'],default='json')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--ratios',type=float,nargs=3,default=[0.7,0.15,0.15])
args = parser.parse_args()
try:
    result = build_dataset(load_raw(args.input.read_text(encoding='utf-8'),args.format),args.seed,args.ratios)
    args.output.mkdir(parents=True,exist_ok=True)
    target = args.output/result['metadata']['version']
    # Reserve the destination atomically; never overwrite an immutable export.
    target.mkdir(exist_ok=False)
    try:
        for name,content in result['files'].items():
            (target/name).write_text(content,encoding='utf-8')
        (target/'metadata.json').write_text(json.dumps(result['metadata'],indent=2)+'\n',encoding='utf-8')
    except BaseException:
        shutil.rmtree(target)
        raise
except (ValueError,OSError) as exc:
    parser.exit(1,f'Dataset build failed: {exc}\n')
print(target)
print('Status:',result['metadata']['status'])
for warning in result['metadata']['warnings']:
    print('Warning:',warning)
