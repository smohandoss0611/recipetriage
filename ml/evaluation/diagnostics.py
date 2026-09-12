"""Run title-only and adversarial diagnostics separately using one loaded provider."""
import json
from pathlib import Path
from uuid import uuid4
from . import shortcut_tests, red_team
from .benchmark import descriptor
from .base_provider import BaseProvider


def run(provider=None,identity=None,payload=None,on_update=None):
    from datetime import datetime,timezone
    result=payload or {'run_id':str(uuid4()),'created_at':datetime.now(timezone.utc).isoformat()}
    result.update(status='running',model_config=identity or descriptor('hf-base'),shortcut=None,red_team=None,error=None)
    provider=provider or BaseProvider()
    def update_shortcut(value):
        result['shortcut']=value
        if on_update:on_update(result)
    try:
        shortcut_tests.run_suite(provider,result['model_config'],update_shortcut)
        result['red_team']=red_team.run_suite(provider,result['model_config'])
        result['status']='completed' if result['shortcut']['status']==result['red_team']['status']=='completed' else 'partial'
    except Exception as exc:
        import logging;logging.getLogger(__name__).exception('Diagnostic suite failed')
        result.update(status='failed',error=f'{type(exc).__name__}: diagnostic worker failed; inspect logs')
    if on_update:on_update(result)
    return result


def main():
    import argparse,os,torch
    from dotenv import load_dotenv
    from recipetriage_ml.training.sft import load_adapter
    load_dotenv('.env');os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()));torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    provider=load_adapter(args.adapter) if args.adapter else BaseProvider();identity=provider.identity if args.adapter else descriptor('hf-base')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(value):
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(args.output)
    result=run(provider,identity,on_update=save)
    print(json.dumps({'status':result['status'],'shortcut':result['shortcut']['summary'] if result['shortcut'] else None,'red_team_passed':(result['red_team'] or {}).get('passed')},indent=2))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
