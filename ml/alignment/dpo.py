"""TRL DPO from explicit human choices, initialized from the saved QLoRA adapter."""
import argparse,json
from pathlib import Path
from .runner import LearningConfig,run


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--source',type=Path)
    args=parser.parse_args();config=LearningConfig.model_validate_json(args.config.read_text()) if args.config else LearningConfig()
    result=run('dpo',config,json.loads(args.pairs.read_text()),args.output,args.source)
    print(json.dumps({'run_id':result['run_id'],'status':result['status'],'comparison':result['comparison']},indent=2))


if __name__=='__main__':main()
