"""Two-step CPU GRPO experiment; never changes the production pointer."""
import argparse,json
from pathlib import Path
from .runner import LearningConfig,run


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--source',type=Path)
    args=parser.parse_args();config=LearningConfig.model_validate_json(args.config.read_text()) if args.config else LearningConfig()
    result=run('grpo',config,root=args.output,source=args.source)
    print(json.dumps({'run_id':result['run_id'],'status':result['status'],'comparison':result['comparison']},indent=2))


if __name__=='__main__':main()
