import argparse
from pathlib import Path
from dotenv import load_dotenv
from .benchmark import RunConfig, new_run, run_benchmark, save_file

def main():
    parser = argparse.ArgumentParser(description='Run the fixed RecipeTriage benchmark; no fine-tuning')
    parser.add_argument('--provider', choices=['hf-base', 'fireworks', 'both'], default='both')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--output', type=Path, default=Path('ml/evaluation/results'))
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--reasoning', choices=['provider-default', 'disabled'], default='provider-default')
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    providers = ['hf-base', 'fireworks'] if args.provider == 'both' else [args.provider]
    failed = False
    for provider in providers:
        config = RunConfig(provider=provider, temperature=args.temperature, max_new_tokens=args.max_new_tokens, reasoning=args.reasoning)
        run = new_run(config)
        last_count = -1
        def persist(payload):
            nonlocal last_count
            save_file(payload, args.output)
            count = len(payload['rows'])
            if count != last_count:
                print(f"{provider}: {count}/{len(payload['benchmark']['cases'])} cases recorded", flush=True)
                last_count = count
        try:
            result = run_benchmark(config, on_update=persist, run=run)
        except BaseException:
            run.update(status='interrupted', error='Runner interrupted; partial evidence retained')
            save_file(run, args.output)
            raise
        print(f"Saved {save_file(result, args.output)}; status={result['status']}", flush=True)
        score = result['summary']['labels']
        print(f"Usable: {result['summary']['usable_responses']}/{len(result['rows'])}; micro F1: {score['micro']['f1'] if score else None}; exact match: {score['exact_match'] if score else None}", flush=True)
        failed |= result['status'] != 'completed' or result['summary']['usable_responses'] != len(result['rows'])
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
