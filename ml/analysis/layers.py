"""Descriptive activation/adapter statistics and matched early/late layer experiments."""
import argparse,gc,json,os
from pathlib import Path
from recipetriage_ml.training.data import seed_snapshot,verify_snapshot
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.sft import run_local,load_adapter
from recipetriage_ml.evaluation.base_provider import BaseProvider,resources,render_prompt
from recipetriage_ml.inference.prompts import build_messages
from recipetriage_ml.data.pipeline import digest


def activation_stats(model,tokenizer,examples):
    import torch
    was_training=model.training;model.eval();values=[]
    try:
        for example in examples:
            prompt=render_prompt(build_messages(example.recipe.inference_recipe()))
            tokens=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
            with torch.inference_mode():out=model(**tokens,output_hidden_states=True,use_cache=False)
            for index,hidden in enumerate(out.hidden_states):
                h=hidden.float()
                values.append({'recipe_id':example.recipe.id,'hidden_state_index':index,'tokens':h.shape[1],
                    'mean':h.mean().item(),'std':h.std(unbiased=False).item(),'rms':h.square().mean().sqrt().item(),
                    'max_abs':h.abs().max().item(),'input_ids_sha256':digest(tokens.input_ids.tolist())})
    finally:model.train(was_training)
    return values


def adapter_stats(model):
    import torch
    values=[]
    with torch.no_grad():
        for name,layer in model.named_modules():
            if hasattr(layer,'lora_A') and 'default' in layer.lora_A:
                a=layer.lora_A['default'].weight.float();b=layer.lora_B['default'].weight.float()
                delta=(b@a)*layer.scaling['default'];base=layer.base_layer.weight.float()
                values.append({'module':name,'a_norm':a.norm().item(),'b_norm':b.norm().item(),
                    'delta_frobenius':delta.norm().item(),'relative_update_norm':(delta.norm()/base.norm().clamp_min(1e-12)).item()})
    return values


def run(output):
    import torch
    torch.set_num_threads(4);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'comparison.json').exists():raise ValueError('Choose a new analysis directory; do not overwrite evidence')
    snapshot=seed_snapshot();examples=verify_snapshot(snapshot)['train'][:2]
    tokenizer,base=resources();before=activation_stats(base,tokenizer,examples)
    (output/'before.json').write_text(json.dumps(before,indent=2)+'\n');resources.cache_clear();del base;gc.collect()
    baseline=json.loads((Path(__file__).resolve().parents[1]/'training/results/experiments-v1/baseline.json').read_text())
    results=[]
    for name,layers in [('early-six',list(range(6))),('late-six',list(range(18,24)))]:
        config=SFTConfig(dataset_version=snapshot['metadata']['version'],lora_rank=4,lora_layers=layers)
        result=run_local(config,snapshot,root=output/'runs',reference_baseline=baseline)
        provider=load_adapter(output/'runs'/result['run_id']);after=activation_stats(provider.model,provider.tokenizer,examples)
        statistics={'before':before,'after':after,'adapter':adapter_stats(provider.model),
                    'scope':'Same two training recipes, token positions and eval-mode forward pass. Descriptive diagnostics, not causal attributions.'}
        (output/f'{name}-statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
        results.append({'name':name,'layers':layers,'run_id':result['run_id'],'trainable_parameters':result['model_parameters']['trainable'],
            'training_seconds':result['training_resources']['train_wall_seconds'],'micro_f1':result['benchmark']['summary']['labels']['micro']['f1'],
            'macro_f1':result['benchmark']['summary']['labels']['macro']['f1'],'statistics_file':f'{name}-statistics.json'})
        del provider;gc.collect()
    payload={'dataset_version':snapshot['metadata']['version'],'benchmark_sha256':baseline['benchmark']['sha256'] if 'sha256' in baseline['benchmark'] else baseline['protocol']['benchmark_sha256'],
             'calibration_recipe_ids':[x.recipe.id for x in examples],'results':results,
             'limitations':['Two recipes and one run per subset do not support causal explanations.','Activation scale changes can reflect coordinate/scaling changes without better task behavior.',
                            'Subset assignment is a controlled configuration intervention; observed differences are not proof that a particular layer encodes a concept.']}
    (output/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n');return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()));print(json.dumps(run(args.output),indent=2))
