"""Sequential deployment measurements and constraint-aware, quality-first selection."""
import argparse,gc,json,math,os,time
from pathlib import Path
from pydantic import BaseModel,ConfigDict,Field
from recipetriage_ml.training.data import file_hash
from recipetriage_ml.training.sft import TrainedProvider
from recipetriage_ml.training.lora import TrainingResources
from recipetriage_ml.evaluation.base_provider import MODEL_ID,MODEL_REVISION,FORMAT_VERSION
from recipetriage_ml.evaluation.benchmark import new_run,run_benchmark,RunConfig
from recipetriage_ml.evaluation.shortcut_tests import run_suite


class Constraints(BaseModel):
    model_config=ConfigDict(extra='forbid')
    min_micro_f1:float=Field(default=.48,ge=0,le=1,allow_inf_nan=False)
    min_schema_validity:float=Field(default=.4,ge=0,le=1,allow_inf_nan=False)
    max_ram_mib:float=Field(default=4096,gt=0,allow_inf_nan=False)
    max_size_mib:float=Field(default=3072,gt=0,allow_inf_nan=False)
    max_p95_ms:float=Field(default=15000,gt=0,allow_inf_nan=False)


def select_candidate(rows,constraints=None):
    c=constraints or Constraints();accepted=[];reasons={}
    for row in rows:
        failures=[]
        for metric,threshold,minimum in [('micro_f1',c.min_micro_f1,True),('schema_validity',c.min_schema_validity,True),
                ('peak_ram_mib',c.max_ram_mib,False),('size_mib',c.max_size_mib,False),('p95_ms',c.max_p95_ms,False)]:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (value<threshold if minimum else value>threshold):failures.append(metric)
        reasons[row['name']]=failures
        if not failures:accepted.append(row)
    chosen=max(accepted,key=lambda x:(x['micro_f1'],x['schema_validity'],-x['p95_ms'])) if accepted else None
    return {'recommended':chosen['name'] if chosen else None,'constraints':c.model_dump(),'failed_constraints':reasons,
            'selection':'Highest Micro-F1, then schema validity, then latency among feasible candidates; size is only a limit.',
            'production_changed':False,'note':'Operational recommendation is separate from the stricter model-registry promotion gate.'}


def load_export(directory):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    directory=Path(directory);manifest=json.loads((directory/'deployment-manifest.json').read_text())
    for name,h in manifest['files'].items():
        if Path(name).name!=name or file_hash(directory/name)!=h:raise ValueError('Deployment export checksum mismatch')
    model=AutoModelForCausalLM.from_pretrained(directory,dtype=torch.float32,device_map={'':'cpu'},trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(directory,trust_remote_code=False)
    identity={'model':MODEL_ID+'+'+manifest['format'],'revision':manifest['artifact_sha256'],'base_model':MODEL_ID,
              'base_revision':MODEL_REVISION,'prompt_format':FORMAT_VERSION,'training_stage':'merged-deployment' if manifest['source_adapter'] else 'pretrained-base-export',
              'quantization':manifest.get('quantization'),'source_adapter':manifest['source_adapter']}
    return TrainedProvider(model,tokenizer,identity)


def measure(name,directory,output):
    import torch
    torch.set_num_threads(4);started=time.perf_counter();provider=load_export(directory);load_seconds=time.perf_counter()-started
    config=RunConfig(provider='hf-base',temperature=0,max_new_tokens=128,reasoning='disabled');payload=new_run(config);payload['model_config']=provider.identity
    # Generation is timed after model load; RSS includes the resident model throughout evaluation.
    with TrainingResources() as measured:benchmark=run_benchmark(config,run=payload,provider_instance=provider)
    shortcut=run_suite(provider,provider.identity)
    s=benchmark['summary'];row={'name':name,'format':json.loads((Path(directory)/'deployment-manifest.json').read_text())['format'],
        'directory':str(Path(directory).resolve()),'size_mib':sum(p.stat().st_size for p in Path(directory).iterdir() if p.is_file())/2**20,
        'peak_ram_mib':measured.result['peak_rss_bytes']/2**20,'vram_mib':None,'device':'cpu','load_seconds':load_seconds,
        'mean_ms':s['latency_all_attempts']['mean_ms'],'p95_ms':s['latency_all_attempts']['p95_ms'],
        'micro_f1':s['labels']['micro']['f1'],'macro_f1':s['labels']['macro']['f1'],'schema_validity':s['schema_validity'],
        'json_validity':s['json_validity'],'benchmark':benchmark,'shortcut':shortcut,'identity':provider.identity,
        'measurement_scope':'One sequential CPU run. Sampled total process RSS during benchmark; model loading excluded from latency. VRAM not applicable.'}
    Path(output).write_text(json.dumps(row,indent=2)+'\n');del provider;gc.collect();return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--name',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);measure(args.name,args.model,args.output)
