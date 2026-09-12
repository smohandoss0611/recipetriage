"""Small isolated TRL learning runs with immutable inputs and automatic evaluation."""
import gc
import json
import math
import os
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel,ConfigDict,Field
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.runs import Journal,now,evaluate
from recipetriage_ml.training.lora import TrainingResources,save_adapter,parameter_report
from recipetriage_ml.training.sft import load_adapter,TrainedProvider
from recipetriage_ml.training.data import file_hash,seed_snapshot,verify_snapshot
from recipetriage_ml.inference.prompts import build_messages
from recipetriage_ml.inference.contracts import Recipe,Generation
from recipetriage_ml.evaluation.base_provider import render_prompt,FORMAT_VERSION
from recipetriage_ml.evaluation import behaviors,shortcut_tests
from .preferences import reference_directory,preference_dataset,ml_directory
from .rewards import make_grpo_reward,WEIGHTS,VERSION


class LearningConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    learning_rate:float=Field(default=5e-6,gt=0,le=1e-4,allow_inf_nan=False)
    max_steps:int=Field(default=2,ge=1,le=10,strict=True)
    beta:float=Field(default=.1,gt=0,le=1,allow_inf_nan=False)
    sequence_length:int=Field(default=1024,ge=512,le=2048,strict=True)
    seed:int=Field(default=42,ge=0,le=2**32-1,strict=True)


def run(method,config=None,pairs=None,root=None,source=None,on_update=None):
    if method not in {'dpo','grpo'}:raise ValueError('Choose dpo or grpo')
    config=config or LearningConfig();source=Path(source or reference_directory())
    if method=='dpo':preference_dataset(pairs or [])  # fail before loading weights without choices
    run={'run_id':str(uuid4()),'kind':method,'status':'queued','phase':'prepare','created_at':now(),
         'config':{'provider':'local',**config.model_dump()},'history':[],'artifacts':{},'error':None,
         'production_changed':False,'reference':json.loads((source/'adapter-provenance.json').read_text())}
    journal=Journal(run,root or Path('ml/alignment/runs'),on_update)
    journal.artifact('input-preferences.json',pairs or [])
    provider=trainer=None;reward_log=[]
    try:
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback,set_seed
        from trl import DPOConfig,DPOTrainer,GRPOConfig,GRPOTrainer
        torch.set_num_threads(int(os.getenv('HF_CPU_THREADS','4')));set_seed(config.seed)
        journal.write(status='preparing')
        provider=load_adapter(source);model,tokenizer=provider.model,provider.tokenizer
        tokenizer.pad_token=tokenizer.eos_token
        # The selected SFT adapter is the policy initialization, not a fresh adapter.
        for name,p in model.named_parameters():p.requires_grad_('.lora_A.default.' in name or '.lora_B.default.' in name)
        model.config.use_cache=False
        params=parameter_report(model);journal.artifact('parameter_report.json',params)
        common=dict(output_dir=str(journal.directory/'checkpoints'),use_cpu=True,fp16=False,bf16=False,
            max_steps=config.max_steps,learning_rate=config.learning_rate,per_device_train_batch_size=1,
            gradient_accumulation_steps=2,per_device_eval_batch_size=1,optim='adamw_torch',weight_decay=0.,
            lr_scheduler_type='constant',warmup_steps=0,max_grad_norm=1.,seed=config.seed,data_seed=config.seed,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            logging_steps=1,save_strategy='no',report_to='none',disable_tqdm=True,dataloader_pin_memory=False)
        class Progress(TrainerCallback):
            def on_log(self,args,state,control,logs=None,**kwargs):
                values={k:float(v) for k,v in (logs or {}).items() if isinstance(v,(int,float))}
                if any(not math.isfinite(v) for v in values.values()):raise ValueError('Non-finite learning metric')
                journal.run['history'].append({'step':state.global_step,**values});journal.write()
        if method=='dpo':
            data=preference_dataset(pairs,tokenizer,config.sequence_length);journal.artifact('preference-dataset.json',data)
            model.load_adapter(source/'adapter',adapter_name='reference',is_trainable=False)
            model.set_adapter('default')
            ref_before={n:p.detach().clone() for n,p in model.named_parameters() if '.reference.' in n}
            args=DPOConfig(**common,beta=config.beta,loss_type='sigmoid',max_length=config.sequence_length,
                max_prompt_length=None,max_completion_length=None,model_adapter_name='default',ref_adapter_name='reference',
                remove_unused_columns=False,eval_strategy='no')
            trainer=DPOTrainer(model=model,args=args,processing_class=tokenizer,
                train_dataset=Dataset.from_list(data['train']),eval_dataset=Dataset.from_list(data['validation']),callbacks=[Progress()])
            journal.run['dataset_lineage']={'preference_version':data['version'],'human_decisions':data['human_decisions'],
                                           'train_pairs':len(data['train']),'validation_pairs':len(data['validation'])}
            journal.run['initial_validation']=trainer.evaluate()
        else:
            snapshot=seed_snapshot();examples=verify_snapshot(snapshot)['train'];rows=[];counterfactuals={}
            for example in examples:
                original=example.recipe.inference_recipe().model_dump();stem=original['title']
                reference={**original,'title':'Traditional '+stem};changed={**original,'title':'Quick '+stem}
                shortcut_tests.assert_title_only(reference,changed,stem)
                prompt=render_prompt(build_messages(Recipe.model_validate(reference)))
                if len(tokenizer.encode(prompt,add_special_tokens=False))>config.sequence_length-128:raise ValueError('GRPO input exceeds budget; no prompt truncation')
                rows.append({'prompt':prompt,'expected_labels':example.labels,'recipe_id':example.recipe.id})
                counterfactuals[prompt]=build_messages(Recipe.model_validate(changed))
            def paired(prompt):
                # Extra greedy title intervention on the SAME current weights; preserve training RNG/mode.
                was_training=model.training;checkpointing=model.is_gradient_checkpointing
                model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
                try:
                    with torch.random.fork_rng(devices=[]):return provider.generate(counterfactuals[prompt],0,128)
                finally:
                    model.train(was_training);model.config.use_cache=False
                    if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            args=GRPOConfig(**common,num_generations=2,max_completion_length=128,max_prompt_length=config.sequence_length-128,
                temperature=.8,top_p=1.,beta=0.,use_vllm=False,loss_type='grpo',scale_rewards='group',
                mask_truncated_completions=True,remove_unused_columns=False)
            trainer=GRPOTrainer(model=model,args=args,processing_class=tokenizer,train_dataset=Dataset.from_list(rows),
                reward_funcs=make_grpo_reward(reward_log,paired,tokenizer.eos_token_id),callbacks=[Progress()])
            journal.run['dataset_lineage']={'version':snapshot['metadata']['version'],'split':'train','recipe_ids':[x.recipe.id for x in examples],
                'annotation_status':'provisional legacy seed; not human preference labels'}
            journal.run['reward']={'version':VERSION,'weights':WEIGHTS,'group_size':2,'kl_beta':0.,
                'warning':'No KL penalty in this two-step CPU probe. Reward is a fallible label-policy proxy, not human preference.'}
            journal.artifact('grpo-inputs.json',rows)
        journal.write(status='training',phase='trl-optimizer-updates')
        with TrainingResources() as measured:trainer.train()
        journal.run['resources']=measured.result
        if method=='dpo':
            journal.run['final_validation']=trainer.evaluate()
            if any(not torch.equal(ref_before[n],p) for n,p in model.named_parameters() if n in ref_before):raise ValueError('DPO reference adapter changed')
            journal.run['reference_frozen_verified']=True;model.delete_adapter('reference');model.set_adapter('default')
        else:
            journal.artifact('reward-components.json',reward_log)
            journal.run['reward_components']=reward_log
        model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
        save_adapter(model,tokenizer,journal.directory/'adapter')
        identity={**provider.identity,'model':provider.identity['base_model']+'+'+method+'/'+run['run_id'],
                  'revision':file_hash(journal.directory/'adapter/adapter_model.safetensors'),'training_stage':method,
                  'parent_checkpoint':provider.identity['revision']}
        journal.artifact('adapter-provenance.json',identity)
        journal.write(status='evaluating',phase='fixed-benchmark-and-title-tests')
        current=TrainedProvider(model,tokenizer,identity)
        benchmark=evaluate(journal,current,identity,'benchmark')
        shortcut=shortcut_tests.run_suite(current,identity);journal.artifact('shortcut.json',shortcut)
        before_bench=json.loads((source/'benchmark.json').read_text())
        before_shortcut=json.loads((ml_directory()/'evaluation/results/diagnostics-v1/qlora.json').read_text())['shortcut']
        before=behaviors.report(before_bench,before_shortcut);after=behaviors.report(benchmark,shortcut)
        journal.run['comparison']=behaviors.compare(before,after)
        journal.write(status='completed',phase='finished',finished_at=now())
    except Exception as exc:
        journal.write(status='failed',error=str(exc),finished_at=now());raise
    finally:
        del trainer,provider;gc.collect()
    return journal.run
