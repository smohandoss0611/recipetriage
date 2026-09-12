"""Real TRL optimizer check on random tiny weights; never a RecipeTriage quality claim."""
import pytest


def test_tiny_dpo_updates_policy_but_keeps_reference_frozen(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('trl');pytest.importorskip('peft')
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers
    from datasets import Dataset
    from peft import LoraConfig,get_peft_model
    from trl import DPOConfig,DPOTrainer
    torch.set_num_threads(1);torch.manual_seed(3)
    tok=Tokenizer(models.WordLevel({'<pad>':0,'<eos>':1,'<unk>':2,'recipe':3,'good':4,'bad':5},unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    config=Qwen2Config(vocab_size=6,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                       max_position_embeddings=64,tie_word_embeddings=True,pad_token_id=0,eos_token_id=1)
    model=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.save_pretrained(tmp_path/'initial');model.load_adapter(tmp_path/'initial',adapter_name='reference',is_trainable=False);model.set_adapter('default')
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if '.reference.' in n}
    policy={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    rows=Dataset.from_list([{'prompt':'recipe','chosen':'good','rejected':'bad'}]*2)
    args=DPOConfig(output_dir=str(tmp_path/'run'),use_cpu=True,bf16=False,fp16=False,max_steps=2,learning_rate=.01,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,report_to='none',save_strategy='no',logging_steps=1,
        max_length=32,max_prompt_length=None,max_completion_length=None,model_adapter_name='default',ref_adapter_name='reference',
        dataloader_pin_memory=False,disable_tqdm=True)
    trainer=DPOTrainer(model=model,args=args,processing_class=tokenizer,train_dataset=rows)
    trainer.train()
    assert any(not torch.equal(policy[n],p) for n,p in model.named_parameters() if n in policy)
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen)
