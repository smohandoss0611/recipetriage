"""Explicit learning defaults. Nothing here selects settings using benchmark scores."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from recipetriage_ml.evaluation.base_provider import MODEL_ID, MODEL_REVISION


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: Literal['local', 'fireworks'] = 'local'
    dataset_version: str = Field(pattern=r'^v1-[0-9a-f]{64}$')
    epochs: int = Field(default=3, ge=1, le=20, strict=True)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01, allow_inf_nan=False)
    sequence_length: int = Field(default=768, ge=128, le=2048, strict=True)
    batch_size: int = Field(default=1, ge=1, le=4, strict=True)
    gradient_accumulation_steps: int = Field(default=2, ge=1, le=32, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    lora_rank: Literal[4, 8, 16, 32] = 8
    lora_alpha: int | None = Field(default=None, ge=1, le=256, strict=True)
    lora_dropout: float = Field(default=0.0, ge=0, le=0.5, allow_inf_nan=False)
    lora_target_policy: Literal['query-value', 'attention', 'attention-and-mlp'] = 'query-value'
    lora_layers: list[int] | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    method: Literal['lora', 'qlora'] = 'lora'
    gradient_checkpointing: bool = Field(default=True, strict=True)
    fireworks_model: str | None = Field(default=None, pattern=r'^accounts/[a-z0-9-]+/models/[a-z0-9_.-]+$')
    fireworks_deployment_shape: str | None = Field(default=None, pattern=r'^accounts/[a-z0-9-]+/deploymentShapes/[a-z0-9_.-]+$')

    @model_validator(mode='after')
    def supported_method(self):
        if self.provider == 'fireworks' and self.method == 'qlora':
            raise ValueError('This bitsandbytes QLoRA implementation is local only')
        return self

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps

    def resolved(self):
        """Record fixed settings as well as editable settings in every run."""
        return {**self.model_dump(), 'effective_batch_size': self.effective_batch_size,
                'base_model': MODEL_ID if self.provider == 'local' else self.fireworks_model,
                'base_revision': MODEL_REVISION if self.provider == 'local' else None,
                'objective': 'assistant-token causal cross-entropy; EOS supervised; padding masked',
                'optimizer': 'adamw_torch' if self.provider == 'local' else 'provider-managed',
                'weight_decay': 0.0, 'scheduler': 'constant', 'warmup_steps': 0,
                'eval_batch_size': 1 if self.provider == 'local' else 'provider-managed',
                'max_grad_norm': 1.0 if self.provider == 'local' else 'provider-managed',
                'lora_alpha': (self.lora_alpha or 2 * self.lora_rank) if self.provider == 'local' else 'provider-managed',
                'lora_dropout': self.lora_dropout if self.provider == 'local' else 'provider-managed',
                'lora_targets': ('inspected full paths: ' + self.lora_target_policy) if self.provider == 'local' else 'provider-managed',
                'device': 'cpu' if self.provider == 'local' else 'provider-managed',
                'dtype': 'float32' if self.provider == 'local' else 'provider-managed',
                'gradient_checkpointing': self.gradient_checkpointing if self.provider == 'local' else 'provider-managed',
                'packing': False if self.provider == 'local' else 'provider-managed',
                'truncation': 'reject overlength locally; managed renderer validation required',
                'checkpoint_selection': 'minimum validation loss, evaluated each epoch' if self.provider == 'local' else 'provider final output model',
                'checkpoint_limit': 2 if self.provider == 'local' else 'provider-managed',
                'benchmark': {'temperature': 0.0, 'max_new_tokens': 128, 'reasoning': 'disabled'}}
