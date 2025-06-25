"""qwen.py

Class definition for Qwen2 models.
"""

from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder

# Registry => support Qwen2.5-0.5B model
QWEN2_MODELS = {
    "qwen2.5-0.5b": {
        "llm_family": "qwen2",
        "llm_cls": Qwen2ForCausalLM,
        "hf_hub_path": "Qwen/Qwen2.5-0.5B",
    }
}


class Qwen2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN2_MODELS[llm_backbone_id],
        )

        # Add PAD token handling similar to other models
        self.tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
