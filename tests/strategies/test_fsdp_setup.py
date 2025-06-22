import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from prismatic.training import strategies as strat_pkg


class DummyFSDP(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DummyTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class DummyLLMBackbone(nn.Module):
    transformer_layer_cls = DummyTransformerLayer

    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(DummyTransformerLayer() for _ in range(num_layers))
        self.half_precision_dtype = torch.bfloat16

    def get_fsdp_wrapping_policy(self):
        from functools import partial

        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        return partial(transformer_auto_wrap_policy, transformer_layer_cls={DummyTransformerLayer})

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DummyVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.half_precision_dtype = torch.bfloat16

    def get_fsdp_wrapping_policy(self):
        from torch.distributed.fsdp.wrap import always_wrap_policy

        return always_wrap_policy

    def forward(self, x):
        return self.linear(x)


class DummyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = DummyVisionBackbone()
        self.llm_backbone = DummyLLMBackbone()
        self.all_module_keys = ["vision_backbone", "llm_backbone"]
        self.trainable_module_keys = self.all_module_keys

    def get_fsdp_wrapping_policy(self):
        from functools import partial

        from torch.distributed.fsdp.wrap import _or_policy

        policies = [
            self.vision_backbone.get_fsdp_wrapping_policy(),
            self.llm_backbone.get_fsdp_wrapping_policy(),
        ]
        return partial(_or_policy, policies=policies)

    def forward(self, *args, **kwargs):
        x = torch.randn(1, 1)
        x = self.llm_backbone(x)
        x = self.vision_backbone(x)
        return x


@pytest.fixture
def strategy(monkeypatch, tmp_path):
    fsdp_module = strat_pkg.fsdp

    monkeypatch.setattr(fsdp_module, "FSDP", DummyFSDP)
    monkeypatch.setattr(fsdp_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_module.torch.cuda, "current_device", lambda: 0)

    model = DummyVLM()
    strategy = fsdp_module.FSDPStrategy(
        vlm=model,
        device_id=0,
        stage="full-finetune",
        epochs=1,
        max_steps=None,
        global_batch_size=1,
        per_device_batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        enable_gradient_checkpointing=True,
        enable_mixed_precision_training=False,
    )

    strategy.run_setup(tmp_path, n_train_examples=2)
    return strategy


def test_fsdp_wraps_model_and_applies_checkpointing(strategy):
    assert isinstance(strategy.vlm, DummyFSDP)
    wrapped = strategy.vlm.module
    for layer in wrapped.llm_backbone.layers:
        assert isinstance(layer, CheckpointWrapper)
