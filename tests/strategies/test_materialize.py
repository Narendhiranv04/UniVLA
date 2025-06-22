import pytest

pytest.importorskip("torch")

import torch.nn as nn
from torch.distributed.fsdp import ShardingStrategy

from prismatic.training.materialize import get_train_strategy
from prismatic.training.strategies.fsdp import FSDPStrategy


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layer_cls = nn.Linear


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm_backbone = TinyBackbone()
        self.all_module_keys = []
        self.trainable_module_keys = []


def test_get_train_strategy_no_shard():
    vlm = TinyVLM()
    strategy = get_train_strategy(
        "no-shard",
        vlm=vlm,
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
        enable_gradient_checkpointing=False,
        enable_mixed_precision_training=False,
    )
    assert isinstance(strategy, FSDPStrategy)
    assert strategy.fsdp_sharding_strategy == ShardingStrategy.NO_SHARD
