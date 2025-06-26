import json
import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn
from pathlib import Path

from prismatic.models import load as load_module
from prismatic.training import strategies as strat_pkg


class DummySelfAttn(nn.Module):
    def __init__(self, embed_dim: int, mismatch: bool = False) -> None:
        super().__init__()
        in_dim = embed_dim + 1 if mismatch else embed_dim
        self.q_proj = nn.Linear(in_dim, embed_dim)


class DummyTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, mismatch: bool = False) -> None:
        super().__init__()
        self.self_attn = DummySelfAttn(embed_dim, mismatch)


class DummyLLM(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, mismatch_first: bool, mismatch_rest: bool) -> None:
        super().__init__()
        layers = []
        for idx in range(num_layers):
            layers.append(DummyTransformerLayer(embed_dim, mismatch_first if idx == 0 else mismatch_rest))
        # mimic HF models which store layers under `model.layers`
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


class DummyLLMBackbone(nn.Module):
    transformer_layer_cls = DummyTransformerLayer

    def __init__(self, embed_dim: int = 4, num_layers: int = 2, mismatch_first: bool = False, mismatch_rest: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.llm = DummyLLM(embed_dim, num_layers, mismatch_first, mismatch_rest)
        self.half_precision_dtype = torch.bfloat16

    def get_fsdp_wrapping_policy(self):
        from functools import partial
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        return partial(transformer_auto_wrap_policy, transformer_layer_cls={DummyTransformerLayer})


class DummyVisionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.half_precision_dtype = torch.bfloat16

    def get_fsdp_wrapping_policy(self):
        from torch.distributed.fsdp.wrap import always_wrap_policy

        return always_wrap_policy


class DummyVLM(nn.Module):
    def __init__(self, llm_backbone: DummyLLMBackbone) -> None:
        super().__init__()
        self.vision_backbone = DummyVisionBackbone()
        self.llm_backbone = llm_backbone
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


class DummyFSDP(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _write_dummy_config(run_dir: Path, *, as_yaml: bool = False) -> None:
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "latest-checkpoint.pt").touch()
    data = {
        "model": {
            "model_id": "dummy",
            "arch_specifier": "gelu-mlp",
            "vision_backbone_id": "dummy-vision",
            "llm_backbone_id": "dummy-llm",
            "image_resize_strategy": "crop",
        }
    }
    if as_yaml:
        with open(run_dir / "config.yaml", "w") as f:
            f.write(
                "model:\n"
                "  model_id: dummy\n"
                "  arch_specifier: gelu-mlp\n"
                "  vision_backbone_id: dummy-vision\n"
                "  llm_backbone_id: dummy-llm\n"
                "  image_resize_strategy: crop\n"
            )
    else:
        with open(run_dir / "config.json", "w") as f:
            json.dump(data, f)


def _write_vla_config(run_dir: Path, base_vlm: str = "dummy") -> None:
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "latest-checkpoint.pt").touch()
    with open(run_dir / "config.json", "w") as f:
        json.dump({"vla": {"base_vlm": base_vlm}}, f)


def test_load_raises_on_bad_q_proj(monkeypatch, tmp_path):
    run_dir = tmp_path / "model"
    _write_dummy_config(run_dir)

    monkeypatch.setattr(load_module, "get_vision_backbone_and_transform", lambda *a, **k: (DummyVisionBackbone(), None))
    monkeypatch.setattr(load_module, "get_llm_backbone_and_tokenizer", lambda *a, **k: (DummyLLMBackbone(mismatch_first=True), None))
    monkeypatch.setattr(
        load_module.PrismaticVLM,
        "from_pretrained",
        classmethod(lambda cls, ckpt, model_id, vision_backbone, llm_backbone, arch_specifier="gelu-mlp", freeze_weights=True: DummyVLM(llm_backbone)),
    )

    with pytest.raises(ValueError):
        load_module.load(run_dir)


def test_load_reinitializes_other_layers(monkeypatch, tmp_path):
    run_dir = tmp_path / "model"
    _write_dummy_config(run_dir)

    llm_backbone = DummyLLMBackbone(mismatch_rest=True)
    monkeypatch.setattr(load_module, "get_vision_backbone_and_transform", lambda *a, **k: (DummyVisionBackbone(), None))
    monkeypatch.setattr(load_module, "get_llm_backbone_and_tokenizer", lambda *a, **k: (llm_backbone, None))
    monkeypatch.setattr(
        load_module.PrismaticVLM,
        "from_pretrained",
        classmethod(lambda cls, ckpt, model_id, vision_backbone, llm_backbone, arch_specifier="gelu-mlp", freeze_weights=True: DummyVLM(llm_backbone)),
    )

    vlm = load_module.load(run_dir)
    hidden_dim = vlm.llm_backbone.embed_dim
    for block in vlm.llm_backbone.llm.model.layers:
        q_proj = block.self_attn.q_proj
        assert tuple(q_proj.weight.shape) == (hidden_dim, hidden_dim)
        if q_proj.bias is not None:
            assert q_proj.bias.numel() == hidden_dim


def test_load_accepts_yaml_config(monkeypatch, tmp_path):
    run_dir = tmp_path / "model"
    _write_dummy_config(run_dir, as_yaml=True)

    monkeypatch.setattr(load_module, "get_vision_backbone_and_transform", lambda *a, **k: (DummyVisionBackbone(), None))
    monkeypatch.setattr(load_module, "get_llm_backbone_and_tokenizer", lambda *a, **k: (DummyLLMBackbone(), None))
    monkeypatch.setattr(
        load_module.PrismaticVLM,
        "from_pretrained",
        classmethod(lambda cls, ckpt, model_id, vision_backbone, llm_backbone, arch_specifier="gelu-mlp", freeze_weights=True: DummyVLM(llm_backbone)),
    )

    vlm = load_module.load(run_dir)
    assert isinstance(vlm, DummyVLM)


def test_load_accepts_vla_config(monkeypatch, tmp_path):
    run_dir = tmp_path / "model"
    _write_vla_config(run_dir)

    class DummyModelCfg:
        def __init__(self):
            self.model_id = "dummy"
            self.arch_specifier = "gelu-mlp"
            self.vision_backbone_id = "dummy-vision"
            self.llm_backbone_id = "dummy-llm"
            self.image_resize_strategy = "crop"

    monkeypatch.setattr(
        load_module.ModelConfig,
        "get_choice_class",
        classmethod(lambda cls, name: DummyModelCfg),
    )
    monkeypatch.setattr(
        load_module,
        "get_vision_backbone_and_transform",
        lambda *a, **k: (DummyVisionBackbone(), None),
    )
    monkeypatch.setattr(
        load_module,
        "get_llm_backbone_and_tokenizer",
        lambda *a, **k: (DummyLLMBackbone(), None),
    )
    monkeypatch.setattr(
        load_module.PrismaticVLM,
        "from_pretrained",
        classmethod(
            lambda cls, ckpt, model_id, vision_backbone, llm_backbone, arch_specifier="gelu-mlp", freeze_weights=True: DummyVLM(llm_backbone)
        ),
    )

    vlm = load_module.load(run_dir)
    assert isinstance(vlm, DummyVLM)


def test_fsdp_setup_repairs_q_proj(monkeypatch, tmp_path):
    llm_backbone = DummyLLMBackbone(mismatch_first=True)
    model = DummyVLM(llm_backbone)

    fsdp_module = strat_pkg.fsdp
    monkeypatch.setattr(fsdp_module, "FSDP", DummyFSDP)
    monkeypatch.setattr(fsdp_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_module.torch.cuda, "current_device", lambda: 0)

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
        enable_gradient_checkpointing=False,
        enable_mixed_precision_training=False,
    )

    strategy.run_setup(tmp_path, n_train_examples=2)

    wrapped = strategy.vlm.module
    hidden_dim = llm_backbone.embed_dim
    for block in wrapped.llm_backbone.llm.model.layers:
        q_proj = block.self_attn.q_proj
        assert tuple(q_proj.weight.shape) == (hidden_dim, hidden_dim)
        if q_proj.bias is not None:
            assert q_proj.bias.numel() == hidden_dim
