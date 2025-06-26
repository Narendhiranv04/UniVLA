"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML optional
    yaml = None

import math
import torch

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for config file (json or yaml) and pretrained checkpoint
        config_json, config_yaml = run_dir / "config.json", run_dir / "config.yaml"
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"

        if not (checkpoint_pt.exists() and (config_json.exists() or config_yaml.exists())):
            # Directory exists but missing expected files -> fall back to HF Hub if possible
            if model_id_or_path not in GLOBAL_REGISTRY:
                raise AssertionError(f"Missing `config.json` or `config.yaml` for `{run_dir = }`")

            overwatch.info(f"Local path `{run_dir}` missing files; falling back to HF Hub entry `{model_id_or_path}`")
            with overwatch.local_zero_first():
                model_id = GLOBAL_REGISTRY[model_id_or_path]["model_id"]
                config_json = hf_hub_download(
                    repo_id=HF_HUB_REPO,
                    filename=f"{model_id}/config.json",
                    cache_dir=cache_dir,
                    token=hf_token,
                )
                checkpoint_pt = hf_hub_download(
                    repo_id=HF_HUB_REPO,
                    filename=f"{model_id}/checkpoints/latest-checkpoint.pt",
                    cache_dir=cache_dir,
                    token=hf_token,
                )
        # Determine which config file to load
        config_file = config_json if config_json.exists() else config_yaml
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(
                repo_id=HF_HUB_REPO,
                filename=f"{model_id}/config.json",
                cache_dir=cache_dir,
                token=hf_token,
            )
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO,
                filename=f"{model_id}/checkpoints/latest-checkpoint.pt",
                cache_dir=cache_dir,
                token=hf_token,
            )
        config_file = config_json

    # Load Model Config
    with open(config_file, "r") as f:
        if config_file.suffix == ".json":
            cfg_data = json.load(f)
        else:
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML config files")
            cfg_data = yaml.safe_load(f)
        if "model" not in cfg_data:
            if "vla" in cfg_data and "base_vlm" in cfg_data["vla"]:
                try:
                    model_cfg = ModelConfig.get_choice_class(cfg_data["vla"]["base_vlm"])().__dict__
                except Exception as e:
                    raise KeyError(
                        f"'model' section missing from config file {config_file} "
                        f"and failed to resolve `vla.base_vlm` = {cfg_data['vla']['base_vlm']}."
                    ) from e
            else:
                raise KeyError(
                    f"'model' section missing from config file {config_file}. "
                    "Ensure `--pretrain_vlm` points to a valid pretrained model "
                    "directory containing this section or use a name from "
                    "`prismatic.available_model_names()`."
                )
        else:
            model_cfg = cfg_data["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    hidden_dim = vlm.llm_backbone.embed_dim

    # Sanity check q_proj dimensions to catch malformed checkpoints early
    try:
        layer = vlm.llm_backbone.llm.model.layers[0].self_attn
        if tuple(layer.q_proj.weight.shape) != (hidden_dim, hidden_dim):
            raise ValueError(
                f"Invalid q_proj weight shape {tuple(layer.q_proj.weight.shape)};"
                f" expected ({hidden_dim}, {hidden_dim}). Make sure `--pretrain_vlm` "
                "points to a directory with the correct `config.json` and checkpoint."
            )
        if layer.q_proj.bias is not None and layer.q_proj.bias.numel() != hidden_dim:
            raise ValueError(
                f"Invalid q_proj bias shape {tuple(layer.q_proj.bias.shape)};"
                f" expected ({hidden_dim},). Ensure the checkpoint is not corrupted."
            )
    except (AttributeError, NameError):
        pass

    # Reinitialize any malformed q_proj layers so training can proceed
    try:
        for idx, block in enumerate(vlm.llm_backbone.llm.model.layers):
            q_proj = block.self_attn.q_proj
            if tuple(q_proj.weight.shape) != (hidden_dim, hidden_dim) or (
                q_proj.bias is not None and q_proj.bias.numel() != hidden_dim
            ):
                overwatch.warning(
                    f"q_proj of layer {idx} had shape {tuple(q_proj.weight.shape)};"
                    f" reinitializing to ({hidden_dim}, {hidden_dim})"
                )
                q_proj.weight.data = torch.empty(hidden_dim, hidden_dim, dtype=q_proj.weight.dtype)
                torch.nn.init.kaiming_uniform_(q_proj.weight, a=math.sqrt(5))
                if q_proj.bias is not None:
                    q_proj.bias.data.zero_()
    except (AttributeError, NameError):
        pass

    return vlm


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
    action_codebook_size: int = 32,
) -> OpenVLA:
    """Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify Checkpoint to Load (via `step_to_load`)
        step_to_load = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load if step_to_load is not None else ''}*.pt")
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO,
                filename=f"{(relpath / 'config.json')!s}",
                cache_dir=cache_dir,
                token=hf_token,
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO,
                filename=f"{(relpath / 'dataset_statistics.json')!s}",
                cache_dir=cache_dir,
                token=hf_token,
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO,
                filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}",
                cache_dir=cache_dir,
                token=hf_token,
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer())

    # Add special tokens and resize embeddings
    # special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(action_codebook_size)]}
    # num_added_toks = action_tokenizer.add_special_tokens(special_tokens_dict)
    # llm_backbone.llm.resize_token_embeddings(32033)

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
    )

    return vla
