import argparse
from pathlib import Path

from prismatic.models import load


def main():
    parser = argparse.ArgumentParser(description="Print q_proj shapes from a checkpoint")
    parser.add_argument("model", type=str, help="Model directory or model name")
    args = parser.parse_args()

    vlm = load(Path(args.model))
    layer = vlm.llm_backbone.llm.model.layers[0].self_attn
    w_shape = tuple(layer.q_proj.weight.shape)
    b_shape = tuple(layer.q_proj.bias.shape) if layer.q_proj.bias is not None else None
    print("q_proj weight:", w_shape)
    print("q_proj bias:", b_shape)


if __name__ == "__main__":
    main()
