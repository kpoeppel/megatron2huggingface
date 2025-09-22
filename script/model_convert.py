import argparse
import yaml
from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.model import ModelConverter

from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import _load_base_checkpoint

import torch.distributed as dist
import os
from types import SimpleNamespace


if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    except RuntimeError as e:
        print(
            f"Could not initialize process group: {e}. Assuming already initialized or not needed for this test."
        )


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--training-dir", type=str)
    # parser.add_argument("--checkpoint-name", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--checkpoint-iteration", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--config-subset", type=str, default="megatron")

    args = parser.parse_args()

    if args.config_file:
        config_file = args.config_file

    with open(config_file) as fp:
        cfg_dict = yaml.safe_load(fp)
        # todo: take care of more nested subsets

    cfg_dict = cfg_dict[args.config_subset]

    cfg_dict["load"] = args.checkpoint_dir
    cfg_dict["pretrained_checkpoint"] = args.checkpoint_dir
    cfg_dict["ckpt_step"] = args.checkpoint_iteration
    cfg_dict["auto_detect_ckpt_format"] = True

    _ = MegatronConfig(**cfg_dict)  # check if configuration actually works.
    megatron_args = SimpleNamespace(world_size=1, rank=0, **cfg_dict)

    initialize_megatron(
        allow_no_cuda=True,
        parsed_args=megatron_args,
    )

    converter = ModelConverter(cfg_dict)

    megatron_model = converter.create_megatron_module()
    sharded_state_dict = megatron_model.state_dict()

    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        args.checkpoint_dir,
        megatron_args,
        rank0=False,
        sharded_state_dict=sharded_state_dict,
        checkpointing_context=None,
    )

    megatron_state_dict = megatron_model.state_dict()

    hf_state_dict = converter.convert_weights(megatron_state_dict)

    del megatron_model
    del megatron_state_dict

    hf_model = converter.create_hf_module()

    hf_model.load(hf_state_dict, strict=True)

    hf_model.save_pretrained(args.output_dir, from_pt=True)


if __name__ == "__main__":
    main()
