import argparse
import yaml
from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.model import ModelConverter

from megatron.training.initialize import initialize_megatron

# from megatron.core.dist_checkpointing import load
from megatron.core.dist_checkpointing.serialization import load, StrictHandling
from torch.distributed.checkpoint import load_state_dict, FileSystemReader
import torch

import torch.distributed as dist
import os
import json
from types import SimpleNamespace
import shutil
import inspect
import sys
from pathlib import Path

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from script.generate_modeling_file import generate_modeling_auto


def signature_to_dict(func_or_cls, drop_self=True):
    # If a class is passed, inspect its __init__
    if inspect.isclass(func_or_cls):
        func = func_or_cls.__init__
    else:
        func = func_or_cls

    sig = inspect.signature(func)
    result = {}
    for name, param in sig.parameters.items():
        if drop_self and name == "self":
            continue
        if param.default is not inspect._empty:
            result[name] = param.default
        else:
            result[name] = None  # or param.default if you want inspect._empty
    return result


def load_and_reshape_distcp(checkpoint_dir, target_model):
    """Load distcp checkpoint and reshape for target model"""

    # Load checkpoint metadata to understand structure
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()

    # Build state dict matching checkpoint structure
    checkpoint_state = {}
    for key in metadata.state_dict_metadata.keys():
        # Initialize with zeros matching saved shape
        if hasattr(metadata.state_dict_metadata[key], "size"):
            saved_shape = metadata.state_dict_metadata[key].size
            checkpoint_state[key] = torch.zeros(saved_shape)
    checkpoint_state = {"model." + key: val for key, val in checkpoint_state.items()}
    # Load actual data
    load_state_dict(
        state_dict=checkpoint_state,
        storage_reader=reader,
    )

    # Reshape to target
    target_state = target_model.state_dict()
    reshaped = {}

    for tgt_key, tgt_param in target_state.items():
        if tgt_key in checkpoint_state:
            ckpt_tensor = checkpoint_state[tgt_key]

            # Extract from ShardedTensor if needed
            if hasattr(ckpt_tensor, "local_shards"):
                local_shards = ckpt_tensor.local_shards()
                if len(local_shards) == 1:
                    ckpt_tensor = local_shards[0].tensor
                else:
                    # Concatenate multiple shards (figure out correct dim)
                    ckpt_tensor = torch.cat([s.tensor for s in local_shards])

            if ckpt_tensor.shape != tgt_param.shape:
                print(f"Reshaping {tgt_key}: {ckpt_tensor.shape} -> {tgt_param.shape}")
                reshaped[tgt_key] = ckpt_tensor.reshape(tgt_param.shape)
            else:
                reshaped[tgt_key] = ckpt_tensor
        else:
            print(f"Missing key: {tgt_key}")

    return reshaped


if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    except RuntimeError as e:
        print(
            f"Could not initialize process group: {e}. Assuming already initialized or not needed for this test."
        )


def _to_base_type(val: str):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--training-dir", type=str)
    # parser.add_argument("--checkpoint-name", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--checkpoint-iteration", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--config-subset", type=str, default="megatron")
    parser.add_argument("--overrides", type=str, nargs="*", default=[])

    args = parser.parse_args()

    if args.config_file:
        config_file = args.config_file

    with open(config_file) as fp:
        cfg_dict = yaml.safe_load(fp)
        # todo: take care of more nested subsets


    cfg_dict_base = signature_to_dict(MegatronConfig)
    cfg_dict_base.update(**cfg_dict[args.config_subset])
    cfg_dict = cfg_dict_base
    print(f"Overrides: {args.overrides}")
    cfg_dict.update(**{arg.split("=")[0]: arg.split("=")[1] for arg in args.overrides})
    # pretty print cfg_dict 
    for key, value in cfg_dict.items():
        print(f"{key}: {value}")
        print("-"*80)
    print("-"*80)
    print("CFG DICT")
    print("-"*80)   
    cfg_dict["load"] = args.checkpoint_dir
    cfg_dict["pretrained_checkpoint"] = args.checkpoint_dir
    cfg_dict["ckpt_step"] = args.checkpoint_iteration
    # cfg_dict["auto_detect_ckpt_format"] = True
    cfg_dict["tensor_model_parallel_size"] = 1
    cfg_dict["sequence_parallel"] = False
    if cfg_dict["expert_tensor_parallel_size"] is not None:
        cfg_dict["expert_tensor_parallel_size"] = 1
    cfg_dict["expert_model_parallel_size"] = 1
    cfg_dict["pipeline_model_parallel_size"] = 1
    cfg_dict["context_parallel_size"] = 1
    cfg_dict["recompute_method"] = None
    cfg_dict["recompute_granularity"] = None
    cfg_dict["gradient_accumulation_fusion"] = False
    cfg_dict["data_parallel_sharding_strategy"] = None
    cfg_dict["use_rotary_position_embeddings"] = (
        True if cfg_dict["position_embedding_type"] == "rope" else False
    )
    
    cfg_dict["auto_detect_ckpt_format"] = False
    cfg_dict["checkpoint_format"] = (
        "dist"  # or "megatron_dist" depending on your codebase
    )
    # cfg_dict["use_distributed_optimizer"] = False  # if not resuming optimizer
    # cfg_dict["overlap_param_gather"] = False
    # cfg_dict["overlap_param_gather_with_optimizer_step"] = False
    # cfg_dict["moe_shared_exper_overlap"] = False
    # cfg_dict["tp_comm_overlap"] = False
    # cfg_dict["use_custom_fsdp"] = False
    # take care of badly saved argument values
    for key in cfg_dict:
        try:
            cfg_dict[key] = int(cfg_dict[key])
        except (ValueError, TypeError):
            try:
                cfg_dict[key] = float(cfg_dict[key])
            except (ValueError, TypeError):
                pass

    _ = MegatronConfig(**cfg_dict)  # check if configuration actually works.
    megatron_args = SimpleNamespace(world_size=1, rank=0, **cfg_dict)
    disable_bias_linear = cfg_dict.get("disable_bias_linear")
    add_bias_linear = cfg_dict.get("add_bias_linear")
    if add_bias_linear:
        if disable_bias_linear:
            cfg_dict['add_bias_linear'] = False
            cfg_dict['disable_bias_linear'] = True
        else:
            cfg_dict['add_bias_linear'] = True
            cfg_dict['disable_bias_linear'] = False 
    cfg_dict["tie_word_embeddings"] = not cfg_dict["untie_embeddings_and_output_weights"]

    initialize_megatron(
        allow_no_cuda=True,
        parsed_args=megatron_args,
    )
    if cfg_dict['tokenizer_type'] == "HuggingFaceTokenizer":
        from transformers import AutoTokenizer
        cfg_dict['vocab_size'] = AutoTokenizer.from_pretrained(cfg_dict['tokenizer_model']).vocab_size


    converter = ModelConverter(cfg_dict)
    print(converter)

    megatron_model = converter.create_megatron_module()
    print(megatron_model)

    def reshape_hook(state_dict, prefix, local_metadata):
        for key in list(state_dict.keys()):
            if key in megatron_model.state_dict():
                target_shape = megatron_model.state_dict()[key].shape
                if state_dict[key].shape != target_shape:
                    state_dict[key] = state_dict[key].reshape(target_shape)

    megatron_model._register_load_state_dict_pre_hook(reshape_hook)

    m = megatron_model
    w = m.decoder.layers[0].self_attention.linear_proj.weight
    print("model expects:", tuple(w.shape))

    strict_map = {
        "assume_ok": StrictHandling.ASSUME_OK_UNEXPECTED,  # fastest; ignores unexpected, errors on missing
        "log_unexpected": StrictHandling.LOG_UNEXPECTED,  # logs unexpected
        "return_all": StrictHandling.RETURN_ALL,  # returns (state, missing, unexpected)
    }
    sharded_state_dict = megatron_model.sharded_state_dict(
        prefix=getattr(args, "sd_prefix", "")
    )
    removed = 0
    for k in list(sharded_state_dict.keys()):
        if '._extra_state' in k:
            sharded_state_dict.pop(k)
            removed += 1
    print(f"[conversion] Stripped {removed} '._extra_state' entries (use --keep-extra-state to retain)." )



    strict = strict_map[getattr(args, "strict_mode", "assume_ok")]

    # load_checkpoint([megatron_model], None, None)
    load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=cfg_dict["load"],
        strict=strict,
    )

    megatron_state_dict = megatron_model.state_dict()

    hf_state_dict = converter.convert_weights(megatron_state_dict)

    # del megatron_model
    # del megatron_state_dict

    hf_model = converter.create_hf_module()

    hf_model.load_state_dict(hf_state_dict, strict=True)

    hf_model.save_pretrained(args.output_dir, from_pt=True)
    # add auto_map to args.output_dir/config.json 
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = json.load(f)
    
    config.setdefault("auto_map", {})
    config["auto_map"].update({
        "AutoConfig": "configuration_megatron.MegatronConfig",
        "AutoModel": "modeling_megatron.MegatronPretrainedModel",
        "AutoModelForCausalLM": "modeling_megatron.MegatronForCausalLM",
    })
    config["model_type"] = "megatron"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    
    # Copy configuration files and generate combined modeling file
    print(f"\nCopying model files to {args.output_dir}...")
    
    # Copy configuration_megatron.py and __init__.py
    for file in ["configuration_megatron.py", "__init__.py"]:
        src_path = f"src/megatron2huggingface/{file}"
        dst_path = os.path.join(args.output_dir, file)
        shutil.copy(src=src_path, dst=dst_path)
        print(f"  ✓ Copied {file}")
    
    # Generate the combined modeling_megatron.py file
    print(f"  Generating modeling_megatron.py...")
    modeling_file = generate_modeling_auto(
        output_dir=args.output_dir,
        verbose=False  # Use quiet mode to avoid cluttering the output
    )
    print(f"  ✓ Generated {modeling_file.name}")
    
    print(f"\n{'='*80}")
    print(f"✓ Model conversion complete!")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
