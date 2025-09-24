import pytest
from types import SimpleNamespace
from megatron2huggingface.configuration_megatron import MegatronConfig
import torch.distributed as dist
import os


def _megatron_config_filled():
    return MegatronConfig(
        **{
            "hidden_size": 128,
            "num_attention_heads": 2,
            "kv_channels": 64,
            "num_query_groups": 1,
            "max_position_embeddings": 128,
            "train_iters": 1,
            "lr_decay_style": "exponential",
            "lr": 1e-3,
            "add_bias_linear": True,
            "add_qkv_bias": False,
            "seq_length": 128,
            "attention_dropout": 0.1,
            "num_layers": 2,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "vocab_size": 256,
            "tokenizer_model": "EleutherAI/gpt-neox-20b",
            "tokenizer_type": "HuggingFaceTokenizer",
            "data_parallel_size": 1,
            "group_query_attention": True,
        }
    )


@pytest.fixture
def megatron_config_filled():
    _megatron_config_filled()


def _megatron_config_filled_dict():
    return {
        k: v
        for k, v in vars(_megatron_config_filled()).items()
        if not k.startswith("_")
    }


@pytest.fixture
def megatron_config_filled_dict():
    return _megatron_config_filled_dict()


@pytest.fixture(scope="session", autouse=True)
def initialize_megatron_ahead_of_tests(request):
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65432"
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        except RuntimeError as e:
            print(
                f"Could not initialize process group: {e}. Assuming already initialized or not needed for this test."
            )

    from megatron.training.initialize import initialize_megatron

    initialize_megatron(
        allow_no_cuda=True,
        parsed_args=SimpleNamespace(
            world_size=1, rank=0, **_megatron_config_filled_dict()
        ),
    )
