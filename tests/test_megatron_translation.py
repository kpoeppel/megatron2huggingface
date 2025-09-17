from megatron2huggingface.configuration import (
    get_megatron_parser,
    get_args_and_types,
    _generate_docstring,
    _generate_kwargs,
    _generate_assigments,
    generate_config,
    generate_megatron_configuration_huggingface,
)


def test_get_megatron_parser():
    parser = get_megatron_parser()
    args = parser.parse_args(args=[])
    assert args is not None
    # Add more specific assertions here to check for expected default values
    assert args.num_query_groups == 1


def test_get_args_and_types():
    parser = get_megatron_parser()
    config = get_args_and_types(parser)
    assert "num_query_groups" in config
    assert config["num_query_groups"] == (int, 1)


def test_generate_docstring():
    res = _generate_docstring(
        {"foo": (int, 1), "bar": (int, None), "name": (str, "name")}
    )
    assert (
        res.strip("\n")
        == """
        foo (`int`, *optional*, defaults to 1):
            Argument foo.
        bar (`int`):
            Argument bar.
        name (`str`, *optional*, defaults to "name"):
            Argument name.
""".strip("\n")
    )


def test_generate_kwargs():
    res = _generate_kwargs({"foo": (int, 1), "bar": (int, None)})
    assert (
        res.strip("\n")
        == """
        foo: int = 1,
        bar: int = None,
""".strip("\n")
    )


def test_generate_assignments():
    res = _generate_assigments({"foo": (int, 1), "bar": (int, None)})
    assert (
        res.strip("\n")
        == """
        self.foo: int = foo
        self.bar: int = bar
""".strip("\n")
    )


def test_generate_config():
    config = {"foo": (int, 1), "bar": (int, None)}

    res = generate_config(config, parser=get_megatron_parser(), class_name="TestConfig")

    assert (
        res.strip("\n")
        == '''
"""Megatron model configuration - generated from megatron2huggingface"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig

class TestConfig(PretrainedConfig):
    r"""
    This configures a MegatronModel.

    Args:
        foo (`int`, *optional*, defaults to 1):
            Argument foo.
        bar (`int`):
            Argument bar.
    """
    def __init__(
        self,
        foo: int = 1,
        bar: int = None,
        **kwargs,
    ):
        self.foo: int = foo
        self.bar: int = bar

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["TestConfig"]
'''.strip("\n")
    )


def test_generate_megatron_configuration_huggingface():
    generate_megatron_configuration_huggingface("exmp_out/configuration_megatron.py")
