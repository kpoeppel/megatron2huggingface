import argparse
import json
from typing import Any, Type

from megatron.training.arguments import add_megatron_arguments


def get_megatron_parser():
    """
    Extracts the arguments from megatron.training.arguments.py.
    """
    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    parser = add_megatron_arguments(parser)
    return parser


def get_args_and_types(
    parser: argparse.ArgumentParser,
    exclude_args: list[str] | None = None,
    override_defaults: dict[str, Any] | None = None,
) -> dict[str, tuple[Type, Any]]:
    """
    Generates a configuration dictionary from the given arguments.

    Args:
        args: The parsed arguments from argparse.
        exclude_args: A list of argument names to exclude from the configuration.
        override_defaults: A dictionary of argument names and values to override the defaults.

    Returns:
        A dictionary representing the configuration.
    """
    args = parser.parse_args(args=[])

    config = {}
    exclude_args = exclude_args or []
    override_defaults = override_defaults or {}

    arg_types = {
        arg.dest: (bool if isinstance(arg, argparse._StoreTrueAction | argparse._StoreFalseAction) else arg.type)
        for arg in parser._actions
    }

    for arg in vars(args):
        if arg not in exclude_args:
            value = getattr(args, arg)
            if arg in override_defaults:
                value = override_defaults.get(arg, value)

            config[arg] = (arg_types[arg], value)

    return config


def _generate_docstring(args: dict[str, tuple[Type, Any]], offset="        ", suboffset="    "):
    lines = []

    def typestr(argtype: Type):
        st = str(argtype)
        if "<class " in st:
            return "`" + st[8:-2] + "`"
        if "<" in st:
            return "Any"

        return st

    def valstr(val: Any) -> str:
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, bool):
            return f"`{val}`"
        return f"{val}"

    for arg, (argtype, default) in args.items():
        lines.append(
            offset
            + arg
            + " ("
            + typestr(argtype=argtype)
            + ((", *optional*, defaults to " + valstr(default)) if default is not None else "")
            + "):"
        )
        lines.append(offset + suboffset + f"Argument {arg}.")
    return "\n".join(lines)


def _generate_kwargs(args: dict[str, tuple[Type, Any]], offset="        "):
    def typestr(argtype: Type):
        st = str(argtype)
        if "<class " in st:
            return st[8:-2]
        if "<" in st:
            return "Any"
        return st

    lines = []
    for arg, (argtype, default) in args.items():
        lines.append(
            offset
            + arg
            + ": "
            + typestr(argtype)
            + " = "
            + (f'"{default}"' if isinstance(default, str) else str(default))
            + ","
        )
    return "\n".join(lines)


def _generate_assigments(args: dict[str, tuple[Type, Any]], offset="        "):
    def typestr(argtype: Type):
        st = str(argtype)
        if "<class " in st:
            return st[8:-2]
        if "<" in st:
            return "Any"
        return st

    lines = []
    for arg, (argtype, default) in args.items():
        lines.append(offset + "self." + arg + ": " + typestr(argtype) + " = " + arg)
    return "\n".join(lines)


def generate_config(
    args: dict[str, tuple[Type, Any]],
    class_name: str = "MegatronConfig",
    base_class_name: str = "PretrainedConfig",
    template: str = '''"""Megatron model configuration - generated from megatron2hugginface"""

from typing import Any

from ...configuration_utils import {base_class_name}

{main}

__all__ = ["{class_name}"]
''',
) -> str:
    # todo
    main = (
        '''class {class_name}({base_class_name}):
    r"""
    This configures a MegatronModel.

    Args:
'''.replace("{class_name}", class_name).replace("{base_class_name}", base_class_name)
        + _generate_docstring(args)
        + '''
    """
    def __init__(
        self,
'''
        + _generate_kwargs(args)
        + """
        **kwargs,
    ):
"""
        + _generate_assigments(args)
        + """

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )"""
    )
    return (
        template.replace("{base_class_name}", base_class_name)
        .replace("{class_name}", class_name)
        .replace("{main}", main)
    )


def generate_megatron_configuration_huggingface(out_file: str = "configuration_megatron.py"):
    parser = get_megatron_parser()
    args = get_args_and_types(parser, exclude_args={}, override_defaults={})
    args["attention_backend"] = (str, "default")
    args["pad_token_id"] = (int, 0)
    args["bos_token_id"] = (int, 0)
    args["eos_token_id"] = (int, 0)
    args["use_cache"] = (bool, True)
    args["tie_word_embeddings"] = (bool, not args["untie_embeddings_and_output_weights"])
    cfg_file_py = generate_config(args, class_name="MegatronConfig", base_class_name="PretrainedConfig")

    with open(out_file, "w") as fp:
        fp.write(cfg_file_py)
