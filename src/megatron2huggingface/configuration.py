import argparse
from typing import Any

from megatron.training.arguments import add_megatron_arguments


def get_megatron_parser():
    """Extracts the arguments from megatron.training.arguments.py."""
    parser = argparse.ArgumentParser(
        description="Megatron-LM Arguments", allow_abbrev=False
    )
    parser = add_megatron_arguments(parser)
    return parser


def _extract_action_type(action: argparse.Action):
    typ = (
        bool
        if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction)
        else action.type
    )
    if action.nargs == "+" or action.nargs == "*":
        if typ is None:
            return list[str]
        else:
            return list[typ]
    else:
        return typ


def get_choices_arg(
    parser: argparse.ArgumentParser,
    arg: str,
):
    for action in parser._actions:
        if action.dest == arg:
            if action.choices:
                return action.choices
    return None


def get_help(
    parser: argparse.ArgumentParser,
    arg: str,
):
    for action in parser._actions:
        if action.dest == arg:
            if action.help:
                return action.help
    return ""


def get_args_and_types(
    parser: argparse.ArgumentParser,
    exclude_args: list[str] | None = None,
    override_defaults: dict[str, Any] | None = None,
) -> dict[str, tuple[type, Any]]:
    """Generates a configuration dictionary from the given arguments.

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
        action.dest: _extract_action_type(action) for action in parser._actions
    }

    for arg in vars(args):
        if arg not in exclude_args:
            value = getattr(args, arg)
            if arg in override_defaults:
                value = override_defaults.get(arg, value)

            config[arg] = (arg_types[arg], value)

    return config


def split_line(offset, line, max_line=120):
    lines = []
    while len(offset + line) > max_line:
        line_sub = line.split(" ")
        cum_length = len(offset + line_sub[0])
        idx = 0
        while cum_length < max_line or idx - 1 > len(line_sub):
            cum_length += 1 + len(line_sub[idx + 1])
            idx += 1
        lines += [offset + " ".join(line_sub[: idx - 1])]
        line = " ".join(line_sub[idx - 1 :])
    return lines + [offset + line]


def _generate_docstring(
    args: dict[str, tuple[type, Any]],
    args_help: dict[str, str] = {},
    offset="        ",
    suboffset="    ",
):
    lines = []

    def typestr(argtype: type):
        st = str(argtype)
        if "<class " in st:
            st = st[8:-2]
        if "<" in st:
            st = "Any"
        if st == "None":
            st = "str"
        return "`" + st + "`"

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
            + (
                (", *optional*, defaults to " + valstr(default))
                if default is not None
                else ""
            )
            + "):"
        )
        lines += split_line(
            offset + suboffset,
            f"Argument {arg}."
            + ((" " + args_help[arg]) if arg in args_help and args_help[arg] else ""),
        )
    return "\n".join(lines)


def _generate_kwargs(args: dict[str, tuple[type, Any]], offset="        "):
    def typestr(argtype: type):
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


def _generate_assigments(args: dict[str, tuple[type, Any]], offset="        "):
    def typestr(argtype: type):
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
    args: dict[str, tuple[type, Any]],
    parser: argparse.ArgumentParser,
    class_name: str = "MegatronConfig",
    base_class_name: str = "PretrainedConfig",
    template: str = '''"""Megatron model configuration - generated from megatron2huggingface"""

from typing import Any

from transformers.configuration_utils import {base_class_name}

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
        + _generate_docstring(
            args, args_help={arg: get_help(arg=arg, parser=parser) for arg in args}
        )
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


def generate_megatron_configuration_huggingface(
    out_file: str = "configuration_megatron.py",
):
    parser = get_megatron_parser()
    args = get_args_and_types(parser, exclude_args={}, override_defaults={})
    args["attention_backend"] = (str, "default")
    args["pad_token_id"] = (int, 0)
    args["bos_token_id"] = (int, 0)
    args["eos_token_id"] = (int, 0)
    args["use_cache"] = (bool, True)
    args["tie_word_embeddings"] = (
        bool,
        not args["untie_embeddings_and_output_weights"],
    )
    cfg_file_py = generate_config(
        args,
        parser=parser,
        class_name="MegatronConfig",
        base_class_name="PretrainedConfig",
    )

    with open(out_file, "w") as fp:
        fp.write(cfg_file_py)
