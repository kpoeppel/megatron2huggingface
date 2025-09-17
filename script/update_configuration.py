from megatron2huggingface.configuration import (
    generate_megatron_configuration_huggingface,
)
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration-file",
        default="src/megatron2huggingface/configuration_megatron.py",
    )

    args = parser.parse_args()

    generate_megatron_configuration_huggingface(args.configuration_file)


if __name__ == "__main__":
    main()
