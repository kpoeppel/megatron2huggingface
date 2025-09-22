# Megatron2HuggingFace

A utility to convert Megatron-LM models and their checkpoints to a HuggingFace-style model.

It automatically uses all megatron arguments for the HuggingFace model configuration (you can choose to exclude certain arguments).
It aims to do a modular conversion, aligning the Megatron modules with HuggingFace modules and do a module-wise conversion mostly. This way sub-component conversion can be tested separately.

This repository is curerntly in alpha state and conversion currently works with the following script:
```bash
PYTHONPATH="src:Megatron-LM:transformers/src" python script/model_convert.py --config-file CONFIG_FILE --checkpoint-dir CHECKPOINT_DIR --output-dir OUTPUT_DIR [ --checkpoint-iteration CHECKPOINT_ITERATION ]
```

The checkpoint directory is the one that contains checkpoints of different iterations, and if the iteration is not specified the latest one is taken. The config file should contain a list of all megatron arguments in yaml format, potentially deeper into the configuration specified by `--config-subset`.

## TODOs

- [x] fully test with dense Transformer model checkpoint (including automated tests)
- [ ] include MoE capabilities
- [ ] include MLA capabilities
- [ ] include other features of Megatron-LM


with the dependencies of `Megatron-LM` and `transformers` all installed in your env.
In the future, for new versions of either of these repositories, tools should be provided here to do adapt the conversion fast for new features. Also by checking out different forks this should easily adapt the behavior. You can already convert a new Megatron argument list to a new huggingface configuration_megatron.py by way of `script/update_configuration.py`.
