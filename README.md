# Megatron2HuggingFace 

A utility to convert Megatron-LM models and their checkpoints to a HuggingFace-style model.

It automatically uses all megatron arguments for the HuggingFace model configuration (you can choose to exclude certain arguments).
It aims to do a modular conversion, aligning the Megatron modules with HuggingFace modules and do a module-wise conversion mostly. This way sub-component conversion can be tested separately.

This repo is curerntly in WIP state and a first test script is provided with `test_runner.py`, use as:
```bash
PYTHONPATH="src:Megatron-LM:transformers/src" python test_runner.py
```

with the dependencies of `Megatron-LM` and `transformers` all installed in your env.
In the future, for new versions of either of these repos, tools should be provided here to do adapt the conversion fast for new features. Also by checking out different forks this should easily adapt the behavior.


