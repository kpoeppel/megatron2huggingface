"""Base utilities for modular checkpoint conversion between Megatron-LM and
HuggingFace.

This module provides the foundation for modular conversion, including
distributed checkpoint loading and a conversion registry system.
"""

import subprocess
from typing import Any, TypeVar
from collections.abc import Callable
from pathlib import Path
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Type definitions
MegatronModule = TypeVar("MegatronModule")
HuggingFaceModule = TypeVar("HuggingFaceModule")


def extract_submodule_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Extract a submodule's state dict by removing the specified prefix from
    matching keys.

    Args:
        state_dict: Full state dictionary
        prefix: Prefix to remove (e.g., "layers.0.self_attention")

    Returns:
        Dictionary with prefix removed from matching keys
    """
    submodule_dict = {}
    prefix_with_dot = prefix + "." if not prefix.endswith(".") else prefix

    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            # Remove the prefix to get the local key
            local_key = key[len(prefix_with_dot) :]
            submodule_dict[local_key] = value
        elif key == prefix:  # Exact match (for cases without dot)
            submodule_dict["weight"] = value  # Common case for single parameter modules

    return submodule_dict


def add_prefix_to_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Add a prefix to all keys in a state dictionary.

    Args:
        state_dict: State dictionary to add prefix to
        prefix: Prefix to add (e.g., "layers.0.self_attention")

    Returns:
        Dictionary with prefix added to all keys
    """
    if not prefix:
        return state_dict

    prefixed_dict = {}
    if prefix:
        prefix_with_dot = prefix + "." if not prefix.endswith(".") else prefix
    else:
        return prefixed_dict

    for key, value in state_dict.items():
        if key == "":  # Handle empty key case
            prefixed_dict[prefix] = value
        else:
            prefixed_dict[prefix_with_dot + key] = value

    return prefixed_dict


def get_git_commit_hash(repo_path: str) -> str | None:
    """Get the current git commit hash for a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(f"Could not get git commit hash for {repo_path}")
        return None


class ConversionRegistry:
    """Registry for component-specific conversion functions."""

    _converters: dict[str, Callable] = {}

    @classmethod
    def register(cls, component_name: str):
        """Decorator to register a converter for a specific component."""

        def decorator(converter_class):
            cls._converters[component_name] = converter_class
            return converter_class

        return decorator

    @classmethod
    def get_converter(cls, component_name: str) -> Callable | None:
        """Get a registered converter by name."""
        return cls._converters.get(component_name)

    @classmethod
    def list_converters(cls) -> list[str]:
        """List all registered converter names."""
        return list(cls._converters.keys())


class MegatronCheckpointLoader:
    """Handles loading of distributed Megatron-LM checkpoints."""

    def __init__(self, checkpoint_path: str, megatron_path: str):
        """Initialize the checkpoint loader.

        Args:
            checkpoint_path: Path to the Megatron checkpoint directory
            megatron_path: Path to Megatron-LM repository
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.megatron_path = Path(megatron_path)

        # Add Megatron to path for imports
        import sys

        if str(self.megatron_path) not in sys.path:
            sys.path.insert(0, str(self.megatron_path))

    def _find_checkpoint_files(self) -> list[Path]:
        """Find all checkpoint files in the directory."""
        checkpoint_files = []

        # Look for latest iteration file
        latest_file = self.checkpoint_path / "latest_checkpointed_iteration.txt"
        if latest_file.exists():
            with open(latest_file) as f:
                iteration = f.read().strip()

            # Find all model files for this iteration
            iter_dir = self.checkpoint_path / f"iter_{iteration:0>7d}"
            if iter_dir.exists():
                checkpoint_files.extend(iter_dir.glob("**/model_optim_rng.pt"))

        # Fallback: look for any model files
        if not checkpoint_files:
            checkpoint_files.extend(self.checkpoint_path.glob("**/model_optim_rng.pt"))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.checkpoint_path}"
            )

        return sorted(checkpoint_files)

    def load_distributed_checkpoint(self) -> dict[str, Any]:
        """Load a distributed Megatron checkpoint using Megatron's utilities.

        Returns:
            Dictionary containing the merged checkpoint data
        """
        try:
            # Try to use Megatron's distributed checkpoint utilities
            # from megatron.core import dist_checkpointing
            # from megatron.training.checkpointing import load_checkpoint

            # This is a simplified approach - in practice, you might need to
            # initialize Megatron's distributed environment properly
            logger.info(f"Loading distributed checkpoint from {self.checkpoint_path}")

            # For now, we'll load individual files and merge them
            checkpoint_files = self._find_checkpoint_files()
            merged_state = {}

            for i, ckpt_file in enumerate(checkpoint_files):
                logger.info(
                    f"Loading checkpoint file {i + 1}/{len(checkpoint_files)}: {ckpt_file}"
                )
                ckpt = torch.load(ckpt_file, map_location="cpu")

                if i == 0:
                    # First file contains the base structure
                    merged_state = ckpt.copy()
                else:
                    # Merge model state from other files
                    if "model" in ckpt:
                        self._merge_model_state(
                            merged_state.get("model", {}), ckpt["model"]
                        )

            return merged_state

        except ImportError:
            logger.warning(
                "Could not import Megatron distributed checkpoint utilities. "
                "Falling back to simple torch.load approach."
            )
            return self._load_simple_checkpoint()

    def _load_simple_checkpoint(self) -> dict[str, Any]:
        """Simple checkpoint loading for non-distributed checkpoints."""
        checkpoint_files = self._find_checkpoint_files()

        if len(checkpoint_files) == 1:
            # Single file checkpoint
            logger.info(f"Loading single checkpoint file: {checkpoint_files[0]}")
            return torch.load(checkpoint_files[0], map_location="cpu")
        else:
            # Multiple files - try to merge them
            logger.info(f"Loading and merging {len(checkpoint_files)} checkpoint files")
            merged_state = {}

            for i, ckpt_file in enumerate(checkpoint_files):
                ckpt = torch.load(ckpt_file, map_location="cpu")

                if i == 0:
                    merged_state = ckpt.copy()
                else:
                    if "model" in ckpt:
                        self._merge_model_state(
                            merged_state.get("model", {}), ckpt["model"]
                        )

            return merged_state

    def _merge_model_state(
        self, base_state: dict[str, torch.Tensor], new_state: dict[str, torch.Tensor]
    ):
        """Merge model state dictionaries, handling tensor parallelism."""
        for key, tensor in new_state.items():
            if key in base_state:
                # Try to concatenate tensors (for tensor parallel weights)
                base_tensor = base_state[key]
                if base_tensor.shape != tensor.shape:
                    # Different shapes - likely tensor parallel, try to concatenate
                    try:
                        # Try concatenating along different dimensions
                        for dim in range(len(base_tensor.shape)):
                            try:
                                concatenated = torch.cat([base_tensor, tensor], dim=dim)
                                base_state[key] = concatenated
                                logger.debug(
                                    f"Concatenated {key} along dimension {dim}"
                                )
                                break
                            except RuntimeError:
                                continue
                        else:
                            logger.warning(
                                f"Could not concatenate tensor {key} with shapes {base_tensor.shape} and {tensor.shape}"
                            )
                    except Exception as e:
                        logger.warning(f"Error merging tensor {key}: {e}")
                else:
                    # Same shape - might be duplicated, keep the first one
                    logger.debug(f"Keeping existing tensor {key} (same shape)")
            else:
                # New tensor
                base_state[key] = tensor

    def get_model_config(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Extract model configuration from checkpoint."""
        if "args" in checkpoint:
            return (
                vars(checkpoint["args"])
                if hasattr(checkpoint["args"], "__dict__")
                else checkpoint["args"]
            )
        elif "hyper_parameters" in checkpoint:
            return checkpoint["hyper_parameters"]
        else:
            logger.warning("Could not find model configuration in checkpoint")
            return {}


class BaseConverter:
    """Base class for component converters."""

    def __init__(self, megatron_config: dict[str, Any]):
        """Initialize the converter with Megatron configuration.

        Args:
            megatron_config: Dictionary containing Megatron model configuration
        """
        self.megatron_config = megatron_config

    def convert_weights(
        self, megatron_weights: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert weights from Megatron format to HuggingFace format.

        Args:
            megatron_weights: Megatron model weights dictionary
            config: Model configuration
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary of converted weights in HuggingFace format
        """
        raise NotImplementedError("Subclasses must implement convert_weights")

    def create_hf_module(self, config, **kwargs):
        """Create a HuggingFace module instance.

        Args:
            config: HuggingFace configuration object
            **kwargs: Additional module parameters

        Returns:
            HuggingFace module instance
        """
        raise NotImplementedError("Subclasses must implement create_hf_module")

    def create_megatron_module(self, **kwargs):
        """Create a Megatron module instance for comparison.

        Args:
            **kwargs: Module parameters

        Returns:
            Megatron module instance
        """
        raise NotImplementedError("Subclasses must implement create_megatron_module")

    def test_conversion(
        self,
        megatron_state: dict[str, torch.Tensor],
        hf_config,
        test_input: torch.Tensor,
        additional_inputs: dict[str, Any] = {},
        assert_error: bool = True,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        permute_megatron_output: list[int] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Test the conversion by comparing outputs of Megatron and HuggingFace
        modules.

        Args:
            megatron_state: Megatron model state dictionary
            hf_config: HuggingFace configuration object
            test_input: Test input tensor
            **kwargs: Additional parameters

        Returns:
            Dictionary containing test results and metrics
        """
        # Convert weights
        hf_weights = self.convert_weights(megatron_state, **kwargs)

        # Create modules
        hf_module = self.create_hf_module(hf_config, **kwargs).to(
            device=test_input.device
        )

        megatron_module = self.create_megatron_module(**kwargs).to(
            device=test_input.device
        )

        # Load weights
        hf_module.load_state_dict(hf_weights, strict=False)

        # Extract and load Megatron weights
        megatron_module.load_state_dict(megatron_state, strict=False)

        # Set to eval mode
        hf_module.eval()
        megatron_module.eval()

        # Forward pass
        with torch.no_grad():
            hf_output = hf_module(test_input, **additional_inputs)
            megatron_output = megatron_module(test_input, **additional_inputs)

        # Compare outputs
        if isinstance(hf_output, tuple):
            hf_output = hf_output[0]
        if isinstance(megatron_output, tuple):
            megatron_output = megatron_output[0]

        if permute_megatron_output:
            megatron_output = megatron_output.permute(*permute_megatron_output)

        # Calculate metrics
        mse = torch.mean((hf_output - megatron_output) ** 2).item()
        max_diff = torch.max(torch.abs(hf_output - megatron_output)).item()
        relative_error = (
            torch.norm(hf_output - megatron_output) / torch.norm(megatron_output)
        ).item()

        if assert_error:
            np.testing.assert_allclose(
                hf_output.detach().cpu().numpy(),
                megatron_output.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )

        return {
            "mse": mse,
            "max_diff": max_diff,
            "relative_error": relative_error,
            "hf_output_shape": hf_output.shape,
            "megatron_output_shape": megatron_output.shape,
            "test_passed": np.allclose(
                hf_output.detach().cpu().numpy(),
                megatron_output.detach().cpu().numpy(),
                atol=atol,
                rtol=rtol,
            ),
        }


def load_megatron_checkpoint(
    checkpoint_path: str, megatron_path: str
) -> dict[str, Any]:
    """Convenience function to load a Megatron checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        megatron_path: Path to Megatron-LM repository

    Returns:
        Loaded checkpoint dictionary
    """
    loader = MegatronCheckpointLoader(checkpoint_path, megatron_path)
    return loader.load_distributed_checkpoint()
