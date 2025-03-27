from typing import Dict, Optional, Tuple, Union

import torch.nn as nn

from src.config import ModelConfig
from src.models.efficientnet import create_model as create_efficientnet
from src.models.efficientnet_attention import (
    create_model as create_efficientnet_attention,
)


class ModelFactory:
    """Factory class for creating different model architectures."""

    # Dictionary mapping model names to their creation functions and default configs
    MODEL_CONFIGS: Dict[str, Dict] = {
        "efficientnet": {
            "create_fn": create_efficientnet,
            "default_config": {
                "num_classes": None,  # Will be set during creation
                "efficientnet_version": "efficientnet-b0",
            },
        },
        "efficientnet_attention": {
            "create_fn": create_efficientnet_attention,
            "default_config": {
                "num_classes": None,  # Will be set during creation
                "efficientnet_version": "efficientnet-b0",
                "kernel_size": (5, 5),
                "cfar_scaling_factors": (5, 20),
            },
        },
    }

    @classmethod
    def create_model(
        cls,
        model_config: ModelConfig,
        num_classes: int,
    ) -> nn.Module:
        """
        Create a model instance based on the model configuration.

        Args:
            model_config: ModelConfig object containing model name and parameters
            num_classes: Number of output classes

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model_name is not found in MODEL_CONFIGS
        """
        if model_config.name not in cls.MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_config.name}' not found. Available models: {list(cls.MODEL_CONFIGS.keys())}"
            )

        # Get the model configuration
        model_config_dict = cls.MODEL_CONFIGS[model_config.name]
        create_fn = model_config_dict["create_fn"]
        default_config = model_config_dict["default_config"].copy()

        # Update default config with provided config
        default_config.update(model_config.params)

        # Set number of classes
        default_config["num_classes"] = num_classes

        # Create and return the model
        return create_fn(**default_config)
