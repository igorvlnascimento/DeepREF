from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from deepref.utils.model_registry import ModelRegistry


class SentenceEncoder(ABC, nn.Module):
    """Abstract base class for all sentence encoders.

    Defines the two-method interface that every encoder must implement:

    * :meth:`tokenize` — converts a raw input item to model inputs.
    * :meth:`forward`  — produces the final tensor representation.
    """
    def __init__(self,
                 model_name,
                 max_length=512,
                 device="cpu",
                 attn_implementation="eager",
                 trainable=False):
        super().__init__()
        self.registry = ModelRegistry()
        self.registry.load(model_name,
                           device=device,
                           trainable=trainable,
                           attn_implementation=attn_implementation)

        self.model_name = model_name
        self.max_length = max_length

    @abstractmethod
    def tokenize(self, item):
        """Convert an input item to model inputs."""
        ...

    @abstractmethod
    def forward(self, item) -> torch.Tensor:
        """Encode an item and return a tensor representation."""
        ...
