from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SentenceEncoder(ABC, nn.Module):
    """Abstract base class for all sentence encoders.

    Defines the two-method interface that every encoder must implement:

    * :meth:`tokenize` — converts a raw input item to model inputs.
    * :meth:`forward`  — produces the final tensor representation.
    """

    @abstractmethod
    def tokenize(self, item):
        """Convert an input item to model inputs."""
        ...

    @abstractmethod
    def forward(self, item) -> torch.Tensor:
        """Encode an item and return a tensor representation."""
        ...
