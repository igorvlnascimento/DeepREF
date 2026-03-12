# ---------------------------------------------------------------------------
# Single-encoder wrapper
# ---------------------------------------------------------------------------

import logging
from types import SimpleNamespace

from torch import nn
import torch

from deepref.encoder.combine_embeddings import CombineEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

class SingleEncoderWrapper(nn.Module):
    """Thin wrapper that satisfies the :class:`SoftmaxMLP` interface for one encoder.

    Provides ``self.model.config.hidden_size`` (used by :class:`MLP` to
    compute layer widths) and a ``forward(items)`` method that dispatches via
    :meth:`CombineEmbeddings._encode_single`, so any encoder type supported by
    :class:`CombineEmbeddings` works here without duplication.

    Used automatically when ``encoder2.type == "none"`` (single-encoder mode).

    Args:
        encoder: the single encoder to wrap.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = CombineEmbeddings._get_hidden_size(encoder)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden))
        logger.info("SingleEncoderWrapper — hidden_size: %d", hidden)

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode a batch and return a ``(B, H)`` float32 tensor."""
        return CombineEmbeddings._encode_batch(self.encoder, items)
