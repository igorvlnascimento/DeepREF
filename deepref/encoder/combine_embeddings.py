# ---------------------------------------------------------------------------
# Combined encoder
# ---------------------------------------------------------------------------

import logging
from types import SimpleNamespace

from torch import nn
import torch

from deepref.encoder.bert_entity_encoder import BertEntityEncoder
from deepref.encoder.llm_encoder import LLMEncoder
from deepref.encoder.relation_encoder import RelationEncoder
from deepref.encoder.sdp_encoder import BoWSDPEncoder, VerbalizedSDPEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class CombineEmbeddings(nn.Module):
    """Concatenate embeddings from two independent encoders.

    Supports any combination of:
    * :class:`~deepref.encoder.relation_encoder.RelationEncoder`
    * :class:`~deepref.encoder.llm_encoder.LLMEncoder`
    * :class:`~deepref.encoder.sdp_encoder.BoWSDPEncoder`
    * :class:`~deepref.encoder.sdp_encoder.VerbalizedSDPEncoder`

    The combined output dimension equals ``hidden_size(encoder1) +
    hidden_size(encoder2)`` and is exposed via ``self.model.config.hidden_size``
    so that :class:`~deepref.model.softmax_mlp.SoftmaxMLP` (and the underlying
    :class:`~deepref.module.nn.mlp.MLP`) can consume this encoder without
    modification.

    Args:
        encoder1: first encoder instance.
        encoder2: second encoder instance.
    """

    def __init__(self, encoder1: nn.Module, encoder2: nn.Module) -> None:
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        h1 = self._get_hidden_size(encoder1)
        h2 = self._get_hidden_size(encoder2)
        combined = h1 + h2

        # Expose model.config.hidden_size for SoftmaxMLP / MLP compatibility.
        self.model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=combined)
        )
        logger.info(
            "CombineEmbeddings — encoder1: %d  encoder2: %d  combined: %d",
            h1, h2, combined,
        )

    # ------------------------------------------------------------------
    # Hidden-size helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hidden_size(encoder: nn.Module) -> int:
        """Return the output dimensionality of *encoder* for a single sample."""
        # BertEntityEncoder (and its subclass RelationEncoder) expose self.hidden_size
        if isinstance(encoder, BertEntityEncoder):
            return encoder.hidden_size
        # BoWSDPEncoder has no neural model; output = dep_vocab length
        if isinstance(encoder, BoWSDPEncoder):
            return len(encoder.dep_vocab)
        # LLMEncoder (and VerbalizedSDPEncoder which inherits it)
        if isinstance(encoder, LLMEncoder):
            return encoder.registry.get_model_hidden_size(encoder.model_name)
        raise ValueError(
            f"Cannot determine hidden size for encoder type: {type(encoder)}"
        )

    # ------------------------------------------------------------------
    # Per-encoder encode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_single(encoder: nn.Module, item: dict) -> torch.Tensor:
        """Encode one sample and return a 1-D float32 embedding tensor.

        Each encoder has a different forward signature; this method dispatches
        appropriately.  Return shape is always ``(H,)`` so outputs can be
        concatenated with ``torch.cat``.

        .. note::
            :class:`RelationEncoder` and :class:`VerbalizedSDPEncoder` are
            checked *before* their parent :class:`LLMEncoder` because both
            are subclasses of it.
        """
        if isinstance(encoder, RelationEncoder):
            token, att_mask, pos_e1, pos_e2, pos_mask = encoder.tokenize(item)
            emb = encoder.forward(token, att_mask, pos_e1, pos_e2, pos_mask)
            return emb.squeeze(0).float()  # (3H,)

        if isinstance(encoder, BertEntityEncoder):
            token, att_mask, pos_e1, pos_e2 = encoder.tokenize(item)
            emb = encoder.forward(token, att_mask, pos_e1, pos_e2)
            return emb.squeeze(0).float()  # (2H,)

        if isinstance(encoder, VerbalizedSDPEncoder):
            emb = encoder.forward(item)    # (1, H)
            return emb.squeeze(0).float()  # (H,)

        if isinstance(encoder, BoWSDPEncoder):
            return encoder.forward(item).float()  # (len_dep_vocab,)

        if isinstance(encoder, LLMEncoder):
            token_ids, attention_mask = encoder.tokenize(item)
            emb = encoder.forward(token_ids, attention_mask)    # (1, H)
            return emb.squeeze(0).float()  # (H,)

        raise ValueError(f"Unsupported encoder type: {type(encoder)}")

    # ------------------------------------------------------------------
    # Forward — called as combine_emb(items=batch_list) by SoftmaxMLP
    # ------------------------------------------------------------------

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode a batch of items and return concatenated embeddings.

        Args:
            items: list of item dicts with ``'token'``, ``'h'``, and ``'t'``
                keys (as returned by :class:`CombineREDataset`).

        Returns:
            Float32 tensor of shape ``(B, H1 + H2)``.
        """
        batch_embs: list[torch.Tensor] = []
        for item in items:
            emb1 = self._encode_single(self.encoder1, item)
            emb2 = self._encode_single(self.encoder2, item).to(emb1.device)
            batch_embs.append(torch.cat([emb1, emb2], dim=0))
        return torch.stack(batch_embs, dim=0)  # (B, H1+H2)