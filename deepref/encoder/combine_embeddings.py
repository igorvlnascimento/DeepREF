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

    @staticmethod
    def _encode_batch(encoder: nn.Module, items: list[dict]) -> torch.Tensor:
        """Encode a full batch in a single forward pass and return ``(B, H)``.

        Tokenises all items, stacks the resulting tensors, and runs the
        encoder once — amortising the GPU kernel-launch overhead that would
        otherwise accumulate when calling :meth:`_encode_single` per item.

        .. note::
            :class:`RelationEncoder` and :class:`VerbalizedSDPEncoder` are
            checked *before* their parent classes for the same reason as in
            :meth:`_encode_single`.
        """
        if isinstance(encoder, RelationEncoder):
            tokenizations = [encoder.tokenize(item) for item in items]
            tokens    = torch.cat([t[0] for t in tokenizations], dim=0)
            att_masks = torch.cat([t[1] for t in tokenizations], dim=0)
            pos_e1    = torch.cat([t[2] for t in tokenizations], dim=0)
            pos_e2    = torch.cat([t[3] for t in tokenizations], dim=0)
            pos_mask  = torch.cat([t[4] for t in tokenizations], dim=0)
            return encoder.forward(tokens, att_masks, pos_e1, pos_e2, pos_mask).float()  # (B, 3H)

        if isinstance(encoder, BertEntityEncoder):
            tokenizations = [encoder.tokenize(item) for item in items]
            tokens    = torch.cat([t[0] for t in tokenizations], dim=0)
            att_masks = torch.cat([t[1] for t in tokenizations], dim=0)
            pos_e1    = torch.cat([t[2] for t in tokenizations], dim=0)
            pos_e2    = torch.cat([t[3] for t in tokenizations], dim=0)
            return encoder.forward(tokens, att_masks, pos_e1, pos_e2).float()  # (B, 2H)

        if isinstance(encoder, VerbalizedSDPEncoder):
            # NLP parsing (spaCy/Stanza) is inherently sequential.  Build the
            # verbalized strings one by one, then batch-tokenize and run the
            # transformer once for the whole batch.
            verbalized = []
            for item in items:
                try:
                    verbalized_item = encoder.verbalize(encoder.mark_sentence(item))
                except RuntimeError:
                    print("Error verbalizing sentence:", item)
                    continue
                verbalized.append(verbalized_item)

            # verbalized = [
            #     encoder.verbalize(encoder.mark_sentence(item), K=1)
            #     for item in items
            # ]
            token_dict = encoder.registry.tokenize(
                encoder.model_name, verbalized,
                max_length=encoder.max_length, padding=True, truncation=True,
            )
            return LLMEncoder.forward(
                encoder, token_dict["input_ids"], token_dict["attention_mask"]
            ).float()  # (B, H)

        if isinstance(encoder, BoWSDPEncoder):
            # No transformer; stacking multi-hot vectors is negligible cost.
            return torch.stack([encoder.forward(item) for item in items], dim=0).float()  # (B, V)

        if isinstance(encoder, LLMEncoder):
            # tokenize_batch pads to the batch's actual max length, not to
            # encoder.max_length — crucial when max_length is large (e.g. 8192)
            # and sentences are short (e.g. 100–300 tokens).
            tokens, att_masks = encoder.tokenize_batch(items)
            return encoder.forward(tokens, att_masks).float()  # (B, H)

        raise ValueError(f"Unsupported encoder type: {type(encoder)}")

    # ------------------------------------------------------------------
    # Forward — called as combine_emb(items=batch_list) by SoftmaxMLP
    # ------------------------------------------------------------------

    def forward(self, items: list[dict]) -> torch.Tensor:
        """Encode a batch of items and return concatenated embeddings.

        Each encoder is called **once** for the entire batch rather than
        once per item, so GPU utilisation scales with ``batch_size`` instead
        of being fixed at 1.

        Args:
            items: list of item dicts with ``'token'``, ``'h'``, and ``'t'``
                keys (as returned by :class:`CombineREDataset`).

        Returns:
            Float32 tensor of shape ``(B, H1 + H2)``.
        """
        embs1 = self._encode_batch(self.encoder1, items)                   # (B, H1)
        embs2 = self._encode_batch(self.encoder2, items).to(embs1.device)  # (B, H2)
        return torch.cat([embs1, embs2], dim=1)                            # (B, H1+H2)