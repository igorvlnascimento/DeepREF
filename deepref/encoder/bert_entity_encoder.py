import logging
import torch

from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.utils.model_registry import ModelRegistry


class BertEntityEncoder(SentenceEncoder):
    """Entity-span encoder backed by a HuggingFace transformer.

    Extends :class:`LLMEncoder` by locating the ``[E1]`` and ``<e2>`` entity
    marker positions in the token sequence and returning the concatenation of
    their hidden states as the sentence representation.

    The output dimension is ``2 * hidden_size`` of the underlying model and is
    exposed via ``self.hidden_size`` for downstream compatibility.

    Args:
        model_name: HuggingFace model name or local path.
        max_length: maximum tokenised sequence length.
        blank_padding: pad / truncate sequences to ``max_length``.
        device: PyTorch device string.
        attn_implementation: attention backend passed to ``AutoModel``.
        trainable: if ``False`` (default) all model parameters are frozen.
    """

    def __init__(
        self,
        model_name,
        max_length=512,
        blank_padding=True,
        has_cls_embedding=False,
        device="cpu",
        attn_implementation="eager",
        trainable=False,
    ):
        logging.info("Loading %s pre-trained checkpoint.", model_name)
        super().__init__(
            model_name,
            max_length=max_length,
            device=device,
            attn_implementation=attn_implementation,
            trainable=trainable,
        )
        self.blank_padding = blank_padding
        self.has_cls_embedding = has_cls_embedding
        # Output: [CLS]+e1+e2 when has_cls_embedding else e1+e2
        self.hidden_size = self.registry.get_model_hidden_size(model_name) * (3 if has_cls_embedding else 2)

    # ------------------------------------------------------------------
    # Shared tokenisation helpers
    # ------------------------------------------------------------------

    def _build_marked_tokens(self, item: dict) -> tuple[list[str], int, int]:
        """Build the entity-marked token list for the sentence content.

        Produces the sequence::

            sent_before  [E1]  head  [/E1]  sent_between  <e2>  tail  </e2>  sent_after

        (or with ``[E1]`` / ``<e2>`` swapped when the tail entity appears first).
        CLS / SEP tokens and any relation prompt are **not** included — callers
        are responsible for wrapping.

        Args:
            item: dict with ``'text'`` or ``'token'``, ``'h'``, and ``'t'``.

        Returns:
            ``(middle_tokens, idx_e1, idx_e2)`` where the indices are positions
            of ``[E1]`` and ``<e2>`` inside *middle_tokens* (zero-based).
        """
        if "text" in item:
            sentence = item["text"]
            is_token = False
        else:
            sentence = item["token"]
            is_token = True

        pos_head = item["h"]["pos"]
        pos_tail = item["t"]["pos"]

        if pos_head[0] <= pos_tail[0]:
            pos_min, pos_max = pos_head, pos_tail
            rev = False
        else:
            pos_min, pos_max = pos_tail, pos_head
            rev = True

        if is_token:
            def _tok(span):
                return self.registry.tokenize_as_str(self.model_name, " ".join(span))
        else:
            def _tok(span):
                return self.registry.tokenize_as_str(self.model_name, span)

        sent0   = _tok(sentence[:pos_min[0]])
        ent_min = _tok(sentence[pos_min[0]:pos_min[1]])
        sent1   = _tok(sentence[pos_min[1]:pos_max[0]])
        ent_max = _tok(sentence[pos_max[0]:pos_max[1]])
        sent2   = _tok(sentence[pos_max[1]:])

        if not rev:
            ent_min_marked = ["[E1]"] + ent_min + ["[/E1]"]
            ent_max_marked = ["[E2]"] + ent_max + ["[/E2]"]
        else:
            ent_min_marked = ["[E2]"] + ent_min + ["[/E2]"]
            ent_max_marked = ["[E1]"] + ent_max + ["[/E1]"]

        middle_tokens = sent0 + ent_min_marked + sent1 + ent_max_marked + sent2
        idx_e1 = middle_tokens.index("[E1]")
        idx_e2 = middle_tokens.index("[E2]")

        return middle_tokens, idx_e1, idx_e2

    def _pad_and_tensorify(
        self, indexed_tokens: list[int], avai_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad / truncate *indexed_tokens* to ``self.max_length`` and tensorify.

        Returns:
            ``(indexed_tokens_tensor, att_mask)`` both of shape ``(1, max_length)``.
        """
        if self.blank_padding:
            pad_id = self.registry.get_tokenizer_pad_token_id(self.model_name) or 0
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(pad_id)
            indexed_tokens = indexed_tokens[:self.max_length]

        t = torch.tensor(indexed_tokens).long().unsqueeze(0)   # (1, L)
        att_mask = torch.zeros(t.size()).long()
        att_mask[0, :avai_len] = 1
        return t, att_mask

    # ------------------------------------------------------------------
    # Encoder interface
    # ------------------------------------------------------------------

    def tokenize(self, item: dict):
        """Tokenise *item* and return tensors ready for :meth:`forward`.

        Args:
            item: dict with ``'text'`` or ``'token'``, ``'h'``, and ``'t'``.

        Returns:
            ``(indexed_tokens, att_mask, pos_e1, pos_e2)`` — each a
            ``(1, L)`` or ``(1, 1)`` long tensor.
        """
        middle_tokens, idx_e1_m, idx_e2_m = self._build_marked_tokens(item)

        cls_token = self.registry.get_tokenizer_cls_token(self.model_name)
        sep_token = self.registry.get_tokenizer_sep_token(self.model_name)
        cls = [cls_token] if cls_token else []
        sep = [sep_token] if sep_token else []

        re_tokens = cls + middle_tokens + sep

        cls_offset = len(cls)
        idx_e1 = min(self.max_length - 1, idx_e1_m + cls_offset)
        idx_e2 = min(self.max_length - 1, idx_e2_m + cls_offset)

        indexed_tokens = self.registry.convert_tokens_to_ids(self.model_name, re_tokens)
        avai_len = len(indexed_tokens)

        t, att_mask = self._pad_and_tensorify(indexed_tokens, avai_len)
        pos_e1 = torch.tensor([[idx_e1]]).long()
        pos_e2 = torch.tensor([[idx_e2]]).long()

        return t, att_mask, pos_e1, pos_e2
    
    def _extract_cls_embedding(
        self,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """Extract the hidden state at the ``[CLS]`` position.

        Args:
            hidden:  ``(B, L, H)`` float32 hidden state tensor.
            pos_cls:  ``(B, 1)`` index of the ``[CLS]`` marker.

        Returns:
            ``(cls_hidden)`` — shape ``(B, H)``.
        """
        onehot_cls = torch.zeros(hidden.size()[:2]).float().to(hidden.device)

        pos_cls = torch.zeros((hidden.size()[0], 1)).long().to(hidden.device)
        onehot_cls = onehot_cls.scatter_(1, pos_cls.to(hidden.device), 1)

        cls_hidden = (onehot_cls.unsqueeze(2) * hidden).sum(1)  # (B, H)
        return cls_hidden

    def _extract_entity_embeddings(
        self,
        hidden: torch.Tensor,
        pos_e1: torch.Tensor,
        pos_e2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the hidden states at the ``[E1]`` and ``<e2>`` positions.

        Args:
            hidden:  ``(B, L, H)`` float32 hidden state tensor.
            pos_e1:  ``(B, 1)`` index of the ``[E1]`` marker.
            pos_e2:  ``(B, 1)`` index of the ``<e2>`` marker.

        Returns:
            ``(e1_hidden, e2_hidden)`` — each of shape ``(B, H)``.
        """
        onehot_e1 = torch.zeros(hidden.size()[:2]).float().to(hidden.device)
        onehot_e2 = torch.zeros(hidden.size()[:2]).float().to(hidden.device)

        onehot_e1 = onehot_e1.scatter_(1, pos_e1.to(hidden.device), 1)
        onehot_e2 = onehot_e2.scatter_(1, pos_e2.to(hidden.device), 1)

        e1_hidden = (onehot_e1.unsqueeze(2) * hidden).sum(1)  # (B, H)
        e2_hidden = (onehot_e2.unsqueeze(2) * hidden).sum(1)  # (B, H)
        return e1_hidden, e2_hidden

    def forward(
        self,
        token: torch.Tensor,
        att_mask: torch.Tensor,
        pos_e1: torch.Tensor,
        pos_e2: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch and return concatenated entity embeddings.

        Args:
            token:    ``(B, L)`` token id tensor.
            att_mask: ``(B, L)`` attention mask.
            pos_e1:   ``(B, 1)`` position of the ``[E1]`` marker.
            pos_e2:   ``(B, 1)`` position of the ``<e2>`` marker.

        Returns:
            Float32 tensor of shape ``(B, 2H)``.
        """
        outputs = self.registry.run_from_input_ids(
            self.model_name, token, attention_mask=att_mask
        )
        hidden = outputs.last_hidden_state.float()  # (B, L, H)
        e1_hidden, e2_hidden = self._extract_entity_embeddings(hidden, pos_e1, pos_e2)
        if self.has_cls_embedding:
            cls_hidden = self._extract_cls_embedding(hidden)
            return torch.cat([cls_hidden, e1_hidden, e2_hidden], dim=1)  # (B, 3H)
        return torch.cat([e1_hidden, e2_hidden], dim=1)  # (B, 2H)
