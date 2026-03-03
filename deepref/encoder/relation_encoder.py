import logging
import torch

from deepref.encoder.bert_entity_encoder import BertEntityEncoder


class RelationEncoder(BertEntityEncoder):
    """Relation encoder that extends :class:`BertEntityEncoder` with a mask prompt.

    Each sample is formatted as::

        {sent_before} <e1> {head} </e1> {sent_between} <e2> {tail} </e2> {sent_after}
        The relation between {head_name} and {tail_name} is [MASK].

    The ``forward`` pass extracts the hidden states at ``<e1>``, ``<e2>``, and
    ``[MASK]`` using the entity-extraction logic inherited from
    :class:`BertEntityEncoder`, then returns their concatenation as the
    sentence representation.

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
        device="cpu",
        attn_implementation="eager",
        trainable=False,
    ):
        logging.info("Loading %s pre-trained checkpoint.", model_name)
        super().__init__(
            model_name,
            max_length=max_length,
            blank_padding=blank_padding,
            device=device,
            attn_implementation=attn_implementation,
            trainable=trainable,
        )
        # Output: e1 + e2 + mask hidden states
        self.hidden_size = self.registry.get_model_hidden_size(model_name) * 3

    def tokenize(self, item: dict):
        """Build the prompt-formatted token sequence for one sample.

        Calls :meth:`BertEntityEncoder._build_marked_tokens` to obtain the
        entity-marked sentence content, then appends the relation prompt and
        the model's native mask token.

        Args:
            item: dict with keys:
                  - ``'text'`` (str) *or* ``'token'`` (list of str)
                  - ``'h'``: ``{'pos': [start, end], 'name': str}``
                  - ``'t'``: ``{'pos': [start, end], 'name': str}``

        Returns:
            ``(indexed_tokens, att_mask, pos_e1, pos_e2, pos_mask)`` —
            each a ``(1, L)`` or ``(1, 1)`` long tensor.
        """
        middle_tokens, idx_e1_m, idx_e2_m = self._build_marked_tokens(item)

        head_name = item["h"]["name"]
        tail_name = item["t"]["name"]

        prompt_tokens = self.registry.tokenize_as_str(
            self.model_name,
            f"The relation between {head_name} and {tail_name} is",
        )
        mask_token = self.registry.get_tokenizer_mask_token(self.model_name)

        cls_token = self.registry.get_tokenizer_cls_token(self.model_name)
        sep_token = self.registry.get_tokenizer_sep_token(self.model_name)
        cls = [cls_token] if cls_token else []
        sep = [sep_token] if sep_token else []

        re_tokens = cls + middle_tokens + prompt_tokens + [mask_token] + sep

        cls_offset = len(cls)
        idx_e1   = min(self.max_length - 1, idx_e1_m + cls_offset)
        idx_e2   = min(self.max_length - 1, idx_e2_m + cls_offset)
        idx_mask = min(self.max_length - 1, re_tokens.index(mask_token))

        indexed_tokens = self.registry.convert_tokens_to_ids(self.model_name, re_tokens)
        avai_len = len(indexed_tokens)

        t, att_mask = self._pad_and_tensorify(indexed_tokens, avai_len)
        pos_e1   = torch.tensor([[idx_e1]]).long()
        pos_e2   = torch.tensor([[idx_e2]]).long()
        pos_mask = torch.tensor([[idx_mask]]).long()

        return t, att_mask, pos_e1, pos_e2, pos_mask

    def forward(
        self,
        token: torch.Tensor,
        att_mask: torch.Tensor,
        pos_e1: torch.Tensor,
        pos_e2: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch and return concatenated entity + mask embeddings.

        Runs the backbone once, extracts the entity hidden states via the
        inherited :meth:`~BertEntityEncoder._extract_entity_embeddings` helper,
        extracts the mask token hidden state, then concatenates all three.

        Args:
            token:    ``(B, L)`` token id tensor.
            att_mask: ``(B, L)`` attention mask.
            pos_e1:   ``(B, 1)`` position of the ``<e1>`` marker.
            pos_e2:   ``(B, 1)`` position of the ``<e2>`` marker.
            pos_mask: ``(B, 1)`` position of the mask token.

        Returns:
            Float32 tensor of shape ``(B, 3H)``.
        """
        outputs = self.registry.run_from_input_ids(
            self.model_name, token, attention_mask=att_mask
        )
        hidden = outputs.last_hidden_state.float()  # (B, L, H)

        # Entity embeddings — reuse parent helper
        e1_hidden, e2_hidden = self._extract_entity_embeddings(hidden, pos_e1, pos_e2)

        # Mask embedding
        onehot_mask = torch.zeros(hidden.size()[:2]).float().to(hidden.device)
        onehot_mask = onehot_mask.scatter_(1, pos_mask.to(hidden.device), 1)
        mask_hidden = (onehot_mask.unsqueeze(2) * hidden).sum(1)  # (B, H)

        return torch.cat([e1_hidden, e2_hidden, mask_hidden], dim=1)  # (B, 3H)
