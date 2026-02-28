import logging
import torch

from deepref.encoder.llm_encoder import LLMEncoder


class RelationEncoder(LLMEncoder):
    """
    Relation encoder for relation classification.

    Each sample is formatted as:

        {text_before} <e1> {head_entity} </e1> {text_between} <e2> {tail_entity} </e2> {text_after}
        The relation between {head_name} and {tail_name} is [MASK].

    where [MASK] is the model's native mask token (e.g. ``[MASK]`` for BERT,
    ``<mask>`` for RoBERTa).

    The ``forward`` pass extracts the hidden states at the ``<e1>`` start
    marker, the ``<e2>`` start marker, and the ``[MASK]`` position, then
    returns their concatenation as the sentence representation.
    """

    def __init__(self, model_name, max_length=512, blank_padding=True):
        """
        Args:
            model_name: HuggingFace model name or local path
            max_length:    maximum tokenized sequence length
            blank_padding: pad/truncate sequences to ``max_length``
        """
        logging.info('Loading {} pre-trained checkpoint.'.format(model_name))
        super().__init__(model_name, max_length=max_length)

        self.blank_padding = blank_padding

        # Output dimension: concatenation of e1, e2, and mask hidden states
        self.hidden_size = self.model.config.hidden_size * 3

    def forward(self, token, att_mask, pos_e1, pos_e2, pos_mask):
        """
        Args:
            token:    (B, L) token id tensor
            att_mask: (B, L) attention mask (1 = real token, 0 = padding)
            pos_e1:   (B, 1) position of the ``<e1>`` marker
            pos_e2:   (B, 1) position of the ``<e2>`` marker
            pos_mask: (B, 1) position of the ``[MASK]`` token
        Return:
            (B, 3H) concatenated e1 / e2 / mask hidden states
        """
        outputs = self.model(token, attention_mask=att_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        onehot_e1   = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_e2   = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_mask = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)

        onehot_e1   = onehot_e1.scatter_(1, pos_e1, 1)
        onehot_e2   = onehot_e2.scatter_(1, pos_e2, 1)
        onehot_mask = onehot_mask.scatter_(1, pos_mask, 1)

        e1_hidden   = (onehot_e1.unsqueeze(2)   * hidden).sum(1)  # (B, H)
        e2_hidden   = (onehot_e2.unsqueeze(2)   * hidden).sum(1)  # (B, H)
        mask_hidden = (onehot_mask.unsqueeze(2) * hidden).sum(1)  # (B, H)

        return torch.cat([e1_hidden, e2_hidden, mask_hidden], dim=1)  # (B, 3H)

    def tokenize(self, item):
        """
        Build the prompt-formatted token sequence for one sample.

        Args:
            item: dict with keys:
                  - ``'text'`` (str) *or* ``'token'`` (list of str)
                  - ``'h'``: ``{'pos': [start, end], 'name': str}``
                  - ``'t'``: ``{'pos': [start, end], 'name': str}``
        Return:
            indexed_tokens: (1, L) token id tensor
            att_mask:       (1, L) attention mask tensor
            pos_e1:         (1, 1) position of ``<e1>`` marker
            pos_e2:         (1, 1) position of ``<e2>`` marker
            pos_mask:       (1, 1) position of mask token
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # Entity surface forms used in the relation prompt
        if is_token:
            head_name = ' '.join(sentence[pos_head[0]:pos_head[1]])
            tail_name = ' '.join(sentence[pos_tail[0]:pos_tail[1]])
        else:
            head_name = sentence[pos_head[0]:pos_head[1]]
            tail_name = sentence[pos_tail[0]:pos_tail[1]]

        # Order entities by their appearance in the sentence
        if pos_head[0] <= pos_tail[0]:
            pos_min, pos_max = pos_head, pos_tail
            rev = False
        else:
            pos_min, pos_max = pos_tail, pos_head
            rev = True

        # Tokenize each sentence segment individually
        if is_token:
            def _tok(span):
                return self.tokenizer.tokenize(' '.join(span))
        else:
            def _tok(span):
                return self.tokenizer.tokenize(span)

        sent0   = _tok(sentence[:pos_min[0]])
        ent_min = _tok(sentence[pos_min[0]:pos_min[1]])
        sent1   = _tok(sentence[pos_min[1]:pos_max[0]])
        ent_max = _tok(sentence[pos_max[0]:pos_max[1]])
        sent2   = _tok(sentence[pos_max[1]:])

        # Wrap entities with their respective boundary markers
        if not rev:  # head entity is closer to the beginning
            ent_min_marked = ['<e1>'] + ent_min + ['</e1>']
            ent_max_marked = ['<e2>'] + ent_max + ['</e2>']
        else:        # tail entity is closer to the beginning
            ent_min_marked = ['<e2>'] + ent_min + ['</e2>']
            ent_max_marked = ['<e1>'] + ent_max + ['</e1>']

        # Relation prompt with the model's native mask token
        prompt_tokens = self.tokenizer.tokenize(
            f"The relation between {head_name} and {tail_name} is"
        )
        mask_token = self.tokenizer.mask_token  # '[MASK]' / '<mask>' / …

        # Model-specific sequence boundaries (CLS / SEP, BOS / EOS, …)
        cls = [self.tokenizer.cls_token] if self.tokenizer.cls_token else []
        sep = [self.tokenizer.sep_token] if self.tokenizer.sep_token else []

        re_tokens = (cls
                     + sent0 + ent_min_marked + sent1 + ent_max_marked + sent2
                     + prompt_tokens + [mask_token]
                     + sep)

        # Locate special token positions (clamp to max_length - 1)
        idx_e1   = min(self.max_length - 1, re_tokens.index('<e1>'))
        idx_e2   = min(self.max_length - 1, re_tokens.index('<e2>'))
        idx_mask = min(self.max_length - 1, re_tokens.index(mask_token))

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Pad or truncate to max_length
        if self.blank_padding:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(pad_id)
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask: 1 for real tokens, 0 for padding
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        pos_e1   = torch.tensor([[idx_e1]]).long()
        pos_e2   = torch.tensor([[idx_e2]]).long()
        pos_mask = torch.tensor([[idx_mask]]).long()

        return indexed_tokens, att_mask, pos_e1, pos_e2, pos_mask
