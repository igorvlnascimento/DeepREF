import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
from .bert_encoder import BERTEncoder
from ..module.nn import CNN

class BERTHiddenStateEncoder(BERTEncoder):
    def __init__(self, pretrain_path, blank_padding=True):
        super().__init__(80, pretrain_path, blank_padding)
        self.bert = BertModel.from_pretrained(pretrain_path, output_hidden_states=True,output_attentions=True)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x, hs ,atts= self.bert(token, attention_mask=att_mask)
        return x, hs, atts

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        # if 'text' in item:
        #     sentence = item['text']
        #     is_token = False
        # else:
        sentence = item['token']
        is_token = True

        tokens = sentence

        re_tokens = ['[CLS]']
        cur_pos = 0
        new_index = [-1]
        for idx, token in enumerate(tokens):
            # token = token.lower()
            re_tokens += self.tokenizer.tokenize(token)
            new_index.extend([idx] * (len(re_tokens) - len(new_index)))
            cur_pos += 1
        re_tokens.append('[SEP]')
        new_index.append(max(new_index) + 1)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.ones(indexed_tokens.size()).long()  # (1, L)

        return indexed_tokens, att_mask, new_index