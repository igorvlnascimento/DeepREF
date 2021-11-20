import logging
import torch
import torch.nn as nn
from ..utils.semantic_knowledge import SemanticKNWL
from transformers import AutoTokenizer, AutoModel

import nltk
nltk.download('wordnet')

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False, sbert=None):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        logging.info('Loading {} pre-trained checkpoint.'.format(pretrain_path.upper()))
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask


class BERTEntityEncoder(nn.Module):
    def __init__(self, upos2id, deps2id, max_length, pretrain_path, blank_padding=True, mask_entity=False, sk_embedding=False, pos_tags_embedding=False, deps_embedding=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.upos2id = upos2id
        self.deps2id = deps2id
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.sk_embedding = sk_embedding
        self.pos_tags_embedding = pos_tags_embedding
        self.deps_embedding = deps_embedding
        hidden_times = (self.pos_tags_embedding + self.deps_embedding + (self.sk_embedding * 2) + 1) * 2
        self.hidden_size = 768 * hidden_times
        self.mask_entity = mask_entity
        logging.info('Loading {} pre-trained checkpoint.'.format(pretrain_path.upper()))
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pos1, pos2, pos_tags, sk1, sk2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask)
        
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        
        concat_list = [head_hidden, tail_hidden]
        
        if self.sk_embedding:
            hidden_sk1, _ = self.bert(sk1, attention_mask=att_mask)
            hidden_sk2, _ = self.bert(sk2, attention_mask=att_mask)
            #SK1
            onehot_head_sk1 = torch.zeros(hidden_sk1.size()[:2]).float().to(hidden_sk1.device)  # (B, L)
            onehot_tail_sk2 = torch.zeros(hidden_sk1.size()[:2]).float().to(hidden_sk1.device)  # (B, L)
            # # onehot_head_sk1 = onehot_head_sk1.scatter_(1, pos1, 1)
            # # onehot_tail_sk2 = onehot_tail_sk2.scatter_(1, pos2, 1)
            
            head_hidden_sk1 = (onehot_head_sk1.unsqueeze(2) * hidden_sk1).sum(1)  # (B, H)
            tail_hidden_sk1 = (onehot_tail_sk2.unsqueeze(2) * hidden_sk1).sum(1)  # (B, H)
            
            # #SK2
            onehot_head_sk2 = torch.zeros(hidden_sk2.size()[:2]).float().to(hidden_sk2.device)  # (B, L)
            onehot_tail_sk2 = torch.zeros(hidden_sk2.size()[:2]).float().to(hidden_sk2.device)  # (B, L)
            # # onehot_head_sk2 = onehot_head_sk2.scatter_(1, pos1, 1)
            # # onehot_tail_sk2 = onehot_tail_sk2.scatter_(1, pos2, 1)
            
            head_hidden_sk2 = (onehot_head_sk2.unsqueeze(2) * hidden_sk2).sum(1)  # (B, H)
            tail_hidden_sk2 = (onehot_tail_sk2.unsqueeze(2) * hidden_sk2).sum(1)  # (B, H)
            
            concat_list.extend([head_hidden_sk1, tail_hidden_sk1, head_hidden_sk2, tail_hidden_sk2])
        
        if self.pos_tags_embedding:
            hidden_pos, _ = self.bert(pos_tags, attention_mask=att_mask)
            
            onehot_head_pos = torch.zeros(hidden_pos.size()[:2]).float().to(hidden_pos.device)  # (B, L)
            onehot_tail_pos = torch.zeros(hidden_pos.size()[:2]).float().to(hidden_pos.device)  # (B, L)
            onehot_head_pos = onehot_head_pos.scatter_(1, pos1, 1)
            onehot_tail_pos = onehot_tail_pos.scatter_(1, pos2, 1)
            
            pos_head_hidden = (onehot_head_pos.unsqueeze(2) * hidden_pos).sum(1)  # (B, H)
            pos_tail_hidden = (onehot_tail_pos.unsqueeze(2) * hidden_pos).sum(1)  # (B, H)
            
            concat_list.extend([pos_head_hidden, pos_tail_hidden])
            
        # if self.deps_embedding:
        #     hidden_deps, _ = self.bert(deps, attention_mask=att_mask)
            
        #     onehot_head_deps = torch.zeros(hidden_deps.size()[:2]).float().to(hidden_deps.device)  # (B, L)
        #     onehot_tail_deps = torch.zeros(hidden_deps.size()[:2]).float().to(hidden_deps.device)  # (B, L)
        #     onehot_head_deps = onehot_head_deps.scatter_(1, pos1, 1)
        #     onehot_tail_deps = onehot_tail_deps.scatter_(1, pos2, 1) 
        
        #     deps_head_hidden = (onehot_head_deps.unsqueeze(2) * hidden_deps).sum(1)  # (B, H)
        #     deps_tail_hidden = (onehot_tail_deps.unsqueeze(2) * hidden_deps).sum(1)  # (B, H)
        
        x = torch.cat(concat_list, 1)  # (B, XH)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        
        if self.pos_tags_embedding:
            pos_tags = item['pos'] if self.pos_tags_embedding else []
        if self.deps_embedding:
            deps = item['deps'] if self.deps_embedding else []

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
            
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        
        indexed_tokens_sk1 = []
        indexed_tokens_sk2 = []
        if self.sk_embedding:
            sk_ents = SemanticKNWL().extract([sentence[pos_head[0]:pos_head[1]][-1], sentence[pos_tail[0]:pos_tail[1]][-1]])
        
            indexed_tokens_sk1 = self.tokenizer.convert_tokens_to_ids(['#'] + sk_ents["ses1"])
            indexed_tokens_sk2 = self.tokenizer.convert_tokens_to_ids(['#'] + sk_ents["ses2"])
        
        #indexed_tokens += indexed_tokens_sk1 + indexed_tokens_sk2
        
        indexed_pos = []
        indexed_deps = []
        
        for pos in pos_tags:
            indexed_pos.append(self.upos2id[pos])
            
        # for dep in deps:
        #     indexed_deps.append(self.deps2id[dep])

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            while len(indexed_tokens_sk1) < self.max_length and self.sk_embedding:
                indexed_tokens_sk1.append(0)  # 0 is id for [PAD]
            while len(indexed_tokens_sk2) < self.max_length and self.sk_embedding:
                indexed_tokens_sk2.append(0)  # 0 is id for [PAD]
            while len(indexed_pos) < self.max_length and self.pos_tags_embedding:
                indexed_pos.append(0)  # 0 is id for [PAD]
            # while len(indexed_deps) < self.max_length and self.deps_embedding:
            #     indexed_deps.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_tokens_sk1 = indexed_tokens_sk1[:self.max_length]
            indexed_tokens_sk2 = indexed_tokens_sk2[:self.max_length]
            indexed_pos = indexed_pos[:self.max_length]
            # indexed_deps = indexed_deps[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        indexed_tokens_sk1 = torch.tensor(indexed_tokens_sk1).long().unsqueeze(0)  # (1, L)
        indexed_tokens_sk2 = torch.tensor(indexed_tokens_sk2).long().unsqueeze(0)  # (1, L)
        indexed_pos = torch.tensor(indexed_pos).long().unsqueeze(0)  # (1, L)
        # indexed_deps = torch.tensor(indexed_deps).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, indexed_pos, indexed_tokens_sk1, indexed_tokens_sk2#, indexed_tokens_sk1, indexed_tokens_sk2, indexed_pos, indexed_deps