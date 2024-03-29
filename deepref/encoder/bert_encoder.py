import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from deepref.encoder.base_bert_encoder import BaseBERTEncoder

class BERTEncoder(nn.Module):
    def __init__(self, 
                 max_length, 
                 pretrain_path,
                 blank_padding=True, 
                 mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        print("bert_cls")
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
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask, return_dict=False)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
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

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


class EBEMEncoder(nn.Module):
    def __init__(self, 
                 pretrain_path, 
                 upos2id,
                 deps2id,
                 dropout_rate=0.5,
                 max_length=128, 
                 blank_padding=True,
                 activation_function=F.relu,
                 mask_entity=False,
                 position_embedding=False,
                 pos_tags_embedding=False, 
                 deps_embedding=False):

        print("ebem")
        super().__init__()
        self.upos2id = upos2id
        self.deps2id = deps2id
        self.max_length = max_length
        self.max_length_embed = 5
        self.word_size = 4
        self.blank_padding = blank_padding
        self.act = activation_function
        
        self.position_embedding = position_embedding
        self.pos_tags_embedding = pos_tags_embedding
        self.deps_embedding = deps_embedding
        self.mask_entity = mask_entity
        
        logging.info('Loading {} pre-trained checkpoint.'.format(pretrain_path.upper()))
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        
        self.input_size = 768 * 4
        self.input_size += self.position_embedding * self.max_length_embed * 2
        self.input_size += (self.pos_tags_embedding + self.deps_embedding) * (self.max_length_embed * 2)
        self.hidden_size = self.input_size // 4
        
        self.linear1 = nn.Linear(self.input_size, self.input_size//2)
        self.linear2 = nn.Linear(self.input_size//2, self.input_size//4)
        self.linear3 = nn.Linear(self.input_size//4, self.hidden_size)
        
        self.position_embed = nn.Embedding(self.max_length, self.max_length_embed, padding_idx=0)
        self.pos_tags_embed = nn.Embedding(len(self.upos2id), self.max_length_embed, padding_idx=0)
        self.deps_tags_embed = nn.Embedding(len(self.deps2id), self.max_length_embed, padding_idx=0)
        
        self.drop = nn.Dropout(dropout_rate)
        
        print("pos-tag:",self.pos_tags_embedding)
        print("deps:",self.deps_embedding)
        print("position:",self.position_embedding)

    def forward(self, token, att_mask, pos1, pos2, sk_pos1, sk_pos2, pos_tag1, pos_tag2, deps1, deps2):
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
        
        sk_pos1 = sk_pos1.squeeze(1)
        sk_pos2 = sk_pos2.squeeze(1)
        onehot_sk_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_sk_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_sk_head = onehot_sk_head.scatter_(1, sk_pos1, 1)
        onehot_sk_tail = onehot_sk_tail.scatter_(1, sk_pos2, 1)
        
        sk_head_hidden = (onehot_sk_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        sk_tail_hidden = (onehot_sk_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
            
        if self.position_embedding:
            pos1_embed = self.position_embed(pos1).squeeze(1)
            pos2_embed = self.position_embed(pos2).squeeze(1)
        
        if self.pos_tags_embedding:
            pos_tag1 = self.pos_tags_embed(pos_tag1).squeeze(1)
            pos_tag2 = self.pos_tags_embed(pos_tag2).squeeze(1)
        
        if self.deps_embedding:
            deps1 = self.deps_tags_embed(deps1).squeeze(1)
            deps2 = self.deps_tags_embed(deps2).squeeze(1)
        
        head_list = [head_hidden, sk_head_hidden]
        tail_list = [tail_hidden, sk_tail_hidden]
        if self.position_embedding:
            head_list.append(pos1_embed)
            tail_list.append(pos2_embed)
        if self.pos_tags_embedding: 
            head_list.append(pos_tag1)
            tail_list.append(pos_tag2)
        if self.deps_embedding: 
            head_list.append(deps1)
            tail_list.append(deps2)  
            
        concat_list = head_list + tail_list

        x = torch.cat(concat_list, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
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
        
        pos_tags = item['pos_tags'] if self.pos_tags_embedding else []
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
        
        sk_ents = item['sk']
        sk1_father = self.tokenizer.tokenize(sk_ents["ses1"][0])
        sk1_grandpa = self.tokenizer.tokenize(sk_ents["ses1"][-1])
        sk2_father = self.tokenizer.tokenize(sk_ents["ses2"][0])
        sk2_grandpa = self.tokenizer.tokenize(sk_ents["ses2"][-1])
        
        ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
        ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        sk1 = ['[unused4]'] + sk1_father + sk1_grandpa + ['[unused5]'] if not rev else ['[unused6]'] + sk1_father + sk1_grandpa + ['[unused7]']
        sk2 = ['[unused6]'] + sk2_father + sk2_grandpa + ['[unused7]'] if not rev else ['[unused4]'] + sk2_father + sk2_grandpa + ['[unused5]']
        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + sk1 + sk2 + ['[SEP]']
        
        sk_pos1_father = re_tokens.index('[unused4]') if not rev else re_tokens.index('[unused6]')
        sk_pos1_grandpa = re_tokens.index('[unused5]') if not rev else re_tokens.index('[unused7]')
        sk_pos2_father = re_tokens.index('[unused6]') if not rev else re_tokens.index('[unused4]')
        sk_pos2_grandpa = re_tokens.index('[unused7]') if not rev else re_tokens.index('[unused5]')
        
        sk_pos1_father = min(self.max_length - 1, sk_pos1_father)
        sk_pos1_grandpa = min(self.max_length - 1, sk_pos1_grandpa)
        sk_pos2_father = min(self.max_length - 1, sk_pos2_father)
        sk_pos2_grandpa = min(self.max_length - 1, sk_pos2_grandpa)
        sk_pos1 = [sk_pos1_father, sk_pos1_grandpa]
        sk_pos2 = [sk_pos2_father, sk_pos2_grandpa]
        if self.mask_entity:
            pos1 = re_tokens.index('[unused4]') if not rev else re_tokens.index('[unused5]')
            pos2 = re_tokens.index('[unused5]') if not rev else re_tokens.index('[unused4]')
        else:
            pos1 = re_tokens.index('[unused0]') if not rev else re_tokens.index('[unused2]')
            pos2 = re_tokens.index('[unused2]') if not rev else re_tokens.index('[unused0]')
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
                
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
            
        pos_tag1 = self.upos2id[pos_tags[pos_head[0]]] if self.pos_tags_embedding else []
        pos_tag2 = self.upos2id[pos_tags[pos_tail[0]]] if self.pos_tags_embedding else []
        
        deps1 = self.deps2id[deps[pos_head[0]]] if self.deps_embedding else []
        deps2 = self.deps2id[deps[pos_tail[0]]] if self.deps_embedding else []

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        
        sk_pos1 = torch.tensor([[sk_pos1]]).long()
        sk_pos2 = torch.tensor([[sk_pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        pos_tag1 = torch.tensor([[pos_tag1]]).long()  # (1, L)
        pos_tag2 = torch.tensor([[pos_tag2]]).long()  # (1, L)
        deps1 = torch.tensor([[deps1]]).long()  # (1, L)
        deps2 = torch.tensor([[deps2]]).long()  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, sk_pos1, sk_pos2, pos_tag1, pos_tag2, deps1, deps2