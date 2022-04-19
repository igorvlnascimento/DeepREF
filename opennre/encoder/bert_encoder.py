import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..utils.semantic_knowledge import SemanticKNWL
from transformers import AutoTokenizer, AutoModel

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
    def __init__(self, 
                 pretrain_path, 
                 upos2id,
                 deps2id,
                 max_length=128, 
                 blank_padding=True,
                 activation_function=F.relu,
                 mask_entity=False,
                 position_embedding=False,
                 sk_embedding=False, 
                 pos_tags_embedding=False, 
                 deps_embedding=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.upos2id = upos2id
        self.deps2id = deps2id
        self.max_length = max_length
        self.max_length_embed = 5
        self.word_size = 4
        self.blank_padding = blank_padding
        self.act = activation_function
        
        self.position_embedding = position_embedding
        self.sk_embedding = sk_embedding
        self.pos_tags_embedding = pos_tags_embedding
        self.deps_embedding = deps_embedding
        
        self.input_size = 768 * 2 + (self.position_embedding * self.max_length) + ((self.pos_tags_embedding + self.deps_embedding) * (self.max_length_embed * 2)) + self.sk_embedding * 768 * 2
        self.hidden_size = self.input_size // 4
        
        self.mask_entity = mask_entity
        
        logging.info('Loading {} pre-trained checkpoint.'.format(pretrain_path.upper()))
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        
        self.position_embed = nn.Embedding(self.max_length, self.max_length_embed, padding_idx=0)
        self.pos_tags_embed = nn.Embedding(len(self.upos2id), self.max_length_embed, padding_idx=0)
        self.deps_tags_embed = nn.Embedding(len(self.deps2id), self.max_length_embed, padding_idx=0)
        
        self.linear1 = nn.Linear(self.input_size, self.input_size//2)
        self.linear2 = nn.Linear(self.input_size//2, self.input_size//4)
        self.linear3 = nn.Linear(self.input_size//4, self.hidden_size)
        #self.drop = nn.Dropout(0.5)
        
        print("pos-tag:",self.pos_tags_embedding)
        print("deps:",self.deps_embedding)
        print("sk:",self.sk_embedding)

    def forward(self, token, att_mask, pos1, pos2, pos1_embed, pos2_embed, sk_pos1, sk_pos2, pos_tag1, pos_tag2, deps1, deps2):
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
        
        sk_pos1 = sk_pos1.squeeze(1)
        sk_pos2 = sk_pos2.squeeze(1)
        onehot_sk_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_sk_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_sk_head = onehot_sk_head.scatter_(1, sk_pos1, 1)
        onehot_sk_tail = onehot_sk_tail.scatter_(1, sk_pos2, 1)
        
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        
        sk_head_hidden = (onehot_sk_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        sk_tail_hidden = (onehot_sk_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        
        pos_tag1 = self.pos_tags_embed(pos_tag1).squeeze(1)
        deps1 = self.deps_tags_embed(deps1).squeeze(1)
        
        pos_tag2 = self.pos_tags_embed(pos_tag2).squeeze(1)
        deps2 = self.deps_tags_embed(deps2).squeeze(1)
        
        head_list = [head_hidden]
        tail_list = [tail_hidden]
        if self.sk_embedding:
            head_list.append(sk_head_hidden)
            tail_list.append(sk_tail_hidden)
        if self.position_embedding:
            head_list.append(self.position_embed(pos1_embed).squeeze(1))
            tail_list.append(self.position_embed(pos2_embed).squeeze(1))
        if self.pos_tags_embedding: 
            head_list.append(pos_tag1)
            tail_list.append(pos_tag2)
        if self.deps_embedding: 
            head_list.append(deps1)
            tail_list.append(deps2)  
            
        concat_list = head_list + tail_list          
        
        # if self.pos_tags_embedding:
        #     concat_list.extend([pos_tags])
            
        # if self.deps_embedding:
        #     concat_list.extend([deps])

        x = torch.cat(concat_list, 1)
        print("shape:",x.shape)
        print("input:",self.input_size)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        #x = self.drop(x)
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
        
        pos_tags = []
        deps = []
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
            
        if self.sk_embedding:
            sk_ents = SemanticKNWL().extract([sentence[pos_head[0]:pos_head[1]][-1], sentence[pos_tail[0]:pos_tail[1]][-1]])
            
            sk0 = self.tokenizer.tokenize(sk_ents["ses1"])
            sk1 = self.tokenizer.tokenize(sk_ents["ses2"])

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
            if self.sk_embedding:
                sk0 = ['[unused0]'] + sk0 + ['[unused1]'] if not rev else ['[unused2]'] + sk0 + ['[unused3]']
                sk1 = ['[unused2]'] + sk1 + ['[unused3]'] if not rev else ['[unused0]'] + sk1 + ['[unused1]']

        if self.sk_embedding:
            re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + sk0 + sk1 + ['[SEP]']
        else:
            re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 2 + len(sent0) if not rev else 2 + len(sent0 + ent0 + sent1)
        pos2 = 2 + len(sent0 + ent0 + sent1) if not rev else 2 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        sk_pos1 = []
        sk_pos2 = []
        if self.sk_embedding:
            sk_pos1 = list(range(2 + len(sent0 + ent0+ sent1 + ent1 + sent2), 4 + len(sent0 + ent0+ sent1 + ent1 + sent2)))
            sk_pos2 = list(range(3 + sk_pos1[-1], len(re_tokens) - 2))
            sk_pos1 = list(range(self.max_length - len(sk_pos1) - 1)) if sk_pos1[-1] > (self.max_length - 1) else sk_pos1
            sk_pos2 = list(range(self.max_length - len(sk_pos2) - 1)) if sk_pos2[-1] > (self.max_length - 1) else sk_pos2
            sk_pos1 = sk_pos1[:2]
            sk_pos2 = sk_pos2[:2]
                
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        
        if self.sk_embedding:
            sk_ents = SemanticKNWL().extract([sentence[pos_head[0]:pos_head[1]][-1], sentence[pos_tail[0]:pos_tail[1]][-1]])
        
            indexed_tokens_sk1 = self.tokenizer.convert_tokens_to_ids(sk_ents["ses1"])
            indexed_tokens_sk2 = self.tokenizer.convert_tokens_to_ids(sk_ents["ses2"])
            
            indexed_tokens = indexed_tokens + indexed_tokens_sk1 + indexed_tokens_sk2
            
            #indexed_tokens_sk = indexed_tokens_sk1 + indexed_tokens_sk2
        
        # indexed_pos = []
        # indexed_deps = []
        pos_tag1 = self.upos2id[pos_tags[pos_head[0]]] if self.pos_tags_embedding else []
        pos_tag2 = self.upos2id[pos_tags[pos_tail[0]]] if self.pos_tags_embedding else []
        
        deps1 = self.deps2id[deps[pos_head[0]]] if self.deps_embedding else []
        deps2 = self.deps2id[deps[pos_tail[0]]] if self.deps_embedding else []
        
        # for pos in pos_tags:
        #     if pos not in self.upos2id:
        #         self.upos2id[pos] = len(self.upos2id)
        #     indexed_pos.append(self.upos2id[pos])
            
        # for dep in deps:
        #     if dep not in self.deps2id:
        #         self.deps2id[dep] = len(self.deps2id)
        #     indexed_deps.append(self.deps2id[dep])

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        
        sk_pos1 = torch.tensor([[sk_pos1]]).long()
        sk_pos2 = torch.tensor([[sk_pos2]]).long()
        
        # Position -> index
        pos1_embed = []
        pos2_embed = []
        pos1_in_index = min(pos1[0], self.max_length)
        pos2_in_index = min(pos2[0], self.max_length)
        for i in range(len(re_tokens)):
            pos1_embed.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2_embed.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        if self.blank_padding:                
            while len(pos1_embed) < self.max_length:
                pos1_embed.append(0)
            while len(pos2_embed) < self.max_length:
                pos2_embed.append(0)
            pos1_embed = pos1_embed[:self.max_length]
            pos2_embed = pos2_embed[:self.max_length]

        pos1_embed = torch.tensor(pos1_embed).long().unsqueeze(0) # (1, L)
        pos2_embed = torch.tensor(pos2_embed).long().unsqueeze(0) # (1, L)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            # while self.sk_embedding and len(indexed_tokens_sk) < self.word_size:
            #     indexed_tokens_sk.append(0)  # 0 is id for [PAD]
            # while self.pos_tags_embedding and len(indexed_pos) < self.max_length_embed:
            #     indexed_pos.append(0)  # 0 is id for [PAD]
            # while self.deps_embedding and len(indexed_deps) < self.max_length_embed:
            #     indexed_deps.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
            # indexed_tokens_sk = indexed_tokens_sk[:self.word_size]
            # indexed_pos = indexed_pos[:self.max_length_embed]
            # indexed_deps = indexed_deps[:self.max_length_embed]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        # indexed_tokens_sk = torch.tensor(indexed_tokens_sk).long().unsqueeze(0)  # (1, L)
        pos_tag1 = torch.tensor([[pos_tag1]]).long()  # (1, L)
        pos_tag2 = torch.tensor([[pos_tag2]]).long()  # (1, L)
        deps1 = torch.tensor([[deps1]]).long()  # (1, L)
        deps2 = torch.tensor([[deps2]]).long()  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2, pos1_embed, pos2_embed, sk_pos1, sk_pos2, pos_tag1, pos_tag2, deps1, deps2#, indexed_tokens_sk1, indexed_tokens_sk2#, indexed_pos, indexed_deps