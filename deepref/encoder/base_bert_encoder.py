import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BaseBERTEncoder(nn.Module):
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
        
        # self.input_size = 768 * 2
        # self.input_size += self.sk_embedding * 768 * 2
        # self.input_size += self.position_embedding * self.max_length_embed * 2
        # self.input_size += (self.pos_tags_embedding + self.deps_embedding) * (self.max_length_embed * 2)
        # self.hidden_size = self.input_size // 4
        
        self.mask_entity = mask_entity
        
        logging.info('Loading {} pre-trained checkpoint.'.format(pretrain_path.upper()))
        self.bert = AutoModel.from_pretrained(pretrain_path, return_dict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        
        self.position_embed = nn.Embedding(self.max_length, self.max_length_embed, padding_idx=0)
        self.pos_tags_embed = nn.Embedding(len(self.upos2id), self.max_length_embed, padding_idx=0)
        self.deps_tags_embed = nn.Embedding(len(self.deps2id), self.max_length_embed, padding_idx=0)
        
        self.drop = nn.Dropout(dropout_rate)
        
        print("pos-tag:",self.pos_tags_embedding)
        print("deps:",self.deps_embedding)
        print("sk:",self.sk_embedding)
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
        pass

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
        
        re_tokens = []
        sk_pos1 = []
        sk_pos2 = []
        if self.sk_embedding:
            sk_ents = item['sk']
            sk1_father = self.tokenizer.tokenize(sk_ents["ses1"][0])
            sk1_grandpa = self.tokenizer.tokenize(sk_ents["ses1"][-1])
            sk2_father = self.tokenizer.tokenize(sk_ents["ses2"][0])
            sk2_grandpa = self.tokenizer.tokenize(sk_ents["ses2"][-1])
            
            sk1 = ['[unused0]'] + ent0 + sk1_father + sk1_grandpa + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + sk1_father + sk1_grandpa + ['[unused3]']
            sk2 = ['[unused2]'] + ent1 + sk2_father + sk2_grandpa + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + sk2_father + sk2_grandpa + ['[unused1]']
            re_tokens = ['[CLS]'] + sent0 + sk1 + sent1 + sk2 +  sent2 + ['[SEP]']
            
            sk_pos1_father = re_tokens.index('[unused0]') + len(ent0) + 1 if not rev else re_tokens.index('[unused2]') + len(ent0) + 1
            sk_pos1_grandpa = re_tokens.index('[unused1]') - len(sk1_grandpa) if not rev else re_tokens.index('[unused3]') - len(sk1_grandpa)
            sk_pos2_father = re_tokens.index('[unused2]') + len(ent1) + 1 if not rev else re_tokens.index('[unused0]') + len(ent1) + 1
            sk_pos2_grandpa = re_tokens.index('[unused3]') - len(sk2_grandpa) if not rev else re_tokens.index('[unused2]') - len(sk2_grandpa)
            sk_pos1_father = min(self.max_length - 1, sk_pos1_father)
            sk_pos1_grandpa = min(self.max_length - 1, sk_pos1_grandpa)
            sk_pos2_father = min(self.max_length - 1, sk_pos2_father)
            sk_pos2_grandpa = min(self.max_length - 1, sk_pos2_grandpa)
            sk_pos1 = [sk_pos1_father, sk_pos1_grandpa]
            sk_pos2 = [sk_pos2_father, sk_pos2_grandpa]
            sk1_father_name = re_tokens[sk_pos1_father] if sk_pos1_father == self.max_length - 1 else sk1_father[0]
            sk1_grandpa_name = re_tokens[sk_pos1_grandpa] if sk_pos1_grandpa == self.max_length - 1 else sk1_grandpa[0]
            sk2_father_name = re_tokens[sk_pos2_father] if sk_pos2_father == self.max_length - 1 else sk2_father[0]
            sk2_grandpa_name = re_tokens[sk_pos2_grandpa] if sk_pos2_grandpa == self.max_length - 1 else sk2_grandpa[0]
            assert re_tokens[sk_pos1_father] == sk1_father_name
            assert re_tokens[sk_pos1_grandpa] == sk1_grandpa_name
            assert re_tokens[sk_pos2_father] == sk2_father_name
            assert re_tokens[sk_pos2_grandpa] == sk2_grandpa_name

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
    
        if not re_tokens:
            re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
                
        pos1 = re_tokens.index('[unused0]') + 1 if not self.mask_entity else re_tokens.index('[unused4]')
        pos2 = re_tokens.index('[unused2]') + 1 if not self.mask_entity else re_tokens.index('[unused5]')
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        ent0_name = re_tokens[pos1] if pos1 == self.max_length - 1 else ent0[1]
        ent1_name = re_tokens[pos2] if pos2 == self.max_length - 1 else ent1[1]
        assert re_tokens[pos1] == ent0_name
        assert re_tokens[pos2] == ent1_name
                
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