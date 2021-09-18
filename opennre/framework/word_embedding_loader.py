import torch
import numpy as np
from gensim.models import KeyedVectors

import io
import os
import json
from tqdm import tqdm


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, embedding_name):
        self.embedding_name = embedding_name
        if 'glove' == embedding_name:
            self.word_dim = 50
            self.path_word = 'pretrain/glove'  # path of pre-trained word embedding
        elif 'fasttext_wiki' == embedding_name:
            self.word_dim = 300
            self.path_word = 'pretrain/fasttext_wiki'
        elif 'fasttext_crawl' == embedding_name:
            self.word_dim = 300
            self.path_word = 'pretrain/fasttext_crawl'
        elif 'elmo' == embedding_name:
            self.path_word = 'pretrain/elmo'
        elif 'senna' == embedding_name:
            self.word_dim = 50
            self.path_word = 'pretrain/senna'
        #self.word_dim = word_dim  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word2vec = list()  # wordID to word embedding

        if 'elmo' == self.embedding_name:
            with open(self.path_word, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip().split()
                    if len(line) != self.word_dim + 1:
                        continue
                    word2id[line[0]] = len(word2id)
                    word2vec.append(np.asarray(line[1:], dtype=np.float32))

            special_emb = np.random.uniform(-1, 1, (2, self.word_dim))
            special_emb[0] = 0  # <pad> is initialize as zero
            word2vec = np.concatenate((special_emb, word2vec), axis=0)
            word2vec = word2vec.astype(np.float32).reshape(-1, self.word_dim)
            word2vec = torch.from_numpy(word2vec)
        
        elif 'fasttext_wiki' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'fasttext_wiki.1M.300d_word2id.json')):
                fasttext_model = KeyedVectors.load_word2vec_format(os.path.join(self.path_word, 'wiki-300d-1M.vec'))
                word2id = fasttext_model.key_to_index
                word2id['PAD'] = len(word2id)  # PAD character
                word2id['UNK'] = len(word2id)  # out of vocabulary
                word2vec = np.array([fasttext_model.word_vec(k) for k in word2id.keys()])
                json.dump(word2id, open(os.path.join(self.path_word,'fasttext_wiki.1M.300d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'fasttext_wiki.1M.300d._mat.npy'), word2vec)
            else:
                word2id = json.load(open(os.path.join(self.path_word, 'fasttext_wiki.1M.300d_word2id.json')))
                word2vec = np.load(os.path.join(self.path_word, 'fasttext_wiki.1M.300d._mat.npy'))
                
        
        elif 'fasttext_crawl' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'fasttext_crawl.2M.300d_word2id.json')):
                fasttext_model = KeyedVectors.load_word2vec_format(os.path.join(self.path_word, 'crawl-300d-2M.vec'))
                word2id = fasttext_model.key_to_index
                word2id['PAD'] = len(word2id)  # PAD character
                word2id['UNK'] = len(word2id)  # out of vocabulary
                word2vec = np.array([fasttext_model.word_vec(k) for k in word2id.keys()])
                json.dump(word2id, open(os.path.join(self.path_word,'fasttext_crawl.2M.300d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'fasttext_crawl.1M.300d._mat.npy'), word2vec)
            else:
                word2id = json.load(open(os.path.join(self.path_word, 'fasttext_crawl.2M.300d_word2id.json')))
                word2vec = np.load(os.path.join(self.path_word, 'fasttext_crawl.2M.300d._mat.npy'))
        
        elif 'glove' == self.embedding_name:
            word2id = json.load(open(os.path.join(self.path_word, 'glove.6B.50d_word2id.json')))
            word2vec = np.load(os.path.join(self.path_word, 'glove.6B.50d_mat.npy'))
            
        elif 'senna' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'senna.50d_word2id.json')):
                fr = open(os.path.join(self.path_word, 'senna', 'embeddings', 'embeddings.txt'), 'r', encoding='utf-8')
                w = open(os.path.join(self.path_word, 'senna', 'hash', 'words.lst'), 'r', encoding='utf-8')
                for i, line in enumerate(fr):
                    line = line.strip().split()
                    if len(line) != self.word_dim + 1:
                        continue
                    word2id[w[i]] = len(word2id)
                    word2id['PAD'] = len(word2id)  # PAD character
                    word2id['UNK'] = len(word2id)  # out of vocabulary
                    word2vec.append(np.asarray(line, dtype=np.float32))
                    
                json.dump(word2id, open(os.path.join(self.path_word,'senna.50d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'senna.50d._mat.npy'), word2vec)
            else:
                word2id = json.load(open(os.path.join(self.path_word, 'senna.50d_word2id.json')))
                word2vec = np.load(os.path.join(self.path_word, 'senna.50d._mat.npy'))
        
        return word2id, word2vec