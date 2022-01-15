import torch
import numpy as np
from gensim.models import KeyedVectors

import io
import os
import json
from tqdm import tqdm
import subprocess


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
        elif 'senna' == embedding_name:
            self.word_dim = 50
            self.path_word = 'pretrain/senna'

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word2vec = list()  # wordID to word embedding
        
        if 'fasttext_wiki' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'fasttext_wiki.1M.300d_word2id.json')):
                if not os.path.exists(os.path.join(self.path_word, 'wiki-news-300d-1M.vec')):
                    subprocess.call(["bash", "pretrain/download_fasttext_wiki.sh"])
                fasttext_model = KeyedVectors.load_word2vec_format(os.path.join(self.path_word, 'wiki-news-300d-1M.vec'))
                word2id = fasttext_model.key_to_index
                word2vec = np.array([fasttext_model.word_vec(k) for k in word2id.keys()])
                json.dump(word2id, open(os.path.join(self.path_word,'fasttext_wiki.1M.300d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'fasttext_wiki.1M.300d._mat.npy'), word2vec)
            word2id = json.load(open(os.path.join(self.path_word, 'fasttext_wiki.1M.300d_word2id.json')))
            word2vec = np.load(os.path.join(self.path_word, 'fasttext_wiki.1M.300d._mat.npy'))
                
        
        elif 'fasttext_crawl' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'fasttext_crawl.2M.300d_word2id.json')):
                if not os.path.exists(os.path.join(self.path_word, 'crawl-300d-2M.vec')):
                    subprocess.call(["bash", "pretrain/download_fasttext_crawl.sh"])
                fasttext_model = KeyedVectors.load_word2vec_format(os.path.join(self.path_word, 'crawl-300d-2M.vec'))
                word2id = fasttext_model.key_to_index
                word2vec = np.array([fasttext_model.word_vec(k) for k in word2id.keys()])
                json.dump(word2id, open(os.path.join(self.path_word,'fasttext_crawl.2M.300d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'fasttext_crawl.1M.300d._mat.npy'), word2vec)
            word2id = json.load(open(os.path.join(self.path_word, 'fasttext_crawl.2M.300d_word2id.json')))
            word2vec = np.load(os.path.join(self.path_word, 'fasttext_crawl.2M.300d._mat.npy'))
        
        elif 'glove' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'glove.6B.50d_mat.npy')):
                subprocess.call(["bash", "pretrain/download_glove.sh"])
            word2id = json.load(open(os.path.join(self.path_word, 'glove.6B.50d_word2id.json')))
            word2vec = np.load(os.path.join(self.path_word, 'glove.6B.50d_mat.npy'))
            
        elif 'senna' == self.embedding_name:
            if not os.path.exists(os.path.join(self.path_word, 'senna.50d_word2id.json')):
                if not os.path.exists(os.path.join(self.path_word, "senna", "embeddings", "embeddings.txt")):
                    subprocess.call(["bash", "pretrain/download_senna.sh"])
                fr = open(os.path.join(self.path_word, 'senna', 'embeddings', 'embeddings.txt'), 'r', encoding='utf-8').readlines()
                w = open(os.path.join(self.path_word, 'senna', 'hash', 'words.lst'), 'r', encoding='utf-8').readlines()
                for i, line in enumerate(fr):
                    line = line.strip().split()
                    if len(line) != self.word_dim:
                        continue
                    word2id[w[i].strip()] = len(word2id)
                    word2vec.append(np.array(line).astype(np.float))
                    
                json.dump(word2id, open(os.path.join(self.path_word,'senna.50d_word2id.json'), 'w'))
                np.save(os.path.join(self.path_word,'senna.50d._mat.npy'), word2vec)
            word2id = json.load(open(os.path.join(self.path_word, 'senna.50d_word2id.json')))
            word2vec = np.load(os.path.join(self.path_word, 'senna.50d._mat.npy'))
        
        return word2id, word2vec