from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
default_root_path = os.path.join(Path(__file__).resolve().parent, '.deepref')

def check_root(root_path=default_root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'benchmark'))
        os.mkdir(os.path.join(root_path, 'pretrain'))
        os.mkdir(os.path.join(root_path, 'pretrain/nre'))

def download_semeval2010(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval2010')):
        os.system('bash ' + os.path.join(root_path, 'bechmark/download_semeval2010.sh'))

def download_semeval20181_1(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval20181-1')):
        os.system('bash ' + os.path.join(root_path, 'bechmark/download_semeval20181-1.sh'))
        
def download_semeval20181_2(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval20181-2')):
        os.system('bash ' + os.path.join(root_path, 'bechmark/download_semeval20181-2.sh'))

def download_ddi(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/ddi')):
        os.system('bash ' + os.path.join(root_path, 'bechmark/download_ddi.sh'))

def download_glove(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/glove')):
        os.mkdir(os.path.join(root_path, 'pretrain/glove'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'deepref/pretrain/glove/glove.6B.50d_mat.npy')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'deepref/pretrain/glove/glove.6B.50d_word2id.json')
        
def download_fasttext_wiki(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/fasttext_wiki')):
        os.system('bash pretrain/download_fasttext_wiki.sh')

def download_fasttext_crawl(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/fasttext_crawl')):
        os.system('bash pretrain/download_fasttext_crawl.sh')

        
def download_senna(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/senna')):
        os.system('bash pretrain/download_senna.sh')
        
def download_bert_base_uncased(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/bert-base-uncased')):
        os.mkdir(os.path.join(root_path, 'pretrain/bert-base-uncased'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'deepref/pretrain/bert-base-uncased/config.json')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'deepref/pretrain/bert-base-uncased/pytorch_model.bin')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'deepref/pretrain/bert-base-uncased/vocab.txt')
        
def download_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if not os.path.exists(ckpt):
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/nre')  + ' ' + root_url + 'deepref/pretrain/nre/' + model_name + '.pth.tar')

def download(name, root_path=default_root_path):
    if not os.path.exists(os.path.join(root_path, 'benchmark')):
        os.mkdir(os.path.join(root_path, 'benchmark'))
    if not os.path.exists(os.path.join(root_path, 'pretrain')):
        os.mkdir(os.path.join(root_path, 'pretrain'))
    if name == 'semeval2010':
        download_semeval2010(root_path=root_path)
    elif name == 'semeval20181-1':
        download_semeval20181_1(root_path=root_path)
    elif name == 'semeval20181-2':
        download_semeval20181_2(root_path=root_path)
    elif name == 'ddi':
        download_ddi(root_path=root_path)
    elif name == 'glove':
        download_glove(root_path=root_path)
    elif name == 'fasttext_wiki':
        download_fasttext_wiki(root_path=root_path)
    elif name == 'fasttext_crawl':
        download_fasttext_crawl(root_path=root_path)
    elif name == 'senna':
        download_senna(root_path=root_path)
    elif name == 'bert_base_uncased':
        download_bert_base_uncased(root_path=root_path)
    
    else:
        raise Exception('Cannot find corresponding data.')

def get_model(model_name, root_path=default_root_path):
    check_root()
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if model_name == 'wiki80_cnn_softmax':
        download_pretrain(model_name, root_path=root_path)
        download('glove', root_path=root_path)
        download('wiki80', root_path=root_path)
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['wiki80_bert_softmax', 'wiki80_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('wiki80', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['tacred_bert_softmax', 'tacred_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('tacred', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/tacred/tacred_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError
