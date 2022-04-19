# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import config
from opennre.dataset.preprocess_dataset import PreprocessDataset
from opennre.framework.word_embedding_loader import WordEmbeddingLoader
from opennre.data.generate_parser_dict import save2json, csv2id
import os
import sys
import argparse
import random

class Training():
        def __init__(self, dataset, parameters, trial=None):
                self.dataset = dataset
                self.trial = trial
        
                self.preprocessing = parameters["preprocessing"]
                self.model = parameters["model"]
                self.max_length = parameters["max_length"]
                self.opt = "adamw" if self.model == "bert" else "sgd"
                self.embedding = parameters["embedding"]
                self.pos_embed = parameters["pos_embed"]
                self.deps_embed = parameters["deps_embed"]
                self.sk_embed = parameters["sk_embed"]
                self.batch_size = parameters["batch_size"]
                self.lr = parameters["lr"]
                self.max_epoch = parameters["max_epoch"]
                
                self.preprocessing_str = 'original'
                if self.preprocessing != 0:
                        self.preprocessing_str = "_".join(sorted(config.PREPROCESSING_COMBINATION[self.preprocessing]))
                        print(self.preprocessing_str)
                        
                self.hyper_params = {
                        "max_length": self.max_length,
                        "max_epoch": self.max_epoch,                        
                        "batch_size": self.batch_size,
                        "lr": self.lr
                }
                
                #if not os.path.exists(os.path.join('opennre', 'data', f'{self.dataset}_upos2id.json')):
                upos2id, deps2id = csv2id(self.dataset)
                save2json(self.dataset, upos2id, deps2id)
                upos2id = json.loads(open(os.path.join('opennre', 'data', f'{self.dataset}_upos2id.json'), 'r').read())
                deps2id = json.loads(open(os.path.join('opennre', 'data', f'{self.dataset}_deps2id.json'), 'r').read())
                        
                
                # Set random seed
                self.set_seed(config.SEED)

                root_path = '.'
                sys.path.append(root_path)
                if not os.path.exists('ckpt'):
                        os.mkdir('ckpt')
                ckpt = '{}_{}'.format(self.dataset, self.model)
                self.ckpt = 'ckpt/{}.pth.tar'.format(ckpt)
                
                self.train_file = ''
                self.val_file = ''
                self.test_file = ''
                self.rel2id_file = ''
                
                if not os.path.exists(os.path.join(root_path, 'benchmark', self.dataset)):
                        opennre.download(self.dataset, root_path=root_path)
                        
                self.train_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset, 
                                                self.preprocessing_str, 
                                                '{}_train_{}.txt'.format(self.dataset, self.preprocessing_str))
                self.val_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset, 
                                                self.preprocessing_str, 
                                                '{}_val_{}.txt'.format(self.dataset, self.preprocessing_str))
                self.test_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset, 
                                                self.preprocessing_str, 
                                                '{}_test_{}.txt'.format(self.dataset, self.preprocessing_str))
                        
                if not (os.path.exists(self.train_file)) or not(os.path.exists(self.val_file)) or not(os.path.exists(self.test_file)):
                        preprocess_dataset = PreprocessDataset(self.dataset, self.preprocessing)
                        preprocess_dataset.preprocess_dataset()
                        
                if not os.path.exists(self.test_file):
                        self.test_file = None
                self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))
                
                rel2id = json.load(open(self.rel2id_file))

                print("embedding:",self.embedding)
                if self.model != 'bert':
                        print(self.model)
                        word2id, word2vec = WordEmbeddingLoader("glove").load_embedding()
                        word_dim = word2vec.shape[1]

                # Define the sentence encoder
                if self.model == "cnn":
                        sentence_encoder = opennre.encoder.CNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=0.5,
                        )

                elif self.model == "pcnn":
                        sentence_encoder = opennre.encoder.PCNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=0.5,
                        )

                elif self.model == "crcnn":
                        sentence_encoder = opennre.encoder.CRCNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=0.5,
                        )

                elif self.model == "gru":
                        sentence_encoder = opennre.encoder.GRUEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=0.5,
                                bidirectional=False
                        )
                        
                elif self.model == "bigru":
                        sentence_encoder = opennre.encoder.GRUEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=0.5,
                                bidirectional=True
                        )

                elif self.model == "lstm":
                        sentence_encoder = opennre.encoder.LSTMEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=0.5,
                                bidirectional=False
                        )

                elif self.model == "bilstm":
                        sentence_encoder = opennre.encoder.LSTMEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=0.5,
                                bidirectional=True
                        )

                elif self.model == "bert":
                        sentence_encoder = opennre.encoder.BERTEntityEncoder(
                                max_length=self.max_length, 
                                pretrain_path=self.embedding,
                                sk_embedding=self.sk_embed,
                                pos_tags_embedding=self.pos_embed,
                                deps_embedding=self.deps_embed,
                                upos2id=upos2id,
                                deps2id=deps2id
                        )

                # Define the model
                self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                        
                
                self.criterion = opennre.model.PairwiseRankingLoss() if self.model == 'crcnn' else None
                
        def set_seed(self, seed):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                        
        def train(self):

                # Define the whole training framework
                framework = opennre.framework.SentenceRE(
                        train_path=self.train_file,
                        val_path=self.val_file,
                        test_path=self.test_file,
                        model=self.model_opennre,
                        ckpt=self.ckpt,
                        batch_size=self.batch_size,
                        max_epoch=self.max_epoch,
                        lr=self.lr,
                        opt=self.opt,
                        criterion=self.criterion,
                        trial=self.trial,
                )

                # Train the model
                try:
                        framework.train_model()
                except RuntimeError as e:
                        if 'out of memory' in str(e):
                                print(" | WARNING: ran out of memory")
                                for p in framework.model.parameters():
                                        if p .grad is not None:
                                                del p.grad
                                torch.cuda.empty_cache()
                                return {"acc": 0, "micro_f1": 0, "macro_f1": 0}
                        
                # Test
                result, pred, ground_truth = framework.eval_model(framework.test_loader)

                # Print the result
                framework.test_set_results(ground_truth, pred, result, self.model, self.embedding, self.hyper_params)
                
                return result
                
if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        # # Data
        parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                 help='Dataset. If not none, the following args can be ignored')

        args = parser.parse_args()
        
        with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
            best_hparams = json.load(f)
        
        train = Training(args.dataset, best_hparams)
        train.train()
