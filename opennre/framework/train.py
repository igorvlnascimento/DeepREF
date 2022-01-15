# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import constants
from opennre.dataset.preprocess_dataset import PreprocessDataset
from opennre.framework.word_embedding_loader import WordEmbeddingLoader
import os
import sys
import argparse
import random

class Training():
        def __init__(self, parameters, trial):
                self.trial = trial
                
                self.dataset = "semeval2010" if parameters["dataset"] is None else parameters["dataset"]
                self.preprocessing = None if parameters["preprocessing"] == 0 else parameters["preprocessing"]
                self.model = "cnn" if parameters["model"] is None else parameters["model"]
                self.metric = "micro_f1" if parameters["metric"] is None else parameters["metric"]
                self.max_length = 128 if parameters["max_length"] is None else parameters["max_length"]
                self.opt = "adamw" if self.model == "bert" else "sgd"
                
                if self.model == "bert":
                        self.embedding = "bert-base-uncased" if parameters["embedding"] is None else parameters["embedding"]
                        self.synt_embeddings = [0,0,0] if parameters["synt_embeddings"] is None else parameters["synt_embeddings"]
                        self.batch_size = 16 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 2e-5 if parameters["lr"] is None else parameters["lr"]
                        self.max_epoch = 3 if parameters["max_epoch"] is None else parameters["max_epoch"]
                else:
                        self.embedding = "glove" if parameters["embedding"] is None else parameters["embedding"]
                        self.batch_size = 160 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 1e-1 if parameters["lr"] is None else parameters["lr"]
                        self.max_epoch = 100 if parameters["max_epoch"] is None else parameters["max_epoch"]
                
                self.preprocessing_str = 'original'
                if self.preprocessing is not None:
                        self.preprocessing_str = "_".join(sorted(constants.PREPROCESSING_COMBINATION[self.preprocessing]))
                        print(self.preprocessing_str)
                        
                self.hyper_params = {
                        "max_length": self.max_length,
                        "max_epoch": self.max_epoch,                        
                        "batch_size": self.batch_size,
                        "lr": self.lr
                }
                
                # Set random seed
                self.set_seed(constants.SEED)

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
                        preprocess_dataset = PreprocessDataset(self.dataset, constants.PREPROCESSING_COMBINATION[self.preprocessing])
                        preprocess_dataset.preprocess_dataset()
                        
                if not os.path.exists(self.test_file):
                        self.test_file = None
                self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))
                
                        
                rel2id = json.load(open(self.rel2id_file))

                print("embedding:",self.embedding)
                if self.model != 'bert':
                        print(self.model)
                        word2id, word2vec = WordEmbeddingLoader("fasttext_crawl").load_embedding()
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


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
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

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "bert":
                        upos2id = json.load(open(os.path.join(root_path, 'opennre/data/upos2id.json')))
                        deps2id = json.load(open(os.path.join(root_path, 'opennre/data/deps2id.json')))
                        sentence_encoder = opennre.encoder.BERTEntityEncoder(
                                upos2id=upos2id,
                                deps2id=deps2id,
                                max_length=self.max_length, 
                                pretrain_path=self.embedding,
                                sk_embedding=self.synt_embeddings[0],
                                pos_tags_embedding=self.synt_embeddings[1],
                                deps_embedding=self.synt_embeddings[2]
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
                #if not self.only_test:
                try:
                        framework.train_model(self.metric)
                except RuntimeError as e:
                        if 'out of memory' in str(e):
                                print(" | WARNING: rran out of memory, retrying batch")
                                for p in framework.model.parameters():
                                        if p .grad is not None:
                                                del p.grad
                                torch.cuda.empty_cache()
                                return 0
                        
                # Test
                result, pred, ground_truth = framework.eval_model(framework.test_loader)

                # Print the result
                framework.test_set_results(ground_truth, pred, result, self.model, self.embedding, self.hyper_params)
                
                return result[self.metric]                
                
if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        # # Data
        parser.add_argument('--metric', default='micro_f1', choices=constants.METRICS,
                help='Metric for picking up best checkpoint')
        parser.add_argument('--dataset', default=None, choices=constants.DATASETS, 
                 help='Dataset. If not none, the following args can be ignored')

        args = parser.parse_args()
        
        with open(constants.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
            best_hparams = json.load(f)
            
        best_hparams["dataset"] = args.dataset
        best_hparams["metric"] = args.metric
        
        train = Training(best_hparams,None)
        train.train()
        
#
