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
                        preprocess_dataset = PreprocessDataset(self.dataset, self.preprocessing)
                        preprocess_dataset.preprocess_dataset()
                        
                if not os.path.exists(self.test_file):
                        self.test_file = None
                self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))
                
                        
                rel2id = json.load(open(self.rel2id_file))

                print("embedding:",self.embedding)
                if self.model != 'bert':
                        print(self.model)
                        word2id, word2vec = WordEmbeddingLoader(self.embedding).load_embedding()
                        word_dim = word2vec.shape[1]
                        
                upos2id = json.load(open(os.path.join(root_path, 'pretrain/upos2id.json')))
                deps2id = json.load(open(os.path.join(root_path, 'pretrain/deps2id.json')))

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
                #framework.load_state_dict(torch.load(self.ckpt)['state_dict'])
                result, pred, ground_truth = framework.eval_model(framework.test_loader)
                
                #framework.get_confusion_matrix(ground_truth, pred, self.model, self.embedding)

                # Print the result
                framework.test_set_results(ground_truth, pred, result, self.model, self.embedding, self.hyper_params)
                
                return result[self.metric]                
                
if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--pretrain_path', default='bert-base-uncased', choices=constants.pretrain_choices,
                help='Pre-trained ckpt path / model name (hugginface)')
        parser.add_argument('--ckpt', default='', 
                help='Checkpoint name')
        parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
                help='Sentence representation pooler')
        parser.add_argument('--only_test', action='store_true', 
                help='Only run test')
        parser.add_argument('--mask_entity', action='store_true', 
                help='Mask entity mentions')

        #Model
        parser.add_argument('--model', default='cnn', choices=['cnn', 'pcnn', 'bert', 'crcnn', 'gru', 'bigru', 'lstm', 'bilstm'],
                help='Model to train')

        #Embedding
        parser.add_argument('--embedding', default='glove', choices=['glove', 'senna', 'elmo'],
                help='Word Embedding')

        # Data
        parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                help='Metric for picking up best checkpoint')
        parser.add_argument('--dataset', default=None, choices=constants.datasets_choices, 
                help='Dataset. If not none, the following args can be ignored')
        parser.add_argument('--preprocessing', nargs="+", default=None,
                help='Preprocessing. If not none, the original dataset is used')
        parser.add_argument('--train_file', default='', type=str,
                help='Training data file')
        parser.add_argument('--val_file', default='', type=str,
                help='Validation data file')
        parser.add_argument('--test_file', default='', type=str,
                help='Test data file')
        parser.add_argument('--rel2id_file', default='', type=str,
                help='Relation to ID file')

        # Hyper-parameters
        parser.add_argument('--batch_size', default=32, type=int,
                help='Batch size')
        parser.add_argument('--lr', default=1e-1, type=float,
                help='Learning rate')
        parser.add_argument('--weight_decay', default=1e-5, type=float,
                help='Weight decay')
        parser.add_argument('--max_length', default=128, type=int,
                help='Maximum sentence length')
        parser.add_argument('--max_epoch', default=50, type=int, # TODO : change default to 100
                help='Max number of training epochs')

        args = parser.parse_args()
        
        BEST_HPARAMS_FILE_PATH = "opennre/optimization/best_hparams_{}.json"
        
        with open(BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
            best_hparams = json.load(f)
        
        train = Training(best_hparams)
        print("Micro-F1:",train.train())
#
