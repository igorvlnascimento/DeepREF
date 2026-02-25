# coding:utf-8
import torch
import numpy as np
import json
import deepref
from deepref import config
from deepref.dataset.dataset import Dataset
from deepref.dataset.preprocessors.stop_word_preprocessor import StopWordPreprocessor
from deepref.dataset.preprocessors.punctuation_preprocessor import PunctuationPreprocessor
from deepref.dataset.preprocessors.brackets_or_parenthesis_preprocessor import BracketsPreprocessor
from deepref.dataset.preprocessors.digit_blinding_preprocessor import DigitBlindingPreprocessor
from deepref.dataset.preprocessors.entity_blinding_preprocessor import EntityBlindingPreprocessor
from deepref.framework.word_embedding_loader import WordEmbeddingLoader
from benchmark.generate_parser import save2json, csv2id
import os
import sys
import argparse
import random

class Training():
        def __init__(self, dataset_name:str, parameters, trial=None, seed=42):
                self.dataset_name = dataset_name
                self.trial = trial
        
                self.preprocessing = parameters["preprocessing"]
                self.model = parameters["model"]
                self.max_length = parameters["max_length"]
                self.opt = "adamw" if self.model == "bert_entity" or self.model == "bert_cls" or self.model == "ebem" else "sgd"
                self.pretrain = parameters["pretrain"]
                self.position_embed = parameters["position_embed"]
                self.pos_tags_embed = parameters["pos_tags_embed"]
                self.deps_embed = parameters["deps_embed"]
                self.batch_size = parameters["batch_size"]
                self.lr = parameters["lr"]
                self.max_epoch = parameters["max_epoch"]
                
                self.preprocessing_str = 'original'
                self.preprocessing = sorted(self.preprocessing)
                print("preprocessing:",self.preprocessing)
                if self.preprocessing != []:
                        self.preprocessing_str = "_".join(self.preprocessing)
                        print(self.preprocessing_str)
                        
                self.hyper_params = {
                        "max_length": self.max_length,
                        "max_epoch": self.max_epoch,                        
                        "batch_size": self.batch_size,
                        "lr": self.lr
                }
                
                upos2id, deps2id = csv2id(self.dataset_name)
                save2json(self.dataset_name, upos2id, deps2id)
                upos2id = json.loads(open(os.path.join('benchmark', self.dataset_name, f"{self.dataset_name}_upos2id.json"), 'r').read())
                deps2id = json.loads(open(os.path.join('benchmark', self.dataset_name, f"{self.dataset_name}_deps2id.json"), 'r').read())
                        
                
                # Set random seed
                self.set_seed(seed)

                root_path = '.'
                sys.path.append(root_path)
                if not os.path.exists('ckpt'):
                        os.mkdir('ckpt')
                ckpt = '{}_{}'.format(self.dataset_name, self.model)
                self.ckpt = 'ckpt/{}.pth.tar'.format(ckpt)
                
                self.train_file = ''
                self.val_file = ''
                self.test_file = ''
                self.rel2id_file = ''
                
                if not os.path.exists(os.path.join(root_path, 'benchmark', self.dataset_name)):
                        deepref.download(self.dataset_name, root_path=root_path)
                        
                self.train_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset_name, 
                                                self.preprocessing_str, 
                                                '{}_train_{}.txt'.format(self.dataset_name, self.preprocessing_str))
                self.val_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset_name, 
                                                self.preprocessing_str, 
                                                '{}_val_{}.txt'.format(self.dataset_name, self.preprocessing_str))
                self.test_file = os.path.join(root_path, 
                                                'benchmark', 
                                                self.dataset_name, 
                                                self.preprocessing_str, 
                                                '{}_test_{}.txt'.format(self.dataset_name, self.preprocessing_str))
                        
                if not (os.path.exists(self.train_file)) or not(os.path.exists(self.val_file)) or not(os.path.exists(self.test_file)):
                        if 'sw' in self.preprocessing:
                                dataset = StopWordPreprocessor(self.dataset_name, self.preprocessing).preprocess_dataset()
                                dataset.write_text(self.preprocessing)
                        if 'p' in self.preprocessing:
                                dataset = PunctuationPreprocessor(self.dataset_name, self.preprocessing).preprocess_dataset()
                                dataset.write_text(self.preprocessing)
                        if 'b' in self.preprocessing:
                                dataset = BracketsPreprocessor(self.dataset_name, self.preprocessing).preprocess_dataset()
                                dataset.write_text(self.preprocessing)
                        if 'd' in self.preprocessing:
                                dataset = DigitBlindingPreprocessor(self.dataset_name, self.preprocessing).preprocess_dataset()
                                dataset.write_text(self.preprocessing)
                        if 'nb' in self.preprocessing and 'eb' in self.preprocessing:
                                if self.dataset_name == 'ddi':
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self.preprocessing, "DRUG").preprocess_dataset()
                                        dataset.write_text(self.preprocessing)
                                else:
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self.preprocessing, "ENTITY").preprocess_dataset()
                                        dataset.write_text(self.preprocessing)
                        elif 'eb' in self.preprocessing:
                                if self.dataset_name == 'ddi':
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self. preprocessing, 'DRUG', 'entity').preprocess_dataset()
                                        dataset.write_text(self.preprocessing)     
                                else:
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self. preprocessing, 'ENTITY', 'entity').preprocess_dataset()
                                        dataset.write_text(self.preprocessing)
                        elif 'nb' in self.preprocessing:
                                if self.dataset_name == 'ddi':
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self.preprocessing, "DRUG").preprocess_dataset()
                                        dataset.write_text(self.preprocessing)
                                else:
                                        dataset = EntityBlindingPreprocessor(self.dataset_name, self.preprocessing, "ENTITY").preprocess_dataset()
                                        dataset.write_text(self.preprocessing)
                        
                if not os.path.exists(self.test_file):
                        self.test_file = None
                self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset_name, '{}_rel2id.json'.format(self.dataset_name))
                
                rel2id = json.load(open(self.rel2id_file))

                print("pretrain:",self.pretrain)
                if 'bert_' not in self.model and self.model != 'ebem':
                        print(self.model)
                        word2id, word2vec = WordEmbeddingLoader("glove").load_embedding()
                        word_dim = word2vec.shape[1]

                # Define the sentence encoder
                if self.model == "cnn":
                        sentence_encoder = deepref.encoder.CNNEncoder(
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
                        sentence_encoder = deepref.encoder.PCNNEncoder(
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
                        sentence_encoder = deepref.encoder.CRCNNEncoder(
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
                        sentence_encoder = deepref.encoder.GRUEncoder(
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
                        sentence_encoder = deepref.encoder.GRUEncoder(
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
                        sentence_encoder = deepref.encoder.LSTMEncoder(
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
                        sentence_encoder = deepref.encoder.LSTMEncoder(
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
                        
                elif self.model == "bert_cls":
                        sentence_encoder = deepref.encoder.BERTEncoder(
                                max_length=self.max_length, 
                                pretrain_path=self.pretrain
                        )
                        
                elif self.model == "bert_entity":
                        sentence_encoder = deepref.encoder.BERTEntityEncoder(
                                max_length=self.max_length,
                                pretrain_path=self.pretrain
                        )
                
                elif self.model == "ebem":
                        sentence_encoder = deepref.encoder.EBEMEncoder(
                                max_length=self.max_length,
                                pretrain_path=self.pretrain,
                                position_embedding=self.position_embed,
                                pos_tags_embedding=self.pos_tags_embed,
                                deps_embedding=self.deps_embed,
                                upos2id=upos2id,
                                deps2id=deps2id
                        )

                # Define the model
                self.model_deepref = deepref.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                
                self.criterion = deepref.model.PairwiseRankingLoss() if self.model == 'crcnn' else None
                
        def set_seed(self, seed):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                        
        def train(self):

                # Define the whole training framework
                framework = deepref.framework.SentenceRE(
                        train_path=self.train_file,
                        val_path=self.val_file,
                        test_path=self.test_file,
                        model=self.model_deepref,
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
                                return {"acc": 0, "micro_p": 0, "micro_r": 0, "micro_f1": 0, "macro_f1": 0}
                        
                # Test
                result, pred, ground_truth = framework.eval_model(framework.test_loader)

                # Print the result
                framework.test_set_results(ground_truth, pred, result, self.model, self.pretrain, self.hyper_params)
                
                torch.cuda.empty_cache()
                
                return result
                
if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        # # Data
        parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                 help='Dataset. If not none, the following args can be ignored')

        args = parser.parse_args()
        
        with open(config.HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
            hparams = json.load(f)
        
        train = Training(args.dataset, hparams)
        train.train()
