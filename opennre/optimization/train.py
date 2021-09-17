# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import constants
from opennre.pre_processing.preprocess_dataset import PreprocessDataset
from opennre.framework.word_embedding_loader import WordEmbeddingLoader
import os
import sys
import argparse
import stanza

class Training():
        def __init__(self, parameters):
                
                self.dataset = "semeval2010" if parameters["dataset"] is None else parameters["dataset"]
                self.preprocessing = None if len(parameters["preprocessing"]) == 0 else parameters["preprocessing"]
                self.model = "cnn" if parameters["model"] is None else parameters["model"]
                self.metric = "micro_f1" if parameters["metric"] is None else parameters["metric"]
                self.max_length = 128 if parameters["max_length"] is None else parameters["max_length"]
                self.pooler = "entity" if parameters["pooler"] is None else parameters["pooler"]
                self.mask_entity = True if parameters["mask_entity"] is None else parameters["mask_entity"]
                self.hidden_size = 230 if parameters["hidden_size"] is None else parameters["hidden_size"]
                self.position_size = 5 if parameters["position_size"] is None else parameters["position_size"]
                self.dropout = 0.5 if parameters["dropout"] is None else parameters["dropout"]
                self.weight_decay = 1e-5 if parameters["weight_decay"] is None else parameters["weight_decay"]
                
                if self.model == "bert":
                        self.embedding = "bert-base-uncased" if parameters["embedding"] is None else parameters["embedding"]
                        self.batch_size = 64 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 2e-5 if parameters["lr"] is None else parameters["lr"]
                        self.max_epoch = 3 if parameters["max_epoch"] is None else parameters["max_epoch"]
                        self.opt = 'adamw'
                else:
                        self.embedding = "glove" if parameters["embedding"] is None else parameters["embedding"]
                        self.batch_size = 160 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 1e-1 if parameters["lr"] is None else parameters["lr"]
                        self.max_epoch = 100 if parameters["max_epoch"] is None else parameters["max_epoch"]
                        self.opt = "sgd" if parameters["opt"] is None else parameters["opt"]
                
                self.preprocessing_str = 'original'
                if self.preprocessing is not None:
                        print(self.preprocessing)
                        self.preprocessing_str = "_".join(sorted(self.preprocessing))
                        
                self.hyper_params = {
                        "max_length": self.max_length,
                        "max_epoch": self.max_epoch,
                        "pooler": self.pooler,
                        "mask_entity": self.mask_entity,
                        "hidden_size": self.hidden_size,
                        "dropout": self.dropout,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                }

                root_path = '.'
                sys.path.append(root_path)
                # if not os.path.exists('ckpt'):
                #         os.mkdir('ckpt')
                # if len(args.ckpt) == 0:
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
                        if self.dataset == "semeval2010":
                                stanza.download('en')
                                nlp = stanza.Pipeline(lang='en', processors="tokenize,ner,pos", tokenize_no_ssplit=True)
                        else:
                                stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})
                                nlp = stanza.Pipeline('en', package="craft", processors={"ner": "bionlp13cg"}, tokenize_no_ssplit=True)
                        preprocess_dataset = PreprocessDataset(self.dataset, self.preprocessing, nlp)
                        preprocess_dataset.preprocess_dataset()
                        
                if not os.path.exists(self.test_file):
                        self.test_file = None
                self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))
                
                        
                rel2id = json.load(open(self.rel2id_file))

                print("embedding:",self.embedding)
                opennre.download(self.embedding, root_path=root_path)
                word2id, word2vec = WordEmbeddingLoader(self.embedding).load_embedding()
                word_dim = word2vec.shape[1]
                
                # if self.embedding == "glove":
                #         print("GLooove!")
                #         # Download glove
                #         opennre.download('glove', root_path=root_path)
                #         word2id, word2vec = WordEmbeddingLoader(self.embedding)
                #         #word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
                #         #word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
                        
                # if self.embedding == "fasttext_wiki":
                #         print("FTWiki!")
                #         # Download glove
                #         opennre.download('fasttext_wiki', root_path=root_path)
                #         word2id, word2vec = WordEmbeddingLoader(self.embedding)
                        
                # if self.embedding == "fasttext_crawl":
                #         print("FTCrawl!")
                #         # Download glove
                #         opennre.download('fasttext_crawl', root_path=root_path)
                #         word2id, word2vec = WordEmbeddingLoader(self.embedding)
                
                # if self.embedding == "senna":
                #         print("Senna!")
                #         # Download glove
                #         opennre.download('glove', root_path=root_path)
                #         word2id, word2vec = WordEmbeddingLoader(self.embedding)
                        
                # if self.embedding == "elmo":
                #         print("GLooove!")
                #         # Download glove
                #         opennre.download('glove', root_path=root_path)
                #         word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
                #         word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

                # Define the sentence encoder
                if self.model == "cnn":
                        sentence_encoder = opennre.encoder.CNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=self.dropout,
                        )


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "pcnn":
                        sentence_encoder = opennre.encoder.PCNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=self.dropout,
                        )


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "crcnn":
                        sentence_encoder = opennre.encoder.CRCNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=self.dropout,
                        )


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "gru":
                        sentence_encoder = opennre.encoder.GRUEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=self.dropout,
                                bidirectional=False
                        )

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "bigru":
                        sentence_encoder = opennre.encoder.GRUEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=self.dropout,
                                bidirectional=True
                        )

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "lstm":
                        print()
                        sentence_encoder = opennre.encoder.LSTMEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=self.dropout,
                                bidirectional=False
                        )

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "bilstm":
                        sentence_encoder = opennre.encoder.LSTMEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=word_dim,
                                position_size=self.position_size,
                                hidden_size=self.hidden_size,
                                blank_padding=True,
                                word2vec=word2vec,
                                dropout=self.dropout,
                                bidirectional=True
                        )

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "bert":
                        if self.pooler == 'entity':
                                sentence_encoder = opennre.encoder.BERTEntityEncoder(
                                        max_length=self.max_length, 
                                        pretrain_path=self.embedding,
                                        mask_entity=self.mask_entity
                                )
                        elif self.pooler == 'cls':
                                sentence_encoder = opennre.encoder.BERTEncoder(
                                        max_length=self.max_length, 
                                        pretrain_path=self.embedding,
                                        mask_entity=self.mask_entity
                                )
                        else:
                                raise NotImplementedError

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                        
                
                self.criterion = opennre.model.PairwiseRankingLoss() if self.model == 'crcnn' else None
                        
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
                        weight_decay=self.weight_decay,
                        opt=self.opt,
                        criterion=self.criterion,
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
                framework.load_state_dict(torch.load(self.ckpt)['state_dict'])
                result, pred, ground_truth = framework.eval_model(framework.test_loader)
                
                framework.get_confusion_matrix(ground_truth, pred, self.model, self.embedding)

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
        parser.add_argument('--max_length', default=40, type=int,
                help='Maximum sentence length')
        parser.add_argument('--max_epoch', default=50, type=int, # TODO : change default to 100
                help='Max number of training epochs')

        args = parser.parse_args()
        
        parameters = {
                "dataset": args.dataset,
                "model": args.model, 
                "metric": args.metric,
                "preprocessing": args.preprocessing,
                "embedding": args.embedding,
                "pretrain_path": args.pretrain_path,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_length": args.max_length,
                "max_epoch": args.max_epoch
        }
        
        train = Training(parameters)
        print("Micro-F1:",train.train())

#args, ckpt = Parser(args).init_args('cnn')

#if args.metric == 'acc':
#    logging.info('Accuracy: {}'.format(result['acc']))
#else:
#    logging.info('Micro precision: {}'.format(result['micro_p']))
#    logging.info('Micro recall: {}'.format(result['micro_r']))
#    logging.info('Micro F1: {}'.format(result['micro_f1']))
#
