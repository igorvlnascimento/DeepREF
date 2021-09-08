# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import model, framework, constants
import os
import sys
import argparse

class Training():
        def __init__(self, parameters):
                
                self.dataset = "semeval2010" if parameters["dataset"] is None else parameters["dataset"]
                self.preprocessing = ["n"] if parameters["preprocessing"] is None else parameters["preprocessing"]
                self.model = "cnn" if parameters["model"] is None else parameters["model"]
                self.metric = "micro_f1" if parameters["metric"] is None else parameters["metric"]
                self.max_length = 128 if parameters["max_length"] is None else parameters["max_length"]
                self.pooler = "entity" if parameters["pooler"] is None else parameters["pooler"]
                self.pretrain_path = "bert-base-uncased" if parameters["pretrain_path"] is None else parameters["pretrain_path"]
                self.mask_entity = True if parameters["mask_entity"] is None else parameters["mask_entity"]
                
                if model == "bert":
                        self.batch_size = 64 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 2e-5 if parameters["lr"] is None else parameters["lr"]
                        self.max_epoch = 3 if parameters["max_epoch"] is None else parameters["max_epoch"]
                else:
                        self.embedding = "glove" if parameters["embedding"] is None else parameters["embedding"]
                        self.batch_size = 160 if parameters["batch_size"] is None else parameters["batch_size"]
                        self.lr = 1e-1 if parameters["lr"] is None else parameters["lr"]
                        self.weight_decay = 1e-5 if parameters["weight_decay"] is None else parameters["weight_decay"]
                        self.max_epoch = 100 if parameters["max_epoch"] is None else parameters["max_epoch"]

                self.opt = 'adamw' if self.model == 'bert' else 'sgd'

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
                
                if self.dataset != None:
                        opennre.download(self.dataset, root_path=root_path)
                        self.train_file = os.path.join(root_path, 'benchmark', self.dataset, 'ner_blinding', '{}_train_ner_blinding.txt'.format(self.dataset))#os.path.join(root_path, 'benchmark', self.dataset, '{}_train.txt'.format(self.dataset))
                        print(self.train_file)
                        self.val_file = os.path.join(root_path, 'benchmark', self.dataset, 'ner_blinding', '{}_val_ner_blinding.txt'.format(self.dataset))#os.path.join(root_path, 'benchmark', self.dataset, '{}_val.txt'.format(self.dataset))
                        self.test_file = os.path.join(root_path, 'benchmark', self.dataset, 'ner_blinding', '{}_test_ner_blinding.txt'.format(self.dataset))#os.path.join(root_path, 'benchmark', self.dataset, '{}_test.txt'.format(self.dataset))
                        self.rel2id_file = os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))#os.path.join(root_path, 'benchmark', self.dataset, '{}_rel2id.json'.format(self.dataset))
                        # if args.dataset == 'wiki80':
                        #     args.metric = 'acc'
                        # else:
                        #     args.metric = 'micro_f1'
                else:
                        if not (os.path.exists(self.train_file) and os.path.exists(self.val_file) and os.path.exists(self.test_file) and os.path.exists(self.rel2id_file)):
                                raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')
                        
                rel2id = json.load(open(self.rel2id_file))

                if self.embedding == "glove":
                        # Download glove
                        opennre.download('glove', root_path=root_path)
                        word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
                        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

                # Define the sentence encoder
                if self.model == "cnn":
                        sentence_encoder = opennre.encoder.CNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=50,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=0.5
                        )


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "pcnn":
                        sentence_encoder = opennre.encoder.PCNNEncoder(
                                token2id=word2id,
                                max_length=self.max_length,
                                word_size=50,
                                position_size=5,
                                hidden_size=230,
                                blank_padding=True,
                                kernel_size=3,
                                padding_size=1,
                                word2vec=word2vec,
                                dropout=0.5
                        )


                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                elif self.model == "bert":
                        if self.pooler == 'entity':
                                sentence_encoder = opennre.encoder.BERTEntityEncoder(
                                        max_length=self.max_length, 
                                        pretrain_path=self.pretrain_path,
                                        mask_entity=self.mask_entity
                                )
                        elif self.pooler == 'cls':
                                sentence_encoder = opennre.encoder.BERTEncoder(
                                        max_length=self.max_length, 
                                        pretrain_path=self.pretrain_path,
                                        mask_entity=self.mask_entity
                                )
                        else:
                                raise NotImplementedError

                        # Define the model
                        self.model_opennre = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
                        
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
                        opt=self.opt
                )

                # Train the model
                #if not self.only_test:
                framework.train_model(self.metric)
                        
                # Test
                framework.load_state_dict(torch.load(self.ckpt)['state_dict'])
                result, _, _ = framework.eval_model(framework.test_loader)
                
                #framework.get_confusion_matrix(ground_truth, pred, model, embedding)

                # Print the result
                #framework.test_set_results(ground_truth, pred, result, model, embedding)
                
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
        parser.add_argument('--model', default='cnn', choices=['cnn', 'pcnn', 'bert'],
                help='Model to train')

        #Embedding
        parser.add_argument('--embedding', default='glove', choices=['glove', 'senna', 'elmo'],
                help='Word Embedding')

        # Data
        parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                help='Metric for picking up best checkpoint')
        parser.add_argument('--dataset', default=None, choices=constants.datasets_choices, 
                help='Dataset. If not none, the following args can be ignored')
        parser.add_argument('--preprocessing', default='none', choices=constants.preprocessing_choices, 
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
