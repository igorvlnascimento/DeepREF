# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import model, framework, constants
import os
import argparse
from parser import Parser

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=constants.datasets_choices, 
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

args, ckpt = Parser(args).init_args('cnn')

rel2id = json.load(open(args.rel2id_file))

root_path = '.'

# Download glove
opennre.download('glove', root_path=root_path)
word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.CNNEncoder(
    token2id=word2id,
    max_length=args.max_length,
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
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt='sgd'
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result, pred, ground_truth = framework.eval_model(framework.test_loader)

framework.get_confusion_matrix(ground_truth, pred, 'cnn', 'glove')

# Print the result
framework.test_set_results(ground_truth, pred, result, 'cnn', 'glove')

#if args.metric == 'acc':
#    logging.info('Accuracy: {}'.format(result['acc']))
#else:
#    logging.info('Micro precision: {}'.format(result['micro_p']))
#    logging.info('Micro recall: {}'.format(result['micro_r']))
#    logging.info('Micro F1: {}'.format(result['micro_f1']))
#
