import os, sys
import subprocess
import argparse
import opennre
from opennre import constants
import logging

class Parser():

    def __init__(self, args):
        self.args = args

    def init_args(self, model):
        # Some basic settings
        root_path = '.'
        sys.path.append(root_path)
        if not os.path.exists('ckpt'):
            os.mkdir('ckpt')
        if len(self.args.ckpt) == 0:
            if model == 'cnn' or model == 'pcnn':
                self.args.ckpt = '{}_{}'.format(self.args.dataset, model)
            elif model == 'bert':
                self.args.ckpt = '{}_{}_{}'.format(self.args.dataset, self.args.pretrain_path, self.args.pooler)
        ckpt = 'ckpt/{}.pth.tar'.format(self.args.ckpt)

        if self.args.preprocessing == 'none':
                self.args.preprocessing = 'original'

        if self.args.dataset != 'none':
            opennre.download(self.args.dataset, root_path=root_path)
            self.args.train_file = os.path.join(root_path, 'benchmark', self.args.dataset, self.args.preprocessing, '{}_train_{}.txt'.format(self.args.dataset, self.args.preprocessing))
            self.args.val_file = os.path.join(root_path, 'benchmark', self.args.dataset, self.args.preprocessing, '{}_val_{}.txt'.format(self.args.dataset, self.args.preprocessing))
            self.args.test_file = os.path.join(root_path, 'benchmark', self.args.dataset, self.args.preprocessing, '{}_test_{}.txt'.format(self.args.dataset, self.args.preprocessing))
            if not os.path.exists(self.args.test_file):
                logging.warn("Test file {} does not exist! Use val file instead".format(self.args.test_file))
                self.args.test_file = self.args.val_file
            self.args.rel2id_file = os.path.join(root_path, 'benchmark', self.args.dataset, '{}_rel2id.json'.format(self.args.dataset))
            if self.args.dataset == 'wiki80':
                self.args.metric = 'acc'
            else:
                self.args.metric = 'micro_f1'
        else:
            if not (os.path.exists(self.args.train_file) and os.path.exists(self.args.val_file) and os.path.exists(self.args.test_file) and os.path.exists(self.args.rel2id_file)):
                raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

        logging.info('Arguments:')
        for arg in vars(self.args):
            logging.info('    {}: {}'.format(arg, getattr(self.args, arg)))

        return self.args, ckpt



if __name__ == '__main__':
    for dataset in constants.datasets_choices:
        if dataset == 'semeval2010':
            pretrain_choices = constants.pretrain_choices[:-2]
        else:
            pretrain_choices = constants.pretrain_choices
        for model in constants.model_choices:
            if model == 'bert':
                for pretrain in pretrain_choices:
                    for preprocessing in constants.preprocessing_choices:
                        subprocess.call(['python', 'example/train_supervised_bert.py', '--dataset', dataset, '--preprocessing', preprocessing, '--pretrain_path', pretrain])
            else:
                for preprocessing in constants.preprocessing_choices:
                    subprocess.call(['python', 'example/train_supervised_{}.py'.format(model), '--dataset', dataset, '--preprocessing', preprocessing])