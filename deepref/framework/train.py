# coding:utf-8
import torch
import numpy as np
import json
import os
import sys
import argparse
import random

from deepref.dataset.re_dataset import REDataset

def __init__(self, dataset_name:str, parameters, trial=None, seed=42):
        self.dataset_name = dataset_name
        self.trial = trial

        self.preprocessing = parameters["preprocessing"]
        self.model = parameters["model"]
        self.max_length = parameters["max_length"]
        self.opt = "adamw" if self.model in ("bert_entity", "bert_cls", "ebem", "prompt_encoder") else "sgd"
        self.pretrain = parameters["pretrain"]
        self.position_embed = parameters["position_embed"]
        self.pos_tags_embed = parameters["pos_tags_embed"]
        self.deps_embed = parameters["deps_embed"]
        self.batch_size = parameters["batch_size"]
        self.lr = parameters["lr"]
        self.max_epoch = parameters["max_epoch"]
        
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
        
        # Load data
        dataset = REDataset(self.dataset_name)
        

        # Set random seed
        self.set_seed(seed)

        root_path = '.'
        sys.path.append(root_path)
        if not os.path.exists('ckpt'):
                os.mkdir('ckpt')
        ckpt = '{}_{}'.format(self.dataset_name, self.model)
        self.ckpt = 'ckpt/{}.pth.tar'.format(ckpt)

        if self.model == "bert_cls":
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
                        #upos2id=upos2id,
                        #deps2id=deps2id
                )

        elif self.model == "sentence_encoder":
                sentence_encoder = deepref.encoder.SentenceEncoder(
                        max_length=self.max_length,
                        pretrain_path=self.pretrain
                )

        elif self.model == "prompt_encoder":
                sentence_encoder = deepref.encoder.PromptEntityEncoder(
                        max_length=self.max_length,
                        pretrain_path=self.pretrain
                )

        elif self.model == "sdp_encoder":
                sentence_encoder = deepref.encoder.SDPEncoder(
                        max_length=self.max_length,
                        pretrain_path=self.pretrain
                )

        # Define the model
        self.model_deepref = deepref.model.SoftmaxMLP(sentence_encoder, len(rel2id), rel2id)
        
        self.criterion = deepref.model.PairwiseRankingLoss() if self.model == 'crcnn' else None
        
def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
                
def train(self):

        # Define the whole training framework
        framework = deepref.framework.SentenceRETrainer(
                train_dataset=self.train_file,
                test_dataset=self.test_file,
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
