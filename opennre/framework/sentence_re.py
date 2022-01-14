import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from datetime import datetime

from sklearn import metrics
import numpy as np

import optuna

from opennre import constants

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 trial,
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd',
                 criterion=None,):
    
        super().__init__()
        self.trial = trial
        self.max_epoch = max_epoch
        # Load data
        self.train_path = train_path
        self.dataset_name = train_path[train_path.rfind('/')+1:train_path.find('_', train_path.rfind('/'))]
        self.preprocessing = train_path[train_path.rfind('/', 0, -(len(train_path)-train_path.rfind('/')))+1:train_path.rfind('/')]
        
        with open(constants.CLASSES_DATASET, 'r') as f:
            classes_dataset = json.load(f)

        self.classes = classes_dataset[self.dataset_name]
            
        print("classes:",self.classes)

        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, warmup=True, metric='acc'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                logits = None 
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                if warmup == True:
                    warmup_step = 300
                    if global_step < warmup_step:
                        warmup_rate = float(global_step) / warmup_step
                    else:
                        warmup_rate = 1.0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * warmup_rate
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            logging.info("=== Epoch %d val ===" % epoch)
            result, _, _ = self.eval_model(self.val_loader)
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if self.trial is not None:
                self.trial.report(result[metric],epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        ground_truth = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                for i in range(label.size(0)):
                    ground_truth.append(label[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result, pred_result, ground_truth

    def test_set_results(self, ground_truth, pred, result, model, embedding, hyper_params):
        embedding = embedding.replace('/', '-').replace('.', '')
        logging.info('Test set results:')
        logging.info('Trained with dataset {}, model {}, embedding {} and preprocessing {}:\n'.format(self.dataset_name, model, embedding, self.preprocessing))
        logging.info('Hyperparams: {}'.format(hyper_params))
        file_path = 'results/{}/ResultsOpenNRE++_{}_{}.txt'.format(self.dataset_name, self.dataset_name, datetime.now().isoformat(timespec="auto"))
        report = metrics.classification_report(ground_truth, pred, target_names=self.classes, digits=5, zero_division=1)
        confusion_matrix = metrics.confusion_matrix(ground_truth, pred)
        logging.info(report)
        logging.info('Accuracy: {}'.format(result['acc']))
        logging.info('Micro precision: {}'.format(result['micro_p']))
        logging.info('Micro recall: {}'.format(result['micro_r']))
        logging.info('Micro F1: {}'.format(result['micro_f1']))
        logging.info('Macro F1: {}'.format(result['macro_f1']))
        os.makedirs(os.path.join(constants.RESULTS_PATH, self.dataset_name), exist_ok=True)
        if os.path.isfile(file_path):
            with open(file_path, 'a') as ablation_file:
                self.write_test_results(ablation_file, model, embedding, hyper_params, result, report, confusion_matrix)
        else:
            with open(file_path, 'w') as ablation_file:
                self.write_test_results(ablation_file, model, embedding, hyper_params, result, report, confusion_matrix)

    def write_test_results(self, file, model, embedding, hyper_params, result, report, confusion_matrix):
        embedding = embedding.replace('/', '-').replace('.', '')
        file.write('Trained with dataset {}, model {}, embedding {} and preprocessing {}:\n'.format(self.dataset_name, model, embedding, self.preprocessing))
        file.write('Hyperparams: {}'.format(hyper_params))
        file.write('Confusion matrix:\n')
        file.write(np.array2string(confusion_matrix)+'\n')
        file.write('Test set results:\n')
        file.write(report+"\n")
        file.write('Accuracy: {}\n'.format(result['acc']))
        file.write('Micro precision: {}\n'.format(result['micro_p']))
        file.write('Micro recall: {}\n'.format(result['micro_r']))
        file.write('Micro F1: {}\n\n'.format(result['micro_f1']))
        file.write('Macro F1: {}\n\n'.format(result['macro_f1']))


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

