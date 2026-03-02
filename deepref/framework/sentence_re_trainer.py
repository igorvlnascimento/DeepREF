import logging
import os
from typing import Any, Dict

from torch import nn, optim
import torch
from tqdm import tqdm

from deepref.dataset.re_dataset import REDataset, RELoader
from deepref.framework.early_stopping import EarlyStopping
from deepref.framework.utils import AverageMeter
from deepref.model.base_model import SentenceRE



class SentenceRETrainer(nn.Module):
    def __init__(self, 
                 model: SentenceRE, 
                 train_dataset: REDataset, 
                 test_dataset: REDataset,
                 ckpt: str,
                 training_parameters: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        self.training_parameters = training_parameters

        self.max_epoch = training_parameters["max_epoch"]
        self.criterion = training_parameters["criterion"]
        self.lr = training_parameters["lr"]

        batch_size = training_parameters["batch_size"]

        self.train_loader = RELoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = RELoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Params and optimizer
        opt = training_parameters["opt"]
        weight_decay = training_parameters["weight_decay"]
        lr = training_parameters["lr"]

        params = self.parameters()
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from torch.optim import AdamW
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
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")

        # Warmup
        warmup_step = training_parameters["warmup_step"]
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = len(train_dataset) // batch_size * self.max_epoch
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
        patience = self.training_parameters.get("patience", 0)
        early_stopper = EarlyStopping(patience=patience)
        for epoch in range(self.max_epoch):
            logging.info("=== Epoch %d train ===" % epoch)
            self.iterate_loader(self.train_loader,
                                warmup=warmup,
                                training=True)

            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            result, _, _  = self.eval_model(self.test_loader)
            #self.results_per_epoch(ground_truth, pred_labels, result, epoch)
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))

            improved = result[metric] > best_metric
            if improved:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]

            if early_stopper.step(improved):
                logging.info(
                    "Early stopping triggered after epoch %d (no improvement for %d epochs)",
                    epoch, patience,
                )
                break
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        with torch.no_grad():
            results = self.iterate_loader(eval_loader, warmup=False, training=False)
        result, pred_result, ground_truth = results
        return result, pred_result, ground_truth
    
    def iterate_loader(self, loader, warmup=True, global_step=0, training=True):
        global_step = 0
        t = tqdm(loader)
        pred_result = []
        ground_truth = []
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        for _, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data["labels"]
            args = {k: v for k, v in data.items() if k != "labels"}
            logits = self.parallel_model(**args)
            _, pred = logits.max(-1) # (B)
            # Log
            acc = float((pred == label).long().sum()) / label.size(0)
            avg_acc.update(acc, 1)
            t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
            if training:
                loss = self.criterion(logits, label)
                avg_loss.update(loss.item(), 1)

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
            else:
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                for i in range(label.size(0)):
                    ground_truth.append(label[i].item())
        if not training:
            result = loader.dataset.eval(pred_result)
            return result, pred_result, ground_truth
        