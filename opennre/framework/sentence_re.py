import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CLASSES = ['Component-Whole(e2,e1)',
            'Other',
            'Instrument-Agency(e2,e1)',
            'Member-Collection(e1,e2)',
            'Cause-Effect(e2,e1)',
            'Entity-Destination(e1,e2)',
            'Content-Container(e1,e2)',
            'Message-Topic(e1,e2)',
            'Product-Producer(e2,e1)',
            'Member-Collection(e2,e1)',
            'Entity-Origin(e1,e2)',
            'Cause-Effect(e1,e2)',
            'Component-Whole(e1,e2)',
            'Message-Topic(e2,e1)',
            'Product-Producer(e1,e2)',
            'Entity-Origin(e2,e1)',
            'Content-Container(e2,e1)',
            'Instrument-Agency(e1,e2)',
            'Entity-Destination(e2,e1)']

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd'):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
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
        self.criterion = nn.CrossEntropyLoss()
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

    def get_confusion_matrix(self, ground_truth, pred_result, image_output_name, only_test=False, output_format='png'):
        c_matrix = confusion_matrix(ground_truth, pred_result)

        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                        display_labels=CLASSES)

        disp.plot(include_values=True, xticks_rotation='vertical')

        if only_test:
            image_output_name = "{}_only_test.{}".format(image_output_name, output_format)

        save_path = os.path.join('results', image_output_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()

    def test_set_results(self, ground_truth, pred, result):
        logging.info('Test set results:')
        report = metrics.classification_report(ground_truth, pred, labels=[i for i in range(19)])
        confusion_matrix = metrics.confusion_matrix(ground_truth, pred)
        logging.info(report)
        #logging.info('Accuracy: {}'.format(result['acc']))
        logging.info('Micro precision: {}'.format(result['micro_p']))
        logging.info('Micro recall: {}'.format(result['micro_r']))
        logging.info('Micro F1: {}'.format(result['micro_f1']))
        with open('AblationStudiesOpenNRE+.txt', 'w') as ablation_file:
            for i in CLASSES:
                ablation_file.write("{}\t\t{}\n\n".format(CLASSES[i], confusion_matrix[i]))
            ablation_file.write(report)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

