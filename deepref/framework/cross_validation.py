import json
import math
import random
import argparse

from tqdm import tqdm

from deepref import config
from deepref.framework.train import Training
from deepref.dataset.dataset import Dataset

class CrossValidation():
    def __init__(self, dataset:Dataset, folds:int=10):
        self.dataset = dataset
        self.folds = folds
        self.training_sets_splitted = []
        self.validation_sets_splitted = []
        self.metrics = []
        
    def set_seed(self, seed):
        random.seed(seed)
        
    def split_random(self):
        self.set_seed(config.SEED)
        train_sentences = self.dataset.train_sentences + self.dataset.val_sentences
        random.shuffle(train_sentences)
        val_sentences_length = len(train_sentences) // self.folds
        for f in range(self.folds):
            self.validation_sets_splitted.append(train_sentences[f*val_sentences_length:(f+1)*val_sentences_length])
            self.training_sets_splitted.append(train_sentences[:f*val_sentences_length]+train_sentences[(f+1)*val_sentences_length:])
        
        
    def validate(self, hparams):
        self.split_random()
        accuracies, micro_f1, macro_f1 = [], [], []
        for f in tqdm(range(self.folds)):
            self.dataset.train_sentences = self.training_sets_splitted[f]
            self.dataset.val_sentences = self.validation_sets_splitted[f]
            
            assert (len(self.dataset.val_sentences) + len(self.dataset.train_sentences)) // self.folds == len(self.dataset.val_sentences)
            
            train = Training(self.dataset, hparams)
            result = train.train()
            accuracies.append(result["acc"])
            micro_f1.append(result["micro_f1"])
            macro_f1.append(result["macro_f1"])
            
        avg_acc = sum(accuracies) / self.folds
        sd_acc = math.sqrt(sum([math.pow(acc - avg_acc, 2) for acc in accuracies]) / self.folds)
        avg_micro_f1 = sum(micro_f1) / self.folds
        sd_micro_f1 = math.sqrt(sum([math.pow(m - avg_micro_f1, 2) for m in micro_f1]) / self.folds)
        avg_macro_f1 = sum(macro_f1) / self.folds
        sd_macro_f1 = math.sqrt(sum([math.pow(m - avg_macro_f1, 2) for m in macro_f1]) / self.folds)
        
        return {"acc": {"avg":avg_acc, "sd": sd_acc}, "micro_f1": {"avg": avg_micro_f1, "sd": sd_micro_f1}, "macro_f1": {"avg": avg_macro_f1, "sd": sd_macro_f1}}
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # # Data
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset. If not none, the following args can be ignored')

    args = parser.parse_args()
    
    with open(config.HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        hparams = json.load(f)
        
    dataset = Dataset(args.dataset)
    dataset.load_dataset_csv()
    
    cv = CrossValidation(dataset)
    results = cv.validate(hparams)
    print(results)
            
            