import os
import json
import argparse
import itertools
from opennre import config

import optuna

from opennre.framework.train import Training
from opennre.framework.cross_validation import CrossValidation
from opennre.optimization.optimizer import Optimizer
from opennre.dataset.dataset import Dataset

class BOOptimizer(Optimizer):
    def __init__(self, dataset:Dataset, metric:str, trials:int=50, cross_validation:bool=False):
        super(BOOptimizer, self).__init__(dataset, metric, trials, cross_validation)
        
        self.study = optuna.create_study(direction="maximize")

    def objective(self, trial):
        
        batch_size =  trial.suggest_int("batch_size", 2, 64, log=True)
        lr =  trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        max_length =  trial.suggest_int("max_length", 16, 256, log=True)
        max_epoch = trial.suggest_int("max_epoch", 2, 8)
        
        parameters = self.hparams
        parameters["dataset"] = self.dataset.name
        parameters["metric"] = self.metric
        parameters["batch_size"] = batch_size
        parameters["lr"] = lr
        parameters["max_length"] = max_length
        parameters["max_epoch"] = max_epoch
        
        print("parameters:",parameters)
        
        if self.cross_validation:
            cv = CrossValidation(self.dataset)
            result = cv.validate(self.hparams)
        else:
            train = Training(self.dataset, parameters, trial)
            result = train.train()
        result_value = result[self.metric]
        if "avg" in result_value:
            result_value = result_value["avg"]
        
        if result_value > self.best_metric_value:
            self.best_metric_value = result_value
            self.best_result = result
        
        return self.best_metric_value
        
    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.trials)
    
        params = self.study.best_params
        
        return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=config.METRICS, 
                help='Metric to optimize')
    parser.add_argument('-t','--trials', default=50, 
                        help='Number of trials to optimize')
    parser.add_argument('-cv','--cross_validation', action='store_true', 
                        help='Optimize using cross validation')
    
    args = parser.parse_args()
    
    dataset = Dataset(args.dataset)
    dataset.load_dataset_csv()
    opt = BOOptimizer(dataset, args.metric, int(args.trials), args.cross_validation)
    best_result = opt.best_result
    best_hparams = opt.optimize()
    
    print("Best params:",best_hparams)
    print("Best result:",best_result)
    
    