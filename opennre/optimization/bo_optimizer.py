import os
import json
import argparse
import itertools
from opennre import config

import optuna

from opennre.framework.train import Training

class Optimizer():
    def __init__(self, dataset, metric, trials=50):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        if not os.path.exists(config.HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(config.HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.hparams = json.load(f)
        
        self.study = optuna.create_study(direction="maximize")
        
        self.params = None
        self.best_metric_value = 0
        self.best_result = None

    def objective(self, trial):
        
        batch_size =  trial.suggest_int("batch_size", 2, 64, log=True)
        lr =  trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        max_length =  trial.suggest_int("max_length", 16, 256, log=True)
        max_epoch = trial.suggest_int("max_epoch", 2, 8)
        
        parameters = self.hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        parameters["batch_size"] = batch_size
        parameters["lr"] = lr
        parameters["max_length"] = max_length
        parameters["max_epoch"] = max_epoch
        
        print("parameters:",parameters)
        
        train = Training(self.dataset, parameters, trial)
        result = train.train()
        result_value = result[self.metric]
        
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
    parser.add_argument('-t','--trials', default=50, help='Number of trials to optimize')
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric, int(args.trials))
    best_result = opt.best_result
    best_hparams = opt.best_hparams
    
    hof = opt.params
    print("Best params:",hof)
    print("Best result:",best_result)
    
    