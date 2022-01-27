import os
import json
import argparse
import itertools
from opennre import config

import optuna

from opennre.framework.train import Training

class Optimizer():
    def __init__(self, dataset, metric, trials=50, opt_type='hyperparams'):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
        
        self.study_model = optuna.create_study(direction="maximize")
        self.study_params = optuna.create_study(direction="maximize")
        
        self.preprocessing = config.PREPROCESSING_TYPES
        self.preprocessing = config.PREPROCESSING_COMBINATION
        
        self.params = None
        self.value = 0
        self.best_result = None
        
        if opt_type == 'model':
            self.params = self.optimize_model()
        elif opt_type == 'hyperparams':
            self.params = self.optimize_hyperparameters()

    def evaluate_hyperparameters(self, trial):
        
        batch_size =  trial.suggest_int("batch_size", 2, 64, log=True)
        lr =  trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        max_length =  trial.suggest_int("max_length", 16, 256, log=True)
        max_epoch = trial.suggest_int("max_epoch", 2, 8)
        
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        parameters["batch_size"] = batch_size
        parameters["lr"] = lr
        parameters["max_length"] = max_length
        parameters["max_epoch"] = max_epoch
        
        print("parameters:",parameters)
        
        train = Training(self.dataset, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
            self.best_result = result
        
        return new_value
    
    def evaluate_model(self, trial):
        
        model = trial.suggest_categorical("model", config.MODELS)
        embedding = trial.suggest_categorical("embedding", config.PRETRAIN_WEIGHTS) if model == 'bert' else trial.suggest_categorical("embedding", config.EMBEDDINGS)
        
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        parameters["model"] = model
        parameters["embedding"] = embedding
        
        print("parameters:",parameters)
        
        train = Training(self.dataset, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
            self.best_result = result
        
        return new_value

    def optimize_model(self):
        self.study_model.optimize(self.evaluate_model, n_trials=self.trials)
    
        params = self.study_model.best_params
        
        return params
        
    def optimize_hyperparameters(self):
        self.study_params.optimize(self.evaluate_hyperparameters, n_trials=self.trials)
    
        params = self.study_params.best_params
        
        return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=config.METRICS, 
                help='Metric to optimize')
    parser.add_argument('-t','--trials', default=50, help='Number of trials to optimize')
    parser.add_argument('-o', '--optimizer_type', default='hyperparams', choices=['hyperparams', 'model'],
                help="Optimization type")
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric, int(args.trials), args.optimizer_type)
    best_result = opt.best_result
    best_hparams = opt.best_hparams
    
    hof = opt.params
    print("Best params:",hof)
    print("Best result:",best_result)
    
    