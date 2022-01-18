import os
import json
import argparse
import itertools
from opennre import constants

import optuna

from opennre.framework.train import Training

class Optimizer():
    def __init__(self, dataset, metric, trials, opt_type):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        if not os.path.exists(constants.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = constants.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
        
        self.study_model = optuna.create_study(direction="maximize")
        self.study_params = optuna.create_study(direction="maximize")
        
        self.preprocessing = constants.PREPROCESSING_TYPES
        self.preprocessing = constants.PREPROCESSING_COMBINATION
        
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
        
        
        
        train = Training(self.dataset, self.metric, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
            self.best_result = result
        
        return new_value
    
    def evaluate_model(self, trial):
        
        model = trial.suggest_categorical("model", constants.MODELS)
        embedding = trial.suggest_categorical("embedding", constants.PRETRAIN_WEIGHTS) if model == 'bert' else trial.suggest_categorical("embedding", constants.EMBEDDINGS)
        preprocessing = 0
        batch_size =  3 if model == 'bert' else 160
        lr =  2e-5 if model == 'bert' else 1e-1
        max_length =  128
        max_epoch = 64 if model == 'bert' else 100
    
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": preprocessing,
            "embedding": embedding,
            "batch_size": batch_size,
            "lr": lr,
            "max_length": max_length,
            "max_epoch": max_epoch,
        }
        
        train = Training(self.dataset, self.metric, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
            self.best_result = result
        
        return result

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
    parser.add_argument('-d','--dataset', default="semeval2010", choices=constants.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=constants.METRICS, 
                help='Metric to optimize')
    parser.add_argument('-t','--trials', default=50, help='Number of trials to optimize')
    parser.add_argument('-o', '--optimizer_type', default='hyperparams', choices=['hyperparams', 'model'],
                help="Optimization type")
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric, int(args.trials), args.optimizer_type)
    best_result = opt.best_result
    best_hparams = opt.best_hparams
    
    hof = opt.params
    print("Best:",hof)
    
    new_value = abs(opt.study_params.best_value)
    json_value = float(best_hparams["{}".format(opt.metric)]) if best_hparams["{}".format(opt.metric)] else 0
    
    if new_value > json_value:
        if args.optimizer_type == 'hyperparams':
            max_epoch = hof["max_epoch"]
            batch_size = hof["batch_size"]
            lr, max_length = hof["lr"], hof["max_length"]
            
            best_hparams["max_epoch"] = max_epoch
            best_hparams["batch_size"] = batch_size
            best_hparams["lr"] = lr
            best_hparams["max_length"] = max_length
            
        elif args.optimizer_type == 'model':
            model = hof["model"]
            embedding = hof["embedding"]
            
            best_hparams["model"] = model
            best_hparams["embedding"] = embedding
        
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(constants.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
            out_f.write(json_object)
    