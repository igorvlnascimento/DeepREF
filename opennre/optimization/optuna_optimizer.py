import os
import json
import argparse
import itertools
from opennre import constants

import optuna

from train import Training

class Optimizer():
    def __init__(self, dataset, metric, trials, opt_type):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        if not os.path.exists(constants.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = {
                "{}".format(self.metric): 0,
                "batch_size": 16,
                "preprocessing": 0,
                "lr": 2e-5,
                "synt_embeddings": [0,0,0],
                "max_length": 128,
                "max_epoch": 3
            }
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
        
        if opt_type == 'model':
            self.params = self.optimize_model()
        elif opt_type == 'hyperparams':
            self.params = self.optimize_hyperparameters()

    def evaluate_hyperparameters(self, individual):
        
        batch_size =  individual.suggest_int("batch_size", 2, 64, log=True)
        lr =  individual.suggest_float("lr", 1e-6, 1e-1, log=True)
        max_length =  individual.suggest_int("max_length", 16, 256, log=True)
        max_epoch = individual.suggest_int("max_epoch", 2, 8)
        
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        parameters["batch_size"] = batch_size
        parameters["lr"] = lr
        parameters["max_length"] = max_length
        parameters["max_epoch"] = max_epoch
        
        print("parameters:",parameters)
        
        train = Training(parameters, individual)
        
        return train.train()
    
    def evaluate_model(self, individual):
        
        model = individual.suggest_categorical("model", constants.MODELS)
        embedding = self.study_model.best_params["pretrain_bert"] if model == 'bert' else self.study_model.best_params["embedding"]
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
        
        train = Training(parameters, individual)
        
        return train.train()

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
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "semeval20181-1", "semeval20181-2", "ddi"], 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=["micro_f1", "macro_f1", "acc"], 
                help='Metric to optimize')
    parser.add_argument('-t','--trials', default=50, help='Number of trials to optimize')
    parser.add_argument('-o', '--optimizer_type', default='hyperparams', choices=['hyperparams', 'model'],
                help="Optimization type")
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric, args.trials, args.optimizer_type)
    best_hparams = opt.best_hparams
    
    hof_hp = opt.params
    print("Best:",hof_hp)
    
    new_value = abs(opt.study_params.best_value)
    json_value = float(best_hparams["{}".format(opt.metric)]) if best_hparams["{}".format(opt.metric)] else 0
    
    if new_value > json_value:
        max_epoch = hof_hp["max_epoch"]
        batch_size = hof_hp["batch_size"]
        lr, max_length = hof_hp["lr"], hof_hp["max_length"]
        
        best_hparams["max_epoch"] = max_epoch
        best_hparams["batch_size"] = batch_size
        best_hparams["lr"] = lr
        best_hparams["max_length"] = max_length
        best_hparams["{}".format(opt.metric)] = new_value
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(constants.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
            out_f.write(json_object)
    