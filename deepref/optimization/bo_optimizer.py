import os
import json
import argparse
import itertools
from random import seed
from deepref import config

import optuna
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate

import plotly

from deepref.framework.train import Training
from deepref.framework.cross_validation import CrossValidation
from deepref.optimization.optimizer import Optimizer
from deepref.dataset.dataset import Dataset

class BOOptimizer(Optimizer):
    def __init__(self, dataset:str, metric:str, trials:int=50, cross_validation:bool=False):
        super(BOOptimizer, self).__init__(dataset, metric, trials, cross_validation)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )

    def objective(self, trial):
        
        batch_size =  trial.suggest_int("batch_size", 2, 64, log=True)
        lr =  trial.suggest_float("lr", 1e-7, 1e-4, log=True)
        #max_length =  trial.suggest_int("max_length", 16, 128, log=True)
        max_epoch = trial.suggest_int("max_epoch", 2, 10)
        
        parameters = self.hparams
        parameters["dataset"] = self.dataset
        parameters["batch_size"] = batch_size
        parameters["lr"] = lr
        #parameters["max_length"] = max_length
        parameters["max_epoch"] = max_epoch
        
        print("parameters:",parameters)
        
        # if self.cross_validation:
        #     cv = CrossValidation(self.dataset)
        #     result = cv.validate(self.hparams)
        # else:
        train = Training(self.dataset, parameters, trial)
        result = train.train()
        result_value = result[self.metric]
        # if "avg" in result_value:
        #     result_value = result_value["avg"]
        
        if result_value > self.best_metric_value:
            self.best_metric_value = result_value
            self.best_result = result
        
        return result_value
        
    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.trials)
        
        fig1 = plot_param_importances(self.study, target_name=self.metric)
        fig2 = plot_parallel_coordinate(self.study, target_name=self.metric)
        RESULT_PATH = f'results/{self.dataset}/'
        fig1.write_image(RESULT_PATH+f'/{self.dataset}_param_inportances.png')
        fig2.write_image(RESULT_PATH+f'/{self.dataset}_parallel_coordinate.png')
    
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
    # parser.add_argument('-cv','--cross_validation', action='store_true', 
    #                     help='Optimize using cross validation')
    
    args = parser.parse_args()
    
    opt = BOOptimizer(args.dataset, args.metric, int(args.trials))
    best_hparams = opt.optimize()
    best_result = opt.best_result
    
    print("Best params:",best_hparams)
    print("Best result:",best_result)
    
    