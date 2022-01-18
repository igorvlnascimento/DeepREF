import itertools
import argparse
import json
import os
from opennre import config

from opennre.framework.train import Training

import optuna

class PreprocessOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.preprocessing = config.PREPROCESSING_TYPES
        self.preprocess_combination = config.PREPROCESSING_COMBINATION
        
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
        self.study_prep = optuna.create_study(direction="maximize")
        
        self.study_prep.optimize(self.preprocessing_optimization, n_trials=len(config.PREPROCESSING_COMBINATION))
        
        self.best_prep = self.study_prep.best_params["preprocessing"]
        self.best_prep_value = self.study_prep.best_value
        
        self.value = 0
        self.best_result = None
        
    def preprocessing_optimization(self, trial):
        
        parameters = self.best_hparams
        
        preprocessing =  trial.suggest_int("preprocessing", 0, len(config.PREPROCESSING_COMBINATION)-1)
            
        parameters["preprocessing"] = preprocessing
        
        train = Training(self.dataset, self.metric, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
            self.best_result = result
        
        return new_value

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=config.METRICS, 
                help='Metric to optimize')
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    prep = PreprocessOptimization(dataset, metric)
    best_result = prep.best_result
    preprocessing, new_value = prep.best_prep, prep.best_prep_values
    print("Type:", prep.preprocess_combination[preprocessing], "Value:", new_value)
    
    best_hparams = {}
    with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(metric)]) if best_hparams["{}".format(metric)] else 0
    
    if new_value > json_value:
        best_hparams["preprocessing"] = preprocessing
        
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as out_f:
            out_f.write(json_object)