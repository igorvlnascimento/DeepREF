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
        self.value = 0
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
        
    def preprocessing_optimization(self, trial):
        
        parameters = self.best_hparams
        
        preprocessing =  trial.suggest_int("preprocessing", 0, len(config.PREPROCESSING_COMBINATION)-1)
            
        parameters["preprocessing"] = preprocessing
        
        train = Training(self.dataset, parameters, trial)
        result = train.train()
        new_value = result[self.metric]
        
        if new_value > self.value:
            self.value = new_value
        
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
    preprocessing, new_value = prep.best_prep, prep.best_prep_value
    print("Type:", prep.preprocess_combination[preprocessing], "Value:", new_value)
    