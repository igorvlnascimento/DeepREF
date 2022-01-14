import itertools
import argparse
import json
import os
from opennre import constants

from train import Training

import optuna

class PreprocessOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.preprocessing = constants.PREPROCESSING_TYPES
        self.preprocess_combination = constants.PREPROCESSING_COMBINATION
        
        if not os.path.exists(constants.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = {
                "{}".format(self.metric): 0,
                "model": "bert",
                "embedding": "bert-base-uncased",
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
            
        self.study_prep = optuna.create_study(direction="maximize")
        
        self.study_prep.optimize(self.preprocessing_optimization, n_trials=len(constants.PREPROCESSING_COMBINATION))
        
        self.best_prep = self.study_prep.best_params["preprocessing"]
        self.best_prep_value = self.study_prep.best_value
        
    def preprocessing_optimization(self, trial):
        
        parameters = self.best_hparams
        
        preprocessing =  trial.suggest_int("preprocessing", 0, len(constants.PREPROCESSING_COMBINATION)-1)
            
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        parameters["preprocessing"] = preprocessing
        
        train = Training(parameters,None)
        
        return train.train()    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=constants.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=constants.METRICS, 
                help='Metric to optimize')
    args = parser.parse_args()
    dataset = args.dataset
    metric = args.metric
    prep = PreprocessOptimization(dataset, metric)
    preprocessing, new_value = prep.best_prep, prep.best_prep_values
    print("Type:", prep.preprocess_combination[preprocessing], "Value:", new_value)
    
    best_hparams = {}
    with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(metric)]) if best_hparams["{}".format(metric)] else 0
    
    if new_value > json_value:
        best_hparams["preprocessing"] = preprocessing
        best_hparams["{}".format(metric)] = new_value
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(constants.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as out_f:
            out_f.write(json_object)