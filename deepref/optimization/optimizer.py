import os
import json

from deepref import config
from deepref.dataset.dataset import Dataset

class Optimizer():
    def __init__(self, dataset:str, metric:str, trials:int=50, cross_validation:bool=False):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        self.cross_validation = cross_validation
        
        self.params = None
        self.best_metric_value = 0
        self.best_result = None
        
        if not os.path.exists(config.HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            json_object = json.dumps(dict, indent=4)
            with open(config.HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(config.HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.hparams = json.load(f)
        