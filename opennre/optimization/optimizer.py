import os
import json

from opennre import config
from opennre.dataset.dataset import Dataset

class Optimizer():
    def __init__(self, dataset:Dataset, metric:str, trials:int=50, cross_validation:bool=False):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        self.cross_validation = cross_validation
        
        self.params = None
        self.best_metric_value = 0
        self.best_result = None
        
        if not os.path.exists(config.HPARAMS_FILE_PATH.format(dataset.name)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.HPARAMS_FILE_PATH.format(dataset.name), 'w') as f:
                f.write(json_object)
        with open(config.HPARAMS_FILE_PATH.format(dataset.name), 'r') as f:
            self.hparams = json.load(f)
        