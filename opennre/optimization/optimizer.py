import os
import json

from opennre import config

class Optimizer():
    def __init__(self, dataset, metric, trials=50):
        self.dataset = dataset
        self.metric = metric
        self.trials = trials
        
        self.params = None
        self.best_metric_value = 0
        self.best_result = None
        
        if not os.path.exists(config.HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        with open(config.HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.hparams = json.load(f)
        