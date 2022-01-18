



import json
from opennre import config
from opennre.framework.train import Training

with open(config.BEST_HPARAMS_FILE_PATH.format('semeval2010'), 'r') as f:
    best_hparams = json.load(f)

train = Training('semeval2010', 'micro_f1', best_hparams)
train.train()





