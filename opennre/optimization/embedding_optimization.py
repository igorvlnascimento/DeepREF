import argparse
import json
import os

from opennre import config

from opennre.framework.train import Training

EMBEDDINGS = [[0,0],[0,1],[1,0],[1,1]]

class EmbeddingOptimization():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        
        if not os.path.exists(config.BEST_HPARAMS_FILE_PATH.format(dataset)):
            dict = config.HPARAMS
            dict["{}".format(self.metric)] = 0
            json_object = json.dumps(dict, indent=4)
            with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'w') as f:
                f.write(json_object)
        self.best_hparams = {}
        with open(config.BEST_HPARAMS_FILE_PATH.format(dataset), 'r') as f:
            self.best_hparams = json.load(f)
            
    def embedding_training(self):
        parameters = self.best_hparams
        parameters["dataset"] = self.dataset
        parameters["metric"] = self.metric
        
        embedding_value = 0
        
        for embed in EMBEDDINGS:
        
            parameters["pos_embed"] = embed[0]
            parameters["deps_embed"] = embed[1]
            
            train = Training(self.dataset, parameters)
            
            result = train.train()
            new_value = result[self.metric]
            
            if new_value > self.value:
                self.value = new_value
                embed_type, self.value = embed, new_value
            
        return embed_type, self.value
    
    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=config.METRICS, 
                help='Metric to optimize')
    args = parser.parse_args()
    
    with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        best_hparams = json.load(f)
    embed = EmbeddingOptimization(args.dataset, args.metric)
    embedding, new_value = embed.embedding_training()
    print("Type:", embedding, "Value:", new_value)
    
    best_hparams = {}
    with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'r') as f:
        best_hparams = json.load(f)
        
    json_value = float(best_hparams["{}".format(args.metric)]) if best_hparams["{}".format(args.metric)] else 0
    
    
    if new_value > json_value:
        best_hparams["pos_embed"] = embedding[0]
        best_hparams["deps_embed"] = embedding[0]
        
        json_object = json.dumps(best_hparams, indent=4)
        
        with open(config.BEST_HPARAMS_FILE_PATH.format(args.dataset), 'w') as out_f:
            out_f.write(json_object)