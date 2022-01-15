import random
import numpy
import json
import argparse
import itertools

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from opennre.framework.train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"
BEST_HPARAMS_FILE_PATH = "opennre/optimization/best_hparams.json"

class Optimizer():
    def __init__(self, dataset, metric):
        self.dataset = dataset
        self.metric = metric
        self.data = json.load(open(CONFIG_FILE_PATH))
        self.best_hparams = json.load(open(BEST_HPARAMS_FILE_PATH))
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.N_CYCLES = 1
    
        self.hof_model = []
        
        self.init_toolbox_model()
        
        self.init_toolbox_hyperparameters()
        
        self.final_combinations = []
        
        self.preprocessing = self.data["preprocessing"]
        self.preprocessing = self.combine_preprocessing(self.preprocessing)
        
        self.synt_embeddings = [[0,0], [0,1], [1,0], [1,1]]
        
    def init_toolbox_model(self):
        
        self.toolbox_model = base.Toolbox()

        self.toolbox_model.register("attr_preprocessing", random.randint, 0, len(self.preprocessing))
        self.toolbox_model.register("attr_model", random.randint, 0, len(self.data["model"])-1)
        self.toolbox_model.register("attr_embedding", random.randint, 0, len(self.data["embedding"])-1)
        self.toolbox_model.register("attr_pretrain_bert", random.randint, 0, len(self.data["pretrain_bert"])-1)
        self.toolbox_model.register("individual", tools.initCycle, creator.Individual,
                        (self.toolbox_model.attr_preprocessing, self.toolbox_model.attr_model, self.toolbox_model.attr_embedding,
                         self.toolbox_model.attr_pretrain_bert), n=self.N_CYCLES)
        self.toolbox_model.register("population", tools.initRepeat, list, self.toolbox_model.individual)
        
        self.toolbox_model.register("evaluate", self.evaluate)
        self.toolbox_model.register("mate", tools.cxTwoPoint)
        self.toolbox_model.register("mutate", tools.mutFlipBit, indpb=0.05)    
        self.toolbox_model.register("select", tools.selTournament, tournsize=3)
        
    def init_toolbox_hyperparameters(self):
        
        self.toolbox_hyperparameters = base.Toolbox()

        self.toolbox_hyperparameters.register("attr_batch_size", random.randint, 0, len(self.data["batch_size"])-1)
        self.toolbox_hyperparameters.register("attr_lr", random.randint, 0, len(self.data["lr"])-1)
        self.toolbox_hyperparameters.register("attr_weight_decay", random.randint, 0, len(self.data["weight_decay"])-1)
        self.toolbox_hyperparameters.register("attr_max_length", random.randint, 0, len(self.data["max_length"])-1)
        self.toolbox_hyperparameters.register("attr_max_epoch", random.randint, 0, len(self.data["max_epoch"])-1)
        self.toolbox_hyperparameters.register("individual", tools.initCycle, creator.Individual,
                        (self.toolbox_hyperparameters.attr_batch_size, self.toolbox_hyperparameters.attr_lr,
                         self.toolbox_hyperparameters.attr_weight_decay, self.toolbox_hyperparameters.attr_max_length,
                         self.toolbox_hyperparameters.attr_max_epoch), n=self.N_CYCLES)
        self.toolbox_hyperparameters.register("population", tools.initRepeat, list, self.toolbox_hyperparameters.individual)
        
        self.toolbox_hyperparameters.register("evaluate", self.evaluate_hyperparameters)
        self.toolbox_hyperparameters.register("mate", tools.cxTwoPoint)
        self.toolbox_hyperparameters.register("mutate", tools.mutFlipBit, indpb=0.05)    
        self.toolbox_hyperparameters.register("select", tools.selTournament, tournsize=3)
        
    def combine_preprocessing(self, preprocessing):
        combinations = []
        for i in range(len(preprocessing)):
            combinations.extend(itertools.combinations(preprocessing, i))
            
        for j, comb in enumerate(combinations):
            if 'eb' in comb and 'nb' in comb:
                comb = list(comb)
                comb.remove('eb')
                combinations[j] = comb
            else:
                combinations[j] = list(comb)
        
        self.final_combinations = [comb for n, comb in enumerate(combinations) if comb not in combinations[:n]]
        print(self.final_combinations)
        return self.final_combinations

    def evaluate_model(self, individual):
        
        
        model = self.data["model"][individual[1]]
        #preprocessing = individual[0]
        embedding = self.data["embedding"][individual[2]]
        pretrain_bert = self.data["pretrain_bert"][individual[3]]
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": [],#self.preprocessing[preprocessing],
            "embedding": pretrain_bert,# if model == "bert" else embedding,
            "synt_embeddings": self.synt_embeddings[synt_embeddings],
            "pooler": None,
            "opt": None,
            "batch_size": batch_size,#_bert if model == "bert" else batch_size,
            "lr": lr,
            "weight_decay": None,#weight_decay,
            "max_length": max_length,
            "max_epoch": max_epoch,#_bert if model == 'bert' else max_epoch,
            "mask_entity": None,
            "hidden_size": None,
            "position_size": None,
            "dropout": None,
        }
        
        print("parameters:",parameters)
        
        train = Training(parameters)
        
        return [train.train()]
        
    def evaluate_preprocessing(self, individual):
        
    
        #preprocessing = self.preprocessing[self.hof_model[0]]
        model = self.data["model"][self.hof_model[1]]
        embedding = self.data["pretrain_bert"][self.hof_model[2]] if model == 'bert' else self.data["embedding"][self.hof_model[2]]

        preprocessing = individual[0] # TODO change toolbox inserting the preprocessing parameter to search
        batch_size =  individual[0]
        lr =  individual[1]
        weight_decay =  individual[2]
        max_length =  individual[3]
        max_epoch =  individual[4]
        
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": self.preprocessing[preprocessing],
            "embedding": pretrain_bert,
            "synt_embeddings": synt_embeddings,
            "batch_size": batch_size,#batch_size_bert if model == 'bert' else batch_size,
            "lr": lr,
            "weight_decay": None,#weight_decay,
            "max_length": max_length,
            "max_epoch": max_epoch,#max_epoch_bert if model == 'bert' else max_epoch,
            "pooler": None,
            "opt": None,
            "mask_entity": None,
            "hidden_size": None,
            "position_size": None,
            "dropout": None,
        }
        
        train = Training(parameters)
        
        return [train.train()]
    
    def evaluate_hyperparameters(self, individual):

        #preprocessing = self.preprocessing[self.hof_model[0]]
        model = self.data["model"][self.hof_model[1]]
        embedding = self.data["pretrain_bert"][self.hof_model[2]] if model == 'bert' else self.data["embedding"][self.hof_model[2]]

        preprocessing = individual[0] # TODO change toolbox inserting the preprocessing parameter to search
        batch_size =  individual[0]
        lr =  individual[1]
        weight_decay =  individual[2]
        max_length =  individual[3]
        max_epoch =  individual[4]
            
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.metric,
            "preprocessing": None,#self.preprocessing[preprocessing],
            "embedding": pretrain_bert,
            "batch_size": batch_size,#batch_size_bert if model == 'bert' else batch_size,
            "lr": lr,
            "weight_decay": None,#weight_decay,
            "max_length": max_length,
            "max_epoch": max_epoch,#max_epoch_bert if model == 'bert' else max_epoch,
            "pooler": None,
            "opt": None,
            "mask_entity": None,
            "hidden_size": None,
            "position_size": None,
            "dropout": None,
        }
        
        train = Training(parameters)
        
        return [train.train()]

    def optimize_model(self):

        pop = self.toolbox_model.population(n=100)
        self.hof_model = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        print("hof_model:",self.hof_model)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox_model, cxpb=0.5, mutpb=0.2, ngen=10, 
                                    stats=stats, halloffame=self.hof_model, verbose=True)
        
        return pop, log, self.hof_model
    
    def optimize_hyperparameters(self):
        pop = self.toolbox_hyperparameters.population(n=10)
        self.hof_hyperparameters = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox_hyperparameters, cxpb=0.5, mutpb=0.2, ngen=10, 
                                    stats=stats, halloffame=self.hof_hyperparameters, verbose=True)
        
        return pop, log, self.hof_hyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "semeval20181-1", "semeval20181-2", "ddi"], 
                help='Dataset')
    parser.add_argument('-m','--metric', default="micro_f1", choices=["micro_f1", "macro_f1", "acc"], 
                help='Metric to optimize')
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset, args.metric)
    hof_model = opt.optimize_model()[2][0]
    hof_hyperparameters = opt.optimize_hyperparameters()[2][0]
    

    preprocessing = []# if len(opt.preprocessing) == 0 else opt.preprocessing[hof_model[0]]
    model = opt.data["model"][hof_model[1]]
    embedding = opt.data["pretrain_bert"][hof_model[2]] if opt.data["model"][hof_model[1]] == 'bert' else opt.data["embedding"][hof_model[2]]
    max_epoch = opt.data["max_epoch_bert"][hof_hyperparameters[6]] if model == 'bert' else opt.data["max_epoch"][hof_hyperparameters[6]]
    pooler, opt, batch_size = opt.data["pooler"][hof_hyperparameters[0]], opt.data["opt"][hof_hyperparameters[1]], opt.data["batch_size"][hof_hyperparameters[2]]
    lr, weight_decay, max_length, mask_entity = opt.data["lr"][hof_hyperparameters[3]], \
                                                opt.data["weight_decay"][hof_hyperparameters[4]], \
                                                opt.data["max_length"][hof_hyperparameters[5]], \
                                                max_epoch, opt.data["mask_entity"][hof_hyperparameters[7]]
    
    
    print("Optimized parameters for dataset {}:".format(args.dataset))
    print("Preprocessing - {}; Model - {}; Embedding - {}; Synt Embedding.".format(preprocessing, model, embedding, synt_embeddings))
    #print("Best model params:",opt.study_model.best_params)
    #print("Best  hyperparams:",opt.study_params.best_params)
    print("Batch size - {};".format(batch_size))
    print("Learning rate - {}; Max Length - {}; Max epoch - {}.".format(lr, max_length, max_epoch))
    print("Best {}:".format(opt.data["optimize"]), abs(opt.study_model.best_value))
    