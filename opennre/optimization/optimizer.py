import random
import numpy
import json
import argparse
import itertools

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from train import Training

CONFIG_FILE_PATH = "opennre/optimization/config_params.json"

class Optimizer():
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = json.load(open(CONFIG_FILE_PATH))
        
        self.final_combinations = []
        
        self.preprocessing = self.data["preprocessing"]
        self.preprocessing = self.combine_preprocessing(self.preprocessing)
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.self.N_CYCLES = 1
        
        self.hof_model = []
        self.hof_hyperparameters = []
        
    def init_toolbox_model(self):
        
        self.toolbox_model = base.Toolbox()

        self.toolbox_model.register("attr_preprocessing", random.randint, 0, len(self.preprocessing))
        self.toolbox_model.register("attr_model", random.randint, 0, len(self.data["model"])-1)
        self.toolbox_model.register("attr_embedding", random.randint, 0, 2)
        #self.toolbox_model.register("attr_pretrain_bert", random.randint, 0, len(self.data["pretrain_bert"])-1)
        self.toolbox_model.register("individual", tools.initCycle, creator.Individual,
                        (self.toolbox_model.attr_preprocessing, self.toolbox_model.attr_model, self.toolbox_model.attr_embedding), n=self.N_CYCLES)
        self.toolbox_model.register("population", tools.initRepeat, list, self.toolbox_model.individual)
        
        self.toolbox_model.register("evaluate", self.evaluate_model)
        self.toolbox_model.register("mate", tools.cxTwoPoint)
        self.toolbox_model.register("mutate", tools.mutFlipBit, indpb=0.05)    
        self.toolbox_model.register("select", tools.selTournament, tournsize=3)
        
    def init_toolbox_hyperparameters(self):
        
        self.toolbox_hyperparameters = base.Toolbox()

        self.toolbox_hyperparameters.register("attr_pooler", random.randint, 0, len(self.data["pooler"])-1)
        self.toolbox_hyperparameters.register("attr_opt", random.randint, 0, len(self.data["opt"])-1)
        self.toolbox_hyperparameters.register("attr_batch_size", random.randint, 0, len(self.data["batch_size"])-1)
        self.toolbox_hyperparameters.register("attr_lr", random.randint, 0, len(self.data["lr"])-1)
        self.toolbox_hyperparameters.register("attr_weight_decay", random.randint, 0, len(self.data["weight_decay"])-1)
        self.toolbox_hyperparameters.register("attr_max_length", random.randint, 0, len(self.data["max_length"])-1)
        self.toolbox_hyperparameters.register("attr_max_epoch", random.randint, 0, len(self.data["max_epoch"])-1)
        self.toolbox_hyperparameters.register("attr_mask_entity", random.randint, 0, len(self.data["mask_entity"])-1)
        self.toolbox_hyperparameters.register("individual", tools.initCycle, creator.Individual,
                        (self.toolbox_hyperparameters.attr_pooler, self.toolbox_hyperparameters.attr_opt,
                         self.toolbox_hyperparameters.attr_batch_size, self.toolbox_hyperparameters.attr_lr,
                         self.toolbox_hyperparameters.attr_weight_decay, self.toolbox_hyperparameters.attr_max_length,
                         self.toolbox_hyperparameters.attr_max_epoch, self.toolbox_hyperparameters.attr_mask_entity), n=self.N_CYCLES)
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
        preprocessing = None if len(self.preprocessing) == 0 else self.preprocessing[individual[0]]
        embedding = self.data["pretrain_bert"][individual[2]] if self.data["model"][individual[1]] == 'bert' else self.data["embedding"][individual[2]]
        
        
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.data["optimize"],
            "preprocessing": preprocessing,
            "embedding": embedding,
            #"pretrain_path": self.data["pretrain_bert"][individual[3]],
            "pooler": None,
            "opt": None,
            "batch_size": None,
            "lr": None,
            "weight_decay": None,
            "max_length": None,
            "max_epoch": None,
            "mask_entity": None
        }
        
        train = Training(parameters)
        return [train.train()]
    
    def evaluate_hyperparameters(self, individual):
        
        preprocessing = None if len(self.preprocessing) == 0 else self.preprocessing[self.hof_model[0]]
        model = self.data["model"][self.hof_model[1]]
        embedding = self.data["pretrain_bert"][self.hof_model[2]] if self.data["model"][self.hof_model[1]] == 'bert' else self.data["embedding"][self.hof_model[2]]
        
        parameters = {
            "dataset": self.dataset,
            "model": model,
            "metric": self.data["optimize"],
            "preprocessing": preprocessing,
            "embedding": embedding,
            "pooler": self.data["pooler"][individual[0]],
            "opt": self.data["opt"][individual[1]],
            "batch_size": self.data["batch_size"][individual[2]],
            "lr": self.data["lr"][individual[3]],
            "weight_decay": self.data["weigth_decay"][individual[4]],
            "max_length": self.data["max_length"][individual[5]],
            "max_epoch": self.data["max_epoch"][individual[6]],
            "mask_entity": self.data["mask_entity"][individual[7]]
        }
        
        train = Training(parameters)
        return [train.train()]

    def optimize_model(self):
        random.seed(64)
        
        pop = self.toolbox.population(n=10)
        self.hof_model = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2, 
                                    stats=stats, halloffame=self.hof_model, verbose=True)
        
        return pop, log, self.hof_model
    
    def optimize_hyperparameters(self):
        random.seed(64)
        
        pop = self.toolbox.population(n=10)
        self.hof_hyperparameters = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2, 
                                    stats=stats, halloffame=self.hof_hyperparameters, verbose=True)
        
        return pop, log, self.hof_hyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "ddi"], 
                help='Dataset')
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset)
    hof_model = opt.optimize_model()[2]
    hof_hyperparameters = opt.optimize_hyperparameters()[2]
    
    preprocessing = 'none' if len(opt.preprocessing) == 0 else opt.preprocessing[hof_model[0]]
    model = opt.data["model"][hof_model[1]]
    embedding = opt.data["pretrain_bert"][hof_model[2]] if opt.data["model"][hof_model[1]] == 'bert' else opt.data["embedding"][hof_model[2]]
    max_epoch = opt.data["max_epoch_bert"] if model == 'bert' else opt.data["max_epoch"]
    
    print("Optimized parameters for dataset {}:".format(args.dataset))
    print("Preprocessing - {}; Model - {}; Embedding - {}.".format(preprocessing, model, embedding))
    print("Pooler - {}; Optimizer - {}; Batch size - {};".format(opt.data["pooler"], opt.data["opt"], opt.data["batch_size"]))
    print("Learning rate - {}; Max Length - {}; Max epoch - {}; Mask entity - {}.".format(opt.data["lr"], opt.data["max_length"], max_epoch, opt.data["mask_entity"]))
    