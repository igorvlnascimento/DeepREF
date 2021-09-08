import random
import numpy
import json
import argparse

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from train import Training


class Optimizer():
    def __init__(self, dataset):
        config_file_path = "opennre/optimization/config_params.json"
        
        self.dataset = dataset
        self.data = json.load(open(config_file_path))
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        
        N_CYCLES = 1

        # self.toolbox.register("attr_preprocessing_replace", random.randint, 0, len(data["preprocessing_replace"]))
        # self.toolbox.register("attr_preprocessing_blinding", random.randint, 0, len(data["preprocessing_blinding"]))
        self.toolbox.register("attr_model", random.randint, 0, len(self.data["model"])-1)
        self.toolbox.register("attr_embedding", random.randint, 0, len(self.data["embedding"])-1)
        self.toolbox.register("attr_pretrain_bert", random.randint, 0, len(self.data["pretrain_bert"])-1)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                        (self.toolbox.attr_model, self.toolbox.attr_embedding, self.toolbox.attr_pretrain_bert), n=N_CYCLES)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_model)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)    
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_model(self, individual):
        
        
        parameters = {
                "dataset": self.dataset,
                "model": self.data["model"][individual[0]], 
                "metric": self.data["optimize"],
                "preprocessing": None,
                "embedding": self.data["embedding"][individual[1]],
                "pretrain_path": self.data["pretrain_bert"][individual[2]],
                "pooler": None,
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
        
        return 0

    def optimize(self):
        random.seed(64)
        
        pop = self.toolbox.population(n=20)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                    stats=stats, halloffame=hof, verbose=True)
        
        return pop, log, hof

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=["semeval2010", "semeval2018", "ddi"], 
                help='Dataset')
    
    args = parser.parse_args()
    
    opt = Optimizer(args.dataset)
    print(opt.optimize())