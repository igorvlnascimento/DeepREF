import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from deepref import config

from collections import Counter

def plot_sentence_lengths_statistics(dataset):
    
    train_csv = pd.read_csv(f'benchmark/{dataset}/original/{dataset}_train_original.csv', sep='\t')
    val_csv = pd.read_csv(f'benchmark/{dataset}/original/{dataset}_val_original.csv', sep='\t')
    test_csv = pd.read_csv(f'benchmark/{dataset}/original/{dataset}_test_original.csv', sep='\t')
    
    sentence_lengths_train = [len(sentence.split()) for sentence in train_csv.loc[:,'original_sentence']]
    sentence_lengths_val = [len(sentence.split()) for sentence in val_csv.loc[:,'original_sentence']]
    sentence_lengths_test = [len(sentence.split()) for sentence in test_csv.loc[:,'original_sentence']]
    
    sentence_lengths = dict(Counter(sentence_lengths_train + sentence_lengths_val + sentence_lengths_test))
    sentence_lengths = [(i, sentence_lengths[i]) for i in sorted(sentence_lengths, reverse=True)]
    X, Y = [], []
    for x, y in sentence_lengths:
        X.append(x)
        Y.append(y)
        
    df = pd.DataFrame({'Nº de tokens': X,
                   'Quantidade': Y})
    sns.set_style('darkgrid')

    sns_plot = sns.barplot(data=df, x='Nº de tokens', y='Quantidade')
    #plt.xticks(rotation=90)
    sns_plot.xaxis.set_major_locator(ticker.MultipleLocator(5))
    sns_plot.xaxis.set_major_formatter(ticker.ScalarFormatter())
    if dataset == 'semeval2010':
        sns_plot.set(title='SemEval 2010')
    elif dataset == 'semeval20181-1':
        sns_plot.set(title='SemEval 2018 ST1')
    elif dataset == 'semeval20181-2':
        sns_plot.set(title='SemEval 2018 ST2')
    elif dataset == 'ddi':
        sns_plot.set(title='DDI')
    fig = sns_plot.get_figure()
    fig.savefig(f"results/{dataset}/{dataset.split('-')[0]}_length_sentence_statistics.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default="semeval2010", choices=config.DATASETS, 
                help='Dataset')
    
    args = parser.parse_args()
    
    plot_sentence_lengths_statistics(args.dataset)
    