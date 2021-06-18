import subprocess
from opennre import constants

if __name__ == '__main__':
    for dataset in constants.datasets_choices:
        if dataset == 'semeval2010':
            pretrain_choices = constants.pretrain_choices[:-2]
        else:
            pretrain_choices = constants.pretrain_choices
        for model in constants.model_choices:
            for pretrain in pretrain_choices:
                for preprocessing in constants.preprocessing_choices:
                    if model == 'bert':
                        subprocess.call(['python', 'example/train_supervised_{}.py'.format(model), '--dataset', dataset, '--preprocessing', preprocessing, '--pretrain_path', pretrain])
                    else:
                        subprocess.call(['python', 'example/train_supervised_{}.py'.format(model), '--dataset', dataset, '--preprocessing', preprocessing])