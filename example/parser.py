import subprocess
from opennre.constants import preprocessing_choices, datasets_choices, semeval_model_choices, ddi_model_choices

if __name__ == '__main__':
    for model in semeval_model_choices:
        for preprocessing in preprocessing_choices:
            subprocess.call(['python', 'example/train_supervised_{}.py'.format(model), '--dataset', 'semeval2010', '--preprocessing', preprocessing])

    #for model in ddi_model_choices:
    #    for preprocessing in preprocessing_choices:
    #        subprocess.call(['python', 'example/train_supervised_{}.py'.format(model), '--dataset', 'ddi', '--preprocessing', preprocessing])