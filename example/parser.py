import subprocess
from opennre.constants import preprocessing_choices, datasets_choices, model_choices

if __name__ == '__main__':
    for model in model_choices:
        for preprocessing in preprocessing_choices:
            subprocess.call(['python', 'example/train_supervised_bert.py', '--dataset', 'semeval2010', '--preprocessing', preprocessing])