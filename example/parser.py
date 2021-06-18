import subprocess
from opennre.constants import preprocessing_choices, datasets_choices

if __name__ == '__main__':
    for preprocessing in preprocessing_choices:
        subprocess.call(['python', 'example/train_supervised_roberta.py', '--dataset', 'semeval2010', '--preprocessing', preprocessing])