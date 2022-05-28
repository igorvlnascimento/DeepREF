import os
import csv
import subprocess

from ast import literal_eval

def test_should_return_word_indexes_with_one_list():
    subprocess.call(['bash', 'benchmark/download_semeval2010.sh'])
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    with open('benchmark/semeval2010/original/semeval2010_test_original.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        row = None
        for i, r in enumerate(csv_reader):
            row = r
            if i == 1:
                break
        assert type(literal_eval(row[4])["e1"]["word_index"][0]) == tuple
    