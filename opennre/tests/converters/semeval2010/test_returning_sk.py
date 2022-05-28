import os
import csv
import subprocess

from ast import literal_eval

def test_should_return_sk_column():
    subprocess.call(['bash', 'benchmark/download_semeval2010.sh'])
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    assert os.stat("benchmark/semeval2010/original/semeval2010_test_original.csv").st_size > 0
    with open('benchmark/semeval2010/original/semeval2010_test_original.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        row = None
        for i, r in enumerate(csv_reader):
            row = r
            break
        assert "sk" in row
    