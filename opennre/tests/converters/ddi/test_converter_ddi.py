from opennre.dataset.converters.converter_ddi import ConverterDDI
import os
import subprocess

def test_should_return_without_errors_wit_stanza_nlp_tool():
    ConverterDDI('stanza', 'scientific')

def test_should_return_without_errors_wit_spacy_nlp_tool():
    ConverterDDI('spacy', 'scientific')

def test_should_return_preprocessed_files_for_ddi_with_stanza():
    subprocess.call(['python', 'opennre/dataset/converters/converter_ddi.py', '--nlp_tool', 'stanza', '--nlp_mode', 'scientific'])
    assert os.path.exists("benchmark/ddi/original/ddi_train_original.txt")
    assert os.path.exists("benchmark/ddi/original/ddi_val_original.txt")
    assert os.path.exists("benchmark/ddi/original/ddi_test_original.txt")
    os.remove("benchmark/ddi/original/ddi_train_original.txt")
    os.remove("benchmark/ddi/original/ddi_val_original.txt")
    os.remove("benchmark/ddi/original/ddi_test_original.txt")
    
def test_should_return_preprocessed_files_for_ddi_with_spacy():
    subprocess.call(['python', 'opennre/dataset/converters/converter_ddi.py', '--nlp_tool', 'spacy', '--nlp_mode', 'scientific'])
    assert os.path.exists("benchmark/ddi/original/ddi_test_original.txt")
    assert os.path.exists("benchmark/ddi/original/ddi_test_original.txt")
    assert os.path.exists("benchmark/ddi/original/ddi_test_original.txt")
    os.remove("benchmark/ddi/original/ddi_train_original.txt")
    os.remove("benchmark/ddi/original/ddi_val_original.txt")
    os.remove("benchmark/ddi/original/ddi_test_original.txt")
    
def test_should_return_non_empty_preprocessed_files_for_ddi_with_stanza():
    subprocess.call(['python', 'opennre/dataset/converters/converter_ddi.py', '--nlp_tool', 'spacy', '--nlp_mode', 'scientific'])
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    os.remove("benchmark/ddi/original/ddi_train_original.txt")
    os.remove("benchmark/ddi/original/ddi_val_original.txt")
    os.remove("benchmark/ddi/original/ddi_test_original.txt")
    
def test_should_return_non_empty_preprocessed_files_for_ddi_with_spacy():
    subprocess.call(['python', 'opennre/dataset/converters/converter_ddi.py', '--nlp_tool', 'spacy', '--nlp_mode', 'scientific'])
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    assert os.stat("benchmark/ddi/original/ddi_test_original.txt").st_size > 0
    os.remove("benchmark/ddi/original/ddi_train_original.txt")
    os.remove("benchmark/ddi/original/ddi_val_original.txt")
    os.remove("benchmark/ddi/original/ddi_test_original.txt")
    
