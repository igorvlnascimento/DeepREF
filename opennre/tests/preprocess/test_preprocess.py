import os

from opennre.dataset.preprocessors.preprocess_dataset import PreprocessDataset

def test_should_return_sw_eb_preprocessed_file_ddi():
    p = PreprocessDataset("ddi", ["sw", "eb"])
    p.preprocess_dataset()
    assert os.path.exists("benchmark/ddi/eb_sw/ddi_test_eb_sw.txt")
    assert os.path.exists("benchmark/ddi/eb_sw/ddi_test_eb_sw.txt")
    assert os.path.exists("benchmark/ddi/eb_sw/ddi_test_eb_sw.txt")
    
def test_should_return_b_d_preprocessed_file_ddi():
    p = PreprocessDataset("ddi", ["b", "d"])
    p.preprocess_dataset()
    assert os.path.exists("benchmark/ddi/b_d/ddi_test_b_d.txt")
    assert os.path.exists("benchmark/ddi/b_d/ddi_test_b_d.txt")
    assert os.path.exists("benchmark/ddi/b_d/ddi_test_b_d.txt")
    
def test_should_return_sw_eb_preprocessed_file_semeval2010():
    p = PreprocessDataset("semeval2010", ["sw", "eb"])
    p.preprocess_dataset()
    assert os.path.exists("benchmark/semeval2010/eb_sw/semeval2010_test_eb_sw.txt")
    assert os.path.exists("benchmark/semeval2010/eb_sw/semeval2010_test_eb_sw.txt")
    assert os.path.exists("benchmark/semeval2010/eb_sw/semeval2010_test_eb_sw.txt")
    
def test_should_return_b_d_preprocessed_file_semeval20181():
    p = PreprocessDataset("semeval20181-1", ["b", "d"])
    p.preprocess_dataset()
    assert os.path.exists("benchmark/semeval20181-1/b_d/semeval20181-1_test_b_d.txt")
    assert os.path.exists("benchmark/semeval20181-1/b_d/semeval20181-1_test_b_d.txt")
    assert os.path.exists("benchmark/semeval20181-1/b_d/semeval20181-1_test_b_d.txt")
    
def test_should_return_b_d_preprocessed_file_semeval20182():
    p = PreprocessDataset("semeval20181-2", ["b", "d"])
    p.preprocess_dataset()
    assert os.path.exists("benchmark/semeval20181-2/b_d/semeval20181-2_test_b_d.txt")
    assert os.path.exists("benchmark/semeval20181-2/b_d/semeval20181-2_test_b_d.txt")
    assert os.path.exists("benchmark/semeval20181-2/b_d/semeval20181-2_test_b_d.txt")
    