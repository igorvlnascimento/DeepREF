import os

def test_should_pass_checking_the_sdp_semeval2010_files_test_dataset():
    assert os.path.isfile("benchmark/semeval2010/sdp/semeval2010_test_sdp.txt")
        
def test_should_pass_checking_the_eb_sdp_semeval2010_files_test_dataset():
    assert os.path.isfile("benchmark/semeval2010/eb_sdp/semeval2010_test_eb_sdp.txt")
        
def test_should_pass_checking_the_nb_sdp_semeval2010_files_test_dataset():
    assert os.path.isfile("benchmark/semeval2010/nb_sdp/semeval2010_test_nb_sdp.txt")
    
def test_should_pass_checking_the_sdp_semeval2010_files_train_dataset():
    assert os.path.isfile("benchmark/semeval2010/sdp/semeval2010_train_sdp.txt")
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_train_dataset():
    assert os.path.isfile("benchmark/semeval2010/eb_sdp/semeval2010_train_eb_sdp.txt")
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_train_dataset():
    assert os.path.isfile("benchmark/semeval2010/nb_sdp/semeval2010_train_nb_sdp.txt")
    
def test_should_pass_checking_the_sdp_semeval2010_files_val_dataset():
    assert os.path.isfile("benchmark/semeval2010/sdp/semeval2010_val_sdp.txt")
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_val_dataset():
    assert os.path.isfile("benchmark/semeval2010/eb_sdp/semeval2010_val_eb_sdp.txt")
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_val_dataset():
    assert os.path.isfile("benchmark/semeval2010/nb_sdp/semeval2010_val_nb_sdp.txt")
    
