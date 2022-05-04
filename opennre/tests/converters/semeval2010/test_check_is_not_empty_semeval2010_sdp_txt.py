import os
    
def test_should_pass_checking_the_sdp_semeval2010_files_train_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/sdp/semeval2010_train_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_train_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/eb_sdp/semeval2010_train_eb_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_train_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/nb_sdp/semeval2010_train_nb_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_sdp_semeval2010_files_val_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/sdp/semeval2010_val_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_val_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/eb_sdp/semeval2010_val_eb_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_val_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/nb_sdp/semeval2010_val_nb_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_sdp_semeval2010_files_test_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/sdp/semeval2010_test_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_test_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/eb_sdp/semeval2010_test_eb_sdp.txt").st_size > 0
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_test_dataset_is_not_empty():
    assert os.stat("benchmark/semeval2010/nb_sdp/semeval2010_test_nb_sdp.txt").st_size > 0
    
##################
    
def test_should_pass_checking_the_sdp_semeval2010_files_train_dataset_right_size():
    assert len(open("benchmark/semeval2010/sdp/semeval2010_train_sdp.txt", 'r').readlines()) == 6400
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_train_dataset_right_size():
    assert len(open("benchmark/semeval2010/eb_sdp/semeval2010_train_eb_sdp.txt").readlines()) == 6400
        
def test_should_pass_checking_the_nb_sdp_semeval2010_files_train_dataset_right_size():
    assert len(open("benchmark/semeval2010/nb_sdp/semeval2010_train_nb_sdp.txt").readlines()) == 6400
    
def test_should_pass_checking_the_sdp_semeval2010_files_val_dataset_right_size():
    assert len(open("benchmark/semeval2010/sdp/semeval2010_val_sdp.txt").readlines()) == 1600
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_val_dataset_right_size():
    assert len(open("benchmark/semeval2010/eb_sdp/semeval2010_val_eb_sdp.txt").readlines()) == 1600
    
def test_should_pass_checking_the_nb_sdp_semeval2010_files_val_dataset_right_size():
    assert len(open("benchmark/semeval2010/nb_sdp/semeval2010_val_nb_sdp.txt").readlines()) == 1600
    
def test_should_pass_checking_the_sdp_semeval2010_files_test_dataset_right_size():
    assert len(open("benchmark/semeval2010/sdp/semeval2010_test_sdp.txt", 'r').readlines()) == 2717
    
def test_should_pass_checking_the_eb_sdp_semeval2010_files_test_dataset_right_size():
    assert len(open("benchmark/semeval2010/eb_sdp/semeval2010_test_eb_sdp.txt").readlines()) == 2717
        
def test_should_pass_checking_the_nb_sdp_semeval2010_files_test_dataset_right_size():
    assert len(open("benchmark/semeval2010/nb_sdp/semeval2010_test_nb_sdp.txt").readlines()) == 2717