from opennre.dataset.dataset import Dataset

class SemEval2010Dataset(Dataset):
    def __init__(self, name:str, train_sentences:list, test_sentences:list, val_perc:float=0.2, preprocessing_type:str ="original"):
        super().__init__(name, train_sentences, test_sentences)
        
        # if preprocessing_type == "original":
        #     self.write_dataframe()
        # self.write_text()
        
    