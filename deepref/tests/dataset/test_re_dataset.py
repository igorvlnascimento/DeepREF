import pytest

from deepref.dataset.re_dataset import REDataset
from deepref.encoder.sentence_encoder import SentenceEncoder

DATASET_SPLITS = ["train", "test"]

DATASET_SPLITS_COUNT = [
    ("train", 8000), 
    ("test", 2717)
]

DATASET_SPLITS_FIRST_RELATION = [
    ("train", "Component-Whole(e2,e1)"), 
    ("test", "Message-Topic(e1,e2)")
]

REL2ID = {"Cause-Effect(e1,e2)": 0, 
          "Cause-Effect(e2,e1)": 1, 
          "Component-Whole(e1,e2)": 2, 
          "Component-Whole(e2,e1)": 3, 
          "Content-Container(e1,e2)": 4, 
          "Content-Container(e2,e1)": 5, 
          "Entity-Destination(e1,e2)": 6, 
          "Entity-Destination(e2,e1)": 7, 
          "Entity-Origin(e1,e2)": 8, 
          "Entity-Origin(e2,e1)": 9, 
          "Instrument-Agency(e1,e2)": 10, 
          "Instrument-Agency(e2,e1)": 11, 
          "Member-Collection(e1,e2)": 12, 
          "Member-Collection(e2,e1)": 13, 
          "Message-Topic(e1,e2)": 14, 
          "Message-Topic(e2,e1)": 15, 
          "Other": 16,
          "Product-Producer(e1,e2)": 17, 
          "Product-Producer(e2,e1)": 18}

@pytest.fixture(scope="module")
def sentence_encoder():
    return SentenceEncoder("HuggingFaceTB/SmolLM-135M-Instruct")

@pytest.mark.parametrize("dataset_split, expected_length", DATASET_SPLITS_COUNT)
def test_re_dataset_count(dataset_split, expected_length, sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer, dataset_split=dataset_split)
    assert len(dataset) == expected_length

@pytest.mark.parametrize("dataset_split, expected_relation", DATASET_SPLITS_FIRST_RELATION)
def test_re_dataset_get_first_item(dataset_split, expected_relation, sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer, dataset_split=dataset_split)
    print("first:", dataset[0])
    assert dataset[0][0] == REL2ID[expected_relation]

@pytest.mark.parametrize("dataset_split", DATASET_SPLITS)
def test_re_dataset_labels_dict(dataset_split, sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer, dataset_split=dataset_split)
    assert dataset.rel2id == REL2ID

def test_re_dataset_eval(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer, dataset_split="test")
    dataset.eval()
