import pytest

from deepref.dataset.re_dataset import REDataset

DATASET_SPLITS_COUNT = [
    ("train", 8000), 
    ("test", 2717)
]

DATASET_SPLITS_FIRST_RELATION = [
    ("train", "Component-Whole(e2,e1)"), 
    ("test", "Message-Topic(e1,e2)")
]

@pytest.mark.parametrize("dataset_split, expected_length", DATASET_SPLITS_COUNT)
def test_re_dataset_count(dataset_split, expected_length):
    dataset = REDataset("benchmark/semeval2010", dataset_split=dataset_split)
    assert len(dataset) == expected_length

@pytest.mark.parametrize("dataset_split, expected_relation", DATASET_SPLITS_FIRST_RELATION)
def test_re_dataset_get_first_item(dataset_split, expected_relation):
    dataset = REDataset("benchmark/semeval2010", dataset_split=dataset_split)
    assert dataset[0]["relation_type"] == expected_relation
