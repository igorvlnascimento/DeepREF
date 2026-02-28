import pytest
import torch

from deepref.dataset.re_dataset import REDataset, RELoader
from deepref.encoder.sentence_encoder import SentenceEncoder

DATASETS = [
    "semeval2010"
]

DATASET_COUNT = [
    ("semeval2010", 10717),
    ("ddi", 4999)
]

DATASET_FIRST_RELATION = [
    ("semeval2010", "Component-Whole(e2,e1)"),
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

@pytest.mark.parametrize("dataset_name, expected_length", DATASET_COUNT)
def test_re_dataset_count(dataset_name, expected_length, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    assert len(dataset) == expected_length

@pytest.mark.parametrize("dataset_name, expected_relation", DATASET_FIRST_RELATION)
def test_re_dataset_get_first_item(dataset_name, expected_relation, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    assert dataset[0]["labels"] == REL2ID[expected_relation]

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_labels_dict(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    assert dataset.rel2id == REL2ID

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_eval(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    pred_result = [torch.randint(0, 18, size=(1,)) for _ in range(len(dataset))]
    results = dataset.eval(pred_result)

    assert len(results) == 6
    assert "acc" in results
    assert "micro_p" in results
    assert "micro_r" in results
    assert "micro_f1" in results
    assert "macro_f1" in results
    assert "cm" in results


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_getitem_has_required_keys(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_getitem_returns_tensors(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    item = dataset[0]
    for value in item.values():
        assert isinstance(value, torch.Tensor)


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_labels_dtype_is_long(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    item = dataset[0]
    assert item["labels"].dtype == torch.long


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_max_length_is_positive(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    assert dataset.max_length > 0

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_format_sentence_inserts_entity_markers(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    sentence = "The system is very fast"
    e1 = "{'name': 'system', 'position': [1, 2]}"
    e2 = "{'name': 'fast', 'position': [4, 5]}"
    result = dataset.format_sentence(sentence, e1, e2)
    assert result == "The <e1> system </e1> is very <e2> fast </e2>"

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_format_sentence_contains_all_markers(dataset_name, sentence_encoder):
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer)
    sentence = "The system is very fast"
    e1 = "{'name': 'system', 'position': [1, 2]}"
    e2 = "{'name': 'fast', 'position': [4, 5]}"
    result = dataset.format_sentence(sentence, e1, e2)
    for marker in ["<e1>", "</e1>", "<e2>", "</e2>"]:
        assert marker in result

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_re_dataset_with_preprocessor(dataset_name, sentence_encoder):
    preprocessor = str.lower
    dataset = REDataset(f"benchmark/{dataset_name}", sentence_encoder.tokenizer, preprocessors_list=[preprocessor])
    item = dataset[0]
    assert "labels" in item
    assert isinstance(item["labels"], torch.Tensor)


def test_re_dataset_preprocessor_does_not_change_labels(sentence_encoder):
    dataset_without = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    dataset_with = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer, preprocessors_list=[str.lower])
    assert dataset_without[0]["labels"] == dataset_with[0]["labels"]


def test_re_dataset_eval_perfect_predictions(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    perfect_preds = [dataset.rel2id[dataset.df.iloc[i]["relation_type"]] for i in range(len(dataset))]
    results = dataset.eval(perfect_preds)
    assert results["acc"] == 1.0


def test_re_dataset_eval_by_name_perfect_predictions(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    perfect_preds = [dataset.df.iloc[i]["relation_type"] for i in range(len(dataset))]
    results = dataset.eval(perfect_preds, use_name=True)
    assert results["acc"] == 1.0


def test_re_dataset_eval_metric_values_in_range(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    pred_result = [0] * len(dataset)
    results = dataset.eval(pred_result)
    assert 0.0 <= results["acc"] <= 1.0
    assert 0.0 <= results["micro_f1"] <= 1.0
    assert 0.0 <= results["macro_f1"] <= 1.0


def test_re_loader_returns_correct_batch_size(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    batch_size = 32
    loader = RELoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == batch_size


def test_re_loader_batch_contains_required_keys(sentence_encoder):
    dataset = REDataset("benchmark/semeval2010", sentence_encoder.tokenizer)
    loader = RELoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch