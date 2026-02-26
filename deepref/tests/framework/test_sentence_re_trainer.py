import os

import pytest
from torch import nn

from deepref.dataset.re_dataset import REDataset
from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.framework.sentence_re_trainer import SentenceRETrainer
from deepref.model.softmax_mlp import SoftmaxMLP

REL2ID = {"Component-Whole(e1,e2)": 0, "Content-Container(e2,e1)": 1, "Entity-Origin(e1,e2)": 2, "Entity-Origin(e2,e1)": 3, "Component-Whole(e2,e1)": 4, "Cause-Effect(e1,e2)": 5, "Content-Container(e1,e2)": 6, "Message-Topic(e2,e1)": 7, "Member-Collection(e2,e1)": 8, "Instrument-Agency(e2,e1)": 9, "Member-Collection(e1,e2)": 10, "Entity-Destination(e2,e1)": 11, "Product-Producer(e1,e2)": 12, "Instrument-Agency(e1,e2)": 13, "Other": 14, "Message-Topic(e1,e2)": 15, "Cause-Effect(e2,e1)": 16, "Product-Producer(e2,e1)": 17, "Entity-Destination(e1,e2)": 18}

MODELS = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "roberta-base",
    "roberta-large"
]

@pytest.fixture(scope="module", params=MODELS)
def datasets_and_encoder(request):
    encoder = SentenceEncoder(request.param)
    train_dataset = REDataset("benchmark/semeval2010", encoder.tokenizer, dataset_split="train")
    train_dataset.df = train_dataset.df[:3]
    test_dataset = REDataset("benchmark/semeval2010", encoder.tokenizer, dataset_split="test")
    test_dataset.df = test_dataset.df[:3]
    return train_dataset, test_dataset, encoder

def test_trainer_sentence_encoder(datasets_and_encoder):
    train_dataset, test_dataset, encoder = datasets_and_encoder

    model = SoftmaxMLP(encoder, len(REL2ID), REL2ID)

    ckpt = f'ckpt/semeval2010_{encoder.model.config.name_or_path.replace("/", "-")}.pth.tar'

    training_parameters = {"batch_size": 32,
                            "max_epoch": 1,
                            "lr": 0.1,
                            "opt": "adamw",
                            "weight_decay": 1e-5, 
                            "warmup_step": 300,
                            "criterion": nn.CrossEntropyLoss()}
    trainer = SentenceRETrainer(model, train_dataset, test_dataset, ckpt, training_parameters)
    trainer.train_model()

    assert os.path.exists(f"ckpt/semeval2010_{encoder.model.config.name_or_path.replace('/', '-')}.pth.tar")

    