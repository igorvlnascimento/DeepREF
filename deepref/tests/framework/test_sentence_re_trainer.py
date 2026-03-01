import os
from unittest import mock
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

# ===========================================================================
# Early stopping — unit tests (no model download; eval_model is mocked)
# ===========================================================================

def _make_trainer(max_epoch: int, patience: int, ckpt: str = "/tmp/test_es_ckpt.pth.tar"):
    """Return a minimal SentenceRETrainer with eval_model available to mock.

    Uses a trivial 1-parameter nn.Linear as model to avoid needing a real
    encoder or dataset download.
    """
    # Tiny linear model that satisfies nn.Module interface
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            # Needed by SentenceRETrainer (model.state_dict())
        def forward(self, **kwargs):
            return self.linear(torch.zeros(1, 2))
        def state_dict(self, **kwargs):
            return self.linear.state_dict()

    tiny_model = _TinyModel()

    # SentenceRETrainer wraps nn.DataParallel; inject model directly
    training_params = {
        "max_epoch": max_epoch,
        "criterion": nn.CrossEntropyLoss(),
        "lr": 1e-3,
        "opt": "adam",
        "weight_decay": 0.0,
        "warmup_step": 0,
        "batch_size": 2,
        "patience": patience,
    }

    # Bypass __init__ to avoid needing real REDataset / tokenizer
    trainer = SentenceRETrainer.__new__(SentenceRETrainer)
    nn.Module.__init__(trainer)
    trainer.model = tiny_model
    trainer.parallel_model = nn.DataParallel(tiny_model)
    trainer.training_parameters = training_params
    trainer.max_epoch = max_epoch
    trainer.criterion = training_params["criterion"]
    trainer.lr = training_params["lr"]
    trainer.optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    trainer.scheduler = None
    trainer.ckpt = ckpt
    # train_loader / test_loader not needed because iterate_loader is mocked
    trainer.train_loader = None
    trainer.test_loader = None
    return trainer


class TestEarlyStopping:

    def test_stops_after_patience_epochs_without_improvement(self):
        """Training should stop once the stagnation counter hits patience."""
        patience = 2
        max_epoch = 10
        trainer = _make_trainer(max_epoch=max_epoch, patience=patience)

        # eval_model returns 0.5 on epoch 0 (improvement), then stays flat
        call_count = 0
        def _fake_eval(loader):
            nonlocal call_count
            metric = 0.5 if call_count == 0 else 0.3
            call_count += 1
            return ({"acc": metric}, [], [])

        with (
            mock.patch.object(trainer, "iterate_loader"),
            mock.patch.object(trainer, "eval_model", side_effect=_fake_eval),
        ):
            trainer.train_model(metric="acc")

        # epoch 0 → improve (counter=0)
        # epoch 1 → stagnate (counter=1)
        # epoch 2 → stagnate (counter=2 >= patience=2) → stop
        # So eval_model is called exactly patience+1 times
        assert call_count == patience + 1

    def test_disabled_when_patience_zero(self):
        """When patience=0, all max_epoch epochs must be run."""
        max_epoch = 5
        trainer = _make_trainer(max_epoch=max_epoch, patience=0)

        call_count = 0
        def _flat_eval(loader):
            nonlocal call_count
            call_count += 1
            return ({"acc": 0.0}, [], [])

        with (
            mock.patch.object(trainer, "iterate_loader"),
            mock.patch.object(trainer, "eval_model", side_effect=_flat_eval),
        ):
            trainer.train_model(metric="acc")

        assert call_count == max_epoch

    def test_checkpoint_saved_on_early_stop(self, tmp_path):
        """Checkpoint must be written before early stopping fires."""
        ckpt = str(tmp_path / "es_ckpt.pth.tar")
        trainer = _make_trainer(max_epoch=10, patience=1, ckpt=ckpt)

        # First epoch improves → checkpoint should be saved
        call_count = 0
        def _fake_eval(loader):
            nonlocal call_count
            metric = 0.8 if call_count == 0 else 0.1
            call_count += 1
            return ({"acc": metric}, [], [])

        with (
            mock.patch.object(trainer, "iterate_loader"),
            mock.patch.object(trainer, "eval_model", side_effect=_fake_eval),
        ):
            trainer.train_model(metric="acc")

        assert os.path.exists(ckpt)


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

    