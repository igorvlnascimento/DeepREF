"""
Tests for the combine-embeddings experiment classes:

    * CombineREDataset   — raw-item dataset backed by REDataset CSV loading.
    * combine_collate_fn — batch collation compatible with SentenceRETrainer.
    * CombineEmbeddings  — dual-encoder concatenation wrapper.
    * CombineRETrainer   — SentenceRETrainer subclass with custom loaders and
                           sklearn-based evaluation.

Test strategy
-------------
* CombineREDataset: tested against benchmark/semeval2010.csv (real file,
  DataFrame sliced to 10 rows so tests run in milliseconds).
* CombineEmbeddings._get_hidden_size / _encode_single: encoder-type dispatch
  verified with lightweight mocks — no LLM download required.
* CombineEmbeddings.forward: tested end-to-end with two BoWSDPEncoder
  instances (spaCy only, no LLM).
* CombineRETrainer: tested end-to-end with two BoWSDPEncoder instances.
  train_model has mlflow patched out; eval_model runs real forward passes on
  4 samples to keep wall-time low.
* Integration tests that would require a real LLM are marked
  ``@pytest.mark.integration``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from deepref.experiments.run_combine_embeddings_experiments import (
    CombineEmbeddings,
    CombineREDataset,
    CombineRETrainer,
    combine_collate_fn,
    make_split_subsets,
)
from deepref.framework.sentence_re_trainer import SentenceRETrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH = "benchmark/semeval2010.csv"

# Item dict format used by all encoders
ITEM_SIMPLE = {
    "token": ["The", "audits", "were", "about", "waste", "."],
    "h": {"name": "audits", "pos": [1, 2]},
    "t": {"name": "waste", "pos": [4, 5]},
}

# Hidden size used by mocked LLM / Relation encoders
MOCK_HIDDEN = 64

# Number of rows kept in the sliced dataset fixture
SLICE_SIZE = 10

# Indices used for tiny train / test subsets inside the trainer fixture
_TRAIN_IDX = list(range(4))
_TEST_IDX = list(range(4, 8))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """10-row slice of semeval2010 for fast tests (no tokenizer required)."""
    ds = CombineREDataset(CSV_PATH)
    ds.df = ds.df.iloc[:SLICE_SIZE].reset_index(drop=True)
    return ds


@pytest.fixture(scope="module")
def bow_encoder():
    """Real BoWSDPEncoder — loads spaCy once; no LLM download."""
    from deepref.encoder.sdp_encoder import BoWSDPEncoder
    return BoWSDPEncoder()


def _make_mock_model(hidden_size: int) -> MagicMock:
    """Return a MagicMock that behaves like a HuggingFace model.

    Crucially, ``.to(device)`` returns *the same object* so that
    ``LLMEncoder.__init__`` (which does ``AutoModel.from_pretrained(...).to(device)``)
    ends up with a mock whose ``config.hidden_size`` is still set correctly.
    """
    mock_model = MagicMock()
    mock_model.config.hidden_size = hidden_size
    mock_model.to.return_value = mock_model          # .to(device) → self
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    return mock_model


@pytest.fixture(scope="module")
def mock_llm_encoder():
    """LLMEncoder instantiated with patched AutoModel / AutoTokenizer."""
    from deepref.encoder.llm_encoder import LLMEncoder
    mock_model = _make_mock_model(MOCK_HIDDEN)
    mock_tokenizer = MagicMock()
    with (
        patch("deepref.encoder.llm_encoder.AutoModel.from_pretrained", return_value=mock_model),
        patch("deepref.encoder.llm_encoder.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        enc = LLMEncoder("mock-llm")
    return enc


@pytest.fixture(scope="module")
def mock_relation_encoder():
    """RelationEncoder instantiated with patched transformer internals."""
    from deepref.encoder.relation_encoder import RelationEncoder
    mock_model = _make_mock_model(MOCK_HIDDEN)
    mock_tokenizer = MagicMock()
    mock_tokenizer.mask_token = "[MASK]"
    mock_tokenizer.cls_token = "[CLS]"
    mock_tokenizer.sep_token = "[SEP]"
    mock_tokenizer.pad_token_id = 0
    with (
        patch("deepref.encoder.llm_encoder.AutoModel.from_pretrained", return_value=mock_model),
        patch("deepref.encoder.llm_encoder.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        enc = RelationEncoder("mock-relation", max_length=16)
    return enc


@pytest.fixture(scope="module")
def mock_verbalized_encoder():
    """VerbalizedSDPEncoder with patched LLM backbone."""
    from deepref.encoder.sdp_encoder import VerbalizedSDPEncoder
    mock_model = _make_mock_model(MOCK_HIDDEN)
    mock_tokenizer = MagicMock()
    with (
        patch("deepref.encoder.llm_encoder.AutoModel.from_pretrained", return_value=mock_model),
        patch("deepref.encoder.llm_encoder.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        enc = VerbalizedSDPEncoder("mock-verbalized")
    return enc


@pytest.fixture(scope="module")
def combine_bow_bow(bow_encoder):
    """CombineEmbeddings wrapping the same BoWSDPEncoder twice (no LLM)."""
    return CombineEmbeddings(bow_encoder, bow_encoder)


@pytest.fixture
def trainer(dataset, bow_encoder):
    """Fresh CombineRETrainer per test — prevents state leaking between tests."""
    from deepref.model.softmax_mlp import SoftmaxMLP

    combine = CombineEmbeddings(bow_encoder, bow_encoder)
    num_class = len(dataset.rel2id)
    model = SoftmaxMLP(
        sentence_encoder=combine,
        num_class=num_class,
        rel2id=dataset.rel2id,
        num_layers=2,
    )

    training_params = {
        "max_epoch": 1,
        "criterion": nn.CrossEntropyLoss(),
        "lr": 1e-3,
        "batch_size": 2,
        "opt": "adam",
        "weight_decay": 0.0,
        "warmup_step": 0,
    }
    ckpt = "/tmp/test_combine_re_trainer.pth.tar"

    return CombineRETrainer(
        model=model,
        train_dataset=Subset(dataset, _TRAIN_IDX),
        test_dataset=Subset(dataset, _TEST_IDX),
        ckpt=ckpt,
        training_parameters=training_params,
    ), ckpt


# ===========================================================================
# 1. CombineREDataset
# ===========================================================================

class TestCombineREDataset:

    def test_len_equals_sliced_rows(self, dataset):
        assert len(dataset) == SLICE_SIZE

    def test_rel2id_is_not_empty(self, dataset):
        assert len(dataset.rel2id) > 0

    def test_rel2id_keys_are_strings(self, dataset):
        for key in dataset.rel2id:
            assert isinstance(key, str)

    def test_rel2id_values_are_integers(self, dataset):
        for val in dataset.rel2id.values():
            assert isinstance(val, int)

    def test_rel2id_values_are_unique(self, dataset):
        values = list(dataset.rel2id.values())
        assert len(values) == len(set(values))

    def test_getitem_returns_tuple_of_two(self, dataset):
        result = dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_getitem_item_is_dict(self, dataset):
        item, _ = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_item_has_token_key(self, dataset):
        item, _ = dataset[0]
        assert "token" in item

    def test_getitem_item_has_h_key(self, dataset):
        item, _ = dataset[0]
        assert "h" in item

    def test_getitem_item_has_t_key(self, dataset):
        item, _ = dataset[0]
        assert "t" in item

    def test_getitem_token_is_list(self, dataset):
        item, _ = dataset[0]
        assert isinstance(item["token"], list)

    def test_getitem_token_elements_are_strings(self, dataset):
        item, _ = dataset[0]
        for tok in item["token"]:
            assert isinstance(tok, str)

    def test_getitem_h_has_name_key(self, dataset):
        item, _ = dataset[0]
        assert "name" in item["h"]

    def test_getitem_h_has_pos_key(self, dataset):
        item, _ = dataset[0]
        assert "pos" in item["h"]

    def test_getitem_t_has_name_key(self, dataset):
        item, _ = dataset[0]
        assert "name" in item["t"]

    def test_getitem_t_has_pos_key(self, dataset):
        item, _ = dataset[0]
        assert "pos" in item["t"]

    def test_getitem_pos_is_list_of_two_ints(self, dataset):
        item, _ = dataset[0]
        for entity in (item["h"], item["t"]):
            pos = entity["pos"]
            assert isinstance(pos, list)
            assert len(pos) == 2
            assert all(isinstance(p, int) for p in pos)

    def test_getitem_label_is_tensor(self, dataset):
        _, label = dataset[0]
        assert isinstance(label, torch.Tensor)

    def test_getitem_label_dtype_is_long(self, dataset):
        _, label = dataset[0]
        assert label.dtype == torch.long

    def test_getitem_label_is_valid_class_id(self, dataset):
        _, label = dataset[0]
        assert 0 <= label.item() < len(dataset.rel2id)

    def test_getitem_entity_name_consistent_with_tokens(self, dataset):
        """Entity name should be the concatenation of tokens at the given pos."""
        item, _ = dataset[0]
        for entity_key in ("h", "t"):
            ent = item[entity_key]
            start, end = ent["pos"]
            expected_name = " ".join(item["token"][start:end])
            assert ent["name"] == expected_name

    def test_missing_csv_returns_zero_length(self):
        ds = CombineREDataset("nonexistent_path/fake.csv")
        assert len(ds) == 0


# ===========================================================================
# 2. combine_collate_fn
# ===========================================================================

class TestCombineCollate:

    @pytest.fixture(autouse=True)
    def _batch(self, dataset):
        item0, label0 = dataset[0]
        item1, label1 = dataset[1]
        self.batch_output = combine_collate_fn([(item0, label0), (item1, label1)])
        self.batch_size = 2

    def test_returns_dict(self):
        assert isinstance(self.batch_output, dict)

    def test_has_labels_key(self):
        assert "labels" in self.batch_output

    def test_has_items_key(self):
        assert "items" in self.batch_output

    def test_labels_is_tensor(self):
        assert isinstance(self.batch_output["labels"], torch.Tensor)

    def test_labels_dtype_is_long(self):
        assert self.batch_output["labels"].dtype == torch.long

    def test_labels_shape_equals_batch_size(self):
        assert self.batch_output["labels"].shape == (self.batch_size,)

    def test_items_is_list(self):
        assert isinstance(self.batch_output["items"], list)

    def test_items_length_equals_batch_size(self):
        assert len(self.batch_output["items"]) == self.batch_size

    def test_items_elements_are_dicts(self):
        for it in self.batch_output["items"]:
            assert isinstance(it, dict)


# ===========================================================================
# 3. CombineEmbeddings — hidden-size helpers
# ===========================================================================

class TestCombineEmbeddingsHiddenSize:

    def test_bow_hidden_size_equals_dep_vocab_len(self, bow_encoder):
        size = CombineEmbeddings._get_hidden_size(bow_encoder)
        assert size == len(bow_encoder.dep_vocab)

    def test_relation_encoder_hidden_size_equals_encoder_attribute(
        self, mock_relation_encoder
    ):
        size = CombineEmbeddings._get_hidden_size(mock_relation_encoder)
        assert size == mock_relation_encoder.hidden_size

    def test_llm_encoder_hidden_size_equals_model_config(self, mock_llm_encoder):
        size = CombineEmbeddings._get_hidden_size(mock_llm_encoder)
        assert size == MOCK_HIDDEN

    def test_verbalized_sdp_hidden_size_equals_model_config(
        self, mock_verbalized_encoder
    ):
        size = CombineEmbeddings._get_hidden_size(mock_verbalized_encoder)
        assert size == MOCK_HIDDEN

    def test_model_config_hidden_size_is_sum_of_both(self, bow_encoder):
        h = len(bow_encoder.dep_vocab)
        combine = CombineEmbeddings(bow_encoder, bow_encoder)
        assert combine.model.config.hidden_size == h + h

    def test_unknown_encoder_raises_value_error(self):
        class _Unknown(nn.Module):
            pass
        with pytest.raises(ValueError, match="Cannot determine hidden size"):
            CombineEmbeddings._get_hidden_size(_Unknown())


# ===========================================================================
# 4. CombineEmbeddings — _encode_single dispatch
# ===========================================================================

class TestCombineEmbeddingsEncodeSingle:
    """
    Verify that _encode_single dispatches correctly to each encoder type and
    always returns a 1-D float32 tensor of the expected length.
    Transformer forward passes are replaced by lightweight mocks so no LLM
    download is required for these tests.
    """

    # ── BoWSDPEncoder ────────────────────────────────────────────────────────

    def test_bow_returns_tensor(self, bow_encoder):
        result = CombineEmbeddings._encode_single(bow_encoder, ITEM_SIMPLE)
        assert isinstance(result, torch.Tensor)

    def test_bow_is_1d(self, bow_encoder):
        result = CombineEmbeddings._encode_single(bow_encoder, ITEM_SIMPLE)
        assert result.ndim == 1

    def test_bow_is_float32(self, bow_encoder):
        result = CombineEmbeddings._encode_single(bow_encoder, ITEM_SIMPLE)
        assert result.dtype == torch.float32

    def test_bow_length_equals_dep_vocab(self, bow_encoder):
        result = CombineEmbeddings._encode_single(bow_encoder, ITEM_SIMPLE)
        assert result.shape[0] == len(bow_encoder.dep_vocab)

    # ── RelationEncoder ──────────────────────────────────────────────────────

    def test_relation_encoder_is_1d(self, mock_relation_encoder):
        fake_tok = (
            torch.zeros(1, 16, dtype=torch.long),
            torch.ones(1, 16, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        )
        fake_emb = torch.zeros(1, MOCK_HIDDEN * 3)
        with (
            mock.patch.object(mock_relation_encoder, "tokenize", return_value=fake_tok),
            mock.patch.object(mock_relation_encoder, "forward", return_value=fake_emb),
        ):
            result = CombineEmbeddings._encode_single(mock_relation_encoder, ITEM_SIMPLE)
        assert result.ndim == 1

    def test_relation_encoder_length_is_3h(self, mock_relation_encoder):
        fake_tok = (
            torch.zeros(1, 16, dtype=torch.long),
            torch.ones(1, 16, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        )
        fake_emb = torch.zeros(1, MOCK_HIDDEN * 3)
        with (
            mock.patch.object(mock_relation_encoder, "tokenize", return_value=fake_tok),
            mock.patch.object(mock_relation_encoder, "forward", return_value=fake_emb),
        ):
            result = CombineEmbeddings._encode_single(mock_relation_encoder, ITEM_SIMPLE)
        assert result.shape[0] == MOCK_HIDDEN * 3

    def test_relation_encoder_is_float32(self, mock_relation_encoder):
        fake_tok = (
            torch.zeros(1, 16, dtype=torch.long),
            torch.ones(1, 16, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        )
        fake_emb = torch.zeros(1, MOCK_HIDDEN * 3, dtype=torch.float16)
        with (
            mock.patch.object(mock_relation_encoder, "tokenize", return_value=fake_tok),
            mock.patch.object(mock_relation_encoder, "forward", return_value=fake_emb),
        ):
            result = CombineEmbeddings._encode_single(mock_relation_encoder, ITEM_SIMPLE)
        assert result.dtype == torch.float32

    # ── LLMEncoder ───────────────────────────────────────────────────────────

    def test_llm_encoder_is_1d(self, mock_llm_encoder):
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_llm_encoder, "forward", return_value=fake_emb):
            result = CombineEmbeddings._encode_single(mock_llm_encoder, ITEM_SIMPLE)
        assert result.ndim == 1

    def test_llm_encoder_length_equals_hidden(self, mock_llm_encoder):
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_llm_encoder, "forward", return_value=fake_emb):
            result = CombineEmbeddings._encode_single(mock_llm_encoder, ITEM_SIMPLE)
        assert result.shape[0] == MOCK_HIDDEN

    def test_llm_encoder_encodes_joined_token_text(self, mock_llm_encoder):
        """LLMEncoder path should join tokens into a string before encoding."""
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_llm_encoder, "forward", return_value=fake_emb) as mock_fwd:
            CombineEmbeddings._encode_single(mock_llm_encoder, ITEM_SIMPLE)
        called_text = mock_fwd.call_args[0][0]
        assert called_text == " ".join(ITEM_SIMPLE["token"])

    # ── VerbalizedSDPEncoder ─────────────────────────────────────────────────

    def test_verbalized_sdp_is_1d(self, mock_verbalized_encoder):
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_verbalized_encoder, "forward", return_value=fake_emb):
            result = CombineEmbeddings._encode_single(mock_verbalized_encoder, ITEM_SIMPLE)
        assert result.ndim == 1

    def test_verbalized_sdp_length_equals_hidden(self, mock_verbalized_encoder):
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_verbalized_encoder, "forward", return_value=fake_emb):
            result = CombineEmbeddings._encode_single(mock_verbalized_encoder, ITEM_SIMPLE)
        assert result.shape[0] == MOCK_HIDDEN

    def test_verbalized_sdp_called_with_item_dict(self, mock_verbalized_encoder):
        """VerbalizedSDPEncoder path passes the raw item dict to forward."""
        fake_emb = torch.zeros(1, MOCK_HIDDEN)
        with mock.patch.object(mock_verbalized_encoder, "forward", return_value=fake_emb) as mock_fwd:
            CombineEmbeddings._encode_single(mock_verbalized_encoder, ITEM_SIMPLE)
        mock_fwd.assert_called_once_with(ITEM_SIMPLE)


# ===========================================================================
# 5. CombineEmbeddings — forward
# ===========================================================================

class TestCombineEmbeddingsForward:
    """
    End-to-end forward tests using two real BoWSDPEncoder instances.
    No LLM download required.
    """

    def test_returns_tensor(self, combine_bow_bow):
        result = combine_bow_bow([ITEM_SIMPLE])
        assert isinstance(result, torch.Tensor)

    def test_dtype_is_float32(self, combine_bow_bow):
        result = combine_bow_bow([ITEM_SIMPLE])
        assert result.dtype == torch.float32

    def test_ndim_is_2(self, combine_bow_bow):
        result = combine_bow_bow([ITEM_SIMPLE])
        assert result.ndim == 2

    def test_single_item_batch_dim_is_1(self, combine_bow_bow):
        result = combine_bow_bow([ITEM_SIMPLE])
        assert result.shape[0] == 1

    def test_batch_size_matches_input_list_length(self, combine_bow_bow):
        batch = [ITEM_SIMPLE, ITEM_SIMPLE]
        result = combine_bow_bow(batch)
        assert result.shape[0] == len(batch)

    def test_embedding_dim_equals_combined_hidden(self, bow_encoder, combine_bow_bow):
        expected_dim = combine_bow_bow.model.config.hidden_size
        result = combine_bow_bow([ITEM_SIMPLE])
        assert result.shape[1] == expected_dim

    def test_forward_callable_with_items_kwarg(self, combine_bow_bow):
        """SoftmaxMLP calls the encoder as encoder(**args) where args={'items': …}."""
        result = combine_bow_bow(items=[ITEM_SIMPLE])
        assert result.shape[0] == 1


# ===========================================================================
# 6. CombineRETrainer — initialisation
# ===========================================================================

class TestCombineRETrainerInit:

    def test_is_instance_of_sentence_re_trainer(self, trainer):
        t, _ = trainer
        assert isinstance(t, SentenceRETrainer)

    def test_has_train_loader(self, trainer):
        t, _ = trainer
        assert hasattr(t, "train_loader")

    def test_train_loader_is_dataloader(self, trainer):
        t, _ = trainer
        assert isinstance(t.train_loader, DataLoader)

    def test_has_test_loader(self, trainer):
        t, _ = trainer
        assert hasattr(t, "test_loader")

    def test_test_loader_is_dataloader(self, trainer):
        t, _ = trainer
        assert isinstance(t.test_loader, DataLoader)

    def test_has_optimizer(self, trainer):
        t, _ = trainer
        assert hasattr(t, "optimizer")

    def test_train_loader_uses_combine_collate(self, trainer):
        """Batches from the train loader must have 'labels' and 'items' keys."""
        t, _ = trainer
        batch = next(iter(t.train_loader))
        assert isinstance(batch, dict)
        assert "labels" in batch
        assert "items" in batch

    def test_test_loader_uses_combine_collate(self, trainer):
        t, _ = trainer
        batch = next(iter(t.test_loader))
        assert isinstance(batch, dict)
        assert "labels" in batch
        assert "items" in batch


# ===========================================================================
# 7. CombineRETrainer — eval_model
# ===========================================================================

class TestCombineRETrainerEvalModel:

    def test_returns_tuple_of_three(self, trainer):
        t, _ = trainer
        result = t.eval_model(t.test_loader)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_first_element_is_dict(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert isinstance(result_dict, dict)

    def test_result_has_acc(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert "acc" in result_dict

    def test_result_has_micro_f1(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert "micro_f1" in result_dict

    def test_result_has_macro_f1(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert "macro_f1" in result_dict

    def test_result_has_micro_p(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert "micro_p" in result_dict

    def test_result_has_micro_r(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert "micro_r" in result_dict

    def test_acc_is_between_0_and_1(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert 0.0 <= result_dict["acc"] <= 1.0

    def test_micro_f1_is_between_0_and_1(self, trainer):
        t, _ = trainer
        result_dict, _, _ = t.eval_model(t.test_loader)
        assert 0.0 <= result_dict["micro_f1"] <= 1.0

    def test_predictions_length_matches_test_size(self, trainer):
        t, _ = trainer
        _, preds, _ = t.eval_model(t.test_loader)
        assert len(preds) == len(_TEST_IDX)

    def test_labels_length_matches_test_size(self, trainer):
        t, _ = trainer
        _, _, labels = t.eval_model(t.test_loader)
        assert len(labels) == len(_TEST_IDX)

    def test_predictions_are_valid_class_ids(self, trainer, dataset):
        t, _ = trainer
        num_class = len(dataset.rel2id)
        _, preds, _ = t.eval_model(t.test_loader)
        for p in preds:
            assert 0 <= p < num_class


# ===========================================================================
# 8. CombineRETrainer — iterate_loader
# ===========================================================================

class TestCombineRETrainerIterateLoader:

    def test_training_returns_float(self, trainer):
        t, _ = trainer
        loss = t.iterate_loader(t.train_loader, training=True)
        assert isinstance(loss, float)

    def test_training_loss_is_non_negative(self, trainer):
        t, _ = trainer
        loss = t.iterate_loader(t.train_loader, training=True)
        assert loss >= 0.0

    def test_inference_returns_none(self, trainer):
        t, _ = trainer
        result = t.iterate_loader(t.test_loader, training=False)
        assert result is None


# ===========================================================================
# 9. CombineRETrainer — train_model
# ===========================================================================

class TestCombineRETrainerTrainModel:

    def test_returns_float(self, trainer):
        t, _ = trainer
        with patch("mlflow.log_metrics"):
            best = t.train_model(metric="micro_f1")
        assert isinstance(best, float)

    def test_best_metric_is_non_negative(self, trainer):
        t, _ = trainer
        with patch("mlflow.log_metrics"):
            best = t.train_model(metric="micro_f1")
        assert best >= 0.0

    def test_saves_checkpoint_when_metric_improves(self, trainer):
        t, ckpt = trainer
        # Force eval_model to return a clearly positive metric so the
        # checkpoint is guaranteed to be written on the first epoch.
        good_result = {
            "acc": 1.0,
            "micro_f1": 1.0,
            "macro_f1": 1.0,
            "micro_p": 1.0,
            "micro_r": 1.0,
        }
        with (
            patch("mlflow.log_metrics"),
            mock.patch.object(t, "eval_model", return_value=(good_result, [], [])),
        ):
            t.train_model(metric="micro_f1")
        assert os.path.exists(ckpt)

    def test_mlflow_log_metrics_called_per_epoch(self, trainer):
        t, _ = trainer
        good_result = {
            "acc": 0.9, "micro_f1": 0.8, "macro_f1": 0.7,
            "micro_p": 0.8, "micro_r": 0.8,
        }
        with (
            patch("mlflow.log_metrics") as mock_log,
            mock.patch.object(t, "eval_model", return_value=(good_result, [], [])),
        ):
            t.train_model(metric="micro_f1")
        # Called once per epoch (max_epoch=1 in the trainer fixture)
        assert mock_log.call_count == t.max_epoch
