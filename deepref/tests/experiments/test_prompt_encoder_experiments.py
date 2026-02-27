"""
Unit tests for deepref/experiments/run_prompt_encoder_experiments.py.

These tests cover:
  1. generate_combinations  — combination cartesian product logic.
  2. build_hparams          — hyperparameter dict construction.
  3. run_experiment         — MLflow logging and Training integration (mocked).
  4. main / parse_args      — CLI arg parsing and dry-run behaviour.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from deepref import config
from deepref.experiments.run_prompt_encoder_experiments import (
    PROMPT_ENCODER_PRETRAIN_WEIGHTS,
    build_hparams,
    generate_combinations,
    main,
    parse_args,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_DATASETS = config.DATASETS
ALL_PRETRAIN = PROMPT_ENCODER_PRETRAIN_WEIGHTS

SINGLE_PREPROCESSING: list[list[str]] = [[]]


# ===========================================================================
# 1. generate_combinations
# ===========================================================================


class TestGenerateCombinations:

    def test_returns_list(self):
        result = generate_combinations(["semeval2010"], ["bert-base-uncased"], [[]])
        assert isinstance(result, list)

    def test_each_element_is_dict(self):
        result = generate_combinations(["semeval2010"], ["bert-base-uncased"], [[]])
        for item in result:
            assert isinstance(item, dict)

    def test_combo_has_required_keys(self):
        result = generate_combinations(["semeval2010"], ["bert-base-uncased"], [[]])
        for item in result:
            assert "dataset" in item
            assert "pretrain" in item
            assert "preprocessing" in item

    def test_single_dataset_single_pretrain_one_combo(self):
        result = generate_combinations(["semeval2010"], ["bert-base-uncased"], [[]])
        assert len(result) == 1

    def test_two_datasets_one_pretrain_two_combos(self):
        result = generate_combinations(
            ["semeval2010", "ddi"], ["bert-base-uncased"], [[]]
        )
        assert len(result) == 2

    def test_one_dataset_two_pretrain_two_combos(self):
        result = generate_combinations(
            ["semeval2010"], ["bert-base-uncased", "dmis-lab/biobert-v1.1"], [[]]
        )
        assert len(result) == 2

    def test_total_count_equals_cartesian_product(self):
        datasets = ["semeval2010", "ddi"]
        pretrain = ["bert-base-uncased", "dmis-lab/biobert-v1.1"]
        preprocessing = [[], ["sw"]]
        result = generate_combinations(datasets, pretrain, preprocessing)
        assert len(result) == len(datasets) * len(pretrain) * len(preprocessing)

    def test_all_config_datasets_covered(self):
        combos = generate_combinations(ALL_DATASETS, ["bert-base-uncased"], [[]])
        covered = {c["dataset"] for c in combos}
        assert covered == set(ALL_DATASETS)

    def test_all_pretrain_weights_covered(self):
        combos = generate_combinations(["semeval2010"], ALL_PRETRAIN, [[]])
        covered = {c["pretrain"] for c in combos}
        assert covered == set(ALL_PRETRAIN)

    def test_each_combination_is_unique(self):
        combos = generate_combinations(ALL_DATASETS, ALL_PRETRAIN, [[]])
        tuples = [(c["dataset"], c["pretrain"], tuple(c["preprocessing"])) for c in combos]
        assert len(tuples) == len(set(tuples))

    def test_empty_datasets_returns_empty(self):
        result = generate_combinations([], ["bert-base-uncased"], [[]])
        assert result == []

    def test_empty_pretrain_returns_empty(self):
        result = generate_combinations(["semeval2010"], [], [[]])
        assert result == []

    def test_preprocessing_value_preserved(self):
        combos = generate_combinations(
            ["semeval2010"], ["bert-base-uncased"], [["sw", "p"]]
        )
        assert combos[0]["preprocessing"] == ["sw", "p"]

    def test_no_preprocessing_produces_empty_list(self):
        combos = generate_combinations(["semeval2010"], ["bert-base-uncased"], [[]])
        assert combos[0]["preprocessing"] == []


# ===========================================================================
# 2. build_hparams
# ===========================================================================


class TestBuildHparams:

    def _make(self, **kwargs):
        defaults = dict(
            pretrain="bert-base-uncased",
            preprocessing=[],
            max_epoch=3,
            batch_size=16,
            lr=2e-5,
            max_length=128,
        )
        defaults.update(kwargs)
        return build_hparams(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._make(), dict)

    def test_model_is_prompt_encoder(self):
        assert self._make()["model"] == "prompt_encoder"

    def test_pretrain_is_set(self):
        hp = self._make(pretrain="dmis-lab/biobert-v1.1")
        assert hp["pretrain"] == "dmis-lab/biobert-v1.1"

    def test_preprocessing_is_set(self):
        hp = self._make(preprocessing=["sw", "p"])
        assert hp["preprocessing"] == ["sw", "p"]

    def test_max_epoch_is_set(self):
        hp = self._make(max_epoch=5)
        assert hp["max_epoch"] == 5

    def test_batch_size_is_set(self):
        hp = self._make(batch_size=32)
        assert hp["batch_size"] == 32

    def test_lr_is_set(self):
        hp = self._make(lr=1e-4)
        assert hp["lr"] == pytest.approx(1e-4)

    def test_max_length_is_set(self):
        hp = self._make(max_length=256)
        assert hp["max_length"] == 256

    def test_has_position_embed_key(self):
        assert "position_embed" in self._make()

    def test_has_pos_tags_embed_key(self):
        assert "pos_tags_embed" in self._make()

    def test_has_deps_embed_key(self):
        assert "deps_embed" in self._make()


# ===========================================================================
# 3. run_experiment (MLflow + Training mocked)
# ===========================================================================


class TestRunExperiment:

    _COMBO = {
        "dataset": "semeval2010",
        "pretrain": "bert-base-uncased",
        "preprocessing": [],
    }
    _FAKE_RESULT = {
        "acc": 0.85,
        "micro_f1": 0.80,
        "macro_f1": 0.78,
        "micro_p": 0.81,
        "micro_r": 0.79,
    }

    def _run(self, combo=None, result=_FAKE_RESULT, side_effect=None):
        combo = combo or self._COMBO.copy()
        mock_training = MagicMock()
        mock_training.train.return_value = result
        if side_effect:
            mock_training.train.side_effect = side_effect

        with (
            patch(
                "deepref.experiments.run_prompt_encoder_experiments.Training",
                return_value=mock_training,
            ) as MockTraining,
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_params") as mock_log_params,
            patch("mlflow.log_metrics") as mock_log_metrics,
            patch("mlflow.set_tag") as mock_set_tag,
        ):
            mock_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            returned = run_experiment(
                combo,
                max_epoch=3,
                batch_size=16,
                lr=2e-5,
                max_length=128,
                mlflow_experiment="test_exp",
            )
        return returned, MockTraining, mock_log_params, mock_log_metrics, mock_set_tag

    def test_returns_result_dict_on_success(self):
        returned, *_ = self._run()
        assert returned == self._FAKE_RESULT

    def test_returns_none_on_exception(self):
        returned, *_ = self._run(side_effect=RuntimeError("boom"))
        assert returned is None

    def test_training_called_with_correct_dataset(self):
        _, MockTraining, *_ = self._run()
        args, _ = MockTraining.call_args
        assert args[0] == "semeval2010"

    def test_training_called_with_prompt_encoder_model(self):
        _, MockTraining, *_ = self._run()
        _, kwargs = MockTraining.call_args
        hparams = MockTraining.call_args[0][1]
        assert hparams["model"] == "prompt_encoder"

    def test_mlflow_log_params_called(self):
        _, _, mock_log_params, *_ = self._run()
        mock_log_params.assert_called_once()

    def test_mlflow_log_params_includes_dataset(self):
        _, _, mock_log_params, *_ = self._run()
        logged = mock_log_params.call_args[0][0]
        assert logged["dataset"] == "semeval2010"

    def test_mlflow_log_params_includes_pretrain(self):
        _, _, mock_log_params, *_ = self._run()
        logged = mock_log_params.call_args[0][0]
        assert logged["pretrain"] == "bert-base-uncased"

    def test_mlflow_log_params_includes_model(self):
        _, _, mock_log_params, *_ = self._run()
        logged = mock_log_params.call_args[0][0]
        assert logged["model"] == "prompt_encoder"

    def test_mlflow_log_metrics_called_on_success(self):
        _, _, _, mock_log_metrics, _ = self._run()
        mock_log_metrics.assert_called_once()

    def test_mlflow_log_metrics_includes_acc(self):
        _, _, _, mock_log_metrics, _ = self._run()
        logged = mock_log_metrics.call_args[0][0]
        assert "acc" in logged

    def test_mlflow_log_metrics_includes_micro_f1(self):
        _, _, _, mock_log_metrics, _ = self._run()
        logged = mock_log_metrics.call_args[0][0]
        assert "micro_f1" in logged

    def test_mlflow_log_metrics_includes_macro_f1(self):
        _, _, _, mock_log_metrics, _ = self._run()
        logged = mock_log_metrics.call_args[0][0]
        assert "macro_f1" in logged

    def test_mlflow_status_tag_success(self):
        _, _, _, _, mock_set_tag = self._run()
        mock_set_tag.assert_any_call("status", "success")

    def test_mlflow_status_tag_failed_on_exception(self):
        _, _, _, _, mock_set_tag = self._run(side_effect=RuntimeError("boom"))
        mock_set_tag.assert_any_call("status", "failed")

    def test_metrics_not_logged_on_exception(self):
        _, _, _, mock_log_metrics, _ = self._run(side_effect=RuntimeError("boom"))
        mock_log_metrics.assert_not_called()

    def test_run_name_contains_dataset(self):
        with (
            patch(
                "deepref.experiments.run_prompt_encoder_experiments.Training",
                return_value=MagicMock(train=MagicMock(return_value=self._FAKE_RESULT)),
            ),
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.set_tag"),
        ):
            mock_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_run.return_value.__exit__ = MagicMock(return_value=False)
            run_experiment(
                self._COMBO,
                max_epoch=3,
                batch_size=16,
                lr=2e-5,
                max_length=128,
                mlflow_experiment="test_exp",
            )
            _, kwargs = mock_run.call_args
            assert "semeval2010" in kwargs.get("run_name", "")


# ===========================================================================
# 4. parse_args and main
# ===========================================================================


class TestParseArgs:

    def test_default_datasets_are_all_config_datasets(self):
        args = parse_args([])
        assert args.datasets == config.DATASETS

    def test_default_pretrain_are_prompt_encoder_weights(self):
        args = parse_args([])
        assert args.pretrain == PROMPT_ENCODER_PRETRAIN_WEIGHTS

    def test_custom_dataset_is_accepted(self):
        args = parse_args(["--datasets", "semeval2010"])
        assert args.datasets == ["semeval2010"]

    def test_custom_pretrain_is_accepted(self):
        args = parse_args(["--pretrain", "bert-base-uncased"])
        assert args.pretrain == ["bert-base-uncased"]

    def test_dry_run_defaults_to_false(self):
        args = parse_args([])
        assert args.dry_run is False

    def test_dry_run_flag_enables_dry_run(self):
        args = parse_args(["--dry_run"])
        assert args.dry_run is True

    def test_max_epoch_default(self):
        args = parse_args([])
        assert args.max_epoch == 3

    def test_batch_size_default(self):
        args = parse_args([])
        assert args.batch_size == 16

    def test_lr_default(self):
        args = parse_args([])
        assert args.lr == pytest.approx(2e-5)

    def test_max_length_default(self):
        args = parse_args([])
        assert args.max_length == 128

    def test_experiment_default(self):
        args = parse_args([])
        assert args.experiment == "prompt_encoder_combinations"


class TestMainDryRun:

    def test_dry_run_prints_all_combinations(self, capsys):
        with (
            patch("mlflow.set_experiment"),
            patch("mlflow.set_tracking_uri"),
        ):
            main(
                [
                    "--datasets", "semeval2010",
                    "--pretrain", "bert-base-uncased",
                    "--dry_run",
                ]
            )
        captured = capsys.readouterr()
        assert "semeval2010" in captured.out
        assert "bert-base-uncased" in captured.out

    def test_dry_run_does_not_call_training(self):
        with (
            patch("mlflow.set_experiment"),
            patch(
                "deepref.experiments.run_prompt_encoder_experiments.Training"
            ) as MockTraining,
        ):
            main(
                [
                    "--datasets", "semeval2010",
                    "--pretrain", "bert-base-uncased",
                    "--dry_run",
                ]
            )
        MockTraining.assert_not_called()

    def test_dry_run_prints_one_line_per_combination(self, capsys):
        with patch("mlflow.set_experiment"):
            main(
                [
                    "--datasets", "semeval2010", "ddi",
                    "--pretrain", "bert-base-uncased",
                    "--dry_run",
                ]
            )
        captured = capsys.readouterr()
        # Two datasets × one pretrain × one preprocessing = 2 lines
        lines = [l for l in captured.out.splitlines() if l.strip()]
        assert len(lines) == 2
