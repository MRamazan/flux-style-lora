import pytest

from flux_style_lora import DatasetConfig, EvaluationConfig, TrainingConfig
from flux_style_lora.errors import ConfigurationError


def test_dataset_config_requires_trigger_word() -> None:
    with pytest.raises(ConfigurationError):
        DatasetConfig(trigger_word="").validate()


def test_training_config_accepts_style_defaults() -> None:
    TrainingConfig().validate()


def test_evaluation_config_requires_prompts() -> None:
    with pytest.raises(ConfigurationError):
        EvaluationConfig().validate()
