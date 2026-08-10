import json

import pytest

from flux_style_lora import assets, constants
from flux_style_lora.paths import ProjectPaths

REQUIRED = constants.MODEL_REQUIRED_ENTRIES


def _populate_snapshot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_index.json").write_text(json.dumps({}), encoding="utf-8")
    for entry in REQUIRED:
        if entry.endswith(".json"):
            continue
        (directory / entry).mkdir(parents=True, exist_ok=True)


def _install_mocks(monkeypatch, populate=True):
    def fake_resolve(repo_id, revision=None):
        assert repo_id == constants.MODEL_REPOSITORY
        return "fluxrev"

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns=None):
        if populate:
            _populate_snapshot(local_dir)
        return local_dir

    monkeypatch.setattr(assets, "resolve_revision", fake_resolve)
    monkeypatch.setattr(assets, "hf_snapshot", fake_snapshot)
    monkeypatch.setattr(assets, "MINIMUM_TRAINING_DISK_BYTES", 0)
    monkeypatch.setattr(assets, "MINIMUM_INFERENCE_DISK_BYTES", 0)


def test_model_constants_target_flux():
    assert constants.MODEL_REPOSITORY == "black-forest-labs/FLUX.1-dev"
    assert constants.MODEL_ARCHITECTURE == "flux"
    assert "transformer" in constants.MODEL_REQUIRED_ENTRIES


def test_no_separate_text_encoder_or_vae_repositories():
    assert not hasattr(constants, "TEXT_ENCODER_REPOSITORY")
    assert not hasattr(constants, "VAE_REPOSITORY")
    assert not hasattr(constants, "TRAINING_CHECKPOINT_FILENAME")
    assert not hasattr(constants, "TARGET_LORA_MODULES")


def test_snapshot_excludes_single_file_checkpoints():
    patterns = set(constants.MODEL_ALLOW_PATTERNS)
    assert "transformer/*" in patterns
    assert "text_encoder_2/*" in patterns
    assert not any(pattern.endswith("flux1-dev.safetensors") for pattern in patterns)


def test_prepare_training_assets_downloads_one_snapshot(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_mocks(monkeypatch)
    manifest = assets.prepare_training_assets(paths)
    model = manifest["training_model"]
    assert model["repository"] == constants.MODEL_REPOSITORY
    assert model["revision"] == "fluxrev"
    assert model["format"] == "diffusers_pipeline"
    assert model["bundled_text_encoder"] is True
    assert model["bundled_vae"] is True
    assert model["local_directory"].endswith("flux1_dev")
    assert "text_encoder" not in manifest
    assert "vae" not in manifest


def test_incomplete_snapshot_is_rejected(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_mocks(monkeypatch, populate=False)
    with pytest.raises(assets.AssetError):
        assets.prepare_training_assets(paths)


def test_inference_reuses_training_snapshot(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_mocks(monkeypatch)
    training = assets.prepare_training_assets(paths)
    inference = assets.prepare_inference_assets(paths)
    assert (
        inference["inference_model"]["local_directory"]
        == (training["training_model"]["local_directory"])
    )
    assert inference["official_defaults"]["num_inference_steps"] == 20
    assert inference["official_defaults"]["guidance_scale"] == 3.5


def test_flux_defaults_match_recommendations():
    from flux_style_lora import EvaluationConfig, TrainingConfig

    training = TrainingConfig()
    assert training.learning_rate == 0.0001
    assert training.lora_rank == 16
    assert training.lora_alpha == 16
    assert training.quantize_transformer is True
    evaluation = EvaluationConfig(prompts=["x"])
    assert evaluation.inference_steps == 20
    assert evaluation.guidance_scale == 3.5
