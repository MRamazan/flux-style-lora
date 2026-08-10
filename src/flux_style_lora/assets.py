from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

from .constants import (
    MINIMUM_INFERENCE_DISK_BYTES,
    MINIMUM_TRAINING_DISK_BYTES,
    MODEL_ALLOW_PATTERNS,
    MODEL_LOCAL_DIRECTORY,
    MODEL_REPOSITORY,
    MODEL_REQUIRED_ENTRIES,
)
from .errors import AssetError
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths


def hf_model_info(repo_id: str, revision: str | None = None) -> Any:
    import huggingface_hub

    return huggingface_hub.HfApi().model_info(
        repo_id=repo_id, revision=revision or "main", token=None
    )


def hf_snapshot(
    repo_id: str, revision: str, local_dir: Path, allow_patterns: list[str] | None = None
) -> Path:
    import huggingface_hub

    return Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=None,
        )
    )


def resolve_revision(repo_id: str, revision: str | None = None) -> str:
    information = hf_model_info(repo_id, revision)
    resolved = getattr(information, "sha", None)
    if not resolved:
        raise AssetError(f"Unable to resolve a commit for repository {repo_id}.")
    return resolved


def read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with Path(path).open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise AssetError(f"The safetensors file is truncated: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise AssetError(f"The safetensors header is truncated: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.pop("__metadata__", {})
    return metadata, header


def verify_model_directory(model_directory: Path) -> list[str]:
    missing = [entry for entry in MODEL_REQUIRED_ENTRIES if not (model_directory / entry).exists()]
    if missing:
        raise AssetError(
            f"The FLUX.1-dev model snapshot at {model_directory} is incomplete. "
            f"Missing entries: {missing}. A Diffusers FLUX pipeline repository is required. "
            f"If the download was empty, confirm that the gated repository {MODEL_REPOSITORY} "
            f"has been accepted for your Hugging Face account."
        )
    return list(MODEL_REQUIRED_ENTRIES)


def prepare_model(paths: ProjectPaths, minimum_free_bytes: int) -> dict[str, Any]:
    model_directory = paths.models / MODEL_LOCAL_DIRECTORY
    already_present = (model_directory / "model_index.json").is_file()
    free_bytes = shutil.disk_usage(paths.root).free
    if free_bytes < minimum_free_bytes and not already_present:
        raise AssetError(
            f"At least {minimum_free_bytes / 1024**3:.0f} GiB of free disk is required for the "
            f"FLUX model. Available: {free_bytes / 1024**3:.2f} GiB."
        )
    model_directory.mkdir(parents=True, exist_ok=True)
    revision = resolve_revision(MODEL_REPOSITORY)
    hf_snapshot(MODEL_REPOSITORY, revision, model_directory, list(MODEL_ALLOW_PATTERNS))
    verified = verify_model_directory(model_directory)
    return {
        "repository": MODEL_REPOSITORY,
        "revision": revision,
        "local_directory": str(model_directory),
        "format": "diffusers_pipeline",
        "verified_entries": verified,
        "bundled_text_encoder": True,
        "bundled_vae": True,
    }


def prepare_training_assets(paths: ProjectPaths) -> dict[str, Any]:
    paths.models.mkdir(parents=True, exist_ok=True)
    manifest = {"training_model": prepare_model(paths, MINIMUM_TRAINING_DISK_BYTES)}
    write_json_atomic(paths.training_asset_manifest, manifest)
    return manifest


def prepare_inference_assets(paths: ProjectPaths) -> dict[str, Any]:
    model = None
    if paths.training_asset_manifest.is_file():
        existing = read_json(paths.training_asset_manifest)["training_model"]
        directory = Path(existing["local_directory"])
        if (directory / "model_index.json").is_file():
            verify_model_directory(directory)
            model = existing
    if model is None:
        model = prepare_model(paths, MINIMUM_INFERENCE_DISK_BYTES)
    manifest = {
        "inference_model": model,
        "official_defaults": {"num_inference_steps": 20, "guidance_scale": 3.5},
    }
    write_json_atomic(paths.inference_asset_manifest, manifest)
    return manifest
