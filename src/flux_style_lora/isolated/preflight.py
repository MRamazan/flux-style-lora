import json
import os
from pathlib import Path

import yaml
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.stable_diffusion_model import StableDiffusion

configuration_path = Path(os.environ["FLUX_TRAINING_CONFIGURATION"])
configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
processes = configuration.get("config", {}).get("process", [])
if not isinstance(processes, list) or len(processes) != 1:
    raise RuntimeError("The configuration must contain exactly one process.")
process = processes[0]
if process.get("type") != "sd_trainer":
    raise RuntimeError("The process type must be sd_trainer.")
model_section = process.get("model", {})
network_section = process.get("network", {})
train_section = process.get("train", {})
dataset_section = process.get("datasets", [])
if model_section.get("arch") != "flux":
    raise RuntimeError("The model architecture must be flux.")
if network_section.get("type") != "lora":
    raise RuntimeError("This pipeline supports LoRA network training only.")
if not network_section.get("transformer_only"):
    raise RuntimeError("FLUX LoRA training must target the transformer.")
if train_section.get("train_text_encoder"):
    raise RuntimeError("FLUX text-encoder training is not supported by this pipeline.")
if train_section.get("merge_network_on_save"):
    raise RuntimeError("Permanent LoRA merging must remain disabled.")
if not isinstance(dataset_section, list) or len(dataset_section) != 1:
    raise RuntimeError("Exactly one canonical dataset directory is required.")
dataset_directory = Path(dataset_section[0]["folder_path"])
images = [
    path for path in dataset_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
]
captions = list(dataset_directory.glob("*.txt"))
if not images or len(images) != len(captions):
    raise RuntimeError("The canonical dataset directory does not contain matching pairs.")
model_configuration = ModelConfig(**model_section)
network_configuration = NetworkConfig(**network_section)
probe = StableDiffusion(
    device="cuda:0",
    model_config=model_configuration,
    dtype=train_section.get("dtype", "bf16"),
)
if not model_configuration.is_flux:
    raise RuntimeError("The resolved model configuration is not a FLUX configuration.")
result = {
    "run_name": configuration["config"]["name"],
    "dataset_pairs": len(images),
    "model_arch": model_configuration.arch,
    "is_flux": bool(model_configuration.is_flux),
    "is_transformer": bool(getattr(probe, "is_transformer", True)),
    "target_lora_modules": list(getattr(probe, "target_lora_modules", []) or []),
    "network_type": network_configuration.type,
    "network_rank": network_configuration.linear,
    "network_alpha": network_configuration.linear_alpha,
    "training_steps": train_section.get("steps"),
    "train_text_encoder": train_section.get("train_text_encoder"),
    "merge_network_on_save": train_section.get("merge_network_on_save"),
}
print(json.dumps(result))
