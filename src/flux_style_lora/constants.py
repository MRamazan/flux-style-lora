from dataclasses import dataclass

MODEL_REPOSITORY = "black-forest-labs/FLUX.1-dev"
MODEL_LOCAL_DIRECTORY = "flux1_dev"
MODEL_REQUIRED_ENTRIES = (
    "model_index.json",
    "transformer",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "vae",
    "scheduler",
)
MODEL_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
)

AI_TOOLKIT_REPOSITORY = "https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_REVISION = "main"
VIRTUALENV_VERSION = "21.6.1"
TORCH_VERSION = "2.9.1"
TORCHVISION_VERSION = "0.24.1"
TORCHAUDIO_VERSION = "2.9.1"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

MODEL_ARCHITECTURE = "flux"
PRIMARY_ADAPTER_NAME = "style_adapter"
MAX_TEXT_LENGTH = 512
SUPPORTS_NEGATIVE_PROMPT = False

DEFAULT_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_SAMPLE_GUIDANCE_SCALE = 4.0
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_LORA_RANK = 16

MINIMUM_TRAINING_DISK_BYTES = 60 * 1024**3
MINIMUM_INFERENCE_DISK_BYTES = 45 * 1024**3

ACCEPTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
KNOWN_UNSUPPORTED_IMAGE_EXTENSIONS = (".webp", ".bmp", ".tif", ".tiff", ".gif")
CAPTION_EXTENSION = ".txt"

BUNDLE_TYPE = "flux_style_lora_evaluation_bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
MAX_BUNDLE_FILES = 5000
MAX_BUNDLE_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_LORA_FILE_BYTES = 1024**3
PACKAGE_DISTRIBUTION = "flux-style-lora"
FALLBACK_PACKAGE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class RepositoryAsset:
    repository: str
    filename: str | None = None
    subfolder: str | None = None
