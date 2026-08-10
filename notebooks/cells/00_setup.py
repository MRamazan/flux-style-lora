# ============================================================
# Setup
# ============================================================

import subprocess
import sys

REPOSITORY_URL = "https://github.com/YOUR_USERNAME/flux-style-lora.git"
PIPELINE_REVISION = "main"
WORKSPACE = "/content/flux_style_lora"

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        f"git+{REPOSITORY_URL}@{PIPELINE_REVISION}",
    ]
)

from flux_style_lora import StyleLoraPipeline

pipeline = StyleLoraPipeline(
    workspace=WORKSPACE,
    repository_revision=PIPELINE_REVISION,
)

setup_report = pipeline.setup(
    verify_environment=True,
    prepare_training_assets=True,
    prepare_inference_assets=False,
)

setup_report.display()
