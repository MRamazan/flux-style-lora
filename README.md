# FLUX Style LoRA

This repository turns the reference FLUX LoRA notebook into a style-only Python package with a compact Google Colab interface.

## Two workflows

There are two Colab notebooks that share one package and one public API.

**Training workflow — `notebooks/flux_style_lora_colab.ipynb` (four cells):**

```text
Main notebook
→ train style LoRA
→ evaluate
→ create a portable evaluation ZIP
```

1. Setup
2. Dataset preparation and visualization
3. Training
4. Evaluation and export

**Evaluation-only workflow — `notebooks/flux_style_lora_evaluation_colab.ipynb` (three cells):**

```text
Evaluation notebook
→ upload the portable ZIP
→ validate and import the LoRA
→ download public evaluation assets
→ run base / checkpoint / scale evaluation
→ optionally export a new portable ZIP
```

1. Setup
2. Upload and import evaluation bundle
3. Evaluation and export

The implementation remains in the package. User decisions remain visible in the notebooks.

## Portable evaluation bundle

`evaluation.export(...)` produces a single self-contained ZIP that is directly uploadable into the evaluation-only notebook through `pipeline.import_evaluation_bundle(zip_path=...)`. The returned object is accepted by `pipeline.evaluate(run=..., config=...)` exactly like a `TrainingRun`. The bundle carries the selected LoRA (and optionally all checkpoints), normalized manifests, provenance, capabilities, and a per-file SHA-256 index. It never contains base model, text-encoder, or VAE weights, dataset images, secrets, or absolute operational paths. See [docs/EVALUATION_BUNDLE.md](docs/EVALUATION_BUNDLE.md) for the schema and validation rules.

Bundles exported with `include_all_checkpoints=True` support the full checkpoint sweep. Selected-checkpoint-only bundles still support base comparison and scale sweep; a requested checkpoint sweep evaluates the single selected checkpoint with a clear notice.

## Style-only scope

The package does not expose a concept type, style mode, object mode, product mode, or generic training mode. Style LoRA training is the only supported workflow. The trigger word is defined in the dataset cell and persisted in dataset and run manifests.

## Model access and the FLUX.1-dev license

The pipeline contains no Hugging Face token handling of its own. There is no Hugging Face token, `getpass` prompt, Colab Secrets integration, or `.env` support anywhere in the package or notebook.

The pipeline trains and evaluates `black-forest-labs/FLUX.1-dev`, a self-contained Diffusers pipeline repository that bundles the CLIP and T5 text encoders, the tokenizers, the scheduler, and the VAE. Setup resolves the exact repository revision and downloads only the Diffusers component folders, deliberately skipping the ~24 GB single-file checkpoints at the repository root. The repository, revision, and local directory are recorded in the asset manifest. The same snapshot is used for training and evaluation, and the transformer is quantized by default because FLUX.1-dev is a 12B model. Model weights are never committed to this repository or included in export archives.

`black-forest-labs/FLUX.1-dev` is a gated repository under the FLUX.1 [dev] Non-Commercial License. The package itself contains no token handling: downloads pass `token=None`, which makes `huggingface_hub` use whatever Hugging Face credentials already exist in the runtime. Accept the model license on Hugging Face and authenticate in your own notebook cell before running setup.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/build_notebook.py
ruff check .
pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Repository workflow

1. Develop and test the package locally.
2. Build the committed notebook with `python scripts/build_notebook.py`.
3. Push the repository to GitHub.
4. Set `REPOSITORY_URL` and `PIPELINE_REVISION` in the setup cell.
5. During development, install a specific commit or branch.
6. For stable Colab runs, install a version tag or immutable commit.

Example:

```python
REPOSITORY_URL = "https://github.com/YOUR_USERNAME/flux-style-lora.git"
PIPELINE_REVISION = "v0.1.0"
```

## Reference implementation

The original notebook is stored in `reference/universal_pipeline_v2_1.ipynb`. Its code cells are also extracted into `reference/extracted_cells` to make migration and comparison easier.

## Claude Code

Read `CLAUDE.md`, then paste the task from `INITIAL_PROMPT.md` into Claude Code from the repository root.
