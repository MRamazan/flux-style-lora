# Requirements

## Functional requirements

### Setup

- Install the package from a user-visible GitHub repository URL and immutable revision or branch.
- Use no Hugging Face authentication. All configured repositories are public and downloaded anonymously.
- Verify the GPU and runtime.
- Create or repair the isolated AI Toolkit environment.
- Resolve and record exact source and model revisions.
- Verify the active FLUX integration before training.
- Prepare the FLUX base model for training.
- Prepare the FLUX model by downloading only the Diffusers component folders of `black-forest-labs/FLUX.1-dev` and verifying them.
- Record the model repository, resolved revision, and local directory in the asset manifest.
- Use the same model snapshot for latent caching, training, samples, and evaluation.
- Reuse the same FLUX snapshot for training and evaluation.

### Dataset

- Accept exactly one uploaded ZIP file.
- Discover image-caption pairs in nested directories.
- Handle repeated basenames safely.
- Reject unsupported, corrupt, unmatched, or empty inputs with actionable messages.
- Require a non-empty style trigger word.
- Audit trigger presence without silently modifying captions by default.
- Optionally prefix missing triggers only when explicitly enabled and after creating backups.
- Fingerprint the canonical dataset.
- Audit exact and perceptual duplicates.
- Display every image with filename, dimensions, caption, and highlighted trigger word.
- Display a summary and issue report before training.

### Training

- Train a transformer-only LoRA on the FLUX base model.
- Keep the text encoder frozen.
- Write production and smoke-test AI Toolkit configurations.
- Validate the generated configuration and source contract before training.
- Support smoke-only, interrupted, completed, and resumed runs.
- Persist enough state to reload a run after a Python runtime restart.
- Discover checkpoints from actual files rather than expected filenames.
- Never label an incomplete run as production-complete.

### Evaluation

- Reuse the FLUX base model for evaluation; download it only when needed.
- Load LoRA weights non-destructively.
- Verify representative base parameters remain unchanged.
- Generate deterministic base-versus-LoRA comparisons.
- Compare actual checkpoints with identical prompts, seeds, dimensions, steps, guidance, VAE, and adapter scale.
- Compare editable LoRA scales without merging weights.
- Never choose the best checkpoint automatically.
- Allow explicit manual checkpoint selection after visual review.

### Export and bundle

- Produce a single portable evaluation bundle ZIP through the existing `evaluation.export(...)` call.
- Include the selected LoRA, optional checkpoints, normalized manifests, provenance, capabilities, and a per-file SHA-256 index using relative POSIX paths.
- Never include base model, text encoder, or VAE weights, dataset images, secrets, or absolute operational paths.
- Detect secret fields by normalized exact name and sensitive suffix, never by substring, so `keep_tokens` and similar fields export successfully.
- Import a bundle through `pipeline.import_evaluation_bundle(zip_path=...)` into an isolated directory with strict validation, returning an `ImportedRun` accepted by `evaluate`.
- Reconstruct the checkpoint inventory from validated bundle paths relative to the extraction root, never from absolute source paths.
- Re-exports from the evaluation-only notebook use the same format and remain re-importable without nesting the original ZIP.

## Non-functional requirements

- All project language is English.
- There is no Google Drive integration.
- The training notebook has exactly four executable code cells and the evaluation-only notebook has exactly three.
- Package code, scripts, and tests contain no comment tokens.
- Notebook comments only separate sections.
- Public errors are actionable and specific.
- Pure Python behavior is locally testable without downloading FLUX.
- Colab-only behavior has a documented smoke-test procedure.
