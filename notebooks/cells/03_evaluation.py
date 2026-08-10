# ============================================================
# Evaluation and export
# ============================================================

from flux_style_lora import EvaluationConfig

PROMPTS = [
    f"a serene mountain landscape at golden hour, in {training_run.trigger_word} style",
    f"a portrait of an elderly fisherman with a weathered face, "
    f"in {training_run.trigger_word} style",
    f"a quiet city street market in the rain at night, in {training_run.trigger_word} style",
    f"a still life of fruit and a ceramic vase on a wooden table, "
    f"in {training_run.trigger_word} style",
    f"a fantasy castle on a cliff above the sea, in {training_run.trigger_word} style",
    f"a cozy bookshop interior with warm lighting, in {training_run.trigger_word} style",
]
SEEDS = [42, 12345, 987654321]
WIDTH = 1024
HEIGHT = 1024
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
NEGATIVE_PROMPT = ""
CHECKPOINT_MODE = "auto"
MAXIMUM_CHECKPOINTS = 8
MANUAL_CHECKPOINT_STEPS = []
PRIMARY_ADAPTER_SCALE = 1.0
SCALE_SWEEP = [0.6, 0.8, 1.0]
COMPARE_BASE_MODEL = True
INCLUDE_BASE_IN_CHECKPOINT_GRID = True
RUN_CHECKPOINT_SWEEP = True
RUN_SCALE_SWEEP = True
INCLUDE_ALL_CHECKPOINTS_IN_EXPORT = False
DOWNLOAD_EXPORTS = False

evaluation_config = EvaluationConfig(
    prompts=PROMPTS,
    seeds=SEEDS,
    width=WIDTH,
    height=HEIGHT,
    inference_steps=INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    negative_prompt=NEGATIVE_PROMPT,
    checkpoint_mode=CHECKPOINT_MODE,
    maximum_checkpoints=MAXIMUM_CHECKPOINTS,
    manual_checkpoint_steps=MANUAL_CHECKPOINT_STEPS,
    primary_adapter_scale=PRIMARY_ADAPTER_SCALE,
    scale_sweep=SCALE_SWEEP,
    compare_base_model=COMPARE_BASE_MODEL,
    include_base_in_checkpoint_grid=INCLUDE_BASE_IN_CHECKPOINT_GRID,
    run_checkpoint_sweep=RUN_CHECKPOINT_SWEEP,
    run_scale_sweep=RUN_SCALE_SWEEP,
)

pipeline.prepare_evaluation_assets()

evaluation = pipeline.evaluate(
    run=training_run,
    config=evaluation_config,
)

evaluation.show_base_comparison()
evaluation.show_checkpoint_grid()
evaluation.show_scale_grid()
evaluation.show_summary()

exports = evaluation.export(
    include_selected_lora=True,
    include_all_checkpoints=INCLUDE_ALL_CHECKPOINTS_IN_EXPORT,
    include_images=True,
    include_logs=True,
    include_manifests=True,
)

exports.display()

if DOWNLOAD_EXPORTS:
    exports.download()
