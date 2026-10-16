# Behavioral Counterfactuals

This codebase accompanies *Interpreting Model Reasoning through
Behavioral Counterfactuals*.

## Quick Start

Run every command from the repository root. The scripts add the repository to
`sys.path`; installing the package itself is not required.

On a Linux NVIDIA GPU machine with Python 3.11, run:

```bash
bash setup.sh
source .venv-cu126/bin/activate
```

Setup installs the CUDA 12.6 dependencies, downloads the official
[Scruples anecdotes dataset](https://github.com/allenai/scruples#data) into
`data/scruples/anecdotes`, and checks access to both models using the IDs already
in the configs. Existing dataset files are preserved. Model weights download on
first use, not during setup. Use `bash setup.sh gemma` or `bash setup.sh deepseek`
to check only one model; both use the same environment and dataset.

For Gemma, first accept the terms/request access on its
[model page](https://huggingface.co/google/gemma-4-E4B-it). Setup prompts for a
[Hugging Face read token](https://huggingface.co/settings/tokens) without displaying
it; press Enter to use an existing login or public access. Noninteractive runs
can supply `HF_TOKEN`. A supplied token is saved in the pipeline's Hugging Face
cache for subsequent commands, not in the configs. Do not share `.cache/` or
tokens with your code. Choose available CUDA device indices in the commands below.

Command-line entry points automatically place Hugging Face, vLLM, Triton,
TorchInductor, CUDA, and pip caches under `.cache/`. Set
`FAITHFUL_INTERVENTIONS_CACHE_ROOT` before running a script to use a different
disk (set it before setup and keep the same value for subsequent commands).
Setup also uses this location for pip downloads. An explicitly set `HF_HOME` or
`PIP_CACHE_DIR` takes precedence for that cache.

## Pipeline

### 1. Prepare paired prompts

Prompt preparation is model-independent and preserves the native Scruples
train/dev/test split. Downstream code exposes `dev` as `val`. Every
`source_id` has one control and one intervention prompt.

```bash
SCRUPLES=data/scruples/anecdotes

python scripts/prepare_dataset.py \
  --intervention suggest_wrong \
  --input_dir "$SCRUPLES" \
  --output_dir .behavior_data/prompts/scruples/suggest_wrong \
  --seed 0
```

### 2. Generate responses

Gemma:

```bash
MODEL=google/gemma-4-E4B-it

python scripts/generate_responses.py \
  --input_dir .behavior_data/prompts/scruples/suggest_wrong \
  --output_dir .behavior_data/traces/gemma-4-E4B-it/suggest_wrong/n1_seed0/generated \
  --model_name "$MODEL" \
  --devices cuda:0,cuda:1 \
  --n_rollouts 1 \
  --seed 0
```

DeepSeek:

```bash
MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

python scripts/generate_responses.py \
  --input_dir .behavior_data/prompts/scruples/suggest_wrong \
  --output_dir .behavior_data/traces/deepseek-r1-0528-qwen3-8b/suggest_wrong/n1_seed0/generated \
  --model_name "$MODEL" \
  --devices cuda:0,cuda:1 \
  --n_rollouts 1 \
  --max_new_tokens 32768 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k -1 \
  --seed 0
```

### 3. Build the teacher cache

The cache evaluates every fixed response prefix under both prompts. At each
response position it stores the union of source top-64, target top-64, and the
observed response token.

Gemma:

```bash
python scripts/build_topk_kl_distill_cache.py \
  --data_dir .behavior_data/traces/gemma-4-E4B-it/suggest_wrong/n1_seed0/generated \
  --cache_dir runs/gemma4_full/suggest_wrong/topk_kl_cache \
  --model_name google/gemma-4-E4B-it \
  --top_k 64 \
  --max_response_tokens 32768 \
  --devices cuda:0 cuda:1
```

DeepSeek:

```bash
python scripts/build_topk_kl_distill_cache.py \
  --data_dir .behavior_data/traces/deepseek-r1-0528-qwen3-8b/suggest_wrong/n1_seed0/generated \
  --cache_dir runs/deepseek_r1_0528_qwen3_8b/suggest_wrong/topk_kl_cache \
  --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --top_k 64 \
  --max_response_tokens 32768 \
  --devices cuda:0 cuda:1
```

Completed caches are reused only when their metadata matches the requested
build.

### 4. Train and select editors

The production sweep trains three methods at learning rates `2e-5`, `5e-5`,
and `1e-4`:

| Method | Role |
| --- | --- |
| `fixed_steering` | Global-direction sanity check |
| `directional_linear` | Input-dependent linear sanity check |
| `edit_module_base_mlp` | Primary per-token residual editor |

Each method is selected by validation recovery. Training never loads the test
split. The canonical checkpoint is `model.pt`; `last_model.pt` is diagnostic.

Input-dependent editors use per-token RMS normalization only; no normalization
statistics file or preparation pass is needed.

Gemma:

```bash
python scripts/run_topk_kl_training_harness.py \
  --config configs/topk_kl_methods_suggest_wrong.yaml \
  --devices cuda:0 cuda:1
```

DeepSeek:

```bash
python scripts/run_topk_kl_training_harness.py \
  --config configs/topk_kl_methods_suggest_wrong_deepseek_r1_qwen3_8b.yaml \
  --devices cuda:0 cuda:1
```

Device arguments override the YAML for that invocation. Resume is enabled, so
completed runs with matching content fingerprints are reused.


### 5. Train the activation-reconstruction comparison

Build paired residual activations from the same teacher-cache rows, then train
the activation-reconstruction MLP. Complete this step before unified evaluation.

Gemma:

```bash
python scripts/build_activation_reconstruction_cache.py \
  --source_cache_dir runs/gemma4_full/suggest_wrong/topk_kl_cache \
  --cache_dir runs/gemma4_full/suggest_wrong/activation_reconstruction_cache \
  --model_name google/gemma-4-E4B-it \
  --layer 20 --devices cuda:0 cuda:1 --batch_size 16 \
  --dtype bf16 --activation_dtype bf16

python scripts/run_activation_nmse_training_harness.py \
  --config configs/activation_nmse_mlp_suggest_wrong.yaml \
  --devices cuda:0 cuda:1
```

DeepSeek:

```bash
python scripts/build_activation_reconstruction_cache.py \
  --source_cache_dir runs/deepseek_r1_0528_qwen3_8b/suggest_wrong/topk_kl_cache \
  --cache_dir runs/deepseek_r1_0528_qwen3_8b/suggest_wrong/activation_reconstruction_cache \
  --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --layer 17 --devices cuda:0 cuda:1 --batch_size 16 \
  --dtype bf16 --activation_dtype bf16

python scripts/run_activation_nmse_training_harness.py \
  --config configs/activation_nmse_mlp_suggest_wrong_deepseek_r1_qwen3_8b.yaml \
  --devices cuda:0 cuda:1
```

This objective reconstructs paired activations from the final question through
the response and selects the checkpoint by validation NMSE. Its editor is
applied at that span during evaluation; the behavior-trained editor is applied
at all positions. Activation recovery is evaluated on the paired question and
response positions; behavior recovery is evaluated on response predictions.

### 6. Evaluate the held-out test set

```bash
python scripts/evaluate_reconstruction_objectives.py \
  --kl_config configs/topk_kl_methods_suggest_wrong.yaml \
  --nmse_config configs/activation_nmse_mlp_suggest_wrong.yaml \
  --device cuda:0
```

For DeepSeek, use the corresponding `_deepseek_r1_qwen3_8b.yaml` configs for
both arguments. Both training sweeps must have selected checkpoints, and the
activation cache must contain the same ordered test rows as the behavior cache.
Existing selected checkpoints can be used without retraining.

### 7. Run interpretation and causal validation

Gemma:

```bash
python scripts/run_counterfactual_analysis.py \
  --config configs/counterfactual_analysis_suggest_wrong.yaml \
  --devices cuda:0 cuda:1
```

DeepSeek:

```bash
python scripts/run_counterfactual_analysis.py \
  --config configs/counterfactual_analysis_suggest_wrong_deepseek_r1_qwen3_8b.yaml \
  --devices cuda:0 cuda:1
```

The resumable stages are:

| Stage | Output |
| --- | --- |
| 0 | Validated run manifest |
| 1 | Teacher-forced traces and fidelity |
| 2 | Total-response edit-event IG/RelP attribution |
| 3 | Exact edit-event ablations |
| 4 | Native activation-patching hypothesis tests |
| 5 | Exhaustive-token attribution/ablation validation |
| 6 | Keyword sentence collection and browser |
| 7 | HTML report |

The default runs stages `0-7`. Use `--stages 5,7`
for exhaustive validation and the report. Add `--force` only when intentionally
recomputing completed requested stages. A report-only refresh is:

```bash
python scripts/run_counterfactual_analysis.py \
  --config configs/counterfactual_analysis_suggest_wrong.yaml \
  --stages 7 \
  --force
```

## Outputs

The principal artifacts are:

```text
runs/<model>/suggest_wrong/
  topk_kl_cache/
  activation_reconstruction_cache/
  topk_kl_training/
    training_selection.json
    selected/<method>/model.pt
    final_evaluation/test_comparison.json
  activation_nmse_training/
    training_selection.json
    selected/edit_module_base_mlp/model.pt
  reconstruction_objective_comparison/test_comparison.json
  counterfactual_analysis/
    attribution/
    ablations/edit_event_ablation.json
    native_analysis/
    report/index.html
```

Each training run also stores `train_history.json`, `eval_history.json`,
plots, and `train.log`.

## Code Map

| Path | Responsibility |
| --- | --- |
| `behavior_pipeline/topk_kl.py` | Conditional-support KL and recovery metrics |
| `behavior_pipeline/support_projection.py` | Shared sparse conditional-support LM-head projection |
| `behavior_pipeline/edit_modules.py` | Steering, linear, and residual-MLP editors |
| `behavior_pipeline/training_harness.py` | LR sweep, resume, and checkpoint selection |
| `scripts/train_topk_conditional_kl_edit_module.py` | Single training run |
| `scripts/evaluate_reconstruction_objectives.py` | Unified held-out method and objective comparisons |
| `behavior_pipeline/method_evaluation.py` | Fresh behavior-method metrics, directional recovery, support mass, and bootstrap intervals |
| `scripts/run_counterfactual_analysis.py` | Resumable analysis orchestration |
| `behavior_pipeline/counterfactual_runtime*.py` | Teacher-forced tracing, attribution, ablation |
| `behavior_pipeline/native_hypothesis*.py` | Native activation-patching analysis |
| `behavior_pipeline/counterfactual_report*.py` | HTML reports and report plots |
