# Behavioral Counterfactuals

This codebase accompanies *Interpreting Model Reasoning through
Behavioral Counterfactuals*.

## Setup

You'll need Python 3.11 and a Linux machine with an NVIDIA GPU. From the
repository directory, run:

```bash
bash setup.sh
source .venv-cu126/bin/activate
```

This installs the dependencies and downloads
[Scruples](https://github.com/allenai/scruples#data) to `data/scruples/anecdotes`.
It also checks access to Gemma and DeepSeek; the model names and paths are already
filled in below. The weights download when you first run a model.
If you're only using one model, you can run `bash setup.sh gemma` or
`bash setup.sh deepseek` instead.

For Gemma, you'll need to accept the terms on its
[Hugging Face page](https://huggingface.co/google/gemma-4-E4B-it).
Setup will ask for a [read token](https://huggingface.co/settings/tokens), which
is hidden as you type. You can also pass it through `HF_TOKEN`, or press Enter
to use a login already saved in the pipeline's cache. Setup saves the token for
later commands, so keep `.cache/` private.

Downloads and runtime caches go in `.cache/` by default. To use another disk,
set `FAITHFUL_INTERVENTIONS_CACHE_ROOT` before setup and keep it set when running
the experiments. Existing `HF_HOME` and `PIP_CACHE_DIR` settings are respected.

## Running the experiments

Follow the steps below in order, choosing the commands for the model you want
to use. Run them from the repository directory with the environment activated.
The examples use GPUs 0 and 1; change `--devices` to match your machine.

### 1. Prepare paired prompts

First, create a control prompt and an intervention prompt for each Scruples
example. This step is shared by both models and keeps the original dataset
splits (`dev` becomes `val`).

```bash
SCRUPLES=data/scruples/anecdotes

python scripts/prepare_dataset.py \
  --intervention suggest_wrong \
  --input_dir "$SCRUPLES" \
  --output_dir .behavior_data/prompts/scruples/suggest_wrong \
  --seed 0
```

### 2. Generate responses

Generate responses to both versions of each prompt:

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

Next, evaluate the saved responses under both prompts and cache the token
probabilities used for training. At each position, we keep the top 64 tokens
from each distribution plus the observed response token.

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

### 4. Train and select editors

These commands train the MLP edit module and the steering-vector and linear
baselines. Each is trained at learning rates `2e-5`, `5e-5`, and `1e-4`, with
the best checkpoint chosen by validation KL recovery. The test set is reserved
for evaluation in step 6.

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

You can change the training settings in the YAML configs. Rerunning a command
reuses completed runs when their inputs and settings match.

### 5. Train the activation-reconstruction comparison

For the activation-reconstruction comparison, first cache the paired activations,
then train an MLP to reconstruct them. This uses the responses from step 3.

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

The checkpoint is selected by validation NMSE. This editor acts on the final
question and response, while the behavior-trained editor acts on all positions.
We evaluate activation recovery over the question and response, and behavior
recovery over response predictions.

### 6. Evaluate the held-out test set

Once both training steps are finished, evaluate the selected checkpoints:

```bash
python scripts/evaluate_reconstruction_objectives.py \
  --kl_config configs/topk_kl_methods_suggest_wrong.yaml \
  --nmse_config configs/activation_nmse_mlp_suggest_wrong.yaml \
  --device cuda:0
```

For DeepSeek, use the corresponding `_deepseek_r1_qwen3_8b.yaml` configs for
both arguments. You can rerun evaluation without retraining the models.

### 7. Run the analysis

Finally, run attribution, ablation, and activation-patching analyses and build
the HTML report:

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

The analysis is split into stages:

| Stage | Output |
| --- | --- |
| 0 | Prepare the analysis run |
| 1 | Evaluate edits on fixed responses |
| 2 | IG and RelP attribution |
| 3 | Edit ablations |
| 4 | Native activation patching |
| 5 | Exhaustive token-level attribution validation |
| 6 | Keyword sentence browser |
| 7 | HTML report |

By default, all stages run, and completed stages are reused when their inputs
and settings match. Use `--stages` to choose specific stages and `--force` to
rerun them. For example, to refresh just the Gemma report:

```bash
python scripts/run_counterfactual_analysis.py \
  --config configs/counterfactual_analysis_suggest_wrong.yaml \
  --stages 7 \
  --force
```

Open `runs/gemma4_full/suggest_wrong/counterfactual_analysis/report/index.html`
to view the report. For DeepSeek, replace `gemma4_full` with
`deepseek_r1_0528_qwen3_8b`.
