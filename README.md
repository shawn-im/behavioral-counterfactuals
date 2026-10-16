# Interpreting Model Reasoning through Behavioral Counterfactuals

## Setup

```bash
bash setup.sh
source .venv-cu126/bin/activate
```

`setup.sh` installs dependencies, downloads the
[Scruples dataset](https://github.com/allenai/scruples#data), and checks model
access. Use `bash setup.sh gemma` or `bash setup.sh deepseek` to check one model
only. Model weights are downloaded on first use.

Gemma requires accepting the terms on its
[Hugging Face page](https://huggingface.co/google/gemma-4-E4B-it).
Enter a [Hugging Face read token](https://huggingface.co/settings/tokens) when
prompted, set `HF_TOKEN`, or press Enter to use credentials already saved in the
pipeline's cache. Tokens are stored in the Hugging Face cache, not in the configs.

Caches are stored under `.cache/` by default; do not include this directory when
sharing the code. Set `FAITHFUL_INTERVENTIONS_CACHE_ROOT` before setup and subsequent
runs to use another location. Existing `HF_HOME` and `PIP_CACHE_DIR` settings take
precedence for those caches.

## Experiments

Run commands from the repository root with the environment activated, in the
order below. Commands are provided for both models. Adjust `--devices` to select
available GPUs.

### Prompt preparation

Prepare control and intervention prompts from Scruples. 

```bash
SCRUPLES=data/scruples/anecdotes

python scripts/prepare_dataset.py \
  --intervention suggest_wrong \
  --input_dir "$SCRUPLES" \
  --output_dir .behavior_data/prompts/scruples/suggest_wrong \
  --seed 0
```

### Response generation

Generate responses for the control and intervention prompts.

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

### Teacher cache

Cache token probabilities for the generated responses under both prompts.
The support is the union of each distribution's top 64 tokens and the observed token.

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

### Behavior reconstruction

Train the MLP edit module, steering-vector baseline, and linear baseline. The
configs sweep learning rates `2e-5`, `5e-5`, and `1e-4` and select checkpoints by
validation KL recovery.

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

### Activation reconstruction

Cache paired activations from the teacher-cache examples, then train the
activation-reconstruction MLP. Checkpoints are selected by validation NMSE.

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

### Evaluation

Evaluate both objectives on the held-out test set after completing both training
sweeps. The command below uses the Gemma configs.

```bash
python scripts/evaluate_reconstruction_objectives.py \
  --kl_config configs/topk_kl_methods_suggest_wrong.yaml \
  --nmse_config configs/activation_nmse_mlp_suggest_wrong.yaml \
  --device cuda:0
```

For DeepSeek, use the corresponding `_deepseek_r1_qwen3_8b.yaml` configs for
both arguments.

### Analysis

Run attribution, edit ablations, and native activation patching, and generate
the HTML report.

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

| Stage | Analysis |
| --- | --- |
| 0 | Prepare the analysis run |
| 1 | Evaluate edits on fixed responses |
| 2 | IG and RelP attribution |
| 3 | Edit ablations |
| 4 | Native activation patching |
| 5 | Exhaustive token-level attribution validation |
| 6 | Keyword sentence browser |
| 7 | HTML report |

Completed stages are reused when inputs and settings match. Use `--stages` to
select stages and `--force` to rerun them. To regenerate only the Gemma report:

```bash
python scripts/run_counterfactual_analysis.py \
  --config configs/counterfactual_analysis_suggest_wrong.yaml \
  --stages 7 \
  --force
```

Open `runs/gemma4_full/suggest_wrong/counterfactual_analysis/report/index.html`
to view the report. For DeepSeek, replace `gemma4_full` with
`deepseek_r1_0528_qwen3_8b`.
