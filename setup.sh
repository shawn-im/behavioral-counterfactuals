#!/usr/bin/env bash
# Install dependencies and data without changing experiment configurations.
set -euo pipefail

MODEL="${1:-both}"
case "$MODEL" in
  gemma|deepseek|both) ;;
  -h|--help)
    echo 'Usage: bash setup.sh [gemma|deepseek|both]'
    echo 'Requires Linux, Python 3.11, and an NVIDIA GPU. Default: both models.'
    echo 'Installs .venv-cu126, downloads Scruples, and checks Hugging Face access.'
    echo 'Model weights download on first pipeline use. HF_TOKEN is supported.'
    exit 0 ;;
  *) echo 'Expected gemma, deepseek, or both. See --help.' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo 'Too many arguments. See --help.' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run setup on your Linux NVIDIA GPU machine, not a local Mac.' >&2
  exit 1
fi
command -v python3.11 >/dev/null || { echo 'Install Python 3.11 with venv support first.' >&2; exit 1; }
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${FAITHFUL_INTERVENTIONS_CACHE_ROOT:-$PWD/.cache}/pip}"
python3.11 -m venv .venv-cu126
source .venv-cu126/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cu126.txt
python -m pip install -r requirements.txt
python -m pip check

# Keep credentials out of shell history and files in the repository.
if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
  echo 'For Gemma, accept model terms at https://huggingface.co/google/gemma-4-E4B-it'
  echo 'Create a read token at https://huggingface.co/settings/tokens'
  read -r -s -p 'Hugging Face token (Enter to use existing login/public access): ' HF_TOKEN
  echo
  if [[ -n "$HF_TOKEN" ]]; then export HF_TOKEN; else unset HF_TOKEN; fi
fi

python - "$MODEL" <<'PY'
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from behavior_pipeline.runtime_env import configure_runtime_caches

# Use the same token/cache directory as the pipeline entry points.
configure_runtime_caches(Path.cwd())
import torch
import yaml
from huggingface_hub import hf_hub_download, login

# Persist an explicitly supplied token in the pipeline's Hugging Face cache,
# so subsequent commands work after this setup subprocess exits.
import os
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check your NVIDIA driver and GPU allocation.")
print(f"CUDA available: {torch.cuda.device_count()} GPU(s).", flush=True)

destination = Path("data/scruples/anecdotes")
names = [f"{split}.scruples-anecdotes.jsonl" for split in ("train", "dev", "test")]
missing = [name for name in names if not (destination / name).is_file()]
if missing:
    url = "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz"
    print("Downloading the official Scruples anecdotes dataset...", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    # Copy only the expected regular files; never extract arbitrary archive paths.
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        archive = Path(temporary) / "anecdotes.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with tarfile.open(archive, "r:gz") as bundle:
            for name in missing:
                matches = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == name]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected exactly one {name} in the Scruples download")
                staged = Path(temporary) / name
                with bundle.extractfile(matches[0]) as source, staged.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if not staged.stat().st_size:
                    raise RuntimeError(f"Downloaded empty dataset file: {name}")
            for name in missing:
                (Path(temporary) / name).replace(destination / name)
for name in names:
    if not (destination / name).stat().st_size:
        raise SystemExit(f"Empty dataset file: {destination / name}")
print(f"Scruples ready at {destination}.", flush=True)

configs = {
    "gemma": "configs/topk_kl_methods_suggest_wrong.yaml",
    "deepseek": "configs/topk_kl_methods_suggest_wrong_deepseek_r1_qwen3_8b.yaml",
}
selected = configs if sys.argv[1] == "both" else [sys.argv[1]]
for model in selected:
    config = yaml.safe_load(Path(configs[model]).read_text())
    repo = config["training"]["model_name"]
    try:
        hf_hub_download(repo, "config.json")
    except Exception as error:
        print(f"Could not access {repo}: {error}", file=sys.stderr)
        print(f"Check access/accept model terms at https://huggingface.co/{repo}", file=sys.stderr)
        raise SystemExit(
            "Create a read token at https://huggingface.co/settings/tokens, then rerun setup "
            "and enter it at the prompt (or set HF_TOKEN for a noninteractive run)."
        )
    print(f"Model access checked: {repo}", flush=True)
PY

echo 'Setup complete. Run: source .venv-cu126/bin/activate'
echo 'Then follow the README pipeline commands; model IDs and dataset paths are already supplied.'
