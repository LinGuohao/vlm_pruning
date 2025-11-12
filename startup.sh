#!/usr/bin/env bash
set -e

ENV_NAME="vlm_pruning"

echo "Creating environment: ${ENV_NAME}"
mamba env create -f vlm_pruning.yml

echo "Activating environment: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Installing python dependencies via uv"
uv pip install -r requirements-uv.txt

uv pip install --no-deps "git+https://github.com/haotian-liu/LLaVA.git"

echo "Done."
echo "Activate environment with:"
echo "  conda activate ${ENV_NAME}"