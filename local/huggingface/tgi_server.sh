#!/bin/bash

# Get the directory of the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Load environment variables from .env file in the parent directory
env $(grep -v '^#' "$script_dir/../.env" | xargs)

# Get the port from the environment variable or use default value
port=${FLASK_PORT_HF_TGI:-5001}

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bench_hf

# Run the Python script
python "$script_dir/llm_bench_hf/server.py" --port "$port"