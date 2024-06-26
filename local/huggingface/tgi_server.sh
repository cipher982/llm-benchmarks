#!/bin/bash

# Get the directory of the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Load environment variables from .env file in the parent directory
env $(grep -v '^#' "$script_dir/../.env" | xargs)

# Get the port from the environment variable or use default value
port=${FLASK_PORT_HF_TGI:5001}

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bench_hf

# set env for LOG_DIR
export LOG_DIR="/home/drose/git/llm-benchmarks/local/logs/"

# Run the Python script
python "/home/drose/git/llm-benchmarks/api/llm_bench/local/hf/server.py" --port "$port"