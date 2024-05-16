# Local Benchmarks

## Overview
This directory contains benchmarks that are run locally on the machine. Each framework runs within a standalone container that are all integrated into a single docker compose file. The benchmarks are run using a relevant script from the /scripts directory. The current options are:
- run_hf.py
    - can be used for both Transformers and Text-Generation-Inference benchmarks
    - **Options:**
        - `--framework TEXT`: LLM API to call. Must be one of 'transformers', 'hf-tgi'
        - `--limit INTEGER`: Limit the number of models run.
        - `--max-size-billion INTEGER`: Maximum size of models in billion parameters.
        - `--run-always`: Flag to always run benchmarks.
        - `--fetch-new-models`: Fetch latest HF-Hub models.
        - `--help`: Show this message and exit.
- run_vllm.py
    - Used for the VLLM benchmarks
    - **Options:**
        - `--framework TEXT`: Framework to use, must be 'vllm'.
        - `--limit INTEGER`: Limit the number of models fetched.
        - `--max-size-billion INTEGER`: Maximum size of models in billion parameters.
        - `--run-always`: Flag to always run benchmarks.
        - `--fetch-new-models`: Fetch latest HF-Hub models.
        - `--help`: Show this message and exit.
- run_gguf.py
    - Used for the GGUF/llama-cpp benchmarks
    - **Options:**
        - `--limit INTEGER`: Limit the number of models to run for debugging.
        - `--run-always`: Flag to always run benchmarks.
        - `--log-level TEXT`: Log level for the benchmarking server.
        - `--help`: Show this message and exit.


## Getting Started
It should be as simple as setting the correct `.env` variables and building the docker containers with the following commands:
```bash
cp .env.example .env # fill out the .env file with the correct values
docker compose -f docker-compose.local.yml up --build
```

## Example Usage

To run the Huggingface Transformers benchmark, use the following command:
```bash
python scripts/run_hf.py --framework transformers --limit 5 --max-size-billion 10 --run-always
```

To run the Huggingface Text-Generation-Inference benchmark, use the following command:
```bash
python scripts/run_hf.py --framework hf-tgi --limit 5 --max-size-billion 10 --run-always
```

To run the VLLM benchmark, use the following command:
```bash
python scripts/run_vllm.py --framework vllm --limit 5 --max-size-billion 10 --run-always
```

To run the GGUF/llama-cpp benchmark, use the following command:
```bash
python scripts/run_gguf.py --limit 5 --run-always --log-level DEBUG
```