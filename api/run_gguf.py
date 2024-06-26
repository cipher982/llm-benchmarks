import os
from typing import Dict

import click
import requests
from llm_bench.local.gguf import fetch_gguf_files

FLASK_PORT = 5003
GGUF_DIR = "/gemini/gguf/"
assert GGUF_DIR, "GGUF_DIR environment variable not set"


@click.command()
@click.option("--limit", default=50, type=int, help="Limit the number of models to run for debugging.")
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
@click.option("--log-level", default="INFO", help="Log level for the benchmarking server.")
def bench_gguf(limit: int, run_always: bool, log_level: str = "INFO"):
    """Benchmark all models on the gguf server."""

    # Fetch all models
    model_names = fetch_gguf_files(model_dir=GGUF_DIR)
    print(f"Fetched {len(model_names)} GGUF models")

    # Limit the number of models to run
    model_names = model_names[:limit]
    print(f"Will run benchmarks for {len(model_names)} models")

    # Run benchmarks
    model_status: Dict[str, int] = {}
    stop = False
    for model in model_names:
        if stop:
            break
        quant = get_quant_type(model)
        print(f"Running benchmark: {model}, quant: {quant[0]}, bits: {quant[1]}")

        config = {
            "model_name": model,
            "quant_method": "gguf",
            "quant_type": quant[0],
            "quant_bits": int(quant[1]),
            "query": "User: Tell me a long story about the history of the world.\nAI:",
            "max_tokens": 256,
            "n_gpu_layers": -1,
            "run_always": run_always,
            "log_level": log_level,
        }
        request_path = f"http://localhost:{FLASK_PORT}/benchmark"
        response = requests.post(request_path, data=config)

        response_code = response.status_code
        print(f"Finished benchmark: {model} with Status Code: {response_code}")

        model_status[model] = response_code

        if len(model_status) >= limit:
            stop = True

    print("All benchmark runs are finished.")

    # Summary of benchmark runs
    print("Summary of benchmark runs:")
    for model, code in model_status.items():
        print(f"Model: {model}, HTTP Response Code: {code} {'✅' if code == 200 else '❌'}")
    print("🎊 Done 🎊")


def get_quant_type(file: str) -> tuple:
    """Get quantization type and number of bits from file name."""
    parts = file.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid file name format: {file}")

    quant_type = parts[-2]
    if not quant_type.startswith("Q"):
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    bits_str = quant_type.split("_")[0][1:]
    if not bits_str.isdigit():
        raise ValueError(f"Invalid number of bits: {bits_str}")

    bits = int(bits_str)
    return quant_type, bits


def get_models_and_quant_types(model_dir: str) -> tuple:
    """Get list of .gguf models and their quant types from any dirs in model_dir."""

    model_names = []
    quant_types = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".gguf"):
                model_name = os.path.join(os.path.basename(root), file)
                model_names.append(model_name)
                quant_type = get_quant_type(file)
                quant_types.append(quant_type)
    return model_names, quant_types


if __name__ == "__main__":
    bench_gguf()
