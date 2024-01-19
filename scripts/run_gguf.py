import os
from typing import Dict

import click
import requests

FLASK_PORT = 5003


@click.command()
@click.option("--limit", default=50, type=int, help="Limit the number of models to run for debugging.")
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
def bench_gguf(limit: int, run_always: bool) -> None:
    """Benchmark all models on the gguf server."""

    # Get all models and quant types from disk
    model_dir = "/gemini/gguf"
    model_names, quant_types = get_models_and_quant_types(model_dir)

    # Manually drop some models
    drop_models = [
        "meta-llama--Llama-2-70b-chat-hf/m-f16.gguf",
        "meta-llama--Llama-2-13b-chat-hf/m-f16.gguf",
    ]
    model_names = [model for model in model_names if model not in drop_models]

    # Run benchmarks
    model_status: Dict[str, int] = {}
    stop = False
    for model in model_names:
        if stop:
            break
        quant = get_quant_type(model)
        print(f"Running benchmark: {model}, quant: {quant}")

        config = {
            "model_name": model,
            "query": "User: Tell me a long story about the history of the world.\nAI:",
            "max_tokens": 256,
            "n_gpu_layers": -1,
            "run_always": run_always,
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


def get_quant_type(file: str) -> str:
    """Get quantization type from file name."""
    if "f16" in file:
        return "f16"
    elif "int8" in file:
        return "8bit"
    elif "int4" in file:
        return "4bit"
    else:
        raise ValueError(f"Unknown quant type for file: {file}")


def get_models_and_quant_types(model_dir: str = "/gemini/gguf") -> tuple:
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
                print(f"Found model: {model_name} with quant: {quant_type}")

    print(f"Found {len(model_names)} models.")
    return model_names, quant_types


if __name__ == "__main__":
    bench_gguf()
