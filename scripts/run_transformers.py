from typing import Dict

import click
import requests

from llm_benchmarks.utils import filter_model_size


@click.command()
@click.option("--limit", default=10, type=int, help="Limit the number of models fetched.")
@click.option("--max-size-billion", default=5, type=int, help="Maximum size of models in billion parameters.")
@click.option("--quantization-bits", default=None, type=str, help="Quantization bits for the model.")
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
def fetch_and_bench_tf(limit: int, max_size_billion: int, quantization_bits: str, run_always: bool) -> None:
    """
    Fetch models from Hugging Face, filter them based on parameter count,
    and benchmark the valid models.
    """
    # Fetch models
    params = {"sort": "downloads", "direction": "-1", "limit": limit, "filter": "text-generation"}
    response = requests.get("https://huggingface.co/api/models", params=params)
    model_data = response.json()

    # Extract model IDs
    model_ids = [entry["id"] for entry in model_data]

    # Filter models based on parameter count
    valid_models = filter_model_size(model_ids, max_size_billion * 1_000)

    # Benchmarking
    model_status: Dict[str, int] = {}
    for model_id in valid_models:
        encoded_model = model_id.replace("/", "%2F")

        print(f"Running benchmark: {model_id}")

        # Define params
        params = {"run_always": str(run_always).lower(), "quantization_bits": quantization_bits}

        # POST request with params
        response = requests.post(f"http://localhost:5000/benchmark/{encoded_model}", params=params)

        response_code = response.status_code
        print(f"Finished benchmark: {model_id} with Status Code: {response_code}")

        model_status[model_id] = response_code

    print("All benchmark runs are finished.")

    # Summary of benchmark runs
    print("Summary of benchmark runs:")
    for model, code in model_status.items():
        if code == 200:
            print(f"Model: {model}, {code} ✅ (Benchmark Successful)")
        elif code == 304:
            pass
        elif code == 500:
            print(f"Model: {model}, {code} ❌ (Benchmark Failed)")
        else:
            print(f"Model: {model}, {code} ❓ (Unknown Status)")
    print("🎊 Done 🎊")


if __name__ == "__main__":
    fetch_and_bench_tf()
