from typing import Dict
from typing import Optional

import click
import requests
from requests.exceptions import HTTPError

from llm_benchmarks.utils import filter_model_size
from llm_benchmarks.utils import get_cached_models


QUANT_TYPES = [
    # "4bit",
    # "8bit",
    None,
]
QUERY_TEXT = "User: Tell me a long story about the history of the world.\nAI:"
MAX_TOKENS = 512
BENCHMARK_URL = "http://localhost:5002/benchmark/{}"


def run_benchmark(model: str, quant: Optional[str], limit: int, model_status: Dict[str, int]) -> bool:
    """
    Run benchmark for a given model and quantization type.
    Returns True if the limit is reached, False otherwise.
    """
    print(f"Running benchmark: {model}, quant: {quant}")

    config = {
        "query": QUERY_TEXT,
        "quant_bits": quant,
        "max_tokens": MAX_TOKENS,
    }
    try:
        model_encoded = model.replace("/", "%2F")
        response = requests.post(BENCHMARK_URL.format(model_encoded), data=config)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        model_status[f"{model}_{quant}"] = 500  # Log the failed model
        return False
    except Exception as err:
        print(f"Other error occurred: {err}")
        model_status[f"{model}_{quant}"] = 500  # Log the failed model
        return False
    else:
        response_code = response.status_code
        print(f"Finished benchmark: {model}, quant: {quant} with Status Code: {response_code}")

        model_status[f"{model}_{quant}"] = response_code
        return len(model_status) >= limit


def print_summary(model_status: Dict[str, int]) -> None:
    """
    Print a summary of the benchmark runs.
    """
    print("Summary of benchmark runs:")
    for model, code in model_status.items():
        if code == 200:
            print(f"Model: {model}, {code} ✅ (Benchmark Successful)")
        elif code == 500:
            print(f"Model: {model}, {code} ❌ (Benchmark Failed)")
        else:
            print(f"Model: {model}, {code} ❓ (Unknown Status)")
    print("🎊 Done 🎊")


@click.command()
@click.option("--max-size-billion", default=5, type=int, help="Maximum size of models in billion parameters.")
@click.option("--limit", default=50, type=int, help="Limit the number of models to run for debugging.")
def bench_tgi(limit: int, max_size_billion: int) -> None:
    """
    Benchmark models on the HuggingFace Text-Generation-Inference server.
    """
    model_status: Dict[str, int] = {}

    model_names = get_cached_models("/rocket/hf")
    model_names = filter_model_size(model_names, max_size_billion * 1_000)

    print(f"Running benchmarks for {len(model_names)} models.")
    for model in model_names[:limit]:
        for quant in QUANT_TYPES:
            if run_benchmark(model, quant, limit, model_status):
                break

    print("All benchmark runs are finished.")
    print_summary(model_status)


if __name__ == "__main__":
    bench_tgi()
