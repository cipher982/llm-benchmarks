import os
from typing import Optional

import click
import requests
from llm_bench_api.utils import filter_model_size
from llm_bench_api.utils import get_cached_models
from requests.exceptions import HTTPError


QUANT_TYPES = [
    "4bit",
    "8bit",
    None,
]
QUERY_TEXT = "User: Tell me a long story about the history of the world.\nAI:"
MAX_TOKENS = 512
TEMPERATURE = 0.1
FLASK_URL = "http://localhost:5000/benchmark/{}"
CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"


@click.command()
@click.option("--framework", help="LLM API to call. Must be one of 'transformers', 'hf-tgi', 'vllm'.")
@click.option("--fetch-new-models", default=False, help="Fetch latest HF-Hub models.")
@click.option("--limit", default=100, type=int, help="Limit the number of models fetched.")
@click.option(
    "--max-size-billion",
    default=5,
    type=int,
    help="Maximum size of models in billion parameters.",
)
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
def main(
    framework: str,
    fetch_new_models: bool,
    limit: int,
    max_size_billion: int,
    run_always: bool,
) -> None:
    """
    Main entrypoint for benchmarking HuggingFace Transformers models.
    Can fetch latest models from the Hub or use the cached models.
    """
    model_status: dict[str, int] = {}

    # Gather models to run
    model_names = get_models_to_run(fetch_new_models)
    print(f"Fetched {len(model_names)} models")
    valid_models = filter_model_size(model_names, max_size_billion * 1_000)
    print(f"Filtered down to {len(valid_models)} models")

    # valid_models = [
    #     "facebook/opt-125m",
    #     # "TheBloke/Llama-2-7B-Chat-GPTQ",
    #     # "EleutherAI/pythia-160m",
    #     # "TheBloke/Llama-2-7B-Chat-AWQ",
    # ]

    # Run benchmarks
    bench_all_models(framework, valid_models, model_status, limit, run_always)

    # Print summary
    print("All benchmark runs are finished.")
    print_summary(model_status)


def bench_all_models(framework, model_names, model_status, limit, run_always):
    print(f"Running benchmarks for {len(model_names)} models.")
    for model in model_names[:limit]:
        if "GPTQ" in model:
            is_limit_reached = bench_gptq(framework, model, model_status, limit, run_always)
        elif "AWQ" in model:
            is_limit_reached = bench_awq(framework, model, model_status, limit, run_always)
        else:
            is_limit_reached = bench_other(framework, model, model_status, limit, run_always)

        if is_limit_reached:
            break


def run_benchmark(
    framework: str,
    model: str,
    model_status: dict[str, int],
    quant_method: Optional[str],
    quant_bits: Optional[str],
    limit: int,
    run_always: bool,
) -> bool:
    """
    Run benchmark for a given model and quantization type.
    Returns True if the limit is reached, False otherwise.
    """
    quant_str = f"{quant_method}_{quant_bits}" if quant_method is not None else "none"
    print(f"Running benchmark: {model}, quant: {quant_str}")

    flask_data = {
        "framework": framework,
        "query": QUERY_TEXT,
        "quant_method": quant_method,
        "quant_bits": quant_bits,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "run_always": run_always,
    }
    try:
        model_encoded = model.replace("/", "%2F")
        response = requests.post(FLASK_URL.format(model_encoded), data=flask_data)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        model_status[f"{model}_{quant_str}"] = 500  # Log the failed model
        return False
    except Exception as err:
        print(f"Other error occurred: {err}")
        model_status[f"{model}_{quant_str}"] = 500  # Log the failed model
        return False
    else:
        response_code = response.status_code
        print(f"Finished benchmark: {model}, quant: {quant_str} with Status Code: {response_code}")

        model_status[f"{model}_{quant_str}"] = response_code
        return len(model_status) >= limit


def bench_gptq(framework, model, model_status, limit, run_always):
    is_limit_reached = run_benchmark(
        framework,
        model,
        model_status=model_status,
        quant_method="gptq",
        quant_bits="4bit",
        limit=limit,
        run_always=run_always,
    )
    if is_limit_reached:
        return True
    return False


def bench_awq(framework, model, model_status, limit, run_always):
    is_limit_reached = run_benchmark(
        framework,
        model,
        model_status=model_status,
        quant_method="awq",
        quant_bits="4bit",
        limit=limit,
        run_always=run_always,
    )
    if is_limit_reached:
        return True
    return False


def bench_other(framework, model, model_status, limit, run_always):
    for quant in QUANT_TYPES:
        quant_method = "bitsandbytes" if quant is not None else None
        is_limit_reached = run_benchmark(
            framework,
            model,
            model_status=model_status,
            quant_method=quant_method,
            quant_bits=quant,
            limit=limit,
            run_always=run_always,
        )
        if is_limit_reached:
            return True
    return False


def get_models_to_run(fetch_hub: bool) -> list[str]:
    if fetch_hub:
        params = {
            "sort": "downloads",
            "direction": "-1",
            "limit": 1_000,
            "filter": "text-generation",
        }
        response = requests.get("https://huggingface.co/api/models", params=params)
        model_data = response.json()

        # Extract model IDs
        model_names = [entry["id"] for entry in model_data]
    else:
        model_names = get_cached_models(CACHE_DIR)

    return model_names


def print_summary(model_status: dict[str, int]) -> None:
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


if __name__ == "__main__":
    main()
