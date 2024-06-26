import os

import click
from llm_bench.api import bench_all_models
from llm_bench.api import print_summary
from llm_bench.utils import fetch_hf_models
from llm_bench.utils import filter_model_size

QUANT_TYPES = [
    None,
]
QUERY_TEXT = "User: Tell me a long story about the history of the world.\nAI:"
MAX_TOKENS = 256
TEMPERATURE = 0.1
FLASK_PORT = 5002
CACHE_DIR = os.environ.get("HF_HUB_CACHE")
assert CACHE_DIR, "HF_HUB_CACHE environment variable not set"


@click.command()
@click.option("--framework", help="Framework to use, must be 'vllm'.")
@click.option("--limit", default=100, type=int, help="Limit the number of models fetched.")
@click.option(
    "--max-size-billion",
    default=5,
    type=int,
    help="Maximum size of models in billion parameters.",
)
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
@click.option("--fetch-new-models", is_flag=True, help="Fetch latest HF-Hub models.")
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
    print(f"Initial run_always value: {run_always}")

    # Gather models to run
    model_names = fetch_hf_models(
        fetch_new=fetch_new_models,
        cache_dir=CACHE_DIR,
        library="transformers",
        created_days_ago=30,
    )
    print(f"Fetched {len(model_names)} models")

    # Filter based on parameter count
    valid_models = filter_model_size(model_names, max_size_billion * 1_000)
    print(f"Filtered down to {len(valid_models)} models")

    valid_models = [
        # "facebook/opt-125m",
        # "TheBloke/Llama-2-7B-Chat-GPTQ",
        # "EleutherAI/pythia-160m",
        # "TheBloke/Llama-2-7B-Chat-AWQ",
        "meta-llama/Meta-Llama-3-8B",
    ]

    # Run benchmarks
    model_status: dict[str, dict] = {}
    bench_all_models(
        framework,
        QUANT_TYPES,
        valid_models,
        model_status,
        limit,
        run_always,
        QUERY_TEXT,
        MAX_TOKENS,
        TEMPERATURE,
        FLASK_PORT,
    )

    # Print summary
    print("All benchmark runs are finished.")
    print_summary(model_status)


if __name__ == "__main__":
    main()
