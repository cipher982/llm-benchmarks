import os

import click
from llm_bench.api import bench_all_models
from llm_bench.api import print_summary
from llm_bench.utils import fetch_hf_models
from llm_bench.utils import filter_model_size

QUANT_TYPES = [
    "4bit",
    "8bit",
    None,
]
QUERY_TEXT = "User: Tell me a long story about the history of the world.\nAI:"
MAX_TOKENS = 256
TEMPERATURE = 0.1
FLASK_PORT_HF_TF = int(os.environ.get("FLASK_PORT_HF_TF", 0))
FLASK_PORT_HF_TGI = int(os.environ.get("FLASK_PORT_HF_TGI", 0))
CACHE_DIR = os.environ.get("HF_HUB_CACHE")
assert CACHE_DIR, "HF_HUB_CACHE environment variable not set"


@click.command()
@click.option(
    "--framework",
    type=str,
    help="LLM API to call. Must be one of 'transformers', 'hf-tgi'",
)
@click.option(
    "--limit",
    default=100,
    type=int,
    help="Limit the number of models run.",
)
@click.option(
    "--max-size-billion",
    default=5,
    type=int,
    help="Maximum size of models in billion parameters.",
)
@click.option(
    "--run-always",
    is_flag=True,
    help="Flag to always run benchmarks.",
)
@click.option(
    "--fetch-new-models",
    is_flag=True,
    help="Fetch latest HF-Hub models.",
)
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

    # Gather models to run
    model_names = fetch_hf_models(fetch_new_models, CACHE_DIR)
    print(f"Fetched {len(model_names)} models")

    # Filter based on parameter count
    valid_models = filter_model_size(model_names, max_size_billion * 1_000)
    print(f"Filtered down to {len(valid_models)} models")

    # Set port
    if framework == "transformers":
        flask_port = FLASK_PORT_HF_TF
    elif framework == "hf-tgi":
        flask_port = FLASK_PORT_HF_TGI
    else:
        raise ValueError(f"Invalid framework: {framework}")

    valid_models = [
        # "facebook/opt-125m",
        # "TheBloke/Llama-2-7B-Chat-GPTQ",
        # "EleutherAI/pythia-160m",
        # "TheBloke/Llama-2-7B-Chat-AWQ",
        # "meta-llama/Llama-2-7b-chat-hf",
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
        flask_port,
    )

    # Print summary
    print("All benchmark runs are finished.")
    print_summary(model_status)


if __name__ == "__main__":
    main()
