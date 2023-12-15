import json
import logging
import os

import click
import requests
from requests.exceptions import HTTPError


log_path = "/var/log/llm_benchmarks.log"
try:
    logging.basicConfig(filename=log_path, level=logging.INFO)
except PermissionError:
    logging.basicConfig(filename="./logs/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

QUERY_TEXT = "Tell me a long story about the history of WW2."
MAX_TOKENS = 512
TEMPERATURE = 0.1
FLASK_URL = "http://localhost:{}/benchmark"
FLASK_PORT = 5004


script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "../models/cloud.json")
with open(json_file_path) as f:
    provider_models = json.load(f)


@click.command()
@click.option("--provider", help="Provider to use, must be 'openai' or 'anthropic'.")
@click.option("--limit", default=100, type=int, help="Limit the number of models run.")
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
def main(
    provider: str,
    limit: int,
    run_always: bool,
) -> None:
    """
    Main entrypoint for benchmarking cloud models.
    """

    assert provider in ["openai", "anthropic"], "provider must be either 'openai' or 'anthropic'"

    # Gather models to run
    model_names = provider_models[provider]
    print(f"Fetched {len(model_names)} models")

    # Run benchmarks
    model_status: dict[str, dict] = {}

    for model in model_names:
        flask_data = {
            "provider": provider,
            "model_name": model,
            "query": QUERY_TEXT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "run_always": run_always,
        }

        try:
            response = requests.post(FLASK_URL.format(FLASK_PORT), data=flask_data)
            response.raise_for_status()
        except HTTPError as http_err:
            print(f"❌ Fail {model}, HTTP error: {http_err}")
            model_status[model] = {"status_code": 500, "json": {}}
        except Exception as err:
            print(f"❌ Fail {model}, other error: {err}")
            model_status[model] = {"status_code": 500, "json": {}}
        else:
            response_code = response.status_code
            response_json = response.json()
            print(f"✅ Pass {model}, {model}: {response_code}")

            model_status[model] = {"status_code": response_code, "json": response_json}

        if len(model_status) >= limit:
            break


if __name__ == "__main__":
    main()
