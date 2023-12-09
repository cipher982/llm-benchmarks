import logging

import click
import requests
from requests.exceptions import HTTPError


log_path = "/var/log/llm_benchmarks.log"
try:
    logging.basicConfig(filename=log_path, level=logging.INFO)
except PermissionError:
    logging.basicConfig(filename="./logs/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

QUERY_TEXT = "Tell me a long story about the history of the world."
MAX_TOKENS = 256
TEMPERATURE = 0.1
FLASK_URL = "http://localhost:{}/benchmark"
FLASK_PORT = 5004


@click.command()
@click.option("--provider", help="Provider to use, must be 'openai'.")
@click.option("--limit", default=100, type=int, help="Limit the number of models run.")
@click.option("--run-always", is_flag=True, help="Flag to always run benchmarks.")
def main(
    provider: str,
    limit: int,
    run_always: bool,
) -> None:
    """
    Main entrypoint for benchmarking OpenAI models.
    """

    assert provider == "openai", "Provider must be 'openai'"

    # Gather models to run
    model_names = [
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-vision-preview",
        # "gpt-4-32k",  # no access
        "gpt-4-0613",
        # "gpt-4-32k-0613",  # no access
        "gpt-4-0314",
        # "gpt-4-32k-0314",  # no model
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-instruct",  # use v1-completions (not chat)
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
    ]
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
