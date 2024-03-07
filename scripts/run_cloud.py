import asyncio
import json
import os

import httpx
import typer
from llm_bench_api.types import BenchmarkRequest

# Constants
QUERY_TEXT = "Tell me a long story of the history of the world."
MAX_TOKENS = 256
TEMPERATURE = 0.1
SERVER_PATH = os.environ.get("SERVER_PATH", "http://localhost:8000/benchmark")

# Load provider models from JSON
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "../models/cloud.json")
with open(json_file_path) as f:
    provider_models = json.load(f)


async def post_benchmark(request: BenchmarkRequest):
    async with httpx.AsyncClient() as client:
        print(f"Sending request to {SERVER_PATH}")
        response = await client.post(SERVER_PATH, json=request.model_dump())
        response.raise_for_status()
        return response.json()


app = typer.Typer()


@app.command()
def main(
    providers: list[str] = typer.Option(["all"], "--providers", help="Providers to use for benchmarking."),
    limit: int = typer.Option(100, "--limit", help="Limit the number of models run."),
    run_always: bool = typer.Option(False, "--run-always", is_flag=True, help="Flag to always run benchmarks."),
    debug: bool = typer.Option(False, "--debug", is_flag=True, help="Flag to enable debug mode."),
) -> None:
    """
    Main entrypoint for benchmarking cloud models.
    """

    async def async_main(providers, limit, run_always, debug):
        if "all" in providers:
            providers = list(provider_models.keys())

        print(f"Running benchmarks for provider(s): {providers}")

        for provider in providers:
            # Gather models to run
            model_names = provider_models.get(provider, [])
            print(f"Fetched {len(model_names)} models for provider: {provider}")

            # Run benchmarks
            model_status: list[dict] = []
            for model in model_names[:limit]:
                request_config = BenchmarkRequest(
                    provider=provider,
                    model=model,
                    query=QUERY_TEXT,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    run_always=run_always,
                    debug=debug,
                )
                try:
                    response = await post_benchmark(request_config)
                    print(f"✅ Pass {model}, {response}")
                    model_status.append({"model": model, "status": "success", "response": response})
                except httpx.HTTPStatusError as http_err:
                    print(f"❌ Fail {model}, HTTP error: {http_err}")
                    model_status.append({"model": model, "status": "error", "error": str(http_err)})
                except Exception as err:
                    print(f"❌ Fail {model}, other error: {err}")
                    model_status.append({"model": model, "status": "error", "error": str(err)})

                if len(model_status) >= limit:
                    break

    asyncio.run(async_main(providers, limit, run_always, debug))


if __name__ == "__main__":
    app()
