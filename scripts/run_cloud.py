import asyncio
import json
import os
from datetime import datetime

import httpx
import typer
from llm_bench_api.cloud.logging import log_benchmark_request
from llm_bench_api.cloud.logging import log_benchmark_status
from llm_bench_api.cloud.logging import log_error
from llm_bench_api.cloud.logging import log_info
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
    log_benchmark_request(request)
    timeout = httpx.Timeout(180.0, connect=180.0)
    try:
        start_time = datetime.now()
        async with httpx.AsyncClient(timeout=timeout) as client:
            log_info(f"Sending request to {SERVER_PATH}")
            response = await client.post(SERVER_PATH, json=request.model_dump())
            response.raise_for_status()
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        return response.json(), response_time
    except httpx.HTTPStatusError as e:
        log_error(f"HTTP error occurred: {str(e)}")
        return {"error": f"HTTP error: {str(e)}"}, None
    except Exception as e:
        log_error(f"Unexpected error occurred: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}, None


app = typer.Typer()


@app.command()
def main(
    providers: list[str] = typer.Option(["all"], "--providers", help="Providers to use for benchmarking."),
    limit: int = typer.Option(100, "--limit", help="Limit the number of models run."),
    run_always: bool = typer.Option(False, "--run-always", is_flag=True, help="Flag to always run benchmarks."),
    debug: bool = typer.Option(False, "--debug", is_flag=True, help="Flag to enable debug mode."),
) -> None:
    """Main entrypoint for benchmarking cloud models."""

    async def benchmark_provider(provider, limit, run_always, debug):
        model_names = provider_models.get(provider, [])
        log_info(f"Fetched {len(model_names)} models for provider: {provider}")
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
            response, response_time = await post_benchmark(request_config)
            # Create status entry
            status_entry = {
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "provider": provider,
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                },
                "response_time": response_time,
            }
            if "error" in response:
                log_error(f"❌ Fail {model}, error: {response['error']}")
                status_entry.update({"status": "error", "error": response["error"]})
            else:
                log_info(f"✅ Pass {model}, {response}")
                status_entry.update({"status": "success", "response": response})
            model_status.append(status_entry)
        return model_status

    async def async_main(providers, limit, run_always, debug):
        if "all" in providers:
            providers = list(provider_models.keys())
        log_info(f"Running benchmarks for provider(s): {providers}")

        # Run benchmarks for each provider asynchronously
        tasks = [benchmark_provider(provider, limit, run_always, debug) for provider in providers]
        results = await asyncio.gather(*tasks)

        # Flatten the list of model statuses and log final results
        combined_model_status = [status for provider_status in results if provider_status for status in provider_status]
        log_benchmark_status(combined_model_status)

    asyncio.run(async_main(providers, limit, run_always, debug))


if __name__ == "__main__":
    app()
