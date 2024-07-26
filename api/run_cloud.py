import asyncio
import os
from datetime import datetime
from typing import List
from typing import Optional

import dotenv
import httpx
import json5 as json
import typer
from llm_bench.cloud.logging import Logger
from llm_bench.types import BenchmarkRequest
from tenacity import RetryError
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

dotenv.load_dotenv()


# Initialize Logger
logger = Logger(
    logs_dir=os.getenv("LOGS_DIR", "./logs"),
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", 6379)),
    redis_db=int(os.getenv("REDIS_DB", 0)),
    redis_password=os.getenv("REDIS_PASSWORD", ""),
)

# Constants
QUERY_TEXT = "Tell a long and happy story about the history of the world."
MAX_TOKENS = 256
TEMPERATURE = 0.1
FASTAPI_PORT = os.environ.get("FASTAPI_PORT_CLOUD")
assert FASTAPI_PORT, "FASTAPI_PORT environment variable not set"
server_path = f"http://localhost:{FASTAPI_PORT}/benchmark"

# Load provider models from JSON
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "../cloud/models.json")
with open(json_file_path) as f:
    provider_models = json.load(f)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
async def post_benchmark(request: BenchmarkRequest):
    timeout = httpx.Timeout(60.0, connect=10.0)
    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=timeout) as client:
        logger.log_info(f"Sending request to {server_path}")
        response = await client.post(server_path, json=request.model_dump())
        response.raise_for_status()

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()
    response_data = response.json()

    if "error" in response_data or response_data.get("status") == "error":
        raise ValueError(response_data.get("message", "Unknown error"))

    return response_data, response_time


# Wrap the post_benchmark function to handle final output
async def benchmark_with_retries(request: BenchmarkRequest):
    retry_count = 0
    try:
        while True:
            try:
                response_data, response_time = await post_benchmark(request)
                return {
                    "status": "success",
                    "data": response_data,
                    "response_time": response_time,
                    "retry_count": retry_count,
                }
            except Exception:
                retry_count += 1
                if retry_count >= 3:  # Max retries
                    raise
    except RetryError as e:
        return {"status": "error", "message": str(e.last_attempt.exception()), "retry_count": retry_count}
    except Exception as e:
        return {"status": "error", "message": str(e), "retry_count": retry_count}


async def benchmark_provider(provider, limit, run_always, debug):
    model_names = provider_models.get(provider, [])[:limit]
    logger.log_info(f"Fetching {len(model_names)} models for provider: {provider}")

    for model in model_names:
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
            response_data, response_time = await post_benchmark(request_config)
            logger.log_info(f"✅ Success {model}, {response_data}")
            status = "success"
            error = None
        except RetryError as e:
            logger.log_error(f"❌ Error {model}, error: {e.last_attempt.exception()}")
            status = "error"
            error = str(e.last_attempt.exception())
            response_data, response_time = None, 0

        yield {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "request": {"provider": provider, "model": model, "max_tokens": MAX_TOKENS},
            "status": status,
            "response": response_data,
            "error": error,
            "response_time": response_time,
        }


app = typer.Typer()


@app.command()
def main(
    providers: Optional[List[str]] = typer.Option(None, "--providers", help="Providers to use for benchmarking."),
    limit: int = typer.Option(100, "--limit", help="Limit the number of models run."),
    run_always: bool = typer.Option(False, "--run-always", is_flag=True, help="Flag to always run benchmarks."),
    debug: bool = typer.Option(False, "--debug", is_flag=True, help="Flag to enable debug mode."),
) -> None:
    async def async_main():
        nonlocal providers
        if providers is None or "all" in providers:
            providers = list(provider_models.keys())
        logger.log_info(f"Running benchmarks for provider(s): {providers}")

        all_results = []
        for provider in providers:
            async for result in benchmark_provider(provider, limit, run_always, debug):
                all_results.append(result)

        logger.log_benchmark_status(all_results)

    asyncio.run(async_main())


if __name__ == "__main__":
    app()
