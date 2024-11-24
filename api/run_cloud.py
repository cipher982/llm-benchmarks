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
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

dotenv.load_dotenv()


# Initialize Logger
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL environment variable is not set")

logger = Logger(
    logs_dir=os.getenv("LOGS_DIR", "./logs"),
    redis_url=redis_url,
)

# Constants
QUERY_TEXT = "Tell a long and happy story about the history of the world."
MAX_TOKENS = 64
TEMPERATURE = 0.1
FASTAPI_PORT = os.environ.get("FASTAPI_PORT_CLOUD")
assert FASTAPI_PORT, "FASTAPI_PORT environment variable not set"
server_path = f"http://localhost:{FASTAPI_PORT}/benchmark"
MAX_RETRIES = 3

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
        try:
            response = await client.post(server_path, json=request.model_dump())
            response.raise_for_status()
        except httpx.HTTPError as e:
            error_msg = f"HTTP Error: {str(e)} (Status: {e.response.status_code if hasattr(e, 'response') else 'N/A'})"
            if hasattr(e, "response"):
                try:
                    error_data = e.response.json()
                    if error_data:
                        error_msg += f" - {error_data}"
                except Exception as e:
                    pass
            raise ValueError(error_msg)

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()
    response_data = response.json()

    if "error" in response_data or response_data.get("status") == "error":
        error_details = []
        if response_data.get("reason"):
            error_details.append(response_data["reason"])
        if response_data.get("message"):
            error_details.append(response_data["message"])
        if response_data.get("error"):
            error_details.append(str(response_data["error"]))
        if response_data.get("metrics"):
            error_details.append(f"metrics: {response_data['metrics']}")

        error_msg = " | ".join(filter(None, error_details))
        if not error_msg:
            error_msg = f"Server returned error status with response: {response_data}"
        raise ValueError(error_msg)

    return response_data, response_time


async def benchmark_with_retries(request: BenchmarkRequest):
    retry_count = 0
    last_error = None

    while retry_count < MAX_RETRIES:
        try:
            response_data, response_time = await post_benchmark(request)
            return {
                "status": "success",
                "data": response_data,
                "response_time": response_time,
                "retry_count": retry_count,
            }
        except Exception as e:
            retry_count += 1
            # Get the actual error message, not just the RetryError wrapper
            if hasattr(e, "last_attempt") and hasattr(e.last_attempt, "exception"):
                last_error = str(e.last_attempt.exception())
            else:
                last_error = str(e)

            if retry_count >= MAX_RETRIES:
                logger.log_error(f"❌ Error {request.model}: {last_error} (after {retry_count} attempts)")
                break
            else:
                logger.log_info(
                    f"Attempt {retry_count}/{MAX_RETRIES} failed for {request.model}: {last_error}, retrying..."
                )
                await asyncio.sleep(1)

    return {"status": "error", "message": last_error, "retry_count": retry_count, "response_time": None, "data": None}


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

        result = await benchmark_with_retries(request_config)
        if result["status"] == "success":
            logger.log_info(f"✅ Success {model}, {result['data']}")

        yield {
            "model": model,
            "provider": provider,
            "status": result["status"],
            "data": result["data"],
            "response_time": result["response_time"],
            "error": result.get("message"),
            "retry_count": result["retry_count"],
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
