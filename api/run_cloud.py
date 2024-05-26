import asyncio
import os
from datetime import datetime

import dotenv
import httpx
import json5 as json
import typer
from llm_bench.cloud.logging import Logger
from llm_bench.types import BenchmarkRequest

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
QUERY_TEXT = "Tell me a long story of the history of the world."
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


async def post_benchmark(request: BenchmarkRequest):
    logger.log_benchmark_request(request)
    timeout = httpx.Timeout(180.0, connect=180.0)
    try:
        start_time = datetime.now()
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.log_info(f"Sending request to {server_path}")
            response = await client.post(server_path, json=request.model_dump())
            response.raise_for_status()
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        return response.json(), response_time
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error: {e.response.status_code} - {e.response.text}"
        logger.log_error(f"HTTP error occurred: {error_message}")
        return {"error": error_message}, None
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.log_error(f"Unexpected error occurred: {error_message}")
        return {"error": error_message}, None


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
        logger.log_info(f"Fetched {len(model_names)} models for provider: {provider}")
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
            if response.get("status") == "success":
                logger.log_info(f"✅ Success {model}, {response}")
                status_entry.update({"status": "success", "response": response})
            elif "error" in response or response.get("status") == "error":
                error_message = response.get("error", "Unknown error")
                logger.log_error(f"❌ Error {model}, error: {error_message}")
                status_entry.update({"status": "error", "error": error_message})
            else:
                unexpected_status = response.get("status", "Unknown status")
                logger.log_error(f"⚠️ Unexpected status {model}, status: {unexpected_status}")
                status_entry.update(
                    {
                        "status": "unexpected",
                        "error": f"Unexpected status: {unexpected_status}",
                    }
                )
            model_status.append(status_entry)
        return model_status

    async def async_main(providers, limit, run_always, debug):
        if "all" in providers:
            providers = list(provider_models.keys())
        logger.log_info(f"Running benchmarks for provider(s): {providers}")

        # Run benchmarks for each provider asynchronously
        tasks = [benchmark_provider(provider, limit, run_always, debug) for provider in providers]
        results = await asyncio.gather(*tasks)

        # Flatten the list of model statuses and log final results
        combined_model_status = [status for provider_status in results if provider_status for status in provider_status]
        logger.log_benchmark_status(combined_model_status)

    asyncio.run(async_main(providers, limit, run_always, debug))


if __name__ == "__main__":
    app()
