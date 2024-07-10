import logging
import os
from datetime import datetime

from fastapi import FastAPI
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from llm_bench.config import CloudConfig
from llm_bench.config import MongoConfig
from llm_bench.logging import log_metrics
from llm_bench.types import BenchmarkRequest
from llm_bench.types import BenchmarkResponse
from llm_bench.utils import has_existing_run

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR = os.environ.get("LOG_DIR", "/var/log")
LOG_FILE_TXT = os.path.join(LOG_DIR, "benchmarks_cloud.log")
LOG_FILE_JSON = os.path.join(LOG_DIR, "benchmarks_cloud.json")
LOG_TO_MONGO = os.getenv("LOG_TO_MONGO", "False").lower() in ("true", "1", "t")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION_CLOUD = os.environ.get("MONGODB_COLLECTION_CLOUD")

FASTAPI_PORT_CLOUD = os.environ.get("FASTAPI_PORT_CLOUD")
assert FASTAPI_PORT_CLOUD, "FASTAPI_PORT_CLOUD environment variable not set"

logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_FILE_TXT), level=LOG_LEVEL)
logger = logging.getLogger(__name__)

PROVIDER_MODULES = {
    "openai": "llm_bench.cloud.providers.openai",
    "anthropic": "llm_bench.cloud.providers.anthropic",
    "bedrock": "llm_bench.cloud.providers.bedrock",
    "vertex": "llm_bench.cloud.providers.vertex",
    "anyscale": "llm_bench.cloud.providers.anyscale",
    "together": "llm_bench.cloud.providers.together",
    "openrouter": "llm_bench.cloud.providers.openrouter",
    "azure": "llm_bench.cloud.providers.azure",
    "runpod": "llm_bench.cloud.providers.runpod",
    "fireworks": "llm_bench.cloud.providers.fireworks",
    "deepinfra": "llm_bench.cloud.providers.deepinfra",
    "groq": "llm_bench.cloud.providers.groq",
    "databricks": "llm_bench.cloud.providers.databricks",
}

app = FastAPI(
    title="LLM Benchmarking API",
    description="API for benchmarking LLMs on the cloud",
    port=FASTAPI_PORT_CLOUD,
)


@app.post("/benchmark", response_model=BenchmarkResponse)
async def call_cloud(request: BenchmarkRequest):
    retry_count = 0
    try:
        logger.info(f"Received benchmark request: Provider={request.provider}, Model={request.model}")
        provider = request.provider
        model_name = request.model
        query = request.query
        max_tokens = request.max_tokens
        temperature = request.temperature
        run_always = request.run_always
        debug = request.debug

        if provider not in PROVIDER_MODULES:
            error_message = f"Invalid provider: {provider}"
            logger.error(error_message)
            return {"status": "error", "message": error_message}

        if not model_name:
            error_message = "model_name must be set"
            logger.error(error_message)
            return {"status": "error", "message": error_message}

        logger.info(f"Received request for model: {model_name}")

        run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create model config
        model_config = CloudConfig(
            provider=provider,
            model_name=model_name,
            run_ts=run_ts,
            temperature=temperature,
            misc={},
        )

        # Create run config
        run_config = {
            "query": query,
            "max_tokens": max_tokens,
        }

        # Check if model has been benchmarked before
        if LOG_TO_MONGO:
            logger.debug("Logging to MongoDB")
            mongo_config = MongoConfig(
                uri=MONGODB_URI,  # type: ignore
                db=MONGODB_DB,  # type: ignore
                collection=MONGODB_COLLECTION_CLOUD,  # type: ignore
            )
            existing_run = has_existing_run(model_name, model_config, mongo_config)
            if existing_run:
                if run_always:
                    logger.info(f"Model has been benchmarked before: {model_name}")
                    logger.info("Re-running benchmark anyway because run_always is True")
                else:
                    logger.info(f"Model has been benchmarked before: {model_name}")
                    return {"status": "skipped", "reason": "model has been benchmarked before"}
            else:
                logger.info(f"Model has not been benchmarked before: {model_name}")
        else:
            logger.debug("Not logging to MongoDB")

        # Load provider module
        module_name = PROVIDER_MODULES[provider]
        module = __import__(module_name, fromlist=["generate"])
        generate = module.generate

        # Run benchmark
        max_retries = 3

        @retry(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, min=1, max=5))
        def run_benchmark_with_retry():
            nonlocal retry_count
            try:
                metrics = generate(model_config, run_config)
                if not metrics:
                    error_message = "metrics is empty"
                    logger.error(error_message)
                    return {"status": "error", "message": error_message}

                if metrics["tokens_per_second"] <= 0:
                    error_message = "tokens_per_second must be greater than 0"
                    logger.error(error_message)
                    return {"status": "error", "message": error_message}

                return metrics
            except Exception as e:
                retry_count += 1
                logger.warning(f"Retry attempt {retry_count} due to error: {str(e)}")
                raise

        result = run_benchmark_with_retry()

        if isinstance(result, dict) and "status" in result and result["status"] == "error":
            return {**result, "retry_count": retry_count}

        metrics = result
        if debug:
            logger.info(f"Debug mode: {debug}")
            logger.info(f"Metrics: {metrics}")
            logger.info(f"Retry count: {retry_count}")
            return {"status": "success", "metrics": metrics}

        # Log metrics
        log_metrics(
            model_type="cloud",
            config=model_config,
            metrics=metrics,
            file_path=os.path.join(LOG_DIR, LOG_FILE_JSON),
            log_to_mongo=LOG_TO_MONGO,
            mongo_uri=MONGODB_URI,
            mongo_db=MONGODB_DB,
            mongo_collection=MONGODB_COLLECTION_CLOUD,
        )

        # Print metrics
        logger.info(f"===== Model: {provider}/{model_name} =====")
        logger.info(f"provider: {model_config.provider}")
        logger.info(f"Output tokens: {metrics['output_tokens']}")
        logger.info(f"Generate time: {metrics['generate_time']:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second']:.2f}")

        return {"status": "success", "retry_count": retry_count}

    except Exception as e:
        error_message = f"An error occurred during benchmark: {str(e)}"
        logger.exception(error_message)
        return {"status": "error", "message": error_message, "retry_count": retry_count}
