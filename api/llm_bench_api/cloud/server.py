import logging
import os
from datetime import datetime

from fastapi import FastAPI
from llm_bench_api.config import CloudConfig
from llm_bench_api.config import MongoConfig
from llm_bench_api.logging import log_to_mongo
from llm_bench_api.types import BenchmarkRequest
from llm_bench_api.types import BenchmarkResponse
from llm_bench_api.utils import has_existing_run


log_path = "./logs/bench_cloud.log"
logging.basicConfig(filename=log_path, level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION_CLOUD = os.environ.get("MONGODB_COLLECTION_CLOUD")
assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION_CLOUD, "MONGODB_COLLECTION_CLOUD environment variable not set"

PROVIDER_MODULES = {
    "openai": "llm_bench_api.cloud.providers.openai",
    "anthropic": "llm_bench_api.cloud.providers.anthropic",
    "bedrock": "llm_bench_api.cloud.providers.bedrock",
    "vertex": "llm_bench_api.cloud.providers.vertex",
    "anyscale": "llm_bench_api.cloud.providers.anyscale",
    "together": "llm_bench_api.cloud.providers.together",
    "openrouter": "llm_bench_api.cloud.providers.openrouter",
    "azure": "llm_bench_api.cloud.providers.azure",
    "runpod": "llm_bench_api.cloud.providers.runpod",
    "fireworks": "llm_bench_api.cloud.providers.fireworks",
    "deepinfra": "llm_bench_api.cloud.providers.deepinfra",
    "groq": "llm_bench_api.cloud.providers.groq",
    "databricks": "llm_bench_api.cloud.providers.databricks",
}

app = FastAPI()


@app.post("/benchmark", response_model=BenchmarkResponse)
async def call_cloud(request: BenchmarkRequest):
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
        mongo_config = MongoConfig(
            uri=MONGODB_URI,
            db=MONGODB_DB,
            collection=MONGODB_COLLECTION_CLOUD,
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

        # Load provider module
        module_name = PROVIDER_MODULES[provider]
        module = __import__(module_name, fromlist=["generate"])
        generate = module.generate

        # Run benchmark
        metrics = generate(model_config, run_config)
        if not metrics:
            error_message = "metrics is empty"
            logger.error(error_message)
            return {"status": "error", "message": error_message}

        logger.info(f"===== Model: {provider}/{model_name} =====")
        logger.info(f"provider: {model_config.provider}")
        logger.info(f"Output tokens: {metrics['output_tokens']}")
        logger.info(f"Generate time: {metrics['generate_time']:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second']:.2f}")

        if metrics["tokens_per_second"] <= 0:
            error_message = "tokens_per_second must be greater than 0"
            logger.error(error_message)
            return {"status": "error", "message": error_message}

        if debug:
            logger.info(f"Debug mode: {debug}")
            logger.info(f"Metrics: {metrics}")
            return {"status": "success", "metrics": metrics}

        log_to_mongo(
            model_type="cloud",
            config=model_config,
            metrics=metrics,
            uri=mongo_config.uri,
            db_name=mongo_config.db,
            collection_name=mongo_config.collection,
        )
        return {"status": "success"}

    except Exception as e:
        error_message = f"An error occurred during benchmark: {str(e)}"
        logger.exception(error_message)
        return {"status": "error", "message": error_message}
