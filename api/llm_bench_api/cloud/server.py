import logging
import os
from datetime import datetime
from typing import Tuple
from typing import Union

from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response
from llm_bench_api.config import CloudConfig
from llm_bench_api.config import MongoConfig
from llm_bench_api.logging import log_to_mongo
from llm_bench_api.utils import has_existing_run


log_path = "/var/log/bench_cloud.log"
try:
    logging.basicConfig(filename=log_path, level=logging.INFO)
except PermissionError:
    logging.basicConfig(filename="./logs/bench_cloud.log", level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION_CLOUD = os.environ.get("MONGODB_COLLECTION_CLOUD")
assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION_CLOUD, "MONGODB_COLLECTION_CLOUD environment variable not set"


app = Flask(__name__)


@app.route("/benchmark", methods=["POST"])
def call_cloud() -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""

    try:
        provider = request.form.get("provider", type=str)
        model_name = request.form.get("model_name", type=str)
        query = request.form.get("query", default=None, type=str)
        max_tokens = request.form.get("max_tokens", default=512, type=int)
        temperature = request.form.get("temperature", default=0.1, type=float)

        run_always_str = request.form.get("run_always", "False").lower()
        run_always = run_always_str == "true"

        assert provider in ["openai", "anthropic", "bedrock"], "provider must be 'openai', 'anthropic', or 'bedrock'"
        assert model_name, "model_name must be set"

        logger.info(f"Received request for model: {model_name}")

        run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create model config
        model_config = CloudConfig(
            provider=provider,
            model_name=model_name,
            run_ts=run_ts,
            temperature=temperature,
            misc={},  # TODO: anything to add here?
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
                return jsonify({"status": "skipped", "reason": "model has been benchmarked before"}), 200
        else:
            logger.info(f"Model has not been benchmarked before: {model_name}")

        # Main benchmarking function
        if provider == "openai":
            from llm_bench_api.cloud.openai import generate
        elif provider == "anthropic":
            from llm_bench_api.cloud.anthropic import generate
        elif provider == "bedrock":
            from llm_bench_api.cloud.bedrock import generate
        else:
            raise Exception(f"provider {provider} not supported")
        metrics = generate(model_config, run_config)
        assert metrics, "metrics is empty"

        # Log it all
        logger.info(f"===== Model: {provider}/{model_name} =====")
        logger.info(f"provider: {model_config.provider}")
        logger.info(f"Output tokens: {metrics['output_tokens']}")
        logger.info(f"Generate time: {metrics['generate_time']:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second']:.2f}")

        log_to_mongo(
            model_type="cloud",
            config=model_config,
            metrics=metrics,
            uri=mongo_config.uri,
            db_name=mongo_config.db,
            collection_name=mongo_config.collection,
        )
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception(f"Error in openai benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
