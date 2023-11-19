import logging
import os
from datetime import datetime
from typing import Tuple
from typing import Union
from urllib.parse import unquote

from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response
from llm_bench_api.config import ModelConfig
from llm_bench_api.config import MongoConfig
from llm_bench_api.logging import log_to_mongo
from llm_bench_api.utils import check_and_clean_space
from llm_bench_api.utils import has_existing_run


log_path = "/var/log/llm_benchmarks.log"
try:
    logging.basicConfig(filename=log_path, level=logging.INFO)
except PermissionError:
    logging.basicConfig(filename="./logs/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")
assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"

DO_SAMPLE = False


app = Flask(__name__)


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def call_huggingface(model_name: str) -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        model_name = unquote(model_name)
        framework = request.form.get("framework", type=str)
        query = request.form.get("query", default=None, type=str)
        quant_method = request.form.get("quant_method", default=None, type=str)
        quant_bits = request.form.get("quant_bits", default=None, type=str)
        max_tokens = request.form.get("max_tokens", default=512, type=int)
        temperature = request.form.get("temperature", default=0.1, type=float)
        run_always = request.form.get("run_always", default=False, type=bool)

        assert framework is not None, "framework is required"

        quant_str = f"{quant_method}_{quant_bits}" if quant_method is not None else "none"
        logger.info(f"Received request for model: {model_name}, quant: {quant_str}")

        # Create model config
        model_config = ModelConfig(
            framework=framework,
            model_name=model_name,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_dtype="torch.float16",
            quantization_method=quant_method,
            quantization_bits=quant_bits,
            temperature=temperature,
            misc={"do_sample": DO_SAMPLE},
        )

        run_config = {
            "query": query,
            "max_tokens": max_tokens,
        }

        # Check if model has been benchmarked before
        mongo_config = MongoConfig(
            uri=MONGODB_URI,
            db=MONGODB_DB,
            collection=MONGODB_COLLECTION,
        )
        existing_run = has_existing_run(model_name, model_config, mongo_config)
        if not run_always and existing_run:
            logger.info(f"Model has been benchmarked before: {model_name}, quant: {quant_str}")
            return jsonify({"status": "skipped", "reason": "model has been benchmarked before"}), 200

        logger.info(f"Model has not been benchmarked before: {model_name}, quant: {quant_str}")

        # Check and clean disk space if needed
        check_and_clean_space(directory=CACHE_DIR, threshold=90.0)

        if framework == "hf-tgi":
            from llm_bench_huggingface.tgi import generate
        elif framework == "transformers":
            from llm_bench_huggingface.transformers import generate
        else:
            raise ValueError(f"Unknown framework: {framework}")

        # Main benchmarking function
        metrics = generate(
            model_config,
            run_config,
        )
        assert metrics, "metrics is empty"

        # logger.info metrics
        logger.info(f"===== Model: {model_name} =====")
        logger.info(f"Requested tokens: {run_config['max_tokens']}")
        logger.info(f"Output tokens: {metrics['output_tokens'][0]}")
        logger.info(f"GPU mem usage: {(metrics['gpu_mem_usage'][0] / 1024**3) :.2f}GB")
        logger.info(f"Generate time: {metrics['generate_time'][0]:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second'][0]:.2f}")

        # Log metrics to MongoDB
        result = log_to_mongo(
            config=model_config,
            metrics=metrics,
            uri=mongo_config.uri,
            db_name=mongo_config.db,
            collection_name=mongo_config.collection,
        )

        return jsonify(result), 200
    except Exception as e:
        logger.exception(f"Error in call_benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


app.run(host="0.0.0.0", port=5000)
