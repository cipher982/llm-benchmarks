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

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.config import MongoConfig
from llm_benchmarks.logging import log_to_mongo
from llm_benchmarks.transformers import generate
from llm_benchmarks.utils import check_and_clean_space
from llm_benchmarks.utils import has_existing_run


logging.basicConfig(filename="/var/log/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")
assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"


app = Flask(__name__)


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def benchmark_transformers(model_name: str) -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        model_name = unquote(model_name)
        # query = request.args.get("query", default=None, type=str) # TODO: query not needed since we have min tokens
        quant_method = request.form.get("quant_method", default=None, type=str)
        quant_bits = request.form.get("quant_bits", default=None, type=str)
        max_tokens = request.form.get("max_tokens", default=512, type=int)
        run_always = request.form.get("run_always", default=False, type=bool)

        quant_str = f"{quant_method}_{quant_bits}" if quant_method is not None else "none"
        logger.info(f"Received request for model: {model_name}, quant: {quant_str}")

        # Create model config
        model_config = ModelConfig(
            framework="transformers",
            model_name=model_name,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_dtype="torch.float16",
            quantization_method=quant_method,
            quantization_bits=quant_bits,
            temperature=0.1,
        )

        # Check if model has been benchmarked before
        mongo_config = MongoConfig(
            uri=MONGODB_URI,
            db=MONGODB_DB,
            collection=MONGODB_COLLECTION,
        )
        if not run_always and has_existing_run(model_name, model_config, mongo_config):
            logger.info(f"Model has been benchmarked before: {model_name}, quant: {quant_str}")
            return jsonify({"status": "skipped", "reason": "model has been benchmarked before"}), 200
        logger.info(f"Model has not been benchmarked before: {model_name}, quant: {quant_str}")

        # Check and clean disk space if needed
        check_and_clean_space(directory=CACHE_DIR, threshold=90.0)

        # Main benchmarking function
        metrics = generate(
            model_config,
            max_tokens=max_tokens,
            llama=False,
        )

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
