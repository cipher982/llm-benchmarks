import logging
import os
from datetime import datetime
from typing import Tuple
from typing import Union
from urllib.parse import unquote

import torch
from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response
from pymongo import MongoClient

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.logging import log_to_mongo
from llm_benchmarks.transformers import generate
from llm_benchmarks.utils import check_and_clean_space

logging.basicConfig(filename="/var/log/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")

assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def benchmark_transformers(model_name: str) -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        model_name = unquote(model_name)
        run_always = request.args.get("run_always", default="False", type=str).lower() == "true"

        logger.info(f"Received request for model {model_name}")

        # Declare config defaults
        config = ModelConfig(
            model_name=model_name,
            quantization_bits=None,
            torch_dtype=torch.float16,
            temperature=0.1,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Initialize MongoDB client and collection
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]

        # Check if the model has already been benchmarked
        existing_config = collection.find_one(
            {
                "model_name": model_name,
                "torch_dtype": str(config.torch_dtype),
                "quantization_bits": config.quantization_bits,
            }
        )
        if existing_config and not run_always:
            logger.info(f"Model {model_name} has already been benchmarked. Skipping.")
            return jsonify({"status": "skipped", "reason": "Model has already been benchmarked"})

        # Check and clean disk space if needed
        check_and_clean_space()

        # Main benchmarking function
        metrics = generate(
            config,
            custom_token_counts=None,
            llama=False,
        )

        # Log metrics to MongoDB
        result = log_to_mongo(
            config=config,
            metrics=metrics,
            uri=MONGODB_URI,
            db_name=MONGODB_DB,
            collection_name=MONGODB_COLLECTION,
        )

        return jsonify(result), 200
    except Exception as e:
        logger.exception(f"Error in call_benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


app.run(host="0.0.0.0", port=5000)
