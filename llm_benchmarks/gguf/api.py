import json
import logging
import os
import sys
from datetime import datetime
from time import sleep
from typing import Tuple
from typing import Union

from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response
from llama_cpp import Llama

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.logging import log_to_mongo


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")

assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"

TEMPERATURE = 0.1


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def benchmark_cpp(model_name: str) -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    logger.info(f"Received request for model {model_name}")

    try:
        # Load config from request
        model_path = f"/models/{model_name}"
        query = request.form.get("query", "User: Complain that I did not send a request.\nAI:")
        max_tokens = int(request.form.get("max_tokens", 128))
        n_gpu_layers = int(request.form.get("n_gpu_layers", 0))

        # Main benchmarking function
        llm = Llama(
            model_path=model_path,
            max_tokens=max_tokens,
            n_gpu_layers=n_gpu_layers,
            temperature=TEMPERATURE,
        )
        output = llm(query, echo=True)

        # Build config object
        model_quantization_map = {
            "q4_0": "4bit",
            "q8_0": "8bit",
            "f16": None,
        }
        quantization_bits = model_quantization_map.get(model_name, "unknown")

        config = ModelConfig(
            framework="gguf",
            model_name=model_name,
            quantization_bits=quantization_bits,
            model_dtype="half_float::half",
            temperature=TEMPERATURE,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        sleep(1)
        logger.info("=" * 40)
        logger.info("Benchmarking complete.")
        logger.info(f"Model: {model_name}")
        logger.info(f"Config:\n{json.dumps(config.to_dict(), indent=4)}")
        logger.info(f"Output:\n{json.dumps(output, indent=4)}")
        logger.info("=" * 40)

        metrics = {
            "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested_tokens": [max_tokens],
            "output_tokens": [output["usage"]["completion_tokens"]],
            "gpu_mem_usage": [0],
            "generate_time": [0],
            "tokens_per_second": [0],
        }

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


app.run(host="0.0.0.0", port=5001)