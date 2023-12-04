import logging
import os
import time
from datetime import datetime
from typing import Tuple
from typing import Union

import pynvml
from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response
from llama_cpp import Llama
from llm_bench_api.config import ModelConfig
from llm_bench_api.config import MongoConfig
from llm_bench_api.logging import log_to_mongo


logging.basicConfig(filename="/var/log/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")

assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"


@app.route("/benchmark", methods=["POST"])
def benchmark_cpp() -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        # Load config from request
        model_name = request.form.get("model_name")
        model_path = f"/models/{model_name}"
        query = request.form.get("query", "User: Complain that I did not send a request.\nAI:")
        max_tokens = int(request.form.get("max_tokens", 512))
        temperature = request.form.get("temperature", default=0.1, type=float)
        n_gpu_layers = int(request.form.get("n_gpu_layers", 0))

        logger.info(f"Received request for model {model_name}")

        # Main benchmarking function
        llm = Llama(
            model_path=model_path,
            max_tokens=max_tokens,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
        )

        # Get GPU memory usage
        time.sleep(1)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # second GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        # Run benchmark
        time_0 = time.time()
        output = llm(query, echo=True)
        time_1 = time.time()

        # Build config object
        model_quantization_list = [
            ("q4_0", "4bit"),
            ("q8_0", "8bit"),
            ("f16", None),
        ]
        quantization_bits = next(
            (bits for key, bits in model_quantization_list if key in model_name),
            "unknown",
        )

        model_config = ModelConfig(
            framework="gguf",
            model_name=model_name,
            quantization_bits=quantization_bits,
            model_dtype="half_float::half",
            temperature=temperature,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Build metrics object
        output_tokens = output["usage"]["completion_tokens"]

        metrics = {
            "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested_tokens": [max_tokens],
            "output_tokens": [output_tokens],
            "gpu_mem_usage": [info.used],
            "generate_time": [time_1 - time_0],
            "tokens_per_second": [output_tokens / (time_1 - time_0) if time_1 > time_0 else 0],
        }

        mongo_config = MongoConfig(
            uri=MONGODB_URI,
            db=MONGODB_DB,
            collection=MONGODB_COLLECTION,
        )

        # Log metrics to MongoDB
        log_to_mongo(
            model_type="local",
            config=model_config,
            metrics=metrics,
            uri=mongo_config.uri,
            db_name=mongo_config.db,
            collection_name=mongo_config.collection,
        )

        logger.info(f"===== Model: {model_name} =====")
        logger.info(f"Requested tokens: {max_tokens}")
        logger.info(f"Output tokens: {metrics['output_tokens'][0]}")
        logger.info(f"GPU mem usage: {(metrics['gpu_mem_usage'][0] / 1024**3) :.2f}GB")
        logger.info(f"Generate time: {metrics['generate_time'][0]:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second'][0]:.2f}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception(f"Error in call_benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


app.run(host="0.0.0.0", port=5001)
