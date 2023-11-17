import logging
import os
import time
from datetime import datetime
from urllib.parse import unquote

from flask import Flask
from flask import jsonify
from flask import request
from huggingface_hub import InferenceClient

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.config import MongoConfig
from llm_benchmarks.logging import log_to_mongo
from llm_benchmarks.tgi.docker import DockerContainer
from llm_benchmarks.utils import check_and_clean_space
from llm_benchmarks.utils import get_vram_usage
from llm_benchmarks.utils import has_existing_run
from llm_benchmarks.utils import setup_logger


app = Flask(__name__)

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")
assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"
TEMPERATURE = 0.1
GPU_DEVICE = 1


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def benchmark_tgi_route(model_name: str):
    logger.info(f"Received TGI benchmark request for model {model_name}")

    try:
        # Extract and validate request parameters
        model_name = unquote(model_name)
        query = request.args.get("query", default=None, type=str)
        quant_method = request.form.get("quant_method", default=None, type=str)
        quant_bits = request.form.get("quant_bits", default=None, type=str)
        max_tokens = request.form.get("max_tokens", default=512, type=int)
        run_always = request.form.get("run_always", default=False, type=bool)

        assert query is not None, "query string is required"

        quant_str = f"{quant_method}_{quant_bits}" if quant_method is not None else "none"
        logger.info(f"Received request for model: {model_name}, quant: {quant_str}")

        # Create model config
        model_config = ModelConfig(
            framework="hf-tgi",
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
        check_and_clean_space(CACHE_DIR, 90.0)

        # Initialize Docker container and client
        with DockerContainer(model_name, CACHE_DIR, GPU_DEVICE, quant_method, quant_bits) as container:
            if container.is_ready():
                logger.info("Docker container is ready.")
                client = InferenceClient("http://127.0.0.0:8080")

                # Run benchmark
                time0 = time.time()
                response = client.text_generation(
                    prompt=query, max_new_tokens=max_tokens, temperature=TEMPERATURE, details=True
                )
                time1 = time.time()

                # Process metrics
                output_tokens = len(response.details.get("tokens", []))  # type: ignore
                vram_usage = get_vram_usage(GPU_DEVICE)
                metrics = {
                    "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "requested_tokens": [max_tokens],
                    "output_tokens": [output_tokens],
                    "gpu_mem_usage": [vram_usage],
                    "generate_time": [time1 - time0],
                    "tokens_per_second": [output_tokens / (time1 - time0) if time1 > time0 else 0],
                }

                # Log metrics to MongoDB
                result = log_to_mongo(
                    config=model_config,
                    metrics=metrics,
                    uri=MONGODB_URI,
                    db_name=MONGODB_DB,
                    collection_name=MONGODB_COLLECTION,
                )

                logger.info(f"Benchmarking complete, model {model_name}, quant: {quant_bits}.")
                return jsonify(result), 200
            else:
                raise Exception("Docker container not ready")
    except Exception as e:
        logger.exception(f"Error in benchmark_tgi_route: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
