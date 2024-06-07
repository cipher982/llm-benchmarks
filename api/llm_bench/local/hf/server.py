import logging
import os
from datetime import datetime
from typing import Tuple
from typing import Union

import click
from flask import Flask
from flask import jsonify
from flask import request
from flask.wrappers import Response

from llm_bench.config import ModelConfig
from llm_bench.config import MongoConfig
from llm_bench.logging import log_metrics
from llm_bench.utils import check_and_clean_space
from llm_bench.utils import has_existing_run

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR = os.environ.get("LOG_DIR", "/var/log")
LOG_FILE_TXT = os.path.join(LOG_DIR, "benchmarks_local.log")
LOG_FILE_JSON = os.path.join(LOG_DIR, "benchmarks_local.json")
LOG_TO_MONGO = os.getenv("LOG_TO_MONGO", "False").lower() in ("true", "1", "t")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION_LOCAL = os.environ.get("MONGODB_COLLECTION_LOCAL")

CACHE_DIR = os.environ.get("HF_HUB_CACHE")
assert CACHE_DIR, "HF_HUB_CACHE environment variable not set"

logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_FILE_TXT), level=LOG_LEVEL)
logger = logging.getLogger(__name__)

DO_SAMPLE = False


app = Flask(__name__)


@app.route("/benchmark", methods=["POST"])
def call_huggingface() -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        model_name = request.form.get("model_name", type=str)
        framework = request.form.get("framework", type=str)
        query = request.form.get("query", default=None, type=str)
        quant_method = request.form.get("quant_method", default=None, type=str)
        quant_bits = request.form.get("quant_bits", default=None, type=str)
        max_tokens = request.form.get("max_tokens", default=256, type=int)
        temperature = request.form.get("temperature", default=0.1, type=float)

        run_always_str = request.form.get("run_always", "False").lower()
        run_always = run_always_str == "true"

        assert framework is not None, "framework is required"
        assert model_name is not None, "model_name is required"

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
        if LOG_TO_MONGO:
            mongo_config = MongoConfig(
                uri=MONGODB_URI,  # type: ignore
                db=MONGODB_DB,  # type: ignore
                collection=MONGODB_COLLECTION_LOCAL,  # type: ignore
            )
            existing_run = has_existing_run(model_name, model_config, mongo_config)
            if existing_run:
                if run_always:
                    logger.info(f"Model has been benchmarked before: {model_name}, quant: {quant_str}")
                    logger.info("Re-running benchmark anyway because run_always is True")
                else:
                    logger.info(f"Model has been benchmarked before: {model_name}, quant: {quant_str}")
                    return (
                        jsonify(
                            {
                                "status": "skipped",
                                "reason": "model has been benchmarked before",
                            }
                        ),
                        200,
                    )
            else:
                logger.info(f"Model has not been benchmarked before: {model_name}, quant: {quant_str}")

        # Check and clean disk space if needed
        check_and_clean_space(directory=CACHE_DIR, threshold=90.0)

        if framework == "transformers":
            from llm_bench.local.hf.transformers import generate
        elif framework == "hf-tgi":
            from llm_bench.local.hf.tgi import generate
        else:
            raise ValueError(f"Unknown framework: {framework}")

        # Main benchmarking function
        metrics = generate(model_config, run_config)
        assert metrics, "metrics is empty"

        # Log metrics
        log_metrics(
            model_type="local",
            config=model_config,
            metrics=metrics,
            file_path=os.path.join(LOG_DIR, LOG_FILE_JSON),
            log_to_mongo=LOG_TO_MONGO,
            mongo_uri=MONGODB_URI,
            mongo_db=MONGODB_DB,
            mongo_collection=MONGODB_COLLECTION_LOCAL,
        )

        # print metrics
        logger.info(f"===== Model: {model_name} =====")
        logger.info(f"Requested tokens: {run_config['max_tokens']}")
        logger.info(f"Output tokens: {metrics['output_tokens'][0]}")
        logger.info(f"GPU mem usage: {(metrics['gpu_mem_usage'][0] / 1024**3) :.2f}GB")
        logger.info(f"Generate time: {metrics['generate_time'][0]:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second'][0]:.2f}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception(f"Error in call_benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@click.command()
@click.option("--port", required=True, help="Port to run the server on")
def main(port):
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
