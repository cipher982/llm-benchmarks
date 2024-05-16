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
from llm_bench_api.logging import log_metrics
from llm_bench_api.utils import has_existing_run

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR = os.environ.get("LOG_DIR", "/var/log")
LOG_FILE_TXT = os.path.join(LOG_DIR, "benchmarks_local.log")
LOG_FILE_JSON = os.path.join(LOG_DIR, "benchmarks_local.json")
LOG_TO_MONGO = os.getenv("LOG_TO_MONGO", "False").lower() in ("true", "1", "t")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION_LOCAL = os.environ.get("MONGODB_COLLECTION_LOCAL")
FLASK_PORT = 5003

logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_FILE_TXT), level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/benchmark", methods=["POST"])
def benchmark_gguf() -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        # Load config from request
        framework = "gguf"
        model_name = request.form.get("model_name")
        model_path = f"/models/gguf/{model_name}"
        query = request.form.get("query", "User: Complain that I did not send a request.\nAI:")
        max_tokens = int(request.form.get("max_tokens", 512))
        temperature = request.form.get("temperature", default=0.1, type=float)
        quant_method = request.form.get("quant_method", type=str)
        quant_type = request.form.get("quant_type", type=str)
        quant_bits = request.form.get("quant_bits", type=str)
        n_gpu_layers = int(request.form.get("n_gpu_layers", 0))
        run_always_str = request.form.get("run_always", "False").lower()
        run_always = run_always_str == "true"
        log_level = request.form.get("log_level", "INFO")
        logger.setLevel(log_level.upper())

        assert model_name, "model_name not set"

        quant_str = quant_type if quant_method is not None else "none"
        logger.info(f"Received request for model: {model_name}, quant: {quant_str}")

        # Create model config
        model_config = ModelConfig(
            framework=framework,
            model_name=model_name,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_dtype="half_float::half",
            quantization_method=quant_method,
            quantization_bits=quant_bits,
            temperature=temperature,
            misc={"gguf_quant_type": quant_type},
        )

        run_config = {
            "query": query,
            "max_tokens": max_tokens,
        }
        logger.info(f"Run config: {run_config}")

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
                    return jsonify({"status": "skipped", "reason": "model has been benchmarked before"}), 200
            else:
                logger.info(f"Model has not been benchmarked before: {model_name}, quant: {quant_str}")

        # Main benchmarking function
        logger.info(f"Loading llama-cpp model from path: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )

        # Get GPU memory usage
        time.sleep(1)
        pynvml.nvmlInit()
        gpu_device = int(os.environ.get("GPU_DEVICE", 0))
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        # Run benchmark
        time_0 = time.time()
        output = llm(
            run_config["query"],
            echo=True,
            max_tokens=run_config["max_tokens"],
            temperature=temperature,
        )
        time_1 = time.time()

        # # Build config object
        # model_quantization_list = [
        #     ("q4_0", "4bit"),
        #     ("q8_0", "8bit"),
        #     ("f16", None),
        # ]
        # quantization_bits = next(
        #     (bits for key, bits in model_quantization_list if key in model_name),
        #     "unknown",
        # )

        # Build metrics object
        output_tokens = output["usage"]["completion_tokens"]  # type: ignore

        metrics = {
            "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested_tokens": [max_tokens],
            "output_tokens": [output_tokens],
            "gpu_mem_usage": [info.used],
            "generate_time": [time_1 - time_0],
            "tokens_per_second": [output_tokens / (time_1 - time_0) if time_1 > time_0 else 0],
        }

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

        logger.info(f"===== Model: {model_name} =====")
        logger.info(f"Requested tokens: {max_tokens}")
        logger.info(f"Output tokens: {metrics['output_tokens'][0]}")
        logger.info(f"GPU mem usage: {(metrics['gpu_mem_usage'][0] / 1024**3) :.2f}GB")
        logger.info(f"Generate time: {metrics['generate_time'][0]:.2f} s")
        logger.info(f"Tokens per second: {metrics['tokens_per_second'][0]:.2f}")
        logger.debug(f"Full output: {output}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception(f"Error in call_benchmark: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


app.run(host="0.0.0.0", port=FLASK_PORT)
