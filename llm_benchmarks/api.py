import logging
from datetime import datetime

from flask import Flask
from flask import jsonify
from flask.wrappers import Response

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.generation import generate_and_log


logging.basicConfig(filename="/var/log/llm_benchmarks.log", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/benchmark/<model_name>", methods=["POST"])
def call_benchmark(model_name: str) -> Response:
    """Enables the use a POST request to call the benchmarking function."""

    logger.info(f"Received request for model {model_name}")

    # Declare config defaults
    config = ModelConfig(
        model_name=model_name,
        # quantization_bits="8bit",
        quantization_bits=None,
        # torch_dtype="float16",
        torch_dtype="auto",
        temperature=0.1,
        run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Your benchmarking logic here
    result = generate_and_log(config)
    return jsonify(result)


app.run(host="0.0.0.0", port=5000)
