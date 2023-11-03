import json
import logging
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

# from pymongo import MongoClient

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MONGODB_URI = os.environ.get("MONGODB_URI")
# MONGODB_DB = os.environ.get("MONGODB_DB")
# MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")

# assert MONGODB_URI, "MONGODB_URI environment variable not set"
# assert MONGODB_DB, "MONGODB_DB environment variable not set"
# assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"


@app.route("/benchmark/<path:model_name>", methods=["POST"])
def benchmark_cpp(model_name: str) -> Union[Response, Tuple[Response, int]]:
    """Enables the use a POST request to call the benchmarking function."""
    try:
        model_path = f"/models/{model_name}"
        query = request.form.get("query", "Tell me a short story.")
        max_tokens = int(request.form.get("max_tokens", 32))

        logger.info(f"Received request for model {model_name}")

        # Initialize MongoDB client and collection
        # client = MongoClient(MONGODB_URI)
        # db = client[MONGODB_DB]
        # collection = db[MONGODB_COLLECTION]

        # Main benchmarking function
        llm = Llama(model_path=model_path)
        output = llm(query, max_tokens=max_tokens, echo=True)

        # Build config object
        config = {
            "model_name": model_name,
            "query": query,
            "max_tokens": max_tokens,
            "run_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        sleep(1)
        logger.info("=" * 40)
        logger.info("Benchmarking complete.")
        logger.info(f"Model: {model_name}")
        logger.info(f"Config:\n{json.dumps(config, indent=4)}")
        logger.info(f"Output:\n{json.dumps(output, indent=4)}")
        logger.info("=" * 40)

        # Log metrics to MongoDB
        # run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # result_data = {
        #     "model_name": model_name,
        #     "query": query,
        #     "max_tokens": max_tokens,
        #     "run_ts": run_ts,
        #     "metrics": metrics,
        # }
        # result = collection.insert_one(result_data)

        return jsonify({"status": "success", "output": output}), 200
    except Exception as e:
        logger.exception(f"Error in benchmark_cpp: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


app.run(host="0.0.0.0", port=5001)
