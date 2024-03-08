import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()

# Constants
GB = 1024**3
LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
FULL_LOGS_FILE = os.path.join(LOGS_DIR, "cloud_history.log")
CURRENT_STATUS_FILE = os.path.join(LOGS_DIR, "cloud_status.json")

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a rotating file handler
file_handler = RotatingFileHandler(FULL_LOGS_FILE, maxBytes=GB, backupCount=4)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Create a stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log_info(message: str) -> None:
    logger.info(message)


def log_error(message: str) -> None:
    logger.error(message)


def log_benchmark_request(request: Any) -> None:
    log_info(f"Benchmark Request - Provider: {request.provider}, Model: {request.model}")


def get_run_outcome(status: dict[str, Any]) -> str:
    if status.get("status") == "success":
        return "success"
    else:
        return "error"


def log_benchmark_status(model_status: list[dict[str, Any]]) -> None:
    try:
        # Load existing data if the file exists
        if os.path.exists(CURRENT_STATUS_FILE):
            with open(CURRENT_STATUS_FILE) as f:
                existing_data = defaultdict(lambda: {"runs": []}, json.load(f))
        else:
            existing_data = defaultdict(lambda: {"runs": []})

        # Create a set of updated models using set comprehension
        updated_models = {status["model"] for status in model_status}

        # Update the status for each model/provider
        for status in model_status:
            model = status["model"]
            existing_data[model].update(
                {
                    "provider": status["request"]["provider"],
                    "model": model,
                    "last_run_timestamp": status["timestamp"],
                }
            )
            existing_data[model]["runs"].append(get_run_outcome(status))

        # Check for models that were not updated in the current round
        for model in existing_data:
            if model not in updated_models:
                existing_data[model]["runs"].append("did-not-run")

        # Save the updated data
        with open(CURRENT_STATUS_FILE, "w") as f:
            json.dump(existing_data, f, indent=2, cls=CustomJSONEncoder)

        # Update Redis with the latest status data
        with redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
        ) as redis_client:
            redis_client.set("cloud_log_status", json.dumps(existing_data))

    except (FileNotFoundError, PermissionError) as e:
        log_error(f"Error accessing benchmark status file: {str(e)}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        log_error(f"Error connecting to Redis: {str(e)}")
    except json.JSONDecodeError as e:
        log_error(f"Error decoding JSON data: {str(e)}")
    except Exception as e:
        log_error(f"Unexpected error occurred: {str(e)}")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
