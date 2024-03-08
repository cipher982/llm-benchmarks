import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Constants
GB = 1024**3
FULL_LOGS_FILE = "./logs/cloud_history.log"
CURRENT_STATUS_FILE = "./logs/cloud_status.json"


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


def log_info(message):
    logger.info(message)


def log_error(message):
    logger.error(message)


def log_benchmark_request(request):
    log_info(f"Benchmark Request - Provider: {request.provider}, Model: {request.model}")


def log_benchmark_status(model_status):
    # Save model_status to a file for persistence
    try:
        with open(CURRENT_STATUS_FILE, "w") as f:
            json.dump(model_status, f, indent=2, cls=CustomJSONEncoder)
    except OSError as e:
        log_error(f"Error saving benchmark status: {str(e)}")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
