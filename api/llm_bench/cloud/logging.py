import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any
from typing import Dict
from typing import List

import redis


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class Logger:
    def __init__(self, logs_dir: str, redis_host: str, redis_port: int, redis_db: int, redis_password: str):
        self.logs_dir = logs_dir
        self.full_logs_file = os.path.join(logs_dir, "run_history.log")
        self.current_status_file = os.path.join(logs_dir, "run_status.json")
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = RotatingFileHandler(self.full_logs_file, maxBytes=1024**3, backupCount=4)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def log_info(self, message: str) -> None:
        self.logger.info(message)

    def log_error(self, message: str) -> None:
        self.logger.error(message)

    def log_benchmark_request(self, request: Any) -> None:
        self.log_info(f"Benchmark Request - Provider: {request.provider}, Model: {request.model}")

    def get_run_outcome(self, status: Dict[str, Any]) -> str:
        return "success" if status.get("status") == "success" else "error"

    def log_benchmark_status(self, model_status: List[Dict[str, Any]]) -> None:
        try:
            if os.path.exists(self.current_status_file):
                with open(self.current_status_file) as f:
                    existing_data = defaultdict(lambda: {"runs": []}, json.load(f))
            else:
                existing_data = defaultdict(lambda: {"runs": []})

            updated_models = {f"{status['request']['provider']}:{status['model']}" for status in model_status}

            for status in model_status:
                model = status["model"]
                provider = status["request"]["provider"]
                composite_key = f"{provider}:{model}"

                existing_data[composite_key].update(
                    {
                        "provider": provider,
                        "model": model,
                        "last_run_timestamp": status["timestamp"],
                    }
                )
                existing_data[composite_key]["runs"].append(self.get_run_outcome(status))

            # Only mark models as did-not-run if they have the same provider
            for key in existing_data:
                if key not in updated_models and ":" in key:  # Ensure we only process composite keys
                    existing_data[key]["runs"].append("did-not-run")

            with open(self.current_status_file, "w") as f:
                json.dump(existing_data, f, indent=2, cls=CustomJSONEncoder)

            with redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
            ) as redis_client:
                redis_client.set("cloud_log_status", json.dumps(existing_data))
                self.log_info(f"Successfully updated api status to redis on {self.redis_host}")
        except (FileNotFoundError, PermissionError) as e:
            self.log_error(f"Error accessing benchmark status file: {str(e)}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.log_error(
                f"Error logging to Redis: {self.redis_host}:{self.redis_port}, DB: {self.redis_db}. Error: {str(e)}"
            )
        except json.JSONDecodeError as e:
            self.log_error(f"Error decoding JSON data: {str(e)}")
        except Exception as e:
            self.log_error(f"Unexpected error occurred: {str(e)}")
