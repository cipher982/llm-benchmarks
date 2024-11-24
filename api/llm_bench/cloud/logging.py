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
    def __init__(self, logs_dir: str, redis_url: str):
        self.logs_dir = logs_dir
        self.full_logs_file = os.path.join(logs_dir, "run_history.log")
        self.current_status_file = os.path.join(logs_dir, "run_status.json")
        self.redis_url = redis_url

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

            updated_models = set()

            for status in model_status:
                try:
                    # Debug logging
                    self.log_info(f"Processing status: {json.dumps(status, default=str)}")

                    model = status["model"]
                    # Check if request exists in the status object
                    provider = status.get("request", {}).get("provider") or status.get("provider")
                    if not provider:
                        self.log_error(f"No provider found in status: {json.dumps(status, default=str)}")
                        continue

                    composite_key = f"{provider}:{model}"
                    updated_models.add(composite_key)

                    existing_data[composite_key].update(
                        {
                            "provider": provider,
                            "model": model,
                            "last_run_timestamp": status.get("timestamp", datetime.now().isoformat()),
                        }
                    )
                    existing_data[composite_key]["runs"].append(self.get_run_outcome(status))
                except KeyError as e:
                    self.log_error(f"KeyError processing status: {e}, Status: {json.dumps(status, default=str)}")
                    continue
                except Exception as e:
                    self.log_error(
                        f"Error processing individual status: {str(e)}, Status: {json.dumps(status, default=str)}"
                    )
                    continue

            # Only mark models as did-not-run if they have the same provider
            for key in list(existing_data.keys()):
                if key not in updated_models and ":" in key:  # Ensure we only process composite keys
                    existing_data[key]["runs"].append("did-not-run")

            with open(self.current_status_file, "w") as f:
                json.dump(existing_data, f, indent=2, cls=CustomJSONEncoder)

            with redis.Redis.from_url(self.redis_url) as redis_client:
                redis_client.set("cloud_log_status", json.dumps(existing_data))
                self.log_info("Successfully updated api status to redis")
        except (FileNotFoundError, PermissionError) as e:
            self.log_error(f"Error accessing benchmark status file: {str(e)}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.log_error(f"Error logging to Redis: {str(e)}")
        except json.JSONDecodeError as e:
            self.log_error(f"Error decoding JSON data: {str(e)}")
        except Exception as e:
            self.log_error(f"Unexpected error occurred: {str(e)}")
