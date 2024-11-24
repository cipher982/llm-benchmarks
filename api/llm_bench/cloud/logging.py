import json
import logging
from collections import defaultdict
from datetime import datetime
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
        if not redis_url:
            raise ValueError("redis_url must be provided")
        if not redis_url.startswith("redis://"):
            raise ValueError("redis_url must start with 'redis://'")

        self.redis_url = redis_url

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

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
            with redis.Redis.from_url(self.redis_url) as redis_client:
                # Get current data from Redis
                current_data = redis_client.get("cloud_log_status")
                existing_data = defaultdict(lambda: {"runs": []}, json.loads(current_data) if current_data else {})

                # Track which providers we're updating in this run
                current_providers = {status.get("provider") for status in model_status if status.get("provider")}
                self.log_info(f"Updating status for providers: {current_providers}")

                # Remove existing entries for providers we're updating
                existing_keys = list(existing_data.keys())
                for key in existing_keys:
                    if ":" in key:
                        provider = key.split(":")[0]
                        if provider in current_providers:
                            del existing_data[key]

                # Process new results
                for status in model_status:
                    try:
                        self.log_info(f"Processing status: {json.dumps(status, default=str)}")

                        model = status["model"]
                        provider = status.get("provider")
                        if not provider:
                            self.log_error(f"No provider found in status: {json.dumps(status, default=str)}")
                            continue

                        composite_key = f"{provider}:{model}"
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

                # Update Redis with new data
                redis_client.set("cloud_log_status", json.dumps(existing_data, cls=CustomJSONEncoder))
                self.log_info("Successfully updated api status to redis")

        except redis.ConnectionError as e:
            self.log_error(f"Redis connection error: {str(e)}")
        except redis.RedisError as e:
            self.log_error(f"Redis error: {str(e)}")
        except json.JSONDecodeError as e:
            self.log_error(f"Error decoding Redis data: {str(e)}")
        except Exception as e:
            self.log_error(f"Unexpected error occurred: {str(e)}")
