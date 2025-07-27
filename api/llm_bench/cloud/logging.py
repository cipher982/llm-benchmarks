import json
import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class Logger:
    def __init__(self, logs_dir: str, max_runs: int = 10):
        self.logs_dir = logs_dir
        self.max_runs = max_runs  # Number of runs to keep in history

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

    def get_run_outcome(self, status: Dict[str, Any]) -> bool:
        return status.get("status") == "success"

    def log_benchmark_status(self, model_status: List[Dict[str, Any]]) -> None:
        """Log benchmark status. Redis has been removed - this now just logs to console."""
        try:
            # Track which providers we're updating in this run
            current_providers = {status.get("provider") for status in model_status if status.get("provider")}
            self.log_info(f"Updating status for providers: {current_providers}")

            # Process and log results
            for status in model_status:
                try:
                    model = status["model"]
                    provider = status.get("provider")
                    if not provider:
                        self.log_error(f"No provider found in status: {json.dumps(status, default=str)}")
                        continue

                    outcome = self.get_run_outcome(status)
                    outcome_str = "SUCCESS" if outcome else "FAILED"
                    self.log_info(f"Benchmark {outcome_str}: {provider}:{model}")

                except KeyError as e:
                    self.log_error(f"KeyError processing status: {e}, Status: {json.dumps(status, default=str)}")
                    continue
                except Exception as e:
                    self.log_error(
                        f"Error processing individual status: {str(e)}, Status: {json.dumps(status, default=str)}"
                    )
                    continue

            self.log_info("Successfully logged benchmark status")

        except Exception as e:
            self.log_error(f"Unexpected error occurred: {str(e)}")
