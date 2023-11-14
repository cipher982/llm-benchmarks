import logging.config
import time
from typing import cast

import pynvml
import requests


def is_container_ready(timeout: int = 1800) -> bool:
    """Check if the Docker container is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8080")
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False


def get_vram_usage() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # second GPU
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return cast(int, info.used)


# Logger Configuration
def setup_logger():
    """Set up logging configuration."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "tgi - %(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": logging.INFO,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "./logs/llm_benchmarks.log",
                "formatter": "standard",
                "level": logging.DEBUG,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": logging.DEBUG,
        },
    }
    logging.config.dictConfig(logging_config)
