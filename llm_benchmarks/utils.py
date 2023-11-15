import logging.config
import os
import re
import shutil
from typing import cast
from typing import List

import pynvml


logger = logging.getLogger(__name__)


def get_used_space_percent(directory: str) -> float:
    """Get the used space percentage of the file system containing the directory."""
    stat = os.statvfs(directory)
    return ((stat.f_blocks - stat.f_bfree) / stat.f_blocks) * 100


def get_model_directories(directory: str) -> List[str]:
    """Get a list of directories in the given directory, filtered by those starting with 'models--'."""
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.startswith("models--")
    ]


def get_oldest_directory(directories: List[str]) -> str:
    """Find the oldest directory in the given list."""
    oldest_directory = min(directories, key=lambda d: os.path.getmtime(d))
    return oldest_directory


def check_and_clean_space(directory: str, threshold: float = 90.0):
    # Check disk usage
    used_space = get_used_space_percent(directory)
    logger.info(f"Current disk usage: {used_space:.2f}%")

    while used_space > threshold:
        # Get model directories
        model_dirs = get_model_directories(directory)

        # If there are no model directories, exit the loop
        if not model_dirs:
            logger.info("No model directories to remove.")
            break

        # Find the oldest directory
        oldest_dir = get_oldest_directory(model_dirs)

        # Remove the oldest directory
        logger.info(f"Removing: {oldest_dir}")
        shutil.rmtree(oldest_dir)

        # Recheck disk usage
        used_space = get_used_space_percent(directory)
        logger.info(f"Updated disk usage: {used_space:.2f}%")


def get_cached_models(directory: str) -> list[str]:
    """
    Get a list of cached HF models in the given directory.
    """
    print(f"Getting cached models from directory: {directory}")
    files = os.listdir(directory)
    model_files = [f for f in files if f.startswith("models--")]
    formatted_names = [f.removeprefix("models--").replace("--", "/") for f in model_files]
    print(f"Returning {len(formatted_names):,} model names")
    return formatted_names


def filter_model_size(model_ids: List[str], max_size_million: int) -> List[str]:
    """
    Filter models based on parameter count.
    """
    valid_models: List[str] = []
    dropped_models: List[str] = []

    for model_id in model_ids:
        # Use regex to extract the parameter size
        match = re.search(r"([0-9.]+[MmBb])", model_id)
        param_count = match.group(1) if match else None

        if not param_count:
            dropped_models.append(model_id)
            continue

        # Normalize parameter count to millions
        unit = param_count[-1].upper()
        numerical_part = float(param_count[:-1])
        if unit == "B":
            numerical_part *= 1000  # Convert B to M

        # Filter based on parameter count
        if numerical_part <= max_size_million:
            valid_models.append(model_id)
        else:
            dropped_models.append(model_id)

    print(f"Keeping models: {', '.join(valid_models)}\n\n")
    print(f"Dropping models: {', '.join(dropped_models)}\n\n")
    print(f"Model count after filtering {len(model_ids):,} -> {len(valid_models):,}")

    return valid_models


def get_vram_usage(gpu_device: int) -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
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
