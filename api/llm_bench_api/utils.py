import logging.config
import os
import re
import shutil
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pynvml
import requests
from llm_bench_api.config import CloudConfig
from llm_bench_api.config import ModelConfig
from llm_bench_api.config import MongoConfig
from pymongo import MongoClient


logger = logging.getLogger(__name__)


def fetch_hf_models(fetch_hub: bool, cache_dir: str) -> list[str]:
    if fetch_hub:
        try:
            response = requests.get(
                "https://huggingface.co/api/models",
                params={
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": 1_000,
                    "filter": "text-generation",
                },
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models from HuggingFace Hub: {e}")
            return []

        # If the request was successful
        model_data = response.json()
        model_names = [entry["id"] for entry in model_data]
    else:
        try:
            model_names = get_cached_models(cache_dir)
        except Exception as e:
            print(f"Error fetching cached models: {e}")
            return []

    # Filter out gguf models (use these in llama-cpp instead)
    model_names = [name for name in model_names if "gguf" not in name.lower()]

    return model_names


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
    logger.info(f"Current disk usage: {used_space:.2f}% ({directory})")

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


def extract_param_count(model_id: str) -> Optional[Tuple[str, float]]:
    """
    Extract the parameter count from the model name.

    Returns a tuple of the model name and its parameter count in millions,
    or None if the pattern does not match.
    """
    # Special case for 'mixtral' models
    if "mixtral" in model_id.lower():
        # If it's a 'mixtral' model, set the numerical part to 56 billion
        numerical_part = 56.0
        unit = "B"
    else:
        # Use regex to extract the parameter size with a specific pattern
        match = re.search(r"(\d+)x(\d+\.\d+|\d+)([MmBb])", model_id)
        if not match:
            # If no multiplier pattern is found, try matching without multiplier
            match = re.search(r"(\d+\.\d+|\d+)([MmBb])", model_id)
            if not match:
                return None
            numerical_part = float(match.group(1))
            unit = match.group(2).upper()
        else:
            # If multiplier pattern is found, calculate the total size
            multiplier, size_str, unit = match.groups()
            numerical_part = float(size_str) * int(multiplier)
            unit = unit.upper()

    # Normalize parameter count to millions
    if unit == "B":
        numerical_part *= 1000  # Convert B to M

    return model_id, numerical_part


def filter_model_size(model_ids: List[str], max_size_million: int) -> List[str]:
    """
    Filter models based on parameter count.
    """
    valid_models: List[str] = []
    dropped_models: List[str] = []

    for model_id in model_ids:
        result = extract_param_count(model_id)
        if not result:
            dropped_models.append(model_id)
            continue

        # Unpack the model name and its parameter count
        _, param_count_million = result

        # Filter based on parameter count
        if param_count_million <= max_size_million:
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


def has_existing_run(model_name: str, model_config: Union[CloudConfig, ModelConfig], mongo_config: MongoConfig) -> bool:
    # Initialize MongoDB client and collection
    client = MongoClient(mongo_config.uri)
    db = client[mongo_config.db]
    collection = db[mongo_config.collection]

    # Check if model has been benchmarked before
    if isinstance(model_config, CloudConfig):
        existing_config = collection.find_one(
            {
                "provider": model_config.provider,
                "model_name": model_name,
            }
        )
    elif isinstance(model_config, ModelConfig):
        existing_config = collection.find_one(
            {
                "framework": model_config.framework,
                "model_name": model_name,
                "quantization_method": model_config.quantization_method,
                "quantization_bits": model_config.quantization_bits,
            }
        )
    else:
        raise Exception(f"Invalid model_config type: {type(model_config)}")

    if existing_config:
        logger.info("Model already benchmarked.")
        client.close()
        return True
    else:
        logger.info("Model not benchmarked.")
        client.close()
        return False
