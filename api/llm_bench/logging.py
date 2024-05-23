"""Logging utilities for the LLM benchmarks."""

import json
import logging.config
import os
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pymongo
import pytz
from filelock import FileLock
from pymongo.collection import Collection

from llm_bench.config import CloudConfig
from llm_bench.config import ModelConfig

logger = logging.getLogger(__name__)


def log_metrics(
    model_type: str,
    config: Union[ModelConfig, CloudConfig],
    metrics: Dict[str, Any],
    file_path: str,
    log_to_mongo: bool,
    mongo_uri: Optional[str] = None,
    mongo_db: Optional[str] = None,
    mongo_collection: Optional[str] = None,
) -> None:
    """Logs metrics to a JSON file and optionally to MongoDB."""
    log_json(model_type, config, metrics, file_path)

    if log_to_mongo:
        assert mongo_uri, "mongo_uri not provided"
        assert mongo_db, "mongo_db not provided"
        assert mongo_collection, "mongo_collection not provided"
        log_mongo(
            model_type=model_type,
            config=config,
            metrics=metrics,
            uri=mongo_uri,
            db_name=mongo_db,
            collection_name=mongo_collection,
        )


def setup_database(uri: str, db_name: str, collection_name: str) -> Collection:
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def insert_into_benchmark_metrics(data: dict, collection: Collection) -> None:
    collection.insert_one(data)


def log_json(model_type: str, config: Union[ModelConfig, CloudConfig], metrics: Dict[str, Any], file_path: str) -> None:
    """Logs the metrics to a JSON file for a model run."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "requested_tokens": metrics["requested_tokens"],
        "output_tokens": metrics["output_tokens"],
        "generate_time": metrics["generate_time"],
        "tokens_per_second": metrics["tokens_per_second"],
        "misc": config.misc,
    }

    if model_type == "local":
        assert isinstance(config, ModelConfig)
        log_entry.update(
            {
                "framework": config.framework,
                "quantization_method": config.quantization_method,
                "quantization_bits": config.quantization_bits,
                "model_dtype": config.model_dtype,
                "gpu_mem_usage": metrics["gpu_mem_usage"],
            }
        )
    elif model_type == "cloud":
        assert isinstance(config, CloudConfig)
        log_entry.update(
            {
                "provider": config.provider,
                "time_to_first_token": metrics["time_to_first_token"],
                "times_between_tokens": metrics["times_between_tokens"],
            }
        )

    lock_path = f"{file_path}.lock"
    with FileLock(lock_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                try:
                    logs = json.load(file)
                except json.JSONDecodeError:
                    logger.warning("Corrupted JSON detected. Attempting to fix.")
                    logs = fix_corrupted_json(file_path)
        else:
            logs = []

        logs.append(log_entry)

        with open(file_path, "w") as file:
            json.dump(logs, file, indent=4)

        logger.info(f"Logged to file: {file_path}")


def fix_corrupted_json(file_path: str) -> list:
    """Attempts to fix a corrupted JSON file by removing invalid entries."""
    with open(file_path, "r") as file:
        content = file.read()

    # Find the last valid JSON array closing bracket
    last_valid_index = content.rfind("}]")

    if last_valid_index != -1:
        # Calculate the number of lines removed
        original_lines = content.splitlines()
        fixed_content = content[: last_valid_index + 2]
        fixed_lines = fixed_content.splitlines()
        lines_removed = len(original_lines) - len(fixed_lines)

        with open(file_path, "w") as file:
            file.write(fixed_content)

        logger.info(f"Fixed corrupted JSON. Removed {lines_removed} lines.")
        return json.loads(fixed_content)
    else:
        logger.error("Could not find a valid JSON array closing bracket. No changes made.")
        return []


def log_mongo(
    model_type: str,
    config: Union[ModelConfig, CloudConfig],
    metrics: Dict[str, Any],
    uri: str,
    db_name: str,
    collection_name: str,
) -> None:
    """Logs the metrics to MongoDB for a model run."""
    assert model_type in ["local", "cloud"], f"Invalid model_type: {model_type}"

    logger.info(f"Logging metrics to MongoDB for {model_type} model {config.model_name}")
    try:
        collection = setup_database(uri, db_name, collection_name)

        # Settimestamps correctly
        run_ts_utc = datetime.strptime(config.run_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
        gen_ts_utc = datetime.strptime(metrics["gen_ts"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)

        data = {
            "run_ts": run_ts_utc,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "gen_ts": gen_ts_utc,
            "requested_tokens": metrics["requested_tokens"],
            "output_tokens": metrics["output_tokens"],
            "generate_time": metrics["generate_time"],
            "tokens_per_second": metrics["tokens_per_second"],
            "misc": config.misc,
        }

        if model_type == "local":
            assert isinstance(config, ModelConfig)
            data.update(
                {
                    "framework": config.framework,
                    "quantization_method": config.quantization_method,
                    "quantization_bits": config.quantization_bits,
                    "model_dtype": config.model_dtype,
                    "gpu_mem_usage": metrics["gpu_mem_usage"],
                }
            )
        elif model_type == "cloud":
            assert isinstance(config, CloudConfig)
            data.update(
                {
                    "provider": config.provider,
                    "time_to_first_token": metrics["time_to_first_token"],
                    "times_between_tokens": metrics["times_between_tokens"],
                }
            )

        insert_into_benchmark_metrics(data, collection)
        sanitized_uri = uri.split("@")[-1]  # Remove credentials part
        logger.info(f"Logged: {config.model_name} | {sanitized_uri} | {db_name} | {collection_name}")
    except Exception as e:
        logger.exception(f"Error in log_to_mongo: {e}")
