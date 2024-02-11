"""Logging utilities for the LLM benchmarks."""
import logging.config
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Union

import pymongo
import pytz
from llm_bench_api.config import CloudConfig
from llm_bench_api.config import ModelConfig
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


def log_to_mongo(
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
                    "streaming": config.streaming,
                    "time_to_first_token": metrics["time_to_first_token"],
                    "times_between_tokens": metrics["times_between_tokens"],
                }
            )

        insert_into_benchmark_metrics(data, collection)
        logger.info(f"Successfully logged metrics to MongoDB for model {config.model_name}")
    except Exception as e:
        logger.exception(f"Error in log_to_mongo: {e}")


def setup_database(uri: str, db_name: str, collection_name: str) -> Collection:
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def insert_into_benchmark_metrics(data: dict, collection: Collection) -> None:
    collection.insert_one(data)
