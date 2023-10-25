"""Module for benchmarking llm infernce speeds and external logging."""
import gc
import logging.config
from datetime import datetime
from time import time
from typing import Dict
from typing import List
from typing import Optional

import pymongo
import torch
from pymongo.collection import Collection

from llm_benchmarks.config import ModelConfig


logger = logging.getLogger(__name__)


T_S_CUTOFF = 5  # Tokens per second cutoff (to prevent long runs)


def generate_and_log(
    config,
    uri: str,
    db_name: str,
    collection_name: str,
    custom_token_counts: list = [],
    llama: bool = False,
) -> None:
    """Main entry point. Generates and logs the data."""
    logger.info(f"Beginning benchmarking for model {config.model_name}")
    try:
        metrics = generate(config, custom_token_counts, llama)
        log_to_mongo(config, metrics, uri, db_name, collection_name)
    except Exception as e:
        logger.exception(f"Error in generate_and_log: {e}")


def generate(
    config: ModelConfig,
    custom_token_counts: Optional[List[int]] = None,
    llama: bool = False,
) -> dict:
    """Generates the data based on the provided config."""

    logger.info(f"Beginning generations for model {config.model_name}")

    if llama:
        from transformers import LlamaForCausalLM as Model  # type: ignore
    else:
        from transformers import AutoModelForCausalLM as Model  # type: ignore

    # Load Model
    logger.info(f"Loading pretrained model {config.model_name}")
    model = Model.from_pretrained(
        config.model_name,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    input_tokens = torch.tensor([[0, 1, 2]]).to("cuda")

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": [],
        "output_tokens": [],
        "gpu_mem_usage": [],
        "generate_time": [],
        "tokens_per_second": [],
    }

    if custom_token_counts:
        requested_tokens = custom_token_counts
    else:
        requested_tokens = [64, 128, 256, 512, 1024]

    # Generate samples
    should_continue = True
    for ix, token_count in enumerate(requested_tokens):
        if ix == 1 and not should_continue:
            break
        logger.info(f"Generating sample for token count {token_count}")
        time0 = time()
        output = None
        try:
            output = model.generate(
                input_tokens,
                do_sample=True,
                temperature=config.temperature,
                min_length=token_count,
                max_length=token_count,
            )
        except Exception as e:
            logger.exception(f"Error in generating sample for token count {token_count}: {e}")

        time1 = time()

        # Collect metrics
        output_tokens = len(output.cpu().numpy().tolist()[0]) if output is not None and output.numel() > 0 else 0
        gpu_mem_usage = torch.cuda.memory_allocated()
        generate_time = time1 - time0
        tokens_per_second = output_tokens / generate_time

        # Append metrics to the metrics dict
        metrics["requested_tokens"].append(token_count)
        metrics["output_tokens"].append(output_tokens)
        metrics["gpu_mem_usage"].append(gpu_mem_usage)
        metrics["generate_time"].append(generate_time)
        metrics["tokens_per_second"].append(tokens_per_second)

        # logger.info metrics
        logger.info(f"===== Model: {config.model_name} Run: {ix+1}/{len(requested_tokens)} =====")
        logger.info(f"Requested tokens: {token_count}")
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"GPU mem usage: {(gpu_mem_usage / 1024**3) :.2f}GB")
        logger.info(f"Generate time: {generate_time:.2f} s")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        if ix == 0 and tokens_per_second < T_S_CUTOFF:
            logger.info(f"Tokens per second is below {T_S_CUTOFF}. Limiting to the first two iterations.")
            should_continue = False

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def log_to_mongo(
    config: ModelConfig,
    metrics: Dict[str, List[float]],
    uri: str,
    db_name: str,
    collection_name: str,
) -> None:
    """Logs the metrics to MongoDB."""

    logger.info(f"Logging metrics to MongoDB for model {config.model_name}")
    try:
        collection = setup_database(uri, db_name, collection_name)

        data = {
            "run_ts": config.run_ts,
            "model_name": config.model_name,
            "quantization_bits": config.quantization_bits,
            "torch_dtype": str(config.torch_dtype),
            "temperature": config.temperature,
            "gen_ts": metrics["gen_ts"],
            "requested_tokens": metrics["requested_tokens"],
            "output_tokens": metrics["output_tokens"],
            "gpu_mem_usage": metrics["gpu_mem_usage"],
            "generate_time": metrics["generate_time"],
            "tokens_per_second": metrics["tokens_per_second"],
        }
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
