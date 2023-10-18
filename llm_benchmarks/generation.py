"""Module for benchmarking llm infernce speeds and external logging."""
import gc
import logging
import os
from datetime import datetime
from time import time

import pymongo
import torch
from pymongo.collection import Collection

from llm_benchmarks.config import ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/hf/"

logger = logging.getLogger(__name__)


MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")


def generate_and_log(
    config,
    custom_token_counts: list = [],
    llama: bool = False,
    db_name: str = "llm_benchmarks",
) -> None:
    metrics = generate(config, custom_token_counts, llama)
    log_to_mongo(config, metrics, db_name)


def generate(
    config: ModelConfig,
    custom_token_counts: list = [],
    llama: bool = False,
) -> dict:
    if llama:
        from transformers import LlamaForCausalLM as Model  # type: ignore
        from transformers import LlamaTokenizer as Tokenizer  # type: ignore
    else:
        from transformers import AutoModelForCausalLM as Model  # type: ignore
        from transformers import AutoTokenizer as Tokenizer  # type: ignore

    # Load Model
    model = Model.from_pretrained(
        config.model_name,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Tokenize inputs
    tokenizer = Tokenizer.from_pretrained(config.model_name)
    text = "Hi: "
    input_tokens = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

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
    for ix, token_count in enumerate(requested_tokens):
        time0 = time()
        output = model.generate(
            input_tokens,
            do_sample=True,
            temperature=config.temperature,
            min_length=token_count,
            max_length=token_count,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        time1 = time()

        # Collect metrics
        output_tokens = len(output.cpu().numpy().tolist()[0])
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

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def log_to_mongo(config: ModelConfig, metrics: dict, db_name: str) -> None:
    collection = setup_database(db_name)

    data = {
        "run_ts": config.run_ts,
        "model_name": config.model_name,
        "quantization_bits": config.quantization_bits,
        "torch_dtype": config.torch_dtype,
        "temperature": config.temperature,
        "gen_ts": metrics["gen_ts"],
        "requested_tokens": metrics["requested_tokens"],
        "output_tokens": metrics["output_tokens"],
        "gpu_mem_usage": metrics["gpu_mem_usage"],
        "generate_time": metrics["generate_time"],
        "tokens_per_second": metrics["tokens_per_second"],
    }
    insert_into_benchmark_metrics(data, collection)


def setup_database(db_name: str) -> Collection:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[db_name]
    collection = db["benchmark_metrics"]
    return collection


def insert_into_benchmark_metrics(data: dict, collection: Collection) -> None:
    collection.insert_one(data)
