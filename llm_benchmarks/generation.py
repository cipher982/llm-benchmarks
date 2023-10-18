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


def setup_database(db_name: str) -> Collection:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[db_name]
    collection = db["benchmark_metrics"]
    return collection


def insert_into_benchmark_metrics(data: dict, collection: Collection) -> None:
    collection.insert_one(data)


# Functionality deprecated by pip
def get_installed_packages() -> str:
    # packages = {package.project_name: package.version for package in pip.get_installed_distributions()}
    # return json.dumps(packages)
    return ""


def generate(
    model_name: str,
    config: ModelConfig,
    custom_token_counts: list = [],
    llama: bool = False,
    db_name: str = "benchmark_metrics",
) -> dict:
    if llama:
        from transformers import LlamaForCausalLM as Model  # type: ignore
        from transformers import LlamaTokenizer as Tokenizer  # type: ignore
    else:
        from transformers import AutoModelForCausalLM as Model  # type: ignore
        from transformers import AutoTokenizer as Tokenizer  # type: ignore

    # Load Model
    model = Model.from_pretrained(
        model_name,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    # model.eval()  # type: ignore
    # model = torch.compile(model)  # type: ignore

    # Tokenize inputs
    tokenizer = Tokenizer.from_pretrained(model_name)
    text = "Hi: "
    input_tokens = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    metrics = {
        "output_tokens": [],
        "gpu_mem_usage": [],
        "generate_time": [],
        "tokens_per_second": [],
    }

    if custom_token_counts:
        token_counts = custom_token_counts
    else:
        token_counts = [32, 64, 128, 256, 512, 1024]

    # MongoDB
    collection = setup_database(db_name)

    # Get installed packages as JSON string
    lib_versions = get_installed_packages()

    # Generate samples
    for i, token_count in enumerate(token_counts):
        time0 = time()
        # with torch.no_grad():
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
        gpu_mem_usage = torch.cuda.memory_allocated() / 1024**3
        generate_time = time1 - time0
        tokens_per_second = output_tokens / generate_time

        # MongoDB metrics
        data = {
            "run_ts": config.run_ts,
            "model_name": model_name,
            "quantization_bits": config.quantization_bits,
            "torch_dtype": config.torch_dtype,
            "temperature": config.temperature,
            "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "lib_versions": lib_versions,
            "token_count": token_count,
            "output_tokens": output_tokens,
            "gpu_mem_usage": gpu_mem_usage,
            "generate_time": generate_time,
            "tokens_per_second": tokens_per_second,
        }
        insert_into_benchmark_metrics(data, collection)

        # Append metrics to the metrics dict
        metrics["output_tokens"].append(output_tokens)
        metrics["gpu_mem_usage"].append(gpu_mem_usage)
        metrics["generate_time"].append(generate_time)
        metrics["tokens_per_second"].append(tokens_per_second)

        # logger.info metrics
        logger.info(f"===== Model: {model_name} Run: {i+1}/{len(token_counts)} =====")
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"GPU mem usage: {gpu_mem_usage:.2f} GB")
        logger.info(f"Generate time: {generate_time:.2f} s")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics
