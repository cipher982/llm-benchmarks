"""LLM generation and benchmarking for HuggingFace Transformers library."""
import gc
import logging.config
import os
from datetime import datetime
from time import time

import torch
from transformers import AutoModelForCausalLM  # type: ignore

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.utils import get_vram_usage


logger = logging.getLogger(__name__)

GPU_DEVICE = os.environ.get("GPU_DEVICE_TF")
assert GPU_DEVICE, "GPU_DEVICE_TRANSFORMERS environment variable not set"


def generate(
    config: ModelConfig,
    run_config: dict,
) -> dict:
    """Run Transformers inference and return metrics."""

    quant_str = f"{config.quantization_method}_{config.quantization_bits}" or "none"
    logger.info(f"Running benchmark: {config.model_name}, quant: {quant_str}")

    if config.quantization_method == "gptq":
        load_in_4bit = False
        load_in_8bit = False
    elif config.quantization_method == "bitsandbytes":
        load_in_4bit = config.load_in_4bit
        load_in_8bit = config.load_in_8bit
    else:
        load_in_4bit = False
        load_in_8bit = False

    # Load model
    logger.info(f"Loading pretrained model: {config.model_name}, quant: {quant_str}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16 if config.model_dtype == "torch.float16" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Generate samples
    time0 = time()
    output = None
    try:
        output = model.generate(
            torch.tensor([[0, 1, 2]]).to("cuda"),
            do_sample=config.misc.get("do_sample"),
            temperature=config.temperature if config.misc.get("do_sample") else None,
            min_length=run_config["max_tokens"],
            max_length=run_config["max_tokens"],
        )
    except Exception as e:
        logger.error(f"Error generating sample: {e}")
        raise e
    time1 = time()

    # Collect metrics
    output_tokens = len(output.cpu().numpy().tolist()[0]) if output is not None and output.numel() > 0 else 0
    vram_usage = get_vram_usage(int(GPU_DEVICE))

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": [run_config["max_tokens"]],
        "output_tokens": [output_tokens],
        "gpu_mem_usage": [vram_usage],
        "generate_time": [time1 - time0],
        "tokens_per_second": [output_tokens / (time1 - time0) if time1 > time0 else 0],
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics
