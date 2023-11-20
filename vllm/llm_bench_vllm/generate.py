"""LLM generation and benchmarking for vLLM library."""
import gc
import logging.config
import os
from datetime import datetime
from time import time

import torch
from llm_bench_api.config import ModelConfig
from llm_bench_api.utils import get_vram_usage

from vllm import LLM
from vllm import SamplingParams

logger = logging.getLogger(__name__)

GPU_DEVICE = os.environ.get("GPU_DEVICE_VLLM")
assert GPU_DEVICE, "GPU_DEVICE_VLLM environment variable not set"


def generate(config: ModelConfig, run_config: dict) -> dict:
    """Run vLLM inference and return metrics."""

    quant_str = f"{config.quantization_method}_{config.quantization_bits}" or "none"
    logger.info(f"Running benchmark: {config.model_name}, quant: {quant_str}")

    # Load model
    model = LLM(model=config.model_name)

    # Set params
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95)

    # Generate tokens
    time_0 = time()
    try:
        output = model.generate(run_config["query"], sampling_params)
    except Exception as e:
        logger.error(f"Error generating tokens: {e}")
        output = None
    time_1 = time()

    print("=====================================")
    print(output)

    print(type(output))

    print("=====================================")

    # Collect metrics
    # output_tokens = len(output.cpu().numpy().tolist()[0]) if output is not None and output.numel() > 0 else 0
    output_tokens = 0
    vram_usage = get_vram_usage(int(GPU_DEVICE))

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": [run_config["max_tokens"]],
        "output_tokens": [output_tokens],
        "gpu_mem_usage": [vram_usage],
        "generate_time": [time_1 - time_0],
        "tokens_per_second": [output_tokens / (time_1 - time_0) if time_1 > time_0 else 0],
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {}
    return metrics
