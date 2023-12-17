"""LLM generation and benchmarking for vLLM library."""
import gc
import logging.config
import os
import time
from datetime import datetime

import torch
from llm_bench_api.config import ModelConfig
from llm_bench_api.utils import get_vram_usage

from vllm import LLM
from vllm import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


logger = logging.getLogger(__name__)

GPU_DEVICE = os.environ.get("GPU_DEVICE_VLLM")
assert GPU_DEVICE, "GPU_DEVICE_VLLM environment variable not set"


def generate(config: ModelConfig, run_config: dict) -> dict:
    """Run vLLM inference and return metrics."""

    quant_str = f"{config.quantization_method}_{config.quantization_bits}" or "none"
    logger.info(f"Running benchmark: {config.model_name}, quant: {quant_str}")

    output_tokens, vram_usage, time_0, time_1 = 0, 0, 0, 0
    model = None

    with torch.no_grad():
        try:
            # Load model
            model = LLM(
                model=config.model_name,
                download_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
                trust_remote_code=True,
            )
            # Set params
            sampling_params = SamplingParams(temperature=0.1, top_p=0.95)

            # Generate tokens
            time_0 = time.time()
            output = model.generate(run_config["query"], sampling_params)
            time_1 = time.time()

            # Collect metrics
            output_tokens = len(output[0].outputs[0].token_ids)
            vram_usage = get_vram_usage(int(GPU_DEVICE))
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}")
            raise e
        finally:
            # Ensure model and CUDA memory is cleaned up
            destroy_model_parallel()
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(3)

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": [run_config["max_tokens"]],
        "output_tokens": [output_tokens],
        "gpu_mem_usage": [vram_usage],
        "generate_time": [time_1 - time_0],
        "tokens_per_second": [output_tokens / (time_1 - time_0) if time_1 > time_0 else 0],
    }

    return metrics
