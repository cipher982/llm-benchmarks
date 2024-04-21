import logging
import os
import time
from datetime import datetime

from huggingface_hub import InferenceClient
from llm_bench_api.config import ModelConfig
from llm_bench_api.utils import get_vram_usage

from llm_bench_hf.tgi import DockerContainer

logger = logging.getLogger(__name__)


GPU_DEVICE = os.environ.get("GPU_DEVICE")
CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert GPU_DEVICE, "GPU_DEVICE environment variable not set"
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"


def generate(config: ModelConfig, run_config: dict):
    """Run TGI inference and return metrics."""
    time.sleep(1)

    quant_str = f"{config.quantization_method}_{config.quantization_bits}" or "none"
    logger.info(f"Running benchmark: {config.model_name}, quant: {quant_str}")

    # Load model
    with DockerContainer(
        config.model_name,
        CACHE_DIR,
        int(GPU_DEVICE),
        config.quantization_method,
        config.quantization_bits,
    ) as container:
        if container.is_ready():
            logger.info("Docker container is ready.")
            client = InferenceClient("http://127.0.0.0:8080")

            # Generate samples
            time0 = time.time()
            response = client.text_generation(
                prompt=run_config["query"],
                max_new_tokens=run_config["max_tokens"],
                temperature=config.temperature,
                details=True,
            )
            time1 = time.time()

            # Process metrics
            output_tokens = len(response.details.tokens) if response.details is not None else 0
            vram_usage = get_vram_usage(int(GPU_DEVICE))
            metrics = {
                "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "requested_tokens": [run_config["max_tokens"]],
                "output_tokens": [output_tokens],
                "gpu_mem_usage": [vram_usage],
                "generate_time": [time1 - time0],
                "tokens_per_second": [output_tokens / (time1 - time0) if time1 > time0 else 0],
            }

            return metrics
