import logging
from datetime import datetime

import torch

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.transformers import generate

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting benchmarking...")

    model_names = ["gpt2", "gpt2-medium"]

    for model_name in model_names:
        config = ModelConfig(
            model_name=model_name,
            # quantization_bits="8bit",
            quantization_bits=None,
            torch_dtype=torch.float16,
            temperature=0.1,
            run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        _ = generate(
            config=config,
            custom_token_counts=[64, 128],
            llama=False,
        )

    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()