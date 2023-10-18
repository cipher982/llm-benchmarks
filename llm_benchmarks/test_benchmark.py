import logging
from datetime import datetime

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.generation import benchmark_model

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting benchmarking...")

    config = ModelConfig(
        # quantization_bits="8bit",
        quantization_bits=None,
        # torch_dtype="float16",
        torch_dtype="auto",
        temperature=0.1,
        run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    for model_name in model_names:
        _ = benchmark_model(
            model_name=model_name,
            config=config,
            custom_token_counts=[64, 128, 256],
            llama=False,
        )

    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()
