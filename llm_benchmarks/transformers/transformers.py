"""LLM generation and benchmarking for HuggingFace Transformers library."""
import gc
import logging.config
from datetime import datetime
from time import time
from typing import List
from typing import Optional

import torch

from llm_benchmarks.config import ModelConfig


logger = logging.getLogger(__name__)


T_S_CUTOFF = 5  # Tokens per second cutoff (to prevent long runs)


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
        torch_dtype=torch.float16 if config.model_dtype == "torch.float16" else torch.float32,
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
            continue

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
