"""LLM generation and benchmarking for HuggingFace Transformers library."""

import gc
import logging.config
import os
from datetime import datetime
from time import time

import torch
from llm_bench.config import ModelConfig
from llm_bench.utils import get_vram_usage
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
GPU_DEVICE = os.environ.get("GPU_DEVICE")
assert GPU_DEVICE, "GPU_DEVICE environment variable not set"


def generate(
    config: ModelConfig,
    run_config: dict,
) -> dict:
    """Run Transformers inference and return metrics."""

    quant_str = f"{config.quantization_method}_{config.quantization_bits}" or "none"
    logger.info(f"Running benchmark: {config.model_name}, quant: {quant_str}")

    if config.quantization_method == "gptq":
        # gptq comes pre-quantized from the hub, no need to convert when loading
        quant_config = None
    elif config.quantization_method == "awq":
        raise NotImplementedError("AWQ not supported at the moment due to compatibility issues")
    elif config.quantization_method == "bitsandbytes":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quant_config = None

    # Prepare the arguments for loading the model
    load_args = {
        "torch_dtype": torch.float16 if config.model_dtype == "torch.float16" else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True,
        "token": HF_TOKEN,
    }

    # Conditionally add quantization_config if it is not None
    if quant_config is not None:
        load_args["quantization_config"] = quant_config

    # Load model
    logger.info(f"Loading pretrained model: {config.model_name}, quant: {quant_str}")
    logger.info(f"Config: {config.to_dict()}")
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_args)
    model.eval()

    # Checks for correct model loading and dtype (hack to ensure pre-quantized models are loaded correctly)
    if config.quantization_bits == "4bit" and config.quantization_method != "bitsandbytes":
        assert model.config.quantization_config.bits == 4, f"Model quant bits: {model.config.quantization.bits}"
    elif config.quantization_bits == "8bit" and config.quantization_method != "bitsandbytes":
        assert model.config.quantization_config.bits == 8, f"Model quant bits: {model.config.quantization.bits}"

    # Generate samples
    time_0 = time()
    output = None
    try:
        with torch.no_grad():
            output = model.generate(
                torch.tensor([[0, 1, 2]]).to("cuda"),
                do_sample=config.misc.get("do_sample"),
                temperature=config.temperature if config.misc.get("do_sample") else None,
                min_length=run_config["max_tokens"],
                max_length=run_config["max_tokens"],
            )
    except Exception as e:
        logger.error(f"Error generating tokens: {e}")
        raise e
    time_1 = time()

    # Collect metrics
    output_tokens = len(output.cpu().numpy().tolist()[0]) if output is not None and output.numel() > 0 else 0
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

    return metrics
