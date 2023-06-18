import gc
import os
from time import time

import torch

import wandb

# import numpy as np

# Fix deadlock issue
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set GPU device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_samples(
    model_name: str,
    config: dict,
    custom_token_counts: list = [],
    llama: bool = False,
) -> dict:
    if llama:
        from transformers import LlamaTokenizer as Tokenizer
        from transformers import LlamaForCausalLM as Model
    else:
        from transformers import AutoTokenizer as Tokenizer
        from transformers import AutoModelForCausalLM as Model

    wandb.init(project="llm-benchmarks-3", config=config)

    # Load Model
    model = Model.from_pretrained(
        model_name,
        load_in_4bit=True if config["quantization_bits"] == "4bit" else False,
        load_in_8bit=True if config["quantization_bits"] == "8bit" else False,
        torch_dtype=config["torch_dtype"],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()  # type: ignore
    model = torch.compile(model)  # type: ignore

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
        token_counts = [16, 32, 64, 128, 256, 512, 1024, 2048]

    # Generate samples
    for i, token_count in enumerate(token_counts):
        time0 = time()
        with torch.no_grad():
            output = model.generate(
                input_tokens,
                do_sample=True,
                temperature=config["temperature"],
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

        # Log metrics
        metrics["output_tokens"].append(output_tokens)
        metrics["gpu_mem_usage"].append(gpu_mem_usage)
        metrics["generate_time"].append(generate_time)
        metrics["tokens_per_second"].append(tokens_per_second)

        # Send to wandb
        wandb.log(
            {
                "output_tokens": output_tokens,
                "gpu_mem_usage": gpu_mem_usage,
                "generate_time": generate_time,
                "tokens_per_second": tokens_per_second,
            }
        )

        # Print metrics
        print(f"===== Model: {model_name} Run: {i+1}/{len(token_counts)} =====")
        print(f"Output tokens: {output_tokens}")
        print(f"GPU mem usage: {gpu_mem_usage:.2f} GB")
        print(f"Generate time: {generate_time:.2f} s")
        print(f"Tokens per second: {tokens_per_second:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    wandb.finish()

    return metrics
