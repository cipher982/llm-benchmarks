import gc
import os
from time import time

import numpy as np
import torch

import wandb

# Fix deadlock issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_samples(
    model_name: str,
    config: dict,
    num_samples: int = 1,
    llama: bool = False,
) -> dict:
    if llama:
        from transformers import LlamaTokenizer as Tokenizer
        from transformers import LlamaForCausalLM as Model
    else:
        from transformers import AutoTokenizer as Tokenizer
        from transformers import AutoModelForCausalLM as Model

    # Load Model
    model = Model.from_pretrained(
        model_name,
        load_in_8bit=config["load_in_8bit"],
        torch_dtype=config["torch_dtype"],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()  # type: ignore
    model = torch.compile(model)  # type: ignore

    # Tokenize inputs
    tokenizer = Tokenizer.from_pretrained(model_name)
    text = "Question: Tell me a history of WW2 in 3 or 4 paragraphs.\nAnswer: "
    input_tokens = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    metrics = {
        "output_tokens": [],
        "gpu_mem_usage": [],
        "generate_time": [],
        "tokens_per_second": [],
    }

    if config["try_different_lengths"]:
        max_length = np.linspace(config["min_length"], config["max_length"], num_samples)
    else:
        max_length = [config["max_length"]] * num_samples

    for i in range(num_samples):
        # Set up Weights & Biases
        wandb.init(project="llm-benchmarks-2", config=config)

        # Generate
        time0 = time()
        with torch.no_grad():
            output = model.generate(
                input_tokens,
                do_sample=True,
                temperature=config["temperature"],
                min_length=int(max_length[i]),
                max_length=int(max_length[i]),
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
        wandb.finish()

        # Print metrics
        print(f"===== Model: {model_name} Run: {i+1}/{num_samples} =====")
        print(f"Output tokens: {output_tokens}")
        print(f"GPU mem usage: {gpu_mem_usage:.2f} GB")
        print(f"Generate time: {generate_time:.2f} s")
        print(f"Tokens per second: {tokens_per_second:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics
