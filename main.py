import argparse
import gc
import os
from time import time
from typing import Optional
from typing import Union

import torch
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

import wandb
from wandb import Table
from wandb.plot import bar

# Fix deadlock issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize wandb
wandb.init(project="llm-benchmarks")

# Create tables for each metric
output_tokens_table = Table(columns=["Model", "Output Tokens"])
gpu_mem_usage_table = Table(columns=["Model", "GPU Memory Usage"])
time_table = Table(columns=["Model", "Total Time"])
tokens_per_second_table = Table(columns=["Model", "Tokens/Second"])

# Define input
text_input = "Question: Tell me a history of WW2 in 3 or 4 paragraphs.\nAnswer: "


def run_local_model(
    text_input: str,
    model_name: str,
    model_class: Union[LlamaForCausalLM, AutoModelForCausalLM],
    tokenizer_class: Union[AutoTokenizer, LlamaTokenizer],
    load_in_8bit: bool,
):
    # Load model
    model = model_class.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Tokenize inputs
    tokenizer = tokenizer_class.from_pretrained(model_name)
    input_tokens = tokenizer(text_input, return_tensors="pt").input_ids.to("cuda")

    # Generate
    time0 = time()
    output = model.generate(  # type: ignore
        input_tokens,
        do_sample=True,
        temperature=0.9,
        max_length=1024,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed_time = time() - time0

    # Collect metrics
    gpu_mem_usage = torch.cuda.memory_allocated() / 1024**3
    token_count = len(output.cpu().numpy().tolist()[0])  # type: ignore

    # Clear up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed_time, token_count, gpu_mem_usage


def run_cloud_model(text_input: str, model_name: str, chat_model: bool):
    print(f"===== Model: {model_name} =====")

    if chat_model:
        model = ChatOpenAI(model=model_name, client=None)
        text_input = [HumanMessage(content=text_input)]  # type: ignore
    else:
        model = OpenAI(model=model_name, max_tokens=-1, client=None)

    time0 = time()
    output = model(text_input)  # type: ignore
    elapsed_time = time() - time0

    if chat_model:
        output = output.content  # type: ignore

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    token_count = len(tokenizer.encode(output))

    return elapsed_time, token_count, 0


# Define function for running models
def run_model(
    text_input: str,
    model_name: str,
    local_model: bool,
    model_class: Optional[Union[AutoModelForCausalLM, LlamaForCausalLM]] = None,
    tokenizer_class: Optional[Union[AutoTokenizer, LlamaTokenizer]] = None,
    load_in_8bit: Optional[bool] = None,
    chat_model: Optional[bool] = None,
):
    print(f"===== Model: {model_name} =====")

    if local_model:
        assert model_class
        assert tokenizer_class
        assert load_in_8bit is not None

        try:
            elapsed_time, token_count, gpu_mem_usage = run_local_model(
                text_input,
                model_name,
                model_class,
                tokenizer_class,
                load_in_8bit,
            )
        except RuntimeError:
            print("RuntimeError: CUDA out of memory. Skipping model...")
            gc.collect()
            torch.cuda.empty_cache()
    else:
        assert chat_model is not None

        elapsed_time, token_count, gpu_mem_usage = run_cloud_model(
            text_input,
            model_name,
            chat_model,
        )

    # Add metrics to tables
    output_tokens_table.add_data(model_name, token_count)
    gpu_mem_usage_table.add_data(model_name, gpu_mem_usage)
    time_table.add_data(model_name, elapsed_time)
    tokens_per_second_table.add_data(model_name, token_count / elapsed_time)

    # Log metrics
    print(f"===== Model: {model_name} =====")
    print(f"Output tokens: {token_count}")
    print(f"GPU memory usage: {gpu_mem_usage:.2f} GB")
    print(f"Total Time: {elapsed_time:.2f} s")
    print(f"Tokens per second: {(token_count / elapsed_time):.2f}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to models config YAML file")
    parser.add_argument("--n", help="Number of times to run each model", default=1)
    args = parser.parse_args()

    # Load models config
    config_file = args.config if args.config else "models_config.yaml"
    with open(config_file) as file:
        models_data = yaml.safe_load(file)

    # Retrieve model lists from YAML data
    gpt2_models = models_data.get("gpt2_models", [])
    dolly_models = models_data.get("dolly_models", [])
    llama_models = models_data.get("llama_models", [])
    openai_text_models = models_data.get("openai_text_models", [])
    openai_chat_models = models_data.get("openai_chat_models", [])

    # Run loop n times for each model (balance out variance in runtimes)
    for _ in range(int(args.n)):
        # Local models
        for model_name in gpt2_models:
            run_model(
                text_input,
                model_name,
                local_model=True,
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer,
                load_in_8bit=False,
            )

        for model_name in dolly_models:
            run_model(
                text_input,
                model_name,
                local_model=True,
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer,
                load_in_8bit=True,
            )

        for model_name in llama_models:
            run_model(
                text_input,
                model_name,
                local_model=True,
                model_class=LlamaForCausalLM,
                tokenizer_class=LlamaTokenizer,
                load_in_8bit=True,
            )

        # Cloud models
        for model_name in openai_text_models:
            run_model(
                text_input=text_input,
                model_name=model_name,
                local_model=False,
                chat_model=False,
            )

        for model_name in openai_chat_models:
            run_model(
                text_input=text_input,
                model_name=model_name,
                local_model=False,
                chat_model=True,
            )

    # Log tables
    wandb.log(
        {
            "output_tokens_bar_chart": bar(
                output_tokens_table, "Model", "Output Tokens"
            ),
            "gpu_mem_usage_bar_chart": bar(
                gpu_mem_usage_table, "Model", "GPU Memory Usage"
            ),
            "time_bar_chart": bar(time_table, "Model", "Total Time"),
            "tokens_per_second_bar_chart": bar(
                tokens_per_second_table, "Model", "Tokens/Second"
            ),
        }
    )
    # Upload artifacts
    wandb.finish()


if __name__ == "__main__":
    main()
