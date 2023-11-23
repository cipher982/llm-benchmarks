import logging
import os
from enum import auto
from enum import Enum
from typing import Optional

import requests
from requests.exceptions import HTTPError


logger = logging.getLogger(__name__)

FLASK_URL = "http://localhost:{}/benchmark/"

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE")
assert CACHE_DIR, "HUGGINGFACE_HUB_CACHE environment variable not set"


class ModelType(Enum):
    GPTQ = auto()
    AWQ = auto()
    OTHER = auto()


class BenchmarkConfig:
    def __init__(
        self,
        framework: str,
        model: str,
        quant_types: list,
        limit: int,
        run_always: bool,
        query: str,
        max_tokens: int,
        temperature: float,
        flask_port: int,
    ):
        self.framework = framework
        self.model = model
        self.quant_types = quant_types
        self.limit = limit
        self.run_always = run_always
        self.query = query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.flask_port = flask_port


def determine_model_type(model_name: str) -> ModelType:
    if "GPTQ" in model_name:
        return ModelType.GPTQ
    elif "AWQ" in model_name:
        return ModelType.AWQ
    else:
        return ModelType.OTHER


def bench_all_models(
    framework: str,
    quant_types: list,
    model_names: list[str],
    model_status: dict[str, dict],
    limit: int,
    run_always: bool,
    query: str,
    max_tokens: int,
    temperature: float,
    flask_port: int,
) -> None:
    for model in model_names[:limit]:
        model_type = determine_model_type(model)
        is_limit_reached = run_benchmark_for_type(
            framework,
            model,
            quant_types,
            model_status,
            model_type,
            limit,
            run_always,
            query,
            max_tokens,
            temperature,
            flask_port,
        )
        if is_limit_reached:
            break


def run_benchmark_for_type(
    framework: str,
    model: str,
    quant_types: list,
    model_status: dict[str, dict],
    model_type: ModelType,
    limit: int,
    run_always: bool,
    query: str,
    max_tokens: int,
    temperature: float,
    flask_port: int,
) -> bool:
    config = BenchmarkConfig(
        framework,
        model,
        quant_types,
        limit,
        run_always,
        query,
        max_tokens,
        temperature,
        flask_port,
    )
    if model_type == ModelType.GPTQ:
        return run_benchmark(config, model_status, "gptq", "4bit")
    elif model_type == ModelType.AWQ:
        return run_benchmark(config, model_status, "awq", "4bit")
    else:
        for quant in quant_types:
            quant_method = "bitsandbytes" if quant is not None else None
            if run_benchmark(config, model_status, quant_method, quant):
                return True
    return False


def run_benchmark(
    config: BenchmarkConfig,
    model_status: dict[str, dict],
    quant_method: Optional[str],
    quant_bits: Optional[str],
) -> bool:
    """
    Run benchmark for a given model and quantization type.
    Returns True if the limit is reached, False otherwise.
    """
    quant_str = f"{quant_method}_{quant_bits}" if quant_method is not None else "none"
    print(f"Running benchmark: {config.model}, quant: {quant_str}")

    flask_data = {
        "framework": config.framework,
        "model_name": config.model,
        "query": config.query,
        "quant_method": quant_method,
        "quant_bits": quant_bits,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "run_always": config.run_always,
    }
    try:
        response = requests.post(FLASK_URL.format(config.flask_port), data=flask_data)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        model_status[f"{config.model}_{quant_str}"] = {"status_code": 500, "json": {}}
        return False
    except Exception as err:
        print(f"Other error occurred: {err}")
        model_status[f"{config.model}_{quant_str}"] = {"status_code": 500, "json": {}}
        return False
    else:
        response_code = response.status_code
        response_json = response.json()
        print(f"Finished benchmark: {config.model}, quant: {quant_str} with Status Code: {response_code}")

        model_status[f"{config.model}_{quant_str}"] = {"status_code": response_code, "json": response_json}
        return len(model_status) >= config.limit


def print_summary(model_status: dict[str, dict]) -> None:
    """
    Print a summary of the benchmark runs.
    """
    print("Summary of benchmark runs:")
    skipped_models = []
    for model, response in model_status.items():
        status = response["json"]["status"] if "json" in response and "status" in response["json"] else "unknown"
        if status == "skipped":
            skipped_models.append(model)
            continue

    if skipped_models:
        print(f"Skipped models: {', '.join(skipped_models)} ⏭️")

    for model, response in model_status.items():
        status = response["json"]["status"] if "json" in response and "status" in response["json"] else "unknown"
        if status == "skipped":
            continue
        elif response["status_code"] == 200:
            print(f"Model: {model}, {response['status_code']} ✅ (Benchmark Successful)")
        elif response["status_code"] == 500:
            print(f"Model: {model}, {response['status_code']} ❌ (Benchmark Failed)")
        else:
            print(f"Model: {model}, {response['status_code']} ❓ (Unknown Status)")
    print("🎊 Done 🎊")
