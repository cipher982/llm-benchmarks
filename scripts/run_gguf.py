from typing import Dict

import click
import requests


FLASK_PORT = 5003


@click.command()
@click.option("--limit", default=50, type=int, help="Limit the number of models to run for debugging.")
def bench_gguf(limit: int) -> None:
    """Benchmark all models on the gguf server."""

    model_names = [
        "llama-3B",
        "llama-7B",
        "llama-13B",
    ]
    quant_types = [
        "/ggml-model-q4_0.gguf",
        "/ggml-model-q8_0.gguf",
        "/ggml-model-f16.gguf",
    ]

    model_status: Dict[str, int] = {}
    stop = False
    for model in model_names:
        for quant in quant_types:
            if stop:
                break
            full_model = model + quant
            print(f"Running benchmark: {full_model}")

            config = {
                "query": "User: Tell me a long story about the history of the world.\nAI:",
                "max_tokens": 256,
                "n_gpu_layers": -1,
            }
            response = requests.post(f"http://localhost:{FLASK_PORT}/benchmark/{full_model}", data=config)

            response_code = response.status_code
            print(f"Finished benchmark: {full_model} with Status Code: {response_code}")

            model_status[full_model] = response_code

            if len(model_status) >= limit:
                stop = True

    print("All benchmark runs are finished.")

    # Summary of benchmark runs
    print("Summary of benchmark runs:")
    for model, code in model_status.items():
        print(f"Model: {model}, HTTP Response Code: {code} {'✅' if code == 200 else '❌'}")
    print("🎊 Done 🎊")


if __name__ == "__main__":
    bench_gguf()
