import re
from typing import Dict
from typing import List

import requests


def fetch_and_bench_tf(limit: int = 10, max_size_billion: int = 5, run_always: bool = False) -> None:
    # Fetch models
    params = {"sort": "downloads", "direction": "-1", "limit": limit, "filter": "text-generation"}
    response = requests.get("https://huggingface.co/api/models", params=params)
    model_data = response.json()

    # Extract model IDs
    model_ids = [entry["id"] for entry in model_data]

    # model_ids = ["codellama/CodeLlama-34b-Instruct-hf"]

    # Filter models
    valid_models: List[str] = []
    dropped_models: List[str] = []
    for model_id in model_ids:
        # Use regex to extract the parameter size
        match = re.search(r"([0-9.]+[MmBb])", model_id)
        param_count = match.group(1) if match else None

        if not param_count:
            dropped_models.append(model_id)
            continue

        # Normalize parameter count to billions
        unit = param_count[-1].upper()
        numerical_part = float(param_count[:-1])
        if unit == "M":
            numerical_part /= 1000  # Convert M to B

        # Filter based on parameter count
        if numerical_part <= max_size_billion:
            valid_models.append(model_id)
        else:
            dropped_models.append(model_id)

    print(f"Keeping models: {', '.join(valid_models)}\n\n")
    print(f"Dropping models: {', '.join(dropped_models)}\n\n")

    # Benchmarking
    model_status: Dict[str, int] = {}
    for model_id in valid_models:
        encoded_model = model_id.replace("/", "%2F")

        print(f"Running benchmark: {model_id}")

        # Define params
        params = {"run_always": str(run_always).lower()}

        # POST request with params
        response = requests.post(f"http://localhost:5000/benchmark/{encoded_model}", params=params)

        response_code = response.status_code
        print(f"Finished benchmark: {model_id} with Status Code: {response_code}")

        model_status[model_id] = response_code

    print("All benchmark runs are finished.")

    # Summary of benchmark runs
    print("Summary of benchmark runs:")
    for model, code in model_status.items():
        print(f"Model: {model}, HTTP Response Code: {code} {'✅' if code == 200 else '❌'}")
    print("🎊 Done 🎊")


if __name__ == "__main__":
    fetch_and_bench_tf(limit=100, max_size_billion=10, run_always=False)
