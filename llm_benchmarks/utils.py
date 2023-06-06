import csv
from datetime import datetime
from pathlib import Path


def log_metrics_to_csv(model_name: str, config: dict, metrics: dict, output_dir: str) -> None:
    # Create a new CSV file for logging with timestamp
    model_name = model_name.split("/")[-1]

    # Prepare the data to be logged
    data = [
        [
            "model_name",
            "output_tokens",
            "gpu_mem_usage",
            "total_time",
            "tokens_per_second",
            "quantization_bits",
            "torch_dtype",
            "temperature",
        ],
    ]

    # Add the data for each run
    for i in range(len(metrics["output_tokens"])):
        row = [
            model_name,
            metrics["output_tokens"][i],
            f"{metrics['gpu_mem_usage'][i]:.2f}",
            f"{metrics['generate_time'][i]:.2f}",
            f"{metrics['tokens_per_second'][i]:.2f}",
            config["quantization_bits"],
            config["torch_dtype"],
            config["temperature"],
        ]
        data.append(row)

    # Write the data to the CSV file
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_name = f"metrics_{model_name}_{dt}.csv"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / csv_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
