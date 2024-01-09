import argparse
import os
import shutil
import subprocess

from huggingface_hub import snapshot_download

# Define the constant for the final output directory
OUTPUT_DIR = "/gemini/gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub.")
    parser.add_argument("-m", "--model", required=True, help="Model ID from the Hugging Face Hub.")
    args = parser.parse_args()

    # Download the model
    cleaned_model_id = clean_model_id(args.model)
    tmp_dir = "/tmp/" + cleaned_model_id
    download_model(model_id=args.model, local_dir=tmp_dir)

    # Convert the model to the gguf format
    print("Converting model to gguf format...")
    outfile_path = os.path.join(OUTPUT_DIR, cleaned_model_id, "m-f16.gguf")
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    process = subprocess.run(
        [
            "python",
            "../prep/llama.cpp/convert.py",
            tmp_dir,
            "--outfile",
            outfile_path,
            "--padvocab",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    # Filter the output
    for line in process.stdout.split("\n"):
        if "error" in line.lower() or "warning" in line.lower():
            print(line)

    shutil.rmtree(tmp_dir)


def download_model(model_id: str, local_dir: str, revision: str = "main") -> None:
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp"

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=revision,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    print(f"Model {model_id} downloaded to {local_dir}")


def clean_model_id(model_id: str) -> str:
    return model_id.replace("/", "--")


if __name__ == "__main__":
    main()
