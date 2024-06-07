import os

from huggingface_hub import snapshot_download


def get_quant_type(file: str) -> str:
    """Get quantization type from file name."""
    if "f16" in file:
        return "f16"
    elif "int8" in file:
        return "8bit"
    elif "int4" in file:
        return "4bit"
    else:
        raise ValueError(f"Unknown quant type for file: {file}")


def download_gguf_model(model_name: str, model_dir: str) -> None:
    model_path = os.path.join(model_dir, model_name.replace("/", "--"))
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )


def fetch_gguf_files(model_dir: str) -> list[str]:
    """Fetch .gguf files from the given directory."""
    gguf_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".gguf"):
                relative_path = os.path.relpath(os.path.join(root, file), model_dir)
                gguf_files.append(relative_path)
    return gguf_files
