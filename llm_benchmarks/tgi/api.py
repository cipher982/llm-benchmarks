import json
import logging
import subprocess

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def start_docker_container(model: str, volume_path: str) -> None:
    """Starts the text-generation-inference Docker container."""
    try:
        command = [
            "docker",
            "run",
            "-d",  # detached for non-blockings
            "--gpus",
            "all",
            "--shm-size",
            "1g",
            "-p",
            "8080:80",
            "--hostname",
            "0.0.0.0",
            "-v",
            f"{volume_path}:/data",
            "ghcr.io/huggingface/text-generation-inference:1.1.0",
            "--model-id",
            model,
        ]
        subprocess.run(command, check=True)
        logger.info("Docker container started successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker container: {e}")


def stop_docker_container(container_name: str) -> None:
    """Stops the specified Docker container."""
    try:
        subprocess.run(["docker", "stop", container_name], check=True)
        logger.info("Docker container stopped successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop Docker container: {e}")


def query_text_generation_docker(input_text: str, max_new_tokens: int) -> dict:
    """Sends a POST request to the text-generation Docker container."""
    url = "http://127.0.0.1:8080/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"inputs": input_text, "parameters": {"max_new_tokens": max_new_tokens}}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return {}


# Example usage

logger.info("Starting Docker container...")
start_docker_container(model="facebook/opt-1.3b", volume_path="/rocket/hf")
logger.info("Docker container started successfully.")

logger.info("Querying Docker container...")
query_text_generation_docker(input_text="Hello", max_new_tokens=20)
logger.info("Query successful.")


logger.info("Stopping Docker container...")
stop_docker_container(container_name="text-generation-inference")
logger.info("Docker container stopped successfully.")
