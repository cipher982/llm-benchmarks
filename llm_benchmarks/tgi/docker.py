import logging
import subprocess
import sys
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class DockerContainer:
    def __init__(
        self,
        model: str,
        volume_path: str,
        gpu_device: int = 1,
        quantization_bits: Optional[str] = None,
    ):
        self.model = model
        self.volume_path = volume_path
        self.gpu_device = gpu_device
        self.quantization_bits = quantization_bits
        self.container_id = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Starts the Docker container."""
        command = [
            "docker",
            "run",
            "-d",
            "--gpus",
            f"device={self.gpu_device}",
            "--shm-size",
            "1g",
            "-p",
            "8080:80",
            "--hostname",
            "0.0.0.0",
            "-v",
            f"{self.volume_path}:/data",
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id",
            self.model,
        ]

        quantization_map = {
            "gptq": {"command": ["--quantize", "gptq"], "message": "Starting Docker container with GPTQ quantization."},
            "8bit": {
                "command": ["--quantize", "bitsandbytes"],
                "message": "Starting Docker container with 8bit quantization.",
            },
            "4bit": {
                "command": ["--quantize", "bitsandbytes-nf4"],
                "message": "Starting Docker container with 4bit quantization.",
            },
            "default": {"command": [], "message": "Starting Docker container without quantization."},
        }

        quantization_key = self.model.lower() if "gptq" in self.model.lower() else (self.quantization_bits or "default")
        quantization_info = quantization_map.get(quantization_key, quantization_map["default"])
        command.extend(quantization_info["command"])
        logger.info(quantization_info["message"])

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if stdout:
                logger.info(f"Docker process stdout: {stdout.decode()}")
            if stderr:
                logger.error(f"Docker process stderr: {stderr.decode()}")
            process.wait()
            logger.info("Docker container started successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker container: {e}")
            sys.exit(1)

        # Fetch container ID immediately
        self.container_id = stdout.decode().strip()
        if not self.container_id:
            raise RuntimeError("Failed to get Docker container ID")

    def stop(self):
        """Stops the Docker container."""
        if not self.container_id:
            return

        try:
            subprocess.run(["docker", "stop", self.container_id], check=True)
            logger.info("Docker container stopped successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Docker container: {e}")
            sys.exit(1)

    def fetch_logs(self):
        """Fetches logs from the Docker container."""
        if not self.container_id:
            return ""

        try:
            result = subprocess.run(["docker", "logs", self.container_id], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch Docker logs: {e}")
            return ""

    def is_ready(self, timeout: int = 1800):
        """Check if the Docker container is ready."""
        success_message = "Connected"
        error_pattern = "Error:"

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check Docker logs for success or error messages
            logs = self.fetch_logs()
            if success_message in logs:
                return True
            if error_pattern in logs:
                logger.error("Error detected in Docker logs.")
                return False

            # Check if container's service is responding
            try:
                response = requests.get("http://127.0.0.1:8080")
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(5)
        return False
