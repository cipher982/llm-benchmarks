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
        cache_dir: str,
        gpu_device: int = 0,
        quant_method: Optional[str] = None,
        quant_bits: Optional[str] = None,
    ):
        self.model = model
        self.cache_dir = cache_dir
        self.gpu_device = gpu_device
        self.quant_method = quant_method
        self.quant_bits = quant_bits
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
            f"{self.cache_dir}:/data",
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id",
            self.model,
        ]

        quant_info = self.get_quantization_info()
        command.extend(quant_info["command"])
        logger.info(quant_info["message"])

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
            result = subprocess.run(
                ["docker", "logs", self.container_id],
                capture_output=True,
                text=True,
                check=True,
            )
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
                error_log = logs.split("Error:")[1].split("\n")[0]
                logger.error(f"Docker logs error: {error_log}")
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

    def get_quantization_info(self):
        if self.quant_method is None:
            return {
                "command": [],
                "message": "Starting Docker container without quantization.",
            }
        elif self.quant_method == "gptq":
            return {
                "command": ["--quantize", "gptq"],
                "message": "Starting Docker container with GPTQ quantization.",
            }
        elif self.quant_method == "bitsandbytes":
            if self.quant_bits == "4bit":
                return {
                    "command": ["--quantize", "bitsandbytes-nf4"],
                    "message": "Starting Docker container with 4bit quantization.",
                }
            elif self.quant_bits == "8bit":
                return {
                    "command": ["--quantize", "bitsandbytes"],
                    "message": "Starting Docker container with 8bit quantization.",
                }
            else:
                raise ValueError(f"Invalid quant_bits: {self.quant_bits}")
        elif self.quant_method == "awq":
            raise NotImplementedError("AWQ not implemented yet.")
        else:
            raise ValueError(f"Invalid quant_method: {self.quant_method}")
