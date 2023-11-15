import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


class DockerContainer:
    def __init__(self, model: str, volume_path: str, quantization_bits: str):
        self.model = model
        self.quantization_bits = quantization_bits
        self.volume_path = volume_path
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
            "1",
            "--shm-size",
            "1g",
            "-p",
            "8080:80",
            "--hostname",
            "0.0.0.0",
            "-v",
            f"{self.volume_path}:/data",
            "ghcr.io/huggingface/text-generation-inference:1.1.0",
            "--model-id",
            self.model,
        ]

        if self.quantization_bits == "8bit":
            command.extend(["--quantize", "bitsandbytes"])
            logger.info("Starting Docker container with 8bit quantization.")
        elif self.quantization_bits == "4bit":
            command.extend(["--quantize", "bitsandbytes-nf4"])
            logger.info("Starting Docker container with 4bit quantization.")
        else:
            logger.info("Starting Docker container without quantization.")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            self.container_id, _ = process.communicate()
            process.wait()
            logger.info("Docker container started successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker container: {e}")
            sys.exit(1)

    def stop(self):
        """Stops the Docker container."""
        if not self.container_id:
            return

        try:
            subprocess.run(["docker", "stop", self.container_id.decode().strip()], check=True)
            logger.info("Docker container stopped successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Docker container: {e}")
            sys.exit(1)
