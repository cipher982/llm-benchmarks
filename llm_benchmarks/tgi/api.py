import logging.config
import os
import time
from datetime import datetime

from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse

from llm_benchmarks.config import ModelConfig
from llm_benchmarks.logging import log_to_mongo
from llm_benchmarks.tgi.docker import DockerContainer
from llm_benchmarks.tgi.utils import get_vram_usage
from llm_benchmarks.tgi.utils import is_container_ready
from llm_benchmarks.tgi.utils import setup_logger


setup_logger()
logger = logging.getLogger(__name__)

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")

assert MONGODB_URI, "MONGODB_URI environment variable not set"
assert MONGODB_DB, "MONGODB_DB environment variable not set"
assert MONGODB_COLLECTION, "MONGODB_COLLECTION environment variable not set"

TEMPERATURE = 0.1


def benchmark_tgi(client: InferenceClient, input_text: str, max_tokens: int) -> TextGenerationResponse:
    """Sends a POST request to the text-generation Docker container."""
    try:
        # response = []
        response = client.text_generation(
            prompt=input_text,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            details=True,
        )
        # response.append(token)
        logger.info("Query successful.")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise


# Main Execution
if __name__ == "__main__":
    model = os.getenv("MODEL_ID", "facebook/opt-1.3b")
    volume_path = os.getenv("VOLUME_PATH", "/rocket/hf")
    requested_tokens = 20
    quant_bits = "8bit"
    query = "User: Tell me a long story about the history of the world.\nAI:"
    max_tokens = 512

    with DockerContainer(model, volume_path, quant_bits) as container:
        logger.info("Querying Docker container...")
        client = InferenceClient("http://127.0.0.0:8080")
        if is_container_ready():
            logger.info("Docker container is ready.")

            # Run benchmark
            time0 = time.time()
            response = benchmark_tgi(client, input_text=query, max_tokens=max_tokens)
            time1 = time.time()

            # Get metrics
            output_tokens = len(response.details.get("tokens"))  # type: ignore
            vram_usage = get_vram_usage()

            # Build metrics object
            metrics = {
                "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "requested_tokens": [requested_tokens],
                "output_tokens": [output_tokens],
                "gpu_mem_usage": [vram_usage],
                "generate_time": [time1 - time0],
                "tokens_per_second": [output_tokens / (time1 - time0)],
            }
            logger.info(f"Metrics: {metrics}")

            config = ModelConfig(
                framework="hf-tgi",
                model_name=model,
                quantization_bits=quant_bits,
                model_dtype="torch.float16",
                temperature=TEMPERATURE,
                run_ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Log metrics to MongoDB
            result = log_to_mongo(
                config=config,
                metrics=metrics,
                uri=MONGODB_URI,
                db_name=MONGODB_DB,
                collection_name=MONGODB_COLLECTION,
            )
        else:
            logger.error("Docker container did not become ready in time.")
