import logging
import os
import shutil
from typing import List


logger = logging.getLogger(__name__)


def get_used_space_percent(directory: str) -> float:
    """Get the used space percentage of the file system containing the directory."""
    stat = os.statvfs(directory)
    return ((stat.f_blocks - stat.f_bfree) / stat.f_blocks) * 100


def get_model_directories(directory: str) -> List[str]:
    """Get a list of directories in the given directory, filtered by those starting with 'models--'."""
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.startswith("models--")
    ]


def get_oldest_directory(directories: List[str]) -> str:
    """Find the oldest directory in the given list."""
    oldest_directory = min(directories, key=lambda d: os.path.getmtime(d))
    return oldest_directory


def check_and_clean_space():
    threshold = 90.0  # Disk usage threshold
    directory = "/data/hf"  # The directory where the models are stored

    # Check disk usage
    used_space = get_used_space_percent(directory)
    logger.info(f"Current disk usage: {used_space:.2f}%")

    while used_space > threshold:
        # Get model directories
        model_dirs = get_model_directories(directory)

        # If there are no model directories, exit the loop
        if not model_dirs:
            logger.info("No model directories to remove.")
            break

        # Find the oldest directory
        oldest_dir = get_oldest_directory(model_dirs)

        # Remove the oldest directory
        logger.info(f"Removing: {oldest_dir}")
        shutil.rmtree(oldest_dir)

        # Recheck disk usage
        used_space = get_used_space_percent(directory)
        logger.info(f"Updated disk usage: {used_space:.2f}%")
