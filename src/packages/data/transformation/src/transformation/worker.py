"""Worker entry point for transformation package."""

import logging

logger = logging.getLogger(__name__)


def start_worker():
    """Start the transformation worker."""
    logger.info("Starting transformation worker...")
    # Worker implementation here


if __name__ == "__main__":
    start_worker()