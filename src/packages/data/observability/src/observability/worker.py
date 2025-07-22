"""Worker entry point for observability package."""

import logging

logger = logging.getLogger(__name__)


def start_worker():
    """Start the observability worker."""
    logger.info("Starting observability worker...")
    # Worker implementation here


if __name__ == "__main__":
    start_worker()