"""Worker entry point for enterprise_auth package."""

import logging

logger = logging.getLogger(__name__)


def start_worker():
    """Start the enterprise auth worker."""
    logger.info("Starting enterprise auth worker...")
    # Worker implementation here


if __name__ == "__main__":
    start_worker()