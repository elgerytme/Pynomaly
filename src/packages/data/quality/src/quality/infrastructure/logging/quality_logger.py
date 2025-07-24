"""Logging utilities for data quality services."""

import logging
import sys
from typing import Optional, Dict, Any


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get configured logger for data quality services."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set level
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger


def configure_quality_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure logging for the entire data quality package."""
    if config is None:
        config = {}
    
    # Default configuration
    log_level = config.get('log_level', 'INFO')
    log_format = config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger for data quality
    root_logger = logging.getLogger('quality')
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler if not exists
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.propagate = False


# Initialize logging with default configuration
configure_quality_logging()