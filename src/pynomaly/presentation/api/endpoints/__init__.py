"""API endpoints."""

from . import (  # admin,  # Temporarily disabled due to UserModel forward reference issue
    auth,
    # autonomous,  # Temporarily disabled for testing
    # datasets,  # Temporarily disabled due to forward reference issue
    # detection,  # Temporarily disabled for testing
    # detectors,  # Temporarily disabled for testing
    # experiments,  # Temporarily disabled for testing
    # export,  # Temporarily disabled for testing
    health,
    # performance,  # Temporarily disabled for testing
)

__all__ = [
    # "admin",  # Temporarily disabled due to UserModel forward reference issue
    "auth",
    "autonomous",
    # "datasets",  # Temporarily disabled due to forward reference issue
    "detection",
    "detectors",
    "experiments",
    "export",
    "health",
    "performance",
]
