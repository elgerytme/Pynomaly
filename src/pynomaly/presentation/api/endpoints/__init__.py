"""API endpoints."""

from . import (  # admin,  # Temporarily disabled due to UserModel forward reference issue
    auth,
    autonomous,
    datasets,
    detection,
    detectors,
    experiments,
    export,
    health,
    performance,
)

__all__ = [
    # "admin",  # Temporarily disabled due to UserModel forward reference issue
    "auth",
    "autonomous",
    "datasets",
    "detection",
    "detectors",
    "experiments",
    "export",
    "health",
    "performance",
]
