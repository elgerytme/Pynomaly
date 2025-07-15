"""GraphQL resolvers package for Pynomaly API."""

from .auth_resolvers import *
from .dataset_resolvers import *
from .detection_resolvers import *
from .detector_resolvers import *
from .model_resolvers import *
from .subscription_resolvers import *
from .training_resolvers import *
from .user_resolvers import *


def get_resolvers():
    """Get all GraphQL resolvers."""
    return {
        "auth": auth_resolvers,
        "user": user_resolvers,
        "dataset": dataset_resolvers,
        "detector": detector_resolvers,
        "model": model_resolvers,
        "training": training_resolvers,
        "detection": detection_resolvers,
        "subscription": subscription_resolvers,
    }