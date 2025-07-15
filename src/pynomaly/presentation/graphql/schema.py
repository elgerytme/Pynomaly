"""GraphQL schema definition for Pynomaly API."""

from __future__ import annotations

import strawberry
from strawberry.extensions import ParserCache, ValidationCache
from strawberry.schema.config import StrawberryConfig

from .types import (
    AnomalyDetectionResult,
    Detector,
    DetectorInput,
    DetectionJob,
    DetectionJobInput,
    DetectionResultInput,
    User,
    UserInput,
)
from .mutations import Mutation
from .queries import Query
from .subscriptions import Subscription


# Create the main GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    config=StrawberryConfig(
        auto_camel_case=True,
        relay_max_complexity=None,
    ),
    extensions=[
        ParserCache(maxsize=128),
        ValidationCache(maxsize=128),
    ],
)

__all__ = [
    "schema",
    "Query",
    "Mutation", 
    "Subscription",
    "AnomalyDetectionResult",
    "Detector",
    "DetectorInput",
    "DetectionJob",
    "DetectionJobInput",
    "DetectionResultInput",
    "User",
    "UserInput",
]