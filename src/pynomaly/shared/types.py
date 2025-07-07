"""
Shared type definitions for the Pynomaly application.
"""

from typing import NewType, Union
import uuid

# Domain identifier types
DatasetId = NewType('DatasetId', str)
DetectorId = NewType('DetectorId', str)
ModelId = NewType('ModelId', str)
UserId = NewType('UserId', str)
TenantId = NewType('TenantId', str)
RoleId = NewType('RoleId', str)
SessionId = NewType('SessionId', str)

# Numeric types
Score = NewType('Score', float)
Confidence = NewType('Confidence', float)
Threshold = NewType('Threshold', float)

# Data types
FeatureName = NewType('FeatureName', str)
FeatureValue = NewType('FeatureValue', Union[int, float, str, bool])

# Infrastructure types
CacheKey = NewType('CacheKey', str)
StoragePath = NewType('StoragePath', str)
ConfigKey = NewType('ConfigKey', str)

# Utility functions
def generate_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())

def generate_dataset_id() -> DatasetId:
    """Generate a new dataset ID."""
    return DatasetId(generate_id())

def generate_detector_id() -> DetectorId:
    """Generate a new detector ID."""
    return DetectorId(generate_id())

def generate_model_id() -> ModelId:
    """Generate a new model ID."""
    return ModelId(generate_id())

def generate_user_id() -> UserId:
    """Generate a new user ID."""
    return UserId(generate_id())

def generate_tenant_id() -> TenantId:
    """Generate a new tenant ID."""
    return TenantId(generate_id())

def generate_role_id() -> RoleId:
    """Generate a new role ID."""
    return RoleId(generate_id())

def generate_session_id() -> SessionId:
    """Generate a new session ID."""
    return SessionId(generate_id())