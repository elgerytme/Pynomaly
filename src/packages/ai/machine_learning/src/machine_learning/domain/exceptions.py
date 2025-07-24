"""Domain exceptions for machine learning package."""


class DomainError(Exception):
    """Base domain exception."""
    pass


class AutoMLError(DomainError):
    """AutoML-specific domain exception."""
    pass


class ModelTrainingError(DomainError):
    """Model training exception."""
    pass


class ModelValidationError(DomainError):
    """Model validation exception."""
    pass


class UncertaintyQuantificationError(DomainError):
    """Uncertainty quantification exception."""
    pass