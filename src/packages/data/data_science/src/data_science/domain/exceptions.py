"""Domain exceptions for data science package."""


class DataScienceError(Exception):
    """Base domain exception for data science operations."""
    pass


class InvalidValueError(DataScienceError):
    """Exception raised when an invalid value is provided."""
    pass


class DataTypeError(DataScienceError):
    """Exception raised when data type validation fails."""
    pass


class FeatureMismatchError(DataScienceError):
    """Exception raised when feature validation fails."""
    pass


class InvalidDataError(DataScienceError):
    """Exception raised when data validation fails."""
    pass