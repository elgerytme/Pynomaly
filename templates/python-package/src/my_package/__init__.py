"""My Package - A comprehensive Python package."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import Calculator, DataProcessor
from .utils import validate_input, format_output

__all__ = [
    "Calculator",
    "DataProcessor", 
    "validate_input",
    "format_output",
]