"""Protocol definitions for clean architecture interfaces."""

from .detector_protocol import DetectorProtocol
from .repository_protocol import RepositoryProtocol
from .data_loader_protocol import DataLoaderProtocol

__all__ = ["DetectorProtocol", "RepositoryProtocol", "DataLoaderProtocol"]