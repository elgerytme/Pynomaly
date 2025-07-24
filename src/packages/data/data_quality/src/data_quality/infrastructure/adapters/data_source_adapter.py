
from abc import ABC, abstractmethod
from typing import Any, Dict


class DataSourceAdapter(ABC):
    """Abstract base class for data source adapters."""

    @abstractmethod
    def read_data(self, source_config: Dict[str, Any]) -> Any:
        """Reads data from the specified source configuration.

        Args:
            source_config: A dictionary containing configuration details for the data source.

        Returns:
            The data read from the source, in a format suitable for processing (e.g., Pandas DataFrame).
        """
        raise NotImplementedError
