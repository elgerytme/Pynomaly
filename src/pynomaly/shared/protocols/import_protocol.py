"""
Import Protocol for Pynomaly

Defines the interface for importing datasets from various sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from pathlib import Path

from ...domain.entities.dataset import Dataset


class ImportProtocol(ABC):
    """Protocol for importing datasets from various sources."""
    
    @abstractmethod
    def import_dataset(
        self,
        file_path: Union[str, Path],
        options: Dict[str, Any] = None
    ) -> Dataset:
        """
        Import dataset from a file.
        
        Args:
            file_path: Path to the file to import
            options: Import configuration options
            
        Returns:
            Dataset object containing the imported data
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions (e.g., ['.xlsx', '.csv'])
        """
        pass
    
    @abstractmethod
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if the file can be imported by this adapter.
        
        Args:
            file_path: File path to validate
            
        Returns:
            True if file can be imported, False otherwise
        """
        pass