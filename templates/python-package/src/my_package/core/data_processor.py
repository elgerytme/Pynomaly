"""Data processing module for data manipulation."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """A data point with value and metadata."""
    
    value: Union[int, float, str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


class DataProcessor:
    """A data processor for handling collections of data points."""
    
    def __init__(self) -> None:
        self._data: List[DataPoint] = []
    
    def add_data_point(self, data_point: DataPoint) -> None:
        """Add a data point to the processor.
        
        Args:
            data_point: The data point to add
        """
        self._data.append(data_point)
    
    def get_data_points(self) -> List[DataPoint]:
        """Get all data points.
        
        Returns:
            List of all data points
        """
        return self._data.copy()
    
    def filter_by_value(self, min_value: Union[int, float]) -> List[DataPoint]:
        """Filter data points by minimum value.
        
        Args:
            min_value: Minimum value to filter by
            
        Returns:
            List of filtered data points
        """
        return [
            dp for dp in self._data
            if isinstance(dp.value, (int, float)) and dp.value >= min_value
        ]
    
    def count(self) -> int:
        """Count the number of data points.
        
        Returns:
            Number of data points
        """
        return len(self._data)
    
    def clear(self) -> None:
        """Clear all data points."""
        self._data.clear()
    
    def get_numeric_values(self) -> List[Union[int, float]]:
        """Get all numeric values from data points.
        
        Returns:
            List of numeric values
        """
        return [
            dp.value for dp in self._data
            if isinstance(dp.value, (int, float))
        ]
    
    def calculate_average(self) -> Optional[float]:
        """Calculate the average of numeric values.
        
        Returns:
            Average value or None if no numeric values exist
        """
        numeric_values = self.get_numeric_values()
        if not numeric_values:
            return None
        return sum(numeric_values) / len(numeric_values)
    
    def calculate_sum(self) -> Union[int, float]:
        """Calculate the sum of numeric values.
        
        Returns:
            Sum of numeric values
        """
        numeric_values = self.get_numeric_values()
        return sum(numeric_values)
    
    def find_min(self) -> Optional[Union[int, float]]:
        """Find the minimum numeric value.
        
        Returns:
            Minimum value or None if no numeric values exist
        """
        numeric_values = self.get_numeric_values()
        if not numeric_values:
            return None
        return min(numeric_values)
    
    def find_max(self) -> Optional[Union[int, float]]:
        """Find the maximum numeric value.
        
        Returns:
            Maximum value or None if no numeric values exist
        """
        numeric_values = self.get_numeric_values()
        if not numeric_values:
            return None
        return max(numeric_values)