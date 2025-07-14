"""Tests for core functionality."""

import pytest
from my_package.core import Calculator, DataProcessor, DataPoint


class TestCalculator:
    """Test cases for Calculator class."""
    
    def test_add(self):
        """Test addition operation."""
        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0
        assert calc.add(0.5, 0.5) == 1.0
    
    def test_subtract(self):
        """Test subtraction operation."""
        calc = Calculator()
        assert calc.subtract(5, 3) == 2
        assert calc.subtract(1, 1) == 0
        assert calc.subtract(0.5, 0.3) == pytest.approx(0.2)
    
    def test_multiply(self):
        """Test multiplication operation."""
        calc = Calculator()
        assert calc.multiply(2, 3) == 6
        assert calc.multiply(-2, 3) == -6
        assert calc.multiply(0.5, 2) == 1.0
    
    def test_divide(self):
        """Test division operation."""
        calc = Calculator()
        assert calc.divide(6, 3) == 2
        assert calc.divide(1, 2) == 0.5
        assert calc.divide(-6, 3) == -2
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(1, 0)
    
    def test_power(self):
        """Test power operation."""
        calc = Calculator()
        assert calc.power(2, 3) == 8
        assert calc.power(5, 0) == 1
        assert calc.power(4, 0.5) == 2.0


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.count() == 0
        assert processor.get_data_points() == []
    
    def test_add_data_point(self):
        """Test adding data points."""
        processor = DataProcessor()
        data_point = DataPoint(value=10, metadata={"test": True})
        
        processor.add_data_point(data_point)
        assert processor.count() == 1
        assert processor.get_data_points()[0].value == 10
    
    def test_filter_by_value(self):
        """Test filtering data points by value."""
        processor = DataProcessor()
        
        # Add test data
        processor.add_data_point(DataPoint(value=5))
        processor.add_data_point(DataPoint(value=10))
        processor.add_data_point(DataPoint(value=15))
        processor.add_data_point(DataPoint(value="text"))  # Non-numeric
        
        # Filter by minimum value
        filtered = processor.filter_by_value(10)
        assert len(filtered) == 2
        assert all(dp.value >= 10 for dp in filtered if isinstance(dp.value, (int, float)))
    
    def test_get_numeric_values(self):
        """Test getting numeric values only."""
        processor = DataProcessor()
        
        processor.add_data_point(DataPoint(value=5))
        processor.add_data_point(DataPoint(value=10.5))
        processor.add_data_point(DataPoint(value="text"))
        
        numeric_values = processor.get_numeric_values()
        assert numeric_values == [5, 10.5]
    
    def test_calculate_average(self):
        """Test calculating average of numeric values."""
        processor = DataProcessor()
        
        # Empty processor
        assert processor.calculate_average() is None
        
        # Add numeric values
        processor.add_data_point(DataPoint(value=10))
        processor.add_data_point(DataPoint(value=20))
        processor.add_data_point(DataPoint(value=30))
        
        assert processor.calculate_average() == 20.0
    
    def test_calculate_sum(self):
        """Test calculating sum of numeric values."""
        processor = DataProcessor()
        
        # Empty processor
        assert processor.calculate_sum() == 0
        
        # Add numeric values
        processor.add_data_point(DataPoint(value=10))
        processor.add_data_point(DataPoint(value=20))
        processor.add_data_point(DataPoint(value=30))
        
        assert processor.calculate_sum() == 60
    
    def test_find_min_max(self):
        """Test finding minimum and maximum values."""
        processor = DataProcessor()
        
        # Empty processor
        assert processor.find_min() is None
        assert processor.find_max() is None
        
        # Add numeric values
        processor.add_data_point(DataPoint(value=10))
        processor.add_data_point(DataPoint(value=5))
        processor.add_data_point(DataPoint(value=20))
        
        assert processor.find_min() == 5
        assert processor.find_max() == 20
    
    def test_clear(self):
        """Test clearing all data points."""
        processor = DataProcessor()
        
        processor.add_data_point(DataPoint(value=10))
        processor.add_data_point(DataPoint(value=20))
        assert processor.count() == 2
        
        processor.clear()
        assert processor.count() == 0
        assert processor.get_data_points() == []


class TestDataPoint:
    """Test cases for DataPoint class."""
    
    def test_initialization(self):
        """Test DataPoint initialization."""
        dp = DataPoint(value=10)
        assert dp.value == 10
        assert dp.metadata == {}
        assert dp.timestamp is None
    
    def test_with_metadata(self):
        """Test DataPoint with metadata."""
        metadata = {"source": "test", "priority": "high"}
        dp = DataPoint(value=10, metadata=metadata)
        assert dp.metadata == metadata
    
    def test_with_timestamp(self):
        """Test DataPoint with timestamp."""
        timestamp = "2023-01-01T00:00:00Z"
        dp = DataPoint(value=10, timestamp=timestamp)
        assert dp.timestamp == timestamp
    
    def test_different_value_types(self):
        """Test DataPoint with different value types."""
        # Integer
        dp_int = DataPoint(value=10)
        assert dp_int.value == 10
        
        # Float
        dp_float = DataPoint(value=10.5)
        assert dp_float.value == 10.5
        
        # String
        dp_str = DataPoint(value="test")
        assert dp_str.value == "test"