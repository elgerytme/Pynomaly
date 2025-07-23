"""Tests for data type value objects."""

import pytest
# TODO: Replace with proper relative imports or shared interfaces
# from packages.data.data.domain.value_objects.data_type import DataType, PrimitiveDataType
from ....data.domain.value_objects.data_type import DataType, PrimitiveDataType


class TestDataType:
    """Test cases for DataType value object."""
    
    def test_create_string_type(self):
        """Test creating a string data type."""
        data_type = DataType(
            primitive_type=PrimitiveDataType.STRING,
            max_length=255,
            nullable=True
        )
        
        assert data_type.primitive_type == PrimitiveDataType.STRING
        assert data_type.max_length == 255
        assert data_type.nullable is True
        assert not data_type.is_numeric()
        assert not data_type.is_temporal()
    
    def test_create_integer_type(self):
        """Test creating an integer data type."""
        data_type = DataType(
            primitive_type=PrimitiveDataType.INTEGER,
            precision=10,
            nullable=False
        )
        
        assert data_type.primitive_type == PrimitiveDataType.INTEGER
        assert data_type.precision == 10
        assert data_type.nullable is False
        assert data_type.is_numeric()
        assert not data_type.is_temporal()
    
    def test_create_float_type_with_scale(self):
        """Test creating a float data type with precision and scale."""
        data_type = DataType(
            primitive_type=PrimitiveDataType.FLOAT,
            precision=10,
            scale=2,
            nullable=True
        )
        
        assert data_type.primitive_type == PrimitiveDataType.FLOAT
        assert data_type.precision == 10
        assert data_type.scale == 2
        assert data_type.is_numeric()
    
    def test_create_datetime_type(self):
        """Test creating a datetime data type."""
        data_type = DataType(
            primitive_type=PrimitiveDataType.DATETIME,
            nullable=False
        )
        
        assert data_type.primitive_type == PrimitiveDataType.DATETIME
        assert data_type.is_temporal()
        assert not data_type.is_numeric()
    
    def test_scale_validation(self):
        """Test that scale cannot exceed precision."""
        with pytest.raises(ValueError, match="Scale cannot exceed precision"):
            DataType(
                primitive_type=PrimitiveDataType.FLOAT,
                precision=5,
                scale=10  # Scale > precision
            )
    
    def test_max_length_validation_for_inappropriate_type(self):
        """Test that max_length is only valid for string/binary types."""
        with pytest.raises(ValueError, match="max_length not applicable"):
            DataType(
                primitive_type=PrimitiveDataType.INTEGER,
                max_length=10  # Not applicable for integers
            )
    
    def test_numeric_type_detection(self):
        """Test numeric type detection."""
        int_type = DataType(primitive_type=PrimitiveDataType.INTEGER)
        float_type = DataType(primitive_type=PrimitiveDataType.FLOAT)
        string_type = DataType(primitive_type=PrimitiveDataType.STRING)
        
        assert int_type.is_numeric()
        assert float_type.is_numeric()
        assert not string_type.is_numeric()
    
    def test_temporal_type_detection(self):
        """Test temporal type detection."""
        date_type = DataType(primitive_type=PrimitiveDataType.DATE)
        datetime_type = DataType(primitive_type=PrimitiveDataType.DATETIME)
        timestamp_type = DataType(primitive_type=PrimitiveDataType.TIMESTAMP)
        string_type = DataType(primitive_type=PrimitiveDataType.STRING)
        
        assert date_type.is_temporal()
        assert datetime_type.is_temporal()
        assert timestamp_type.is_temporal()
        assert not string_type.is_temporal()
    
    def test_type_compatibility(self):
        """Test data type compatibility checking."""
        int_type = DataType(primitive_type=PrimitiveDataType.INTEGER)
        float_type = DataType(primitive_type=PrimitiveDataType.FLOAT)
        string_type = DataType(primitive_type=PrimitiveDataType.STRING)
        date_type = DataType(primitive_type=PrimitiveDataType.DATE)
        datetime_type = DataType(primitive_type=PrimitiveDataType.DATETIME)
        
        # Same types are compatible
        assert int_type.is_compatible_with(int_type)
        
        # Numeric types are compatible
        assert int_type.is_compatible_with(float_type)
        assert float_type.is_compatible_with(int_type)
        
        # Temporal types are compatible
        assert date_type.is_compatible_with(datetime_type)
        assert datetime_type.is_compatible_with(date_type)
        
        # Different type families are not compatible
        assert not string_type.is_compatible_with(int_type)
        assert not date_type.is_compatible_with(string_type)