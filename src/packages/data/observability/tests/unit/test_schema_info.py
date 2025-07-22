"""Unit tests for SchemaInfo value object."""

import pytest

from data_observability.domain.value_objects.schema_info import ColumnInfo, SchemaInfo


class TestColumnInfo:
    """Test cases for ColumnInfo value object."""
    
    def test_create_column_info_minimal(self):
        """Test creating column info with minimal data."""
        column = ColumnInfo(name="id", data_type="integer")
        
        assert column.name == "id"
        assert column.data_type == "integer"
        assert column.nullable is True  # Default value
        assert column.default_value is None
        assert column.description is None
        assert column.constraints == []
    
    def test_create_column_info_full(self):
        """Test creating column info with full data."""
        column = ColumnInfo(
            name="user_id",
            data_type="bigint",
            nullable=False,
            default_value="0",
            description="User identifier",
            constraints=["PRIMARY KEY", "AUTO_INCREMENT"]
        )
        
        assert column.name == "user_id"
        assert column.data_type == "bigint"
        assert column.nullable is False
        assert column.default_value == "0"
        assert column.description == "User identifier"
        assert column.constraints == ["PRIMARY KEY", "AUTO_INCREMENT"]
    
    def test_column_info_equality(self):
        """Test column info equality."""
        column1 = ColumnInfo(
            name="name",
            data_type="varchar(255)",
            nullable=True
        )
        
        column2 = ColumnInfo(
            name="name",
            data_type="varchar(255)",
            nullable=True
        )
        
        assert column1 == column2
    
    def test_column_info_inequality(self):
        """Test column info inequality."""
        column1 = ColumnInfo(name="name", data_type="varchar(255)")
        column2 = ColumnInfo(name="email", data_type="varchar(255)")
        
        assert column1 != column2
    
    def test_column_info_validation(self):
        """Test column info validation."""
        # Test empty name
        with pytest.raises(ValueError):
            ColumnInfo(name="", data_type="integer")
        
        # Test empty data type
        with pytest.raises(ValueError):
            ColumnInfo(name="id", data_type="")
    
    def test_column_info_to_dict(self):
        """Test converting column info to dictionary."""
        column = ColumnInfo(
            name="created_at",
            data_type="timestamp",
            nullable=False,
            default_value="CURRENT_TIMESTAMP",
            description="Record creation time",
            constraints=["NOT NULL"]
        )
        
        column_dict = column.to_dict()
        
        assert column_dict["name"] == "created_at"
        assert column_dict["data_type"] == "timestamp"
        assert column_dict["nullable"] is False
        assert column_dict["default_value"] == "CURRENT_TIMESTAMP"
        assert column_dict["description"] == "Record creation time"
        assert column_dict["constraints"] == ["NOT NULL"]
    
    def test_column_info_from_dict(self):
        """Test creating column info from dictionary."""
        column_dict = {
            "name": "updated_at",
            "data_type": "timestamp",
            "nullable": True,
            "default_value": None,
            "description": "Record update time",
            "constraints": []
        }
        
        column = ColumnInfo.from_dict(column_dict)
        
        assert column.name == "updated_at"
        assert column.data_type == "timestamp"
        assert column.nullable is True
        assert column.default_value is None
        assert column.description == "Record update time"
        assert column.constraints == []


class TestSchemaInfo:
    """Test cases for SchemaInfo value object."""
    
    def test_create_schema_info_empty(self):
        """Test creating empty schema info."""
        schema = SchemaInfo(columns=[])
        
        assert schema.columns == []
        assert schema.version is None
        assert schema.last_modified is None
    
    def test_create_schema_info_with_columns(self):
        """Test creating schema info with columns."""
        columns = [
            ColumnInfo(name="id", data_type="bigint", nullable=False),
            ColumnInfo(name="name", data_type="varchar(255)", nullable=True),
            ColumnInfo(name="email", data_type="varchar(255)", nullable=False),
        ]
        
        schema = SchemaInfo(
            columns=columns,
            version="1.0",
            last_modified="2024-01-15T10:30:00Z"
        )
        
        assert len(schema.columns) == 3
        assert schema.columns[0].name == "id"
        assert schema.columns[1].name == "name"
        assert schema.columns[2].name == "email"
        assert schema.version == "1.0"
        assert schema.last_modified == "2024-01-15T10:30:00Z"
    
    def test_get_column_by_name(self):
        """Test getting column by name."""
        columns = [
            ColumnInfo(name="id", data_type="bigint"),
            ColumnInfo(name="name", data_type="varchar(255)"),
        ]
        
        schema = SchemaInfo(columns=columns)
        
        id_column = schema.get_column_by_name("id")
        assert id_column is not None
        assert id_column.name == "id"
        assert id_column.data_type == "bigint"
        
        nonexistent_column = schema.get_column_by_name("nonexistent")
        assert nonexistent_column is None
    
    def test_get_column_names(self):
        """Test getting all column names."""
        columns = [
            ColumnInfo(name="user_id", data_type="bigint"),
            ColumnInfo(name="username", data_type="varchar(100)"),
            ColumnInfo(name="created_at", data_type="timestamp"),
        ]
        
        schema = SchemaInfo(columns=columns)
        column_names = schema.get_column_names()
        
        assert column_names == ["user_id", "username", "created_at"]
    
    def test_get_nullable_columns(self):
        """Test getting nullable columns."""
        columns = [
            ColumnInfo(name="id", data_type="bigint", nullable=False),
            ColumnInfo(name="name", data_type="varchar(255)", nullable=True),
            ColumnInfo(name="email", data_type="varchar(255)", nullable=False),
            ColumnInfo(name="phone", data_type="varchar(20)", nullable=True),
        ]
        
        schema = SchemaInfo(columns=columns)
        nullable_columns = schema.get_nullable_columns()
        
        assert len(nullable_columns) == 2
        assert nullable_columns[0].name == "name"
        assert nullable_columns[1].name == "phone"
    
    def test_get_required_columns(self):
        """Test getting required (non-nullable) columns."""
        columns = [
            ColumnInfo(name="id", data_type="bigint", nullable=False),
            ColumnInfo(name="name", data_type="varchar(255)", nullable=True),
            ColumnInfo(name="email", data_type="varchar(255)", nullable=False),
        ]
        
        schema = SchemaInfo(columns=columns)
        required_columns = schema.get_required_columns()
        
        assert len(required_columns) == 2
        assert required_columns[0].name == "id"
        assert required_columns[1].name == "email"
    
    def test_schema_info_equality(self):
        """Test schema info equality."""
        columns1 = [
            ColumnInfo(name="id", data_type="bigint"),
            ColumnInfo(name="name", data_type="varchar(255)"),
        ]
        
        columns2 = [
            ColumnInfo(name="id", data_type="bigint"),
            ColumnInfo(name="name", data_type="varchar(255)"),
        ]
        
        schema1 = SchemaInfo(columns=columns1, version="1.0")
        schema2 = SchemaInfo(columns=columns2, version="1.0")
        
        assert schema1 == schema2
    
    def test_schema_info_inequality(self):
        """Test schema info inequality."""
        columns1 = [ColumnInfo(name="id", data_type="bigint")]
        columns2 = [ColumnInfo(name="user_id", data_type="bigint")]
        
        schema1 = SchemaInfo(columns=columns1)
        schema2 = SchemaInfo(columns=columns2)
        
        assert schema1 != schema2
    
    def test_schema_info_to_dict(self):
        """Test converting schema info to dictionary."""
        columns = [
            ColumnInfo(name="id", data_type="bigint", nullable=False),
            ColumnInfo(name="name", data_type="varchar(255)", nullable=True),
        ]
        
        schema = SchemaInfo(
            columns=columns,
            version="2.0",
            last_modified="2024-01-15T10:30:00Z"
        )
        
        schema_dict = schema.to_dict()
        
        assert "columns" in schema_dict
        assert len(schema_dict["columns"]) == 2
        assert schema_dict["columns"][0]["name"] == "id"
        assert schema_dict["columns"][1]["name"] == "name"
        assert schema_dict["version"] == "2.0"
        assert schema_dict["last_modified"] == "2024-01-15T10:30:00Z"
    
    def test_schema_info_from_dict(self):
        """Test creating schema info from dictionary."""
        schema_dict = {
            "columns": [
                {
                    "name": "product_id",
                    "data_type": "uuid",
                    "nullable": False,
                    "default_value": None,
                    "description": "Product identifier",
                    "constraints": ["PRIMARY KEY"]
                },
                {
                    "name": "product_name",
                    "data_type": "varchar(200)",
                    "nullable": False,
                    "default_value": None,
                    "description": "Product name",
                    "constraints": ["NOT NULL"]
                }
            ],
            "version": "3.1",
            "last_modified": "2024-02-01T14:00:00Z"
        }
        
        schema = SchemaInfo.from_dict(schema_dict)
        
        assert len(schema.columns) == 2
        assert schema.columns[0].name == "product_id"
        assert schema.columns[0].data_type == "uuid"
        assert schema.columns[0].nullable is False
        assert schema.columns[1].name == "product_name"
        assert schema.version == "3.1"
        assert schema.last_modified == "2024-02-01T14:00:00Z"
    
    def test_add_column(self):
        """Test adding column to schema."""
        schema = SchemaInfo(columns=[])
        
        new_column = ColumnInfo(name="status", data_type="varchar(20)")
        schema.add_column(new_column)
        
        assert len(schema.columns) == 1
        assert schema.columns[0].name == "status"
    
    def test_remove_column(self):
        """Test removing column from schema."""
        columns = [
            ColumnInfo(name="id", data_type="bigint"),
            ColumnInfo(name="name", data_type="varchar(255)"),
            ColumnInfo(name="temp", data_type="varchar(50)"),
        ]
        
        schema = SchemaInfo(columns=columns)
        
        removed = schema.remove_column("temp")
        assert removed is True
        assert len(schema.columns) == 2
        assert schema.get_column_by_name("temp") is None
        
        # Try to remove non-existent column
        removed = schema.remove_column("nonexistent")
        assert removed is False