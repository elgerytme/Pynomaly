"""Schema validation and compliance checking service."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from jsonschema import validate, ValidationError as JsonSchemaValidationError, Draft7Validator

from ...domain.services.validation_engine import (
    ValidationRule, ValidationResult, ValidationError, ValidationSeverity, 
    ValidationCategory, ValidationContext, ValidationMetrics
)

logger = logging.getLogger(__name__)


class SchemaType(str, Enum):
    """Schema definition types."""
    JSON_SCHEMA = "json_schema"
    PANDAS_SCHEMA = "pandas_schema"
    GREAT_EXPECTATIONS = "great_expectations"
    CUSTOM_SCHEMA = "custom_schema"


class DataTypeConstraint(str, Enum):
    """Data type constraint definitions."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    UUID_TYPE = "uuid"
    JSON_TYPE = "json"


class ColumnConstraint(BaseModel):
    """Column-level constraints."""
    name: str
    data_type: DataTypeConstraint
    nullable: bool = True
    unique: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    format_constraint: Optional[str] = None
    default_value: Optional[Any] = None
    description: Optional[str] = None
    
    class Config:
        use_enum_values = True


class TableConstraint(BaseModel):
    """Table-level constraints."""
    name: str
    constraint_type: str  # 'primary_key', 'foreign_key', 'unique', 'check'
    columns: List[str]
    referenced_table: Optional[str] = None
    referenced_columns: Optional[List[str]] = None
    check_expression: Optional[str] = None
    description: Optional[str] = None


class SchemaDefinition(BaseModel):
    """Complete schema definition."""
    schema_id: UUID = Field(default_factory=uuid4)
    name: str
    version: str = "1.0.0"
    schema_type: SchemaType
    
    # Column definitions
    columns: List[ColumnConstraint]
    required_columns: Set[str] = Field(default_factory=set)
    
    # Table constraints
    table_constraints: List[TableConstraint] = Field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    # Schema-level rules
    allow_extra_columns: bool = False
    strict_mode: bool = True
    
    class Config:
        use_enum_values = True
    
    @validator('required_columns', pre=True)
    def validate_required_columns(cls, v):
        if isinstance(v, list):
            return set(v)
        return v
    
    def get_column_by_name(self, name: str) -> Optional[ColumnConstraint]:
        """Get column constraint by name."""
        for column in self.columns:
            if column.name == name:
                return column
        return None
    
    def get_required_columns(self) -> Set[str]:
        """Get list of required (non-nullable) columns."""
        required = set(self.required_columns)
        for column in self.columns:
            if not column.nullable:
                required.add(column.name)
        return required
    
    def get_unique_columns(self) -> List[str]:
        """Get list of columns with unique constraints."""
        return [col.name for col in self.columns if col.unique]
    
    def get_primary_key_columns(self) -> List[str]:
        """Get primary key columns."""
        for constraint in self.table_constraints:
            if constraint.constraint_type == 'primary_key':
                return constraint.columns
        return []


class SchemaValidationRule(ValidationRule):
    """Schema validation rule implementation."""
    
    def __init__(
        self,
        rule_id: str,
        schema_definition: SchemaDefinition,
        name: str = "Schema Validation",
        description: str = "Validates data against schema definition",
        **kwargs
    ):
        super().__init__(
            rule_id, 
            name, 
            description, 
            category=ValidationCategory.DATA_TYPE,
            **kwargs
        )
        self.schema = schema_definition
        self._type_validators = self._initialize_type_validators()
    
    def validate_record(self, record: Dict[str, Any], row_index: int) -> bool:
        """Validate a single record against schema."""
        passed = True
        
        # Check required columns
        required_columns = self.schema.get_required_columns()
        for req_col in required_columns:
            if req_col not in record or pd.isna(record[req_col]):
                self.add_error(
                    row_index=row_index,
                    column_name=req_col,
                    invalid_value=record.get(req_col),
                    message=f"Required column '{req_col}' is missing or null",
                    error_code="REQUIRED_COLUMN_MISSING"
                )
                passed = False
        
        # Validate each column
        for column_name, value in record.items():
            column_constraint = self.schema.get_column_by_name(column_name)
            
            if column_constraint is None:
                if not self.schema.allow_extra_columns:
                    self.add_error(
                        row_index=row_index,
                        column_name=column_name,
                        invalid_value=value,
                        message=f"Column '{column_name}' is not defined in schema",
                        error_code="UNDEFINED_COLUMN"
                    )
                    if self.schema.strict_mode:
                        passed = False
                continue
            
            # Validate column value
            if not self._validate_column_value(value, column_constraint, row_index):
                passed = False
        
        return passed
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate entire dataset against schema."""
        passed = True
        
        # Check for missing required columns
        required_columns = self.schema.get_required_columns()
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            self.add_error(
                row_index=None,
                column_name=None,
                invalid_value=list(missing_columns),
                message=f"Required columns missing: {missing_columns}",
                error_code="MISSING_REQUIRED_COLUMNS",
                context={'missing_columns': list(missing_columns)}
            )
            passed = False
        
        # Check for undefined columns
        if not self.schema.allow_extra_columns:
            defined_columns = set(col.name for col in self.schema.columns)
            extra_columns = set(df.columns) - defined_columns
            if extra_columns:
                self.add_error(
                    row_index=None,
                    column_name=None,
                    invalid_value=list(extra_columns),
                    message=f"Undefined columns found: {extra_columns}",
                    error_code="UNDEFINED_COLUMNS",
                    context={'extra_columns': list(extra_columns)}
                )
                if self.schema.strict_mode:
                    passed = False
        
        # Validate data types at column level
        for column in self.schema.columns:
            if column.name not in df.columns:
                continue
            
            col_data = df[column.name]
            
            # Check data type compatibility
            if not self._validate_column_dtype(col_data, column):
                self.add_error(
                    row_index=None,
                    column_name=column.name,
                    invalid_value=str(col_data.dtype),
                    message=f"Column '{column.name}' has incompatible data type {col_data.dtype}",
                    error_code="INCOMPATIBLE_DTYPE",
                    expected_value=column.data_type.value
                )
                passed = False
            
            # Check unique constraints
            if column.unique:
                duplicates = col_data.duplicated()
                if duplicates.any():
                    duplicate_count = duplicates.sum()
                    self.add_error(
                        row_index=None,
                        column_name=column.name,
                        invalid_value=f"{duplicate_count} duplicates",
                        message=f"Unique constraint violated: {duplicate_count} duplicate values in column '{column.name}'",
                        error_code="UNIQUE_CONSTRAINT_VIOLATION",
                        context={'duplicate_count': duplicate_count}
                    )
                    passed = False
        
        # Validate table constraints
        for constraint in self.schema.table_constraints:
            if not self._validate_table_constraint(df, constraint):
                passed = False
        
        return passed
    
    def _validate_column_value(self, value: Any, column: ColumnConstraint, row_index: int) -> bool:
        """Validate individual column value."""
        passed = True
        
        # Handle null values
        if pd.isna(value):
            if not column.nullable:
                self.add_error(
                    row_index=row_index,
                    column_name=column.name,
                    invalid_value=value,
                    message=f"Column '{column.name}' cannot be null",
                    error_code="NULL_VALUE_NOT_ALLOWED"
                )
                passed = False
            return passed  # Skip other validations for null values
        
        # Validate data type
        type_validator = self._type_validators.get(column.data_type)
        if type_validator and not type_validator(value):
            self.add_error(
                row_index=row_index,
                column_name=column.name,
                invalid_value=value,
                message=f"Value '{value}' is not a valid {column.data_type.value}",
                error_code="INVALID_DATA_TYPE",
                expected_value=column.data_type.value
            )
            passed = False
        
        # Validate length constraints
        if column.data_type in [DataTypeConstraint.STRING, DataTypeConstraint.TEXT]:
            str_value = str(value)
            if column.min_length is not None and len(str_value) < column.min_length:
                self.add_error(
                    row_index=row_index,
                    column_name=column.name,
                    invalid_value=value,
                    message=f"Value length {len(str_value)} is below minimum {column.min_length}",
                    error_code="LENGTH_TOO_SHORT",
                    expected_value=f"min_length: {column.min_length}"
                )
                passed = False
            
            if column.max_length is not None and len(str_value) > column.max_length:
                self.add_error(
                    row_index=row_index,
                    column_name=column.name,
                    invalid_value=value,
                    message=f"Value length {len(str_value)} exceeds maximum {column.max_length}",
                    error_code="LENGTH_TOO_LONG",
                    expected_value=f"max_length: {column.max_length}"
                )
                passed = False
        
        # Validate numeric range constraints
        if column.data_type in [DataTypeConstraint.INTEGER, DataTypeConstraint.FLOAT, DataTypeConstraint.NUMERIC]:
            try:
                numeric_value = float(value)
                
                if column.min_value is not None and numeric_value < column.min_value:
                    self.add_error(
                        row_index=row_index,
                        column_name=column.name,
                        invalid_value=value,
                        message=f"Value {numeric_value} is below minimum {column.min_value}",
                        error_code="VALUE_BELOW_MINIMUM",
                        expected_value=f"min: {column.min_value}"
                    )
                    passed = False
                
                if column.max_value is not None and numeric_value > column.max_value:
                    self.add_error(
                        row_index=row_index,
                        column_name=column.name,
                        invalid_value=value,
                        message=f"Value {numeric_value} exceeds maximum {column.max_value}",
                        error_code="VALUE_ABOVE_MAXIMUM",
                        expected_value=f"max: {column.max_value}"
                    )
                    passed = False
                    
            except (ValueError, TypeError):
                # Type validation already handled above
                pass
        
        # Validate pattern constraints
        if column.pattern is not None:
            try:
                pattern = re.compile(column.pattern)
                if not pattern.match(str(value)):
                    self.add_error(
                        row_index=row_index,
                        column_name=column.name,
                        invalid_value=value,
                        message=f"Value '{value}' does not match pattern '{column.pattern}'",
                        error_code="PATTERN_MISMATCH",
                        expected_value=column.pattern
                    )
                    passed = False
            except re.error as e:
                logger.warning(f"Invalid regex pattern for column {column.name}: {e}")
        
        # Validate enum constraints
        if column.enum_values is not None:
            if value not in column.enum_values:
                self.add_error(
                    row_index=row_index,
                    column_name=column.name,
                    invalid_value=value,
                    message=f"Value '{value}' is not in allowed values: {column.enum_values}",
                    error_code="INVALID_ENUM_VALUE",
                    expected_value=column.enum_values
                )
                passed = False
        
        # Validate format constraints
        if column.format_constraint is not None:
            if not self._validate_format(value, column.format_constraint):
                self.add_error(
                    row_index=row_index,
                    column_name=column.name,
                    invalid_value=value,
                    message=f"Value '{value}' does not match format '{column.format_constraint}'",
                    error_code="FORMAT_CONSTRAINT_VIOLATION",
                    expected_value=column.format_constraint
                )
                passed = False
        
        return passed
    
    def _validate_column_dtype(self, col_data: pd.Series, column: ColumnConstraint) -> bool:
        """Validate pandas column data type."""
        expected_type = column.data_type
        actual_dtype = col_data.dtype
        
        if expected_type == DataTypeConstraint.INTEGER:
            return pd.api.types.is_integer_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.FLOAT:
            return pd.api.types.is_float_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.NUMERIC:
            return pd.api.types.is_numeric_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.STRING:
            return pd.api.types.is_string_dtype(actual_dtype) or pd.api.types.is_object_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.BOOLEAN:
            return pd.api.types.is_bool_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.DATETIME:
            return pd.api.types.is_datetime64_any_dtype(actual_dtype)
        elif expected_type == DataTypeConstraint.CATEGORICAL:
            return pd.api.types.is_categorical_dtype(actual_dtype)
        
        return True  # Default to allowing any type for other constraints
    
    def _validate_table_constraint(self, df: pd.DataFrame, constraint: TableConstraint) -> bool:
        """Validate table-level constraints."""
        passed = True
        
        if constraint.constraint_type == 'primary_key':
            # Check for nulls in primary key columns
            pk_columns = [col for col in constraint.columns if col in df.columns]
            if pk_columns:
                null_count = df[pk_columns].isnull().any(axis=1).sum()
                if null_count > 0:
                    self.add_error(
                        row_index=None,
                        column_name=','.join(pk_columns),
                        invalid_value=f"{null_count} null values",
                        message=f"Primary key columns {pk_columns} contain {null_count} null values",
                        error_code="PRIMARY_KEY_NULL_VIOLATION",
                        context={'pk_columns': pk_columns, 'null_count': null_count}
                    )
                    passed = False
                
                # Check for duplicates
                duplicate_count = df.duplicated(subset=pk_columns).sum()
                if duplicate_count > 0:
                    self.add_error(
                        row_index=None,
                        column_name=','.join(pk_columns),
                        invalid_value=f"{duplicate_count} duplicates",
                        message=f"Primary key columns {pk_columns} contain {duplicate_count} duplicate combinations",
                        error_code="PRIMARY_KEY_DUPLICATE_VIOLATION",
                        context={'pk_columns': pk_columns, 'duplicate_count': duplicate_count}
                    )
                    passed = False
        
        elif constraint.constraint_type == 'unique':
            # Check uniqueness constraint
            unique_columns = [col for col in constraint.columns if col in df.columns]
            if unique_columns:
                duplicate_count = df.duplicated(subset=unique_columns).sum()
                if duplicate_count > 0:
                    self.add_error(
                        row_index=None,
                        column_name=','.join(unique_columns),
                        invalid_value=f"{duplicate_count} duplicates",
                        message=f"Unique constraint violated: {duplicate_count} duplicate combinations in {unique_columns}",
                        error_code="UNIQUE_CONSTRAINT_VIOLATION",
                        context={'unique_columns': unique_columns, 'duplicate_count': duplicate_count}
                    )
                    passed = False
        
        elif constraint.constraint_type == 'foreign_key':
            # Foreign key validation would require reference data
            # For now, just log that foreign key validation is not implemented
            logger.info(f"Foreign key validation for {constraint.name} requires reference data - skipping")
        
        elif constraint.constraint_type == 'check':
            # Custom check constraint validation
            if constraint.check_expression:
                try:
                    # Simple expression evaluation - in production, use safer evaluation
                    check_result = df.eval(constraint.check_expression)
                    if hasattr(check_result, 'all') and not check_result.all():
                        violation_count = (~check_result).sum()
                        self.add_error(
                            row_index=None,
                            column_name=','.join(constraint.columns),
                            invalid_value=f"{violation_count} violations",
                            message=f"Check constraint '{constraint.name}' violated: {violation_count} records",
                            error_code="CHECK_CONSTRAINT_VIOLATION",
                            context={
                                'constraint_name': constraint.name,
                                'expression': constraint.check_expression,
                                'violation_count': violation_count
                            }
                        )
                        passed = False
                except Exception as e:
                    logger.warning(f"Error evaluating check constraint {constraint.name}: {e}")
        
        return passed
    
    def _initialize_type_validators(self) -> Dict[DataTypeConstraint, callable]:
        """Initialize type validation functions."""
        return {
            DataTypeConstraint.INTEGER: lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool),
            DataTypeConstraint.FLOAT: lambda x: isinstance(x, (float, np.floating)),
            DataTypeConstraint.NUMERIC: lambda x: isinstance(x, (int, float, np.number)) and not isinstance(x, bool),
            DataTypeConstraint.STRING: lambda x: isinstance(x, str),
            DataTypeConstraint.TEXT: lambda x: isinstance(x, str),
            DataTypeConstraint.BOOLEAN: lambda x: isinstance(x, (bool, np.bool_)),
            DataTypeConstraint.DATETIME: self._validate_datetime,
            DataTypeConstraint.DATE: self._validate_date,
            DataTypeConstraint.TIME: self._validate_time,
            DataTypeConstraint.EMAIL: self._validate_email,
            DataTypeConstraint.URL: self._validate_url,
            DataTypeConstraint.UUID_TYPE: self._validate_uuid,
            DataTypeConstraint.JSON_TYPE: self._validate_json,
            DataTypeConstraint.CATEGORICAL: lambda x: True  # Any value can be categorical
        }
    
    def _validate_datetime(self, value: Any) -> bool:
        """Validate datetime value."""
        try:
            pd.to_datetime(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_date(self, value: Any) -> bool:
        """Validate date value."""
        try:
            dt = pd.to_datetime(value)
            # Check if it's a date (no time component)
            return dt.time() == dt.time().replace(hour=0, minute=0, second=0, microsecond=0)
        except (ValueError, TypeError):
            return False
    
    def _validate_time(self, value: Any) -> bool:
        """Validate time value."""
        try:
            pd.to_datetime(value, format='%H:%M:%S')
            return True
        except (ValueError, TypeError):
            try:
                pd.to_datetime(value, format='%H:%M')
                return True
            except (ValueError, TypeError):
                return False
    
    def _validate_email(self, value: Any) -> bool:
        """Validate email format."""
        if not isinstance(value, str):
            return False
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, value))
    
    def _validate_url(self, value: Any) -> bool:
        """Validate URL format."""
        if not isinstance(value, str):
            return False
        
        url_pattern = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        return bool(re.match(url_pattern, value))
    
    def _validate_uuid(self, value: Any) -> bool:
        """Validate UUID format."""
        if not isinstance(value, str):
            return False
        
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, value.lower()))
    
    def _validate_json(self, value: Any) -> bool:
        """Validate JSON format."""
        try:
            import json
            if isinstance(value, str):
                json.loads(value)
            elif isinstance(value, (dict, list)):
                json.dumps(value)
            else:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_format(self, value: Any, format_constraint: str) -> bool:
        """Validate custom format constraints."""
        format_validators = {
            'email': self._validate_email,
            'url': self._validate_url,
            'uuid': self._validate_uuid,
            'json': self._validate_json,
            'date': self._validate_date,
            'time': self._validate_time,
            'datetime': self._validate_datetime
        }
        
        validator_func = format_validators.get(format_constraint)
        if validator_func:
            return validator_func(value)
        
        # If not a predefined format, treat as regex pattern
        try:
            pattern = re.compile(format_constraint)
            return bool(pattern.match(str(value)))
        except re.error:
            logger.warning(f"Invalid format constraint: {format_constraint}")
            return True


class SchemaComplianceChecker:
    """Check dataset compliance against multiple schema definitions."""
    
    def __init__(self):
        self.schemas: Dict[str, SchemaDefinition] = {}
    
    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register a schema definition."""
        self.schemas[schema.name] = schema
    
    def check_compliance(
        self,
        df: pd.DataFrame,
        schema_name: str,
        validation_context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """Check dataset compliance against a registered schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        rule = SchemaValidationRule(
            rule_id=f"schema_validation_{schema_name}",
            schema_definition=schema,
            name=f"Schema Validation: {schema_name}"
        )
        
        start_time = datetime.utcnow()
        
        # Validate dataset
        dataset_passed = rule.validate_dataset(df)
        
        # Validate records
        records_passed = 0
        records_processed = 0
        
        for index, row in df.iterrows():
            records_processed += 1
            record = row.to_dict()
            
            if rule.validate_record(record, index):
                records_passed += 1
        
        overall_passed = dataset_passed and (records_passed == records_processed)
        pass_rate = records_passed / records_processed if records_processed > 0 else 1.0
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        metrics = ValidationMetrics(
            total_records=len(df),
            records_processed=records_processed,
            records_passed=records_passed,
            records_failed=records_processed - records_passed,
            pass_rate=pass_rate,
            execution_time_ms=execution_time
        )
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            passed=overall_passed,
            metrics=metrics,
            errors=rule.get_errors(),
            context=validation_context or ValidationContext(dataset_name="unknown"),
            statistics={
                'schema_name': schema_name,
                'schema_version': schema.version,
                'columns_defined': len(schema.columns),
                'columns_present': len(set(df.columns) & set(col.name for col in schema.columns)),
                'extra_columns': len(set(df.columns) - set(col.name for col in schema.columns)),
                'missing_columns': len(set(col.name for col in schema.columns) - set(df.columns))
            }
        )
    
    def get_schema_summary(self, schema_name: str) -> Dict[str, Any]:
        """Get summary information about a schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        
        return {
            'schema_id': str(schema.schema_id),
            'name': schema.name,
            'version': schema.version,
            'schema_type': schema.schema_type.value,
            'total_columns': len(schema.columns),
            'required_columns': len(schema.get_required_columns()),
            'unique_columns': len(schema.get_unique_columns()),
            'primary_key_columns': schema.get_primary_key_columns(),
            'table_constraints': len(schema.table_constraints),
            'allow_extra_columns': schema.allow_extra_columns,
            'strict_mode': schema.strict_mode,
            'created_at': schema.created_at.isoformat(),
            'description': schema.description
        }
    
    def compare_schemas(self, schema_name1: str, schema_name2: str) -> Dict[str, Any]:
        """Compare two schema definitions."""
        if schema_name1 not in self.schemas or schema_name2 not in self.schemas:
            raise ValueError("One or both schemas not found")
        
        schema1 = self.schemas[schema_name1]
        schema2 = self.schemas[schema_name2]
        
        columns1 = {col.name: col for col in schema1.columns}
        columns2 = {col.name: col for col in schema2.columns}
        
        common_columns = set(columns1.keys()) & set(columns2.keys())
        only_in_schema1 = set(columns1.keys()) - set(columns2.keys())
        only_in_schema2 = set(columns2.keys()) - set(columns1.keys())
        
        differences = []
        for col_name in common_columns:
            col1 = columns1[col_name]
            col2 = columns2[col_name]
            
            if col1.data_type != col2.data_type:
                differences.append({
                    'column': col_name,
                    'difference_type': 'data_type',
                    'schema1_value': col1.data_type.value,
                    'schema2_value': col2.data_type.value
                })
            
            if col1.nullable != col2.nullable:
                differences.append({
                    'column': col_name,
                    'difference_type': 'nullable',
                    'schema1_value': col1.nullable,
                    'schema2_value': col2.nullable
                })
        
        return {
            'schema1': schema_name1,
            'schema2': schema_name2,
            'common_columns': list(common_columns),
            'only_in_schema1': list(only_in_schema1),
            'only_in_schema2': list(only_in_schema2),
            'column_differences': differences,
            'compatibility_score': len(common_columns) / max(len(columns1), len(columns2)) if columns1 or columns2 else 1.0
        }
    
    def generate_schema_from_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str,
        infer_constraints: bool = True
    ) -> SchemaDefinition:
        """Generate schema definition from DataFrame."""
        columns = []
        
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Infer data type
            data_type = self._infer_data_type(col_data)
            
            # Check nullability
            nullable = col_data.isnull().any()
            
            # Check uniqueness
            unique = not col_data.duplicated().any() if infer_constraints else False
            
            # Infer constraints
            constraints = {}
            if infer_constraints:
                constraints = self._infer_column_constraints(col_data, data_type)
            
            column_constraint = ColumnConstraint(
                name=col_name,
                data_type=data_type,
                nullable=nullable,
                unique=unique,
                **constraints
            )
            
            columns.append(column_constraint)
        
        schema = SchemaDefinition(
            name=schema_name,
            schema_type=SchemaType.CUSTOM_SCHEMA,
            columns=columns,
            description=f"Auto-generated schema from DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        
        self.register_schema(schema)
        return schema
    
    def _infer_data_type(self, col_data: pd.Series) -> DataTypeConstraint:
        """Infer data type from pandas Series."""
        dtype = col_data.dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            return DataTypeConstraint.INTEGER
        elif pd.api.types.is_float_dtype(dtype):
            return DataTypeConstraint.FLOAT
        elif pd.api.types.is_bool_dtype(dtype):
            return DataTypeConstraint.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return DataTypeConstraint.DATETIME
        elif pd.api.types.is_categorical_dtype(dtype):
            return DataTypeConstraint.CATEGORICAL
        else:
            # Check if string data looks like specific formats
            sample_values = col_data.dropna().head(100).astype(str)
            
            # Check for email pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if sample_values.str.match(email_pattern).mean() > 0.8:
                return DataTypeConstraint.EMAIL
            
            # Check for URL pattern
            url_pattern = r'^https?://'
            if sample_values.str.match(url_pattern).mean() > 0.8:
                return DataTypeConstraint.URL
            
            # Check for UUID pattern
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
            if sample_values.str.lower().str.match(uuid_pattern).mean() > 0.8:
                return DataTypeConstraint.UUID_TYPE
            
            return DataTypeConstraint.STRING
    
    def _infer_column_constraints(self, col_data: pd.Series, data_type: DataTypeConstraint) -> Dict[str, Any]:
        """Infer additional constraints for a column."""
        constraints = {}
        
        if data_type in [DataTypeConstraint.STRING, DataTypeConstraint.TEXT]:
            non_null_data = col_data.dropna().astype(str)
            if len(non_null_data) > 0:
                lengths = non_null_data.str.len()
                constraints['min_length'] = int(lengths.min())
                constraints['max_length'] = int(lengths.max())
        
        elif data_type in [DataTypeConstraint.INTEGER, DataTypeConstraint.FLOAT, DataTypeConstraint.NUMERIC]:
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                constraints['min_value'] = float(non_null_data.min())
                constraints['max_value'] = float(non_null_data.max())
        
        # Check for enum-like data (limited distinct values)
        unique_values = col_data.dropna().unique()
        if len(unique_values) <= 20:  # Arbitrary threshold for categorical data
            constraints['enum_values'] = unique_values.tolist()
        
        return constraints


class JSONSchemaValidator:
    """Validate data against JSON Schema definitions."""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_json_schema(self, schema_name: str, json_schema: Dict[str, Any]) -> None:
        """Register a JSON schema."""
        # Validate the schema itself
        Draft7Validator.check_schema(json_schema)
        self.schemas[schema_name] = json_schema
    
    def validate_records(
        self,
        records: List[Dict[str, Any]],
        schema_name: str
    ) -> List[ValidationError]:
        """Validate records against JSON schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        validator = Draft7Validator(schema)
        errors = []
        
        for idx, record in enumerate(records):
            try:
                validator.validate(record)
            except JsonSchemaValidationError as e:
                validation_error = ValidationError(
                    rule_id=f"json_schema_{schema_name}",
                    row_index=idx,
                    column_name=e.absolute_path[-1] if e.absolute_path else None,
                    invalid_value=record.get(e.absolute_path[-1]) if e.absolute_path else None,
                    error_message=e.message,
                    error_code="JSON_SCHEMA_VIOLATION",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_TYPE,
                    context={
                        'schema_path': list(e.absolute_path),
                        'validator': e.validator,
                        'validator_value': e.validator_value
                    }
                )
                errors.append(validation_error)
        
        return errors
    
    def generate_json_schema_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate JSON schema from DataFrame."""
        properties = {}
        required = []
        
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Determine JSON schema type
            if pd.api.types.is_integer_dtype(col_data.dtype):
                col_schema = {"type": "integer"}
            elif pd.api.types.is_float_dtype(col_data.dtype):
                col_schema = {"type": "number"}
            elif pd.api.types.is_bool_dtype(col_data.dtype):
                col_schema = {"type": "boolean"}
            else:
                col_schema = {"type": "string"}
            
            # Add constraints
            if not col_data.isnull().any():
                required.append(col_name)
            
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                col_schema["minimum"] = float(col_data.min())
                col_schema["maximum"] = float(col_data.max())
            
            properties[col_name] = col_schema
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }