"""Data validation domain service."""

from typing import Dict, Any, List, Optional
from uuid import UUID
from ..entities.data_set import DataSet
from ..entities.data_element import DataElement
from ..value_objects.data_schema import DataSchema
from ..value_objects.data_classification import DataQualityDimension


class DataValidationService:
    """Domain service for data validation operations."""
    
    def validate_dataset_against_schema(self, dataset: DataSet, actual_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate dataset data against its schema definition."""
        validation_results = {
            'is_valid': True,
            'validation_timestamp': dataset.updated_at.isoformat() if dataset.updated_at else None,
            'schema_violations': [],
            'quality_metrics': {},
            'record_errors': [],
            'summary': {}
        }
        
        if not actual_data:
            validation_results['is_valid'] = False
            validation_results['schema_violations'].append('No data provided for validation')
            return validation_results
        
        schema = dataset.schema
        total_records = len(actual_data)
        
        # Schema structure validation
        schema_validation = self._validate_schema_structure(schema, actual_data[0])
        validation_results['schema_violations'].extend(schema_validation['violations'])
        validation_results['is_valid'] &= schema_validation['is_valid']
        
        # Data quality metrics calculation
        quality_metrics = self._calculate_quality_metrics(schema, actual_data)
        validation_results['quality_metrics'] = quality_metrics
        
        # Record-level validation
        record_errors = self._validate_records(schema, actual_data)
        validation_results['record_errors'] = record_errors[:100]  # Limit error count
        
        # Summary statistics
        validation_results['summary'] = {
            'total_records': total_records,
            'valid_records': total_records - len(record_errors),
            'error_rate': len(record_errors) / total_records if total_records > 0 else 0,
            'schema_compliance': len(validation_results['schema_violations']) == 0
        }
        
        # Overall validation result
        error_threshold = 0.05  # 5% error threshold
        validation_results['is_valid'] = (
            validation_results['is_valid'] and
            validation_results['summary']['error_rate'] <= error_threshold
        )
        
        return validation_results
    
    def validate_element_data(self, element: DataElement, values: List[Any]) -> Dict[str, Any]:
        """Validate element data against its definition."""
        validation_results = {
            'is_valid': True,
            'element_name': element.name,
            'validation_timestamp': element.updated_at.isoformat() if element.updated_at else None,
            'violations': [],
            'statistics': {},
            'quality_score': 1.0
        }
        
        if not values:
            if element.is_required:
                validation_results['is_valid'] = False
                validation_results['violations'].append('Required element has no values')
            return validation_results
        
        total_values = len(values)
        null_count = sum(1 for v in values if v is None)
        non_null_values = [v for v in values if v is not None]
        
        # Required field validation
        if element.is_required and null_count == total_values:
            validation_results['is_valid'] = False
            validation_results['violations'].append('Required element cannot be entirely null')
        
        # Data type validation
        type_validation = self._validate_element_data_types(element, non_null_values)
        validation_results['violations'].extend(type_validation['violations'])
        validation_results['is_valid'] &= type_validation['is_valid']
        
        # Uniqueness validation
        if element.is_unique:
            unique_count = len(set(str(v) for v in non_null_values))
            if unique_count != len(non_null_values):
                validation_results['is_valid'] = False
                validation_results['violations'].append('Unique constraint violation detected')
        
        # Allowed values validation
        if element.allowed_values:
            invalid_values = [v for v in non_null_values if str(v) not in element.allowed_values]
            if invalid_values:
                validation_results['is_valid'] = False
                validation_results['violations'].append(f'Invalid values found: {invalid_values[:5]}')
        
        # Statistics calculation
        validation_results['statistics'] = {
            'total_count': total_values,
            'null_count': null_count,
            'distinct_count': len(set(str(v) for v in values)),
            'completeness': (total_values - null_count) / total_values if total_values > 0 else 0
        }
        
        # Quality score calculation
        validation_results['quality_score'] = self._calculate_element_quality_score(
            validation_results['statistics'], 
            len(validation_results['violations'])
        )
        
        return validation_results
    
    def validate_schema_compatibility(self, source_schema: DataSchema, target_schema: DataSchema) -> Dict[str, Any]:
        """Validate compatibility between two schemas."""
        compatibility_results = {
            'is_compatible': True,
            'compatibility_issues': [],
            'missing_fields': [],
            'type_mismatches': [],
            'constraint_conflicts': []
        }
        
        # Check for missing required fields
        for field_name, target_field in target_schema.fields.items():
            if target_field.is_required and field_name not in source_schema.fields:
                compatibility_results['missing_fields'].append(field_name)
                compatibility_results['is_compatible'] = False
        
        # Check for data type compatibility
        for field_name, source_field in source_schema.fields.items():
            if field_name in target_schema.fields:
                target_field = target_schema.fields[field_name]
                if not source_field.data_type.is_compatible_with(target_field.data_type):
                    compatibility_results['type_mismatches'].append({
                        'field': field_name,
                        'source_type': source_field.data_type.primitive_type.value,
                        'target_type': target_field.data_type.primitive_type.value
                    })
                    compatibility_results['is_compatible'] = False
        
        # Check constraint conflicts
        for field_name in set(source_schema.fields.keys()) & set(target_schema.fields.keys()):
            source_field = source_schema.fields[field_name]
            target_field = target_schema.fields[field_name]
            
            # Required vs optional conflict
            if source_field.is_required != target_field.is_required:
                compatibility_results['constraint_conflicts'].append({
                    'field': field_name,
                    'conflict': 'required_mismatch',
                    'source': source_field.is_required,
                    'target': target_field.is_required
                })
        
        return compatibility_results
    
    def _validate_schema_structure(self, schema: DataSchema, sample_record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema structure against sample data."""
        result = {'is_valid': True, 'violations': []}
        
        # Check for missing required fields
        for field_name, field_schema in schema.fields.items():
            if field_schema.is_required and field_name not in sample_record:
                result['violations'].append(f'Required field {field_name} missing from data')
                result['is_valid'] = False
        
        # Check for unexpected fields
        schema_fields = set(schema.fields.keys())
        data_fields = set(sample_record.keys())
        unexpected_fields = data_fields - schema_fields
        
        if unexpected_fields:
            result['violations'].append(f'Unexpected fields in data: {list(unexpected_fields)}')
        
        return result
    
    def _calculate_quality_metrics(self, schema: DataSchema, data: List[Dict[str, Any]]) -> Dict[DataQualityDimension, float]:
        """Calculate data quality metrics."""
        metrics = {}
        
        if not data:
            return metrics
        
        total_records = len(data)
        total_fields = len(schema.fields)
        
        # Completeness: percentage of non-null values
        total_cells = total_records * total_fields
        null_cells = 0
        
        for record in data:
            for field_name in schema.fields.keys():
                if record.get(field_name) is None:
                    null_cells += 1
        
        metrics[DataQualityDimension.COMPLETENESS] = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
        
        # Validity: percentage of values that pass data type validation
        valid_cells = 0
        for record in data:
            for field_name, field_schema in schema.fields.items():
                value = record.get(field_name)
                if value is not None and self._is_valid_type(value, field_schema.data_type):
                    valid_cells += 1
                elif value is None and not field_schema.is_required:
                    valid_cells += 1
        
        metrics[DataQualityDimension.VALIDITY] = valid_cells / total_cells if total_cells > 0 else 0
        
        # Uniqueness: percentage of unique records (simplified)
        record_strings = [str(sorted(record.items())) for record in data]
        unique_records = len(set(record_strings))
        metrics[DataQualityDimension.UNIQUENESS] = unique_records / total_records if total_records > 0 else 1.0
        
        # Consistency: all records have the same field structure
        first_record_keys = set(data[0].keys())
        consistent_records = sum(1 for record in data if set(record.keys()) == first_record_keys)
        metrics[DataQualityDimension.CONSISTENCY] = consistent_records / total_records if total_records > 0 else 1.0
        
        return metrics
    
    def _validate_records(self, schema: DataSchema, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate individual records against schema."""
        record_errors = []
        
        for index, record in enumerate(data):
            record_issues = []
            
            for field_name, field_schema in schema.fields.items():
                value = record.get(field_name)
                
                # Required field validation
                if field_schema.is_required and value is None:
                    record_issues.append(f'Required field {field_name} is null')
                
                # Data type validation
                if value is not None and not self._is_valid_type(value, field_schema.data_type):
                    record_issues.append(f'Field {field_name} has invalid type')
                
                # Enum validation
                if (value is not None and field_schema.enum_values and 
                    str(value) not in field_schema.enum_values):
                    record_issues.append(f'Field {field_name} has invalid enum value')
            
            if record_issues:
                record_errors.append({
                    'record_index': index,
                    'issues': record_issues
                })
        
        return record_errors
    
    def _validate_element_data_types(self, element: DataElement, values: List[Any]) -> Dict[str, Any]:
        """Validate data types for element values."""
        result = {'is_valid': True, 'violations': []}
        
        invalid_count = 0
        for value in values:
            if not self._is_valid_type(value, element.data_type):
                invalid_count += 1
        
        if invalid_count > 0:
            result['is_valid'] = False
            result['violations'].append(f'{invalid_count} values do not match expected type {element.data_type.primitive_type.value}')
        
        return result
    
    def _is_valid_type(self, value: Any, data_type) -> bool:
        """Check if value matches expected data type."""
        from ..value_objects.data_type import PrimitiveDataType
        
        if value is None:
            return data_type.nullable
        
        type_mapping = {
            PrimitiveDataType.STRING: str,
            PrimitiveDataType.INTEGER: int,
            PrimitiveDataType.FLOAT: (int, float),
            PrimitiveDataType.BOOLEAN: bool,
        }
        
        expected_type = type_mapping.get(data_type.primitive_type)
        if expected_type:
            return isinstance(value, expected_type)
        
        # For other types (date, datetime, etc.), basic validation
        return isinstance(value, (str, int, float, bool))
    
    def _calculate_element_quality_score(self, statistics: Dict[str, Any], violation_count: int) -> float:
        """Calculate quality score for an element."""
        base_score = statistics.get('completeness', 0.0)
        
        # Penalize for violations
        violation_penalty = min(0.5, violation_count * 0.1)
        
        return max(0.0, base_score - violation_penalty)