"""Data schema domain service."""

from typing import Dict, Any, List, Optional, Set
from uuid import UUID
from ..entities.data_set import DataSet
from ..entities.data_element import DataElement
from ..value_objects.data_schema import DataSchema, DataFieldSchema
from ..value_objects.data_type import DataType, PrimitiveDataType


class DataSchemaService:
    """Domain service for data schema operations."""
    
    def generate_schema_from_data(self, sample_data: List[Dict[str, Any]], schema_name: str) -> DataSchema:
        """Generate schema definition from sample data."""
        if not sample_data:
            raise ValueError("Cannot generate schema from empty data")
        
        # Analyze all records to understand the data structure
        field_analysis = self._analyze_fields(sample_data)
        
        # Create field schemas
        fields = {}
        for field_name, analysis in field_analysis.items():
            data_type = self._infer_data_type(analysis)
            
            field_schema = DataFieldSchema(
                field_name=field_name,
                data_type=data_type,
                description=f"Auto-generated field from data analysis",
                is_required=analysis['required_score'] > 0.95,  # 95% non-null threshold
                is_primary_key=False,  # Cannot infer from data alone
                is_foreign_key=False,  # Cannot infer from data alone
                enum_values=analysis.get('enum_candidates') if analysis.get('is_enum_candidate') else None
            )
            fields[field_name] = field_schema
        
        return DataSchema(
            schema_name=schema_name,
            version="1.0.0",
            fields=fields,
            description=f"Auto-generated schema from {len(sample_data)} sample records"
        )
    
    def merge_schemas(self, schemas: List[DataSchema], merged_name: str) -> DataSchema:
        """Merge multiple schemas into a single compatible schema."""
        if not schemas:
            raise ValueError("Cannot merge empty schema list")
        
        if len(schemas) == 1:
            return schemas[0]
        
        # Collect all fields from all schemas
        all_fields = {}
        field_sources = {}
        
        for schema in schemas:
            for field_name, field_schema in schema.fields.items():
                if field_name not in all_fields:
                    all_fields[field_name] = field_schema
                    field_sources[field_name] = [schema.schema_name]
                else:
                    # Merge field definitions
                    all_fields[field_name] = self._merge_field_schemas(
                        all_fields[field_name], 
                        field_schema
                    )
                    field_sources[field_name].append(schema.schema_name)
        
        # Collect constraints and indexes
        all_constraints = []
        all_indexes = []
        
        for schema in schemas:
            all_constraints.extend(schema.constraints)
            all_indexes.extend(schema.indexes)
        
        return DataSchema(
            schema_name=merged_name,
            version="1.0.0",
            fields=all_fields,
            constraints=list(set(all_constraints)),  # Remove duplicates
            indexes=all_indexes,
            description=f"Merged schema from: {', '.join(s.schema_name for s in schemas)}",
            metadata={'source_schemas': [s.schema_name for s in schemas]}
        )
    
    def evolve_schema(self, current_schema: DataSchema, new_data: List[Dict[str, Any]]) -> DataSchema:
        """Evolve existing schema based on new data patterns."""
        if not new_data:
            return current_schema
        
        # Analyze new data
        field_analysis = self._analyze_fields(new_data)
        
        # Update existing fields and add new ones
        updated_fields = dict(current_schema.fields)
        
        for field_name, analysis in field_analysis.items():
            if field_name in updated_fields:
                # Update existing field
                updated_fields[field_name] = self._evolve_field_schema(
                    updated_fields[field_name], 
                    analysis
                )
            else:
                # Add new field
                data_type = self._infer_data_type(analysis)
                updated_fields[field_name] = DataFieldSchema(
                    field_name=field_name,
                    data_type=data_type,
                    description=f"Added during schema evolution",
                    is_required=analysis['required_score'] > 0.95
                )
        
        # Increment version
        version_parts = current_schema.version.split('.')
        if len(version_parts) >= 2:
            minor = int(version_parts[1]) + 1
            new_version = f"{version_parts[0]}.{minor}.0"
        else:
            new_version = f"{current_schema.version}.1"
        
        return DataSchema(
            schema_name=current_schema.schema_name,
            version=new_version,
            fields=updated_fields,
            description=current_schema.description,
            primary_keys=current_schema.primary_keys,
            foreign_keys=current_schema.foreign_keys,
            indexes=current_schema.indexes,
            constraints=current_schema.constraints,
            metadata={
                **current_schema.metadata,
                'evolution_timestamp': new_data[0].get('timestamp') if new_data else None,
                'evolution_record_count': len(new_data)
            }
        )
    
    def compare_schemas(self, schema1: DataSchema, schema2: DataSchema) -> Dict[str, Any]:
        """Compare two schemas and identify differences."""
        comparison = {
            'schemas': {
                'schema1': {'name': schema1.schema_name, 'version': schema1.version},
                'schema2': {'name': schema2.schema_name, 'version': schema2.version}
            },
            'field_differences': {
                'added_fields': [],
                'removed_fields': [],
                'modified_fields': [],
                'type_changes': []
            },
            'constraint_differences': {
                'added_constraints': [],
                'removed_constraints': []
            },
            'compatibility_assessment': 'compatible'
        }
        
        fields1 = set(schema1.fields.keys())
        fields2 = set(schema2.fields.keys())
        
        # Field additions and removals
        comparison['field_differences']['added_fields'] = list(fields2 - fields1)
        comparison['field_differences']['removed_fields'] = list(fields1 - fields2)
        
        # Modified fields
        common_fields = fields1 & fields2
        for field_name in common_fields:
            field1 = schema1.fields[field_name]
            field2 = schema2.fields[field_name]
            
            if not self._fields_equal(field1, field2):
                comparison['field_differences']['modified_fields'].append(field_name)
                
                if field1.data_type.primitive_type != field2.data_type.primitive_type:
                    comparison['field_differences']['type_changes'].append({
                        'field': field_name,
                        'from_type': field1.data_type.primitive_type.value,
                        'to_type': field2.data_type.primitive_type.value
                    })
        
        # Constraint differences
        constraints1 = set(schema1.constraints)
        constraints2 = set(schema2.constraints)
        
        comparison['constraint_differences']['added_constraints'] = list(constraints2 - constraints1)
        comparison['constraint_differences']['removed_constraints'] = list(constraints1 - constraints2)
        
        # Compatibility assessment
        has_breaking_changes = (
            comparison['field_differences']['removed_fields'] or
            comparison['field_differences']['type_changes'] or
            comparison['constraint_differences']['removed_constraints']
        )
        
        if has_breaking_changes:
            comparison['compatibility_assessment'] = 'breaking_changes'
        elif comparison['field_differences']['added_fields']:
            comparison['compatibility_assessment'] = 'backward_compatible'
        else:
            comparison['compatibility_assessment'] = 'identical'
        
        return comparison
    
    def validate_schema_integrity(self, schema: DataSchema) -> Dict[str, Any]:
        """Validate schema internal integrity and constraints."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Primary key validation
        for pk_field in schema.primary_keys:
            if pk_field not in schema.fields:
                validation['is_valid'] = False
                validation['issues'].append(f"Primary key field '{pk_field}' not found in schema")
            else:
                field = schema.fields[pk_field]
                if not field.is_required:
                    validation['warnings'].append(f"Primary key field '{pk_field}' should be required")
        
        # Foreign key validation
        for fk_field, fk_ref in schema.foreign_keys.items():
            if fk_field not in schema.fields:
                validation['is_valid'] = False
                validation['issues'].append(f"Foreign key field '{fk_field}' not found in schema")
        
        # Index validation
        for index_fields in schema.indexes:
            for field_name in index_fields:
                if field_name not in schema.fields:
                    validation['is_valid'] = False
                    validation['issues'].append(f"Index field '{field_name}' not found in schema")
        
        # Field-level validation
        for field_name, field_schema in schema.fields.items():
            if field_schema.is_foreign_key and not field_schema.foreign_key_reference:
                validation['issues'].append(f"Field '{field_name}' marked as foreign key but missing reference")
        
        return validation
    
    def _analyze_fields(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze fields in data to understand their characteristics."""
        field_analysis = {}
        
        # Get all possible field names
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        for field_name in all_fields:
            values = []
            null_count = 0
            
            for record in data:
                value = record.get(field_name)
                if value is None:
                    null_count += 1
                else:
                    values.append(value)
            
            analysis = {
                'total_count': len(data),
                'null_count': null_count,
                'non_null_count': len(values),
                'required_score': len(values) / len(data) if data else 0,
                'distinct_values': list(set(str(v) for v in values)),
                'distinct_count': len(set(str(v) for v in values)),
                'sample_values': values[:10],
                'inferred_types': self._infer_types_from_values(values)
            }
            
            # Enum candidate detection
            if analysis['distinct_count'] <= 10 and analysis['non_null_count'] > 0:
                analysis['is_enum_candidate'] = True
                analysis['enum_candidates'] = analysis['distinct_values']
            
            field_analysis[field_name] = analysis
        
        return field_analysis
    
    def _infer_data_type(self, field_analysis: Dict[str, Any]) -> DataType:
        """Infer data type from field analysis."""
        inferred_types = field_analysis['inferred_types']
        
        # Choose the most restrictive compatible type
        if 'integer' in inferred_types and len(inferred_types) == 1:
            return DataType(primitive_type=PrimitiveDataType.INTEGER)
        elif 'float' in inferred_types or 'integer' in inferred_types:
            return DataType(primitive_type=PrimitiveDataType.FLOAT)
        elif 'boolean' in inferred_types and len(inferred_types) == 1:
            return DataType(primitive_type=PrimitiveDataType.BOOLEAN)
        else:
            # Default to string
            max_length = max(len(str(v)) for v in field_analysis['sample_values']) if field_analysis['sample_values'] else 255
            return DataType(
                primitive_type=PrimitiveDataType.STRING,
                max_length=min(max_length * 2, 1000)  # Add buffer, cap at 1000
            )
    
    def _infer_types_from_values(self, values: List[Any]) -> Set[str]:
        """Infer possible types from actual values."""
        types = set()
        
        for value in values:
            if isinstance(value, bool):
                types.add('boolean')
            elif isinstance(value, int):
                types.add('integer')
            elif isinstance(value, float):
                types.add('float')
            elif isinstance(value, str):
                # Try to parse as other types
                if value.lower() in ('true', 'false'):
                    types.add('boolean')
                else:
                    try:
                        int(value)
                        types.add('integer')
                        continue
                    except ValueError:
                        pass
                    
                    try:
                        float(value)
                        types.add('float')
                        continue
                    except ValueError:
                        pass
                
                types.add('string')
            else:
                types.add('string')  # Default fallback
        
        return types
    
    def _merge_field_schemas(self, field1: DataFieldSchema, field2: DataFieldSchema) -> DataFieldSchema:
        """Merge two field schemas into a compatible one."""
        # Use the more permissive settings
        return DataFieldSchema(
            field_name=field1.field_name,
            data_type=self._merge_data_types(field1.data_type, field2.data_type),
            description=field1.description or field2.description,
            is_required=field1.is_required and field2.is_required,  # Both must be required
            is_primary_key=field1.is_primary_key or field2.is_primary_key,
            is_foreign_key=field1.is_foreign_key or field2.is_foreign_key,
            foreign_key_reference=field1.foreign_key_reference or field2.foreign_key_reference,
            default_value=field1.default_value if field1.default_value is not None else field2.default_value
        )
    
    def _merge_data_types(self, type1: DataType, type2: DataType) -> DataType:
        """Merge two data types into a compatible one."""
        # If types are the same, return as-is
        if type1.primitive_type == type2.primitive_type:
            return DataType(
                primitive_type=type1.primitive_type,
                max_length=max(type1.max_length or 0, type2.max_length or 0) if type1.max_length or type2.max_length else None,
                nullable=type1.nullable or type2.nullable
            )
        
        # Handle numeric type compatibility
        numeric_types = {PrimitiveDataType.INTEGER, PrimitiveDataType.FLOAT}
        if type1.primitive_type in numeric_types and type2.primitive_type in numeric_types:
            return DataType(primitive_type=PrimitiveDataType.FLOAT, nullable=type1.nullable or type2.nullable)
        
        # Default to string for incompatible types
        return DataType(
            primitive_type=PrimitiveDataType.STRING,
            max_length=1000,
            nullable=type1.nullable or type2.nullable
        )
    
    def _evolve_field_schema(self, current_field: DataFieldSchema, analysis: Dict[str, Any]) -> DataFieldSchema:
        """Evolve a field schema based on new data analysis."""
        # Make field optional if new data shows it can be null
        new_required = current_field.is_required and analysis['required_score'] > 0.95
        
        # Expand data type if needed
        new_data_type = self._evolve_data_type(current_field.data_type, analysis)
        
        return DataFieldSchema(
            field_name=current_field.field_name,
            data_type=new_data_type,
            description=current_field.description,
            is_required=new_required,
            is_primary_key=current_field.is_primary_key,
            is_foreign_key=current_field.is_foreign_key,
            foreign_key_reference=current_field.foreign_key_reference,
            default_value=current_field.default_value
        )
    
    def _evolve_data_type(self, current_type: DataType, analysis: Dict[str, Any]) -> DataType:
        """Evolve data type based on new analysis."""
        inferred_types = analysis['inferred_types']
        
        # If current type is compatible with all inferred types, keep it
        if current_type.primitive_type == PrimitiveDataType.STRING:
            # String can accommodate anything, just update max_length if needed
            new_max_length = current_type.max_length
            if analysis['sample_values']:
                observed_max = max(len(str(v)) for v in analysis['sample_values'])
                if new_max_length is None or observed_max > new_max_length:
                    new_max_length = min(observed_max * 2, 2000)
            
            return DataType(
                primitive_type=current_type.primitive_type,
                max_length=new_max_length,
                nullable=current_type.nullable or analysis['null_count'] > 0
            )
        
        # For other types, check compatibility
        if str(current_type.primitive_type.value) in inferred_types:
            return current_type
        
        # Type needs to be widened
        if current_type.primitive_type == PrimitiveDataType.INTEGER and 'float' in inferred_types:
            return DataType(primitive_type=PrimitiveDataType.FLOAT)
        
        # Fall back to string
        return DataType(primitive_type=PrimitiveDataType.STRING, max_length=1000)
    
    def _fields_equal(self, field1: DataFieldSchema, field2: DataFieldSchema) -> bool:
        """Check if two field schemas are equal."""
        return (
            field1.field_name == field2.field_name and
            field1.data_type.primitive_type == field2.data_type.primitive_type and
            field1.is_required == field2.is_required and
            field1.is_primary_key == field2.is_primary_key and
            field1.is_foreign_key == field2.is_foreign_key
        )