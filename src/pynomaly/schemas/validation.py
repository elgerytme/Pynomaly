"""Schema validation utilities for backward compatibility.

This module provides utilities to validate schema compatibility and ensure
backward compatibility when schemas are updated or migrated.

Classes:
    SchemaCompatibilityError: Exception for schema compatibility issues
    BackwardCompatibilityValidator: Validator for backward compatibility
    
Functions:
    validate_schema_compatibility: Validate schema compatibility between versions
    ensure_backward_compatibility: Ensure backward compatibility is maintained
"""

from __future__ import annotations

from typing import Any, Dict, Type, Optional, List, Union
from datetime import datetime
import json
import difflib

from pydantic import BaseModel, ValidationError


class SchemaCompatibilityError(Exception):
    """Exception raised when schema compatibility issues are detected."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 old_schema: Optional[str] = None, new_schema: Optional[str] = None):
        super().__init__(message)
        self.field_name = field_name
        self.old_schema = old_schema
        self.new_schema = new_schema
        self.timestamp = datetime.utcnow()
        
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = super().__str__()
        if self.field_name:
            base_msg += f" (field: {self.field_name})"
        return base_msg


class BackwardCompatibilityValidator:
    """Validator for ensuring backward compatibility of schemas."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize the validator.
        
        Args:
            strict_mode: If True, any breaking change raises an error.
                        If False, only critical breaking changes raise errors.
        """
        self.strict_mode = strict_mode
        self.validation_log: List[Dict[str, Any]] = []
        
    def validate_field_compatibility(self, old_field: Dict[str, Any], 
                                   new_field: Dict[str, Any], 
                                   field_name: str) -> bool:
        """Validate compatibility between old and new field definitions.
        
        Args:
            old_field: Old field definition
            new_field: New field definition
            field_name: Name of the field being validated
            
        Returns:
            True if compatible, False otherwise
            
        Raises:
            SchemaCompatibilityError: If incompatible changes are detected
        """
        issues = []
        
        # Check if field type changed
        old_type = old_field.get('type')
        new_type = new_field.get('type')
        
        if old_type != new_type:
            issues.append(f"Field type changed from {old_type} to {new_type}")
        
        # Check if required field became optional (allowed)
        # or optional field became required (breaking change)
        old_required = old_field.get('required', False)
        new_required = new_field.get('required', False)
        
        if not old_required and new_required:
            issues.append(f"Optional field became required")
        
        # Check if constraints became more restrictive
        old_constraints = old_field.get('constraints', {})
        new_constraints = new_field.get('constraints', {})
        
        # Check minimum/maximum constraints
        if 'minimum' in old_constraints and 'minimum' in new_constraints:
            if new_constraints['minimum'] > old_constraints['minimum']:
                issues.append(f"Minimum constraint became more restrictive")
        
        if 'maximum' in old_constraints and 'maximum' in new_constraints:
            if new_constraints['maximum'] < old_constraints['maximum']:
                issues.append(f"Maximum constraint became more restrictive")
        
        # Check string length constraints
        if 'min_length' in old_constraints and 'min_length' in new_constraints:
            if new_constraints['min_length'] > old_constraints['min_length']:
                issues.append(f"Minimum length constraint became more restrictive")
        
        if 'max_length' in old_constraints and 'max_length' in new_constraints:
            if new_constraints['max_length'] < old_constraints['max_length']:
                issues.append(f"Maximum length constraint became more restrictive")
        
        # Log issues
        if issues:
            log_entry = {
                'timestamp': datetime.utcnow(),
                'field_name': field_name,
                'issues': issues,
                'severity': 'error' if self.strict_mode else 'warning'
            }
            self.validation_log.append(log_entry)
            
            if self.strict_mode:
                raise SchemaCompatibilityError(
                    f"Breaking changes detected in field '{field_name}': {'; '.join(issues)}",
                    field_name=field_name
                )
        
        return len(issues) == 0
    
    def validate_schema_structure(self, old_schema: Dict[str, Any], 
                                 new_schema: Dict[str, Any]) -> bool:
        """Validate overall schema structure compatibility.
        
        Args:
            old_schema: Old schema definition
            new_schema: New schema definition
            
        Returns:
            True if compatible, False otherwise
        """
        issues = []
        
        # Check if required fields were removed
        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))
        
        removed_required = old_required - new_required
        if removed_required:
            issues.append(f"Required fields removed: {removed_required}")
        
        # Check if field types changed
        old_properties = old_schema.get('properties', {})
        new_properties = new_schema.get('properties', {})
        
        for field_name in old_properties:
            if field_name in new_properties:
                old_field = old_properties[field_name]
                new_field = new_properties[field_name]
                
                try:
                    self.validate_field_compatibility(old_field, new_field, field_name)
                except SchemaCompatibilityError as e:
                    issues.append(str(e))
        
        # Check if additionalProperties became more restrictive
        old_additional = old_schema.get('additionalProperties', True)
        new_additional = new_schema.get('additionalProperties', True)
        
        if old_additional and not new_additional:
            issues.append("additionalProperties became false (more restrictive)")
        
        if issues:
            log_entry = {
                'timestamp': datetime.utcnow(),
                'issues': issues,
                'severity': 'error' if self.strict_mode else 'warning'
            }
            self.validation_log.append(log_entry)
            
            if self.strict_mode:
                raise SchemaCompatibilityError(
                    f"Schema structure compatibility issues: {'; '.join(issues)}"
                )
        
        return len(issues) == 0
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a detailed validation report.
        
        Returns:
            Validation report with issues and recommendations
        """
        error_count = sum(1 for entry in self.validation_log if entry['severity'] == 'error')
        warning_count = sum(1 for entry in self.validation_log if entry['severity'] == 'warning')
        
        return {
            'timestamp': datetime.utcnow(),
            'total_issues': len(self.validation_log),
            'error_count': error_count,
            'warning_count': warning_count,
            'issues': self.validation_log,
            'is_compatible': error_count == 0,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if self.validation_log:
            recommendations.append("Review all identified compatibility issues")
            
            # Count issue types
            field_issues = sum(1 for entry in self.validation_log if 'field_name' in entry)
            if field_issues > 0:
                recommendations.append(f"Consider deprecation strategy for {field_issues} field changes")
            
            # Check for type changes
            type_changes = sum(1 for entry in self.validation_log 
                             if any('type changed' in issue for issue in entry.get('issues', [])))
            if type_changes > 0:
                recommendations.append("Type changes detected - consider data migration scripts")
            
            # Check for constraint changes
            constraint_changes = sum(1 for entry in self.validation_log 
                                   if any('constraint' in issue for issue in entry.get('issues', [])))
            if constraint_changes > 0:
                recommendations.append("Constraint changes detected - validate existing data")
        
        return recommendations


def validate_schema_compatibility(old_schema: Union[Type[BaseModel], Dict[str, Any]], 
                                new_schema: Union[Type[BaseModel], Dict[str, Any]],
                                strict_mode: bool = True) -> bool:
    """Validate schema compatibility between versions.
    
    Args:
        old_schema: Old schema (Pydantic model or dict)
        new_schema: New schema (Pydantic model or dict)
        strict_mode: If True, any breaking change raises an error
        
    Returns:
        True if compatible, False otherwise
        
    Raises:
        SchemaCompatibilityError: If incompatible changes are detected
    """
    validator = BackwardCompatibilityValidator(strict_mode=strict_mode)
    
    # Convert Pydantic models to schema dicts if needed
    if isinstance(old_schema, type) and issubclass(old_schema, BaseModel):
        old_schema_dict = old_schema.schema()
    else:
        old_schema_dict = old_schema
        
    if isinstance(new_schema, type) and issubclass(new_schema, BaseModel):
        new_schema_dict = new_schema.schema()
    else:
        new_schema_dict = new_schema
    
    return validator.validate_schema_structure(old_schema_dict, new_schema_dict)


def ensure_backward_compatibility(schema_class: Type[BaseModel], 
                                 test_data: List[Dict[str, Any]]) -> bool:
    """Ensure backward compatibility by testing with historical data.
    
    Args:
        schema_class: Schema class to test
        test_data: List of historical data samples
        
    Returns:
        True if all test data validates successfully
        
    Raises:
        SchemaCompatibilityError: If validation fails
    """
    validation_errors = []
    
    for i, data in enumerate(test_data):
        try:
            schema_class(**data)
        except ValidationError as e:
            validation_errors.append({
                'sample_index': i,
                'data': data,
                'error': str(e),
                'error_details': e.errors()
            })
    
    if validation_errors:
        error_summary = f"Backward compatibility validation failed for {len(validation_errors)} samples"
        raise SchemaCompatibilityError(
            error_summary,
            old_schema=json.dumps(validation_errors, indent=2, default=str)
        )
    
    return True


def generate_schema_diff(old_schema: Dict[str, Any], 
                        new_schema: Dict[str, Any]) -> List[str]:
    """Generate a human-readable diff between two schemas.
    
    Args:
        old_schema: Old schema definition
        new_schema: New schema definition
        
    Returns:
        List of diff lines
    """
    old_json = json.dumps(old_schema, indent=2, sort_keys=True)
    new_json = json.dumps(new_schema, indent=2, sort_keys=True)
    
    diff = difflib.unified_diff(
        old_json.splitlines(keepends=True),
        new_json.splitlines(keepends=True),
        fromfile='old_schema.json',
        tofile='new_schema.json'
    )
    
    return list(diff)


def validate_data_migration(old_data: List[Dict[str, Any]], 
                          new_schema: Type[BaseModel],
                          migration_function: Optional[callable] = None) -> bool:
    """Validate data migration from old format to new schema.
    
    Args:
        old_data: List of data in old format
        new_schema: New schema class
        migration_function: Optional function to transform old data
        
    Returns:
        True if migration is successful
        
    Raises:
        SchemaCompatibilityError: If migration fails
    """
    migration_errors = []
    
    for i, data in enumerate(old_data):
        try:
            # Apply migration function if provided
            if migration_function:
                migrated_data = migration_function(data)
            else:
                migrated_data = data
            
            # Validate against new schema
            new_schema(**migrated_data)
            
        except Exception as e:
            migration_errors.append({
                'sample_index': i,
                'original_data': data,
                'error': str(e),
                'error_type': type(e).__name__
            })
    
    if migration_errors:
        error_summary = f"Data migration validation failed for {len(migration_errors)} samples"
        raise SchemaCompatibilityError(
            error_summary,
            old_schema=json.dumps(migration_errors, indent=2, default=str)
        )
    
    return True
