"""
Business Logic Template Domain Entity

Represents reusable business logic patterns that can be instantiated
across different business contexts with parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..value_objects.entity_id import EntityId
from ..value_objects.version_number import VersionNumber
from ..value_objects.entity_metadata import EntityMetadata
from ..exceptions.entity_exceptions import EntityValidationError, InvalidEntityError


@dataclass
class TemplateParameter:
    """Represents a parameter definition for a business logic template."""
    
    name: str
    type_hint: str
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation_rules: List[str] = field(default_factory=list)
    
    def validate_value(self, value: Any) -> bool:
        """Validate a parameter value against this parameter definition."""
        if self.required and value is None:
            return False
        
        # Type validation would be implemented here
        # For now, basic validation
        return True


@dataclass
class BusinessLogicTemplate:
    """
    Domain entity representing a reusable business logic template.
    
    Templates encapsulate common domain patterns and can be instantiated
    with specific parameters for different business contexts.
    """
    
    name: str
    description: str
    category: str
    logic_definition: str
    parameters: List[TemplateParameter]
    id: EntityId = field(default_factory=EntityId.generate)
    version: VersionNumber = field(default=VersionNumber("1.0.0"))
    metadata: EntityMetadata = field(default_factory=EntityMetadata.empty)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    parent_template_id: Optional[EntityId] = None
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate the business logic template after initialization."""
        self._validate_template()
    
    def _validate_template(self) -> None:
        """Validate template properties and business rules."""
        if not self.name or not self.name.strip():
            raise InvalidEntityError("Template name cannot be empty")
        
        if not self.description or not self.description.strip():
            raise InvalidEntityError("Template description cannot be empty")
        
        if not self.logic_definition or not self.logic_definition.strip():
            raise InvalidEntityError("Template logic definition cannot be empty")
        
        if not self.category or not self.category.strip():
            raise InvalidEntityError("Template category cannot be empty")
        
        # Validate parameter names are unique
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            raise EntityValidationError("Template parameters must have unique names")
    
    def add_parameter(self, parameter: TemplateParameter) -> BusinessLogicTemplate:
        """Add a parameter to the template."""
        # Check for duplicate parameter names
        existing_names = {p.name for p in self.parameters}
        if parameter.name in existing_names:
            raise EntityValidationError(f"Parameter '{parameter.name}' already exists")
        
        new_parameters = list(self.parameters)
        new_parameters.append(parameter)
        
        return self._create_updated_copy(
            parameters=new_parameters,
            updated_at=datetime.now()
        )
    
    def remove_parameter(self, parameter_name: str) -> BusinessLogicTemplate:
        """Remove a parameter from the template."""
        new_parameters = [p for p in self.parameters if p.name != parameter_name]
        
        if len(new_parameters) == len(self.parameters):
            raise InvalidEntityError(f"Parameter '{parameter_name}' not found")
        
        return self._create_updated_copy(
            parameters=new_parameters,
            updated_at=datetime.now()
        )
    
    def validate_parameters(self, parameter_values: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate parameter values against template parameter definitions.
        
        Returns:
            Dictionary of validation errors (parameter_name -> error_message)
        """
        errors = {}
        
        # Check for required parameters
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(parameter_values.keys())
        
        missing_params = required_params - provided_params
        for param_name in missing_params:
            errors[param_name] = "Required parameter is missing"
        
        # Validate provided parameter values
        param_lookup = {p.name: p for p in self.parameters}
        for param_name, value in parameter_values.items():
            if param_name in param_lookup:
                param_def = param_lookup[param_name]
                if not param_def.validate_value(value):
                    errors[param_name] = f"Invalid value for parameter '{param_name}'"
        
        return errors
    
    def create_instance(self, parameter_values: Dict[str, Any]) -> BusinessLogicInstance:
        """Create an instance of this template with bound parameter values."""
        validation_errors = self.validate_parameters(parameter_values)
        if validation_errors:
            raise EntityValidationError(
                "Template instantiation failed due to parameter validation errors",
                validation_errors=validation_errors
            )
        
        # Fill in default values for optional parameters
        instance_values = dict(parameter_values)
        for param in self.parameters:
            if param.name not in instance_values and param.default_value is not None:
                instance_values[param.name] = param.default_value
        
        return BusinessLogicInstance(
            template_id=self.id,
            template_version=self.version,
            parameter_values=instance_values,
            instance_name=f"{self.name}_instance",
            created_at=datetime.now()
        )
    
    def add_tag(self, tag: str) -> BusinessLogicTemplate:
        """Add a tag to the template."""
        new_tags = set(self.tags)
        new_tags.add(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def remove_tag(self, tag: str) -> BusinessLogicTemplate:
        """Remove a tag from the template."""
        new_tags = set(self.tags)
        new_tags.discard(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def update_metadata(self, metadata: EntityMetadata) -> BusinessLogicTemplate:
        """Update the template metadata."""
        return self._create_updated_copy(
            metadata=metadata,
            updated_at=datetime.now()
        )
    
    def increment_version(self, version_type: str = "patch") -> BusinessLogicTemplate:
        """Create a new version of the template."""
        new_version = self.version.increment(version_type)
        
        return self._create_updated_copy(
            version=new_version,
            updated_at=datetime.now()
        )
    
    def deactivate(self) -> BusinessLogicTemplate:
        """Deactivate the template."""
        return self._create_updated_copy(
            is_active=False,
            updated_at=datetime.now()
        )
    
    def activate(self) -> BusinessLogicTemplate:
        """Activate the template."""
        return self._create_updated_copy(
            is_active=True,
            updated_at=datetime.now()
        )
    
    def _create_updated_copy(self, **kwargs) -> BusinessLogicTemplate:
        """Create an updated copy of the template with specified changes."""
        update_data = {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'logic_definition': self.logic_definition,
            'parameters': self.parameters,
            'id': self.id,
            'version': self.version,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tags': self.tags,
            'parent_template_id': self.parent_template_id,
            'is_active': self.is_active
        }
        update_data.update(kwargs)
        
        return BusinessLogicTemplate(**update_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'logic_definition': self.logic_definition,
            'parameters': [
                {
                    'name': p.name,
                    'type_hint': p.type_hint,
                    'description': p.description,
                    'required': p.required,
                    'default_value': p.default_value,
                    'validation_rules': p.validation_rules
                }
                for p in self.parameters
            ],
            'version': str(self.version),
            'metadata': self.metadata.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': list(self.tags),
            'parent_template_id': str(self.parent_template_id) if self.parent_template_id else None,
            'is_active': self.is_active
        }


@dataclass
class BusinessLogicInstance:
    """
    Represents an instance of a business logic template with bound parameters.
    """
    
    template_id: EntityId
    template_version: VersionNumber
    parameter_values: Dict[str, Any]
    instance_name: str
    id: EntityId = field(default_factory=EntityId.generate)
    created_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None
    
    def execute(self) -> Any:
        """Execute the business logic instance."""
        # This would contain the actual execution logic
        # For now, we'll just update execution tracking
        self.execution_count += 1
        self.last_executed_at = datetime.now()
        
        # Return a placeholder result
        return {"status": "executed", "parameters": self.parameter_values}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary representation."""
        return {
            'id': str(self.id),
            'template_id': str(self.template_id),
            'template_version': str(self.template_version),
            'parameter_values': self.parameter_values,
            'instance_name': self.instance_name,
            'created_at': self.created_at.isoformat(),
            'execution_count': self.execution_count,
            'last_executed_at': self.last_executed_at.isoformat() if self.last_executed_at else None
        }