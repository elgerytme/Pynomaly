"""Data element entity."""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from packages.core.domain.abstractions.base_entity import BaseEntity
from ..value_objects.data_type import DataType
from ..value_objects.data_classification import DataClassification


class ElementType(str, Enum):
    """Types of data elements."""
    ATTRIBUTE = "attribute"
    DIMENSION = "dimension"
    MEASURE = "measure"
    KEY = "key"
    FOREIGN_KEY = "foreign_key"
    CALCULATED = "calculated"
    DERIVED = "derived"
    METADATA = "metadata"


class ElementStatus(str, Enum):
    """Data element status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"


class DataElement(BaseEntity):
    """Represents an individual data element or field within a dataset."""
    
    element_id: UUID = Field(default_factory=uuid4, description="Unique element identifier")
    dataset_id: UUID = Field(..., description="Parent dataset reference")
    name: str = Field(..., min_length=1, max_length=200, description="Element name")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Element description")
    business_definition: Optional[str] = Field(None, description="Business definition")
    element_type: ElementType = Field(..., description="Type of data element")
    status: ElementStatus = Field(default=ElementStatus.ACTIVE, description="Element status")
    data_type: DataType = Field(..., description="Data type definition")
    classification: Optional[DataClassification] = Field(None, description="Data classification")
    position: int = Field(default=0, ge=0, description="Position in dataset schema")
    is_required: bool = Field(default=True, description="Whether element is required")
    is_primary_key: bool = Field(default=False, description="Whether element is primary key")
    is_foreign_key: bool = Field(default=False, description="Whether element is foreign key")
    foreign_key_reference: Optional[str] = Field(None, description="Foreign key reference")
    is_unique: bool = Field(default=False, description="Whether values must be unique")
    is_indexed: bool = Field(default=False, description="Whether element is indexed")
    default_value: Optional[Any] = Field(None, description="Default value")
    allowed_values: List[str] = Field(default_factory=list, description="Enumerated allowed values")
    value_range: Optional[Dict[str, Any]] = Field(None, description="Valid value range")
    format_pattern: Optional[str] = Field(None, description="Format validation pattern")
    calculation_formula: Optional[str] = Field(None, description="Calculation formula if derived")
    source_elements: List[UUID] = Field(default_factory=list, description="Source element references")
    derived_elements: List[UUID] = Field(default_factory=list, description="Derived element references")
    business_rules: List[str] = Field(default_factory=list, description="Business rules")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules")
    quality_checks: List[str] = Field(default_factory=list, description="Quality check rules")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistical information")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")
    null_count: int = Field(default=0, ge=0, description="Number of null values")
    distinct_count: Optional[int] = Field(None, ge=0, description="Number of distinct values")
    min_value: Optional[Any] = Field(None, description="Minimum value")
    max_value: Optional[Any] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value (numeric types)")
    std_dev: Optional[float] = Field(None, description="Standard deviation (numeric types)")
    most_common_values: List[Dict[str, Any]] = Field(default_factory=list, description="Most common values")
    value_distribution: Dict[str, int] = Field(default_factory=dict, description="Value frequency distribution")
    data_samples: List[str] = Field(default_factory=list, description="Sample values")
    lineage_upstream: List[UUID] = Field(default_factory=list, description="Upstream element dependencies")
    lineage_downstream: List[UUID] = Field(default_factory=list, description="Downstream element consumers")
    change_log: List[Dict[str, Any]] = Field(default_factory=list, description="Change history")
    validation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Validation history")
    owner: Optional[str] = Field(None, description="Element owner or steward")
    steward: Optional[str] = Field(None, description="Data steward")
    documentation_url: Optional[str] = Field(None, description="Documentation link")
    tags: List[str] = Field(default_factory=list, description="Element tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate element name format."""
        if not v.strip():
            raise ValueError("Element name cannot be empty")
        return v.strip()
    
    @validator('position')
    def validate_position(cls, v: int) -> int:
        """Validate position is non-negative."""
        if v < 0:
            raise ValueError("Position must be non-negative")
        return v
    
    @validator('foreign_key_reference')
    def validate_foreign_key_reference(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate foreign key reference when is_foreign_key is True."""
        if values.get('is_foreign_key') and not v:
            raise ValueError("foreign_key_reference required when is_foreign_key is True")
        return v
    
    @validator('calculation_formula')
    def validate_calculation_formula(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate calculation formula when element is calculated."""
        element_type = values.get('element_type')
        if element_type == ElementType.CALCULATED and not v:
            raise ValueError("calculation_formula required for calculated elements")
        return v
    
    def activate(self) -> None:
        """Activate the element."""
        self.status = ElementStatus.ACTIVE
        self.updated_at = datetime.utcnow()
        self._log_change("ACTIVATED", "Element activated")
    
    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate the element."""
        self.status = ElementStatus.INACTIVE
        self.updated_at = datetime.utcnow()
        self._log_change("DEACTIVATED", reason or "Element deactivated")
    
    def deprecate(self, replacement_element_id: Optional[UUID] = None, reason: Optional[str] = None) -> None:
        """Deprecate the element."""
        self.status = ElementStatus.DEPRECATED
        self.updated_at = datetime.utcnow()
        
        change_details = {"reason": reason or "Element deprecated"}
        if replacement_element_id:
            change_details["replacement_element_id"] = str(replacement_element_id)
        
        self._log_change("DEPRECATED", "Element deprecated", change_details)
    
    def update_data_type(self, new_data_type: DataType, reason: Optional[str] = None) -> None:
        """Update element data type."""
        old_type = f"{self.data_type.primitive_type.value}"
        self.data_type = new_data_type
        self.updated_at = datetime.utcnow()
        
        self._log_change("DATA_TYPE_UPDATED", 
                        f"Data type changed from {old_type} to {new_data_type.primitive_type.value}",
                        {"reason": reason})
    
    def update_classification(self, new_classification: DataClassification) -> None:
        """Update element classification."""
        old_sensitivity = self.classification.sensitivity_level if self.classification else "none"
        self.classification = new_classification
        self.updated_at = datetime.utcnow()
        
        self._log_change("CLASSIFICATION_UPDATED",
                        f"Classification changed from {old_sensitivity} to {new_classification.sensitivity_level}")
    
    def update_statistics(self, statistics: Dict[str, Any]) -> None:
        """Update element statistics."""
        self.statistics = statistics
        
        # Update specific statistical fields
        self.null_count = statistics.get('null_count', 0)
        self.distinct_count = statistics.get('distinct_count')
        self.min_value = statistics.get('min_value')
        self.max_value = statistics.get('max_value')
        self.avg_value = statistics.get('avg_value')
        self.std_dev = statistics.get('std_dev')
        
        # Update value distribution and common values
        if 'value_distribution' in statistics:
            self.value_distribution = statistics['value_distribution']
        if 'most_common_values' in statistics:
            self.most_common_values = statistics['most_common_values']
        
        self.updated_at = datetime.utcnow()
        self._log_change("STATISTICS_UPDATED", "Element statistics updated")
    
    def update_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """Update quality metrics."""
        self.quality_metrics.update(metrics)
        self.updated_at = datetime.utcnow()
        
        self._log_change("QUALITY_METRICS_UPDATED", "Quality metrics updated", metrics)
    
    def validate_element(self, validation_results: Dict[str, Any]) -> None:
        """Record element validation results."""
        validation_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'results': validation_results,
            'is_valid': validation_results.get('is_valid', True)
        }
        
        self.validation_history.append(validation_entry)
        
        # Keep only last 50 validation entries
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]
        
        self.updated_at = datetime.utcnow()
        self._log_change("VALIDATED", "Element validation completed", validation_results)
    
    def add_business_rule(self, rule: str) -> None:
        """Add a business rule."""
        if rule not in self.business_rules:
            self.business_rules.append(rule)
            self.updated_at = datetime.utcnow()
            self._log_change("BUSINESS_RULE_ADDED", f"Business rule added: {rule}")
    
    def remove_business_rule(self, rule: str) -> None:
        """Remove a business rule."""
        if rule in self.business_rules:
            self.business_rules.remove(rule)
            self.updated_at = datetime.utcnow()
            self._log_change("BUSINESS_RULE_REMOVED", f"Business rule removed: {rule}")
    
    def add_validation_rule(self, rule: str) -> None:
        """Add a validation rule."""
        if rule not in self.validation_rules:
            self.validation_rules.append(rule)
            self.updated_at = datetime.utcnow()
            self._log_change("VALIDATION_RULE_ADDED", f"Validation rule added: {rule}")
    
    def remove_validation_rule(self, rule: str) -> None:
        """Remove a validation rule."""
        if rule in self.validation_rules:
            self.validation_rules.remove(rule)
            self.updated_at = datetime.utcnow()
            self._log_change("VALIDATION_RULE_REMOVED", f"Validation rule removed: {rule}")
    
    def set_position(self, position: int) -> None:
        """Set element position in schema."""
        old_position = self.position
        self.position = max(0, position)
        self.updated_at = datetime.utcnow()
        
        self._log_change("POSITION_CHANGED", f"Position changed from {old_position} to {position}")
    
    def _log_change(self, change_type: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a change to the element."""
        change_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': change_type,
            'description': description,
            'details': details or {}
        }
        
        self.change_log.append(change_entry)
        
        # Keep only last 100 changes
        if len(self.change_log) > 100:
            self.change_log = self.change_log[-100:]
    
    def is_active(self) -> bool:
        """Check if element is currently active."""
        return self.status == ElementStatus.ACTIVE
    
    def is_key_field(self) -> bool:
        """Check if element is a key field."""
        return self.is_primary_key or self.is_foreign_key
    
    def is_numeric(self) -> bool:
        """Check if element is numeric type."""
        return self.data_type.is_numeric()
    
    def is_temporal(self) -> bool:
        """Check if element is temporal type."""
        return self.data_type.is_temporal()
    
    def is_calculated_or_derived(self) -> bool:
        """Check if element is calculated or derived."""
        return self.element_type in [ElementType.CALCULATED, ElementType.DERIVED]
    
    def has_enumerated_values(self) -> bool:
        """Check if element has enumerated allowed values."""
        return bool(self.allowed_values)
    
    def get_completeness_ratio(self, total_records: int) -> float:
        """Calculate data completeness ratio."""
        if total_records == 0:
            return 0.0
        
        non_null_count = total_records - self.null_count
        return non_null_count / total_records
    
    def get_uniqueness_ratio(self, total_records: int) -> float:
        """Calculate uniqueness ratio."""
        if total_records == 0 or self.distinct_count is None:
            return 0.0
        
        return self.distinct_count / total_records
    
    def is_high_cardinality(self, threshold_ratio: float = 0.8) -> bool:
        """Check if element has high cardinality."""
        if self.distinct_count is None:
            return False
        
        # Assume we have access to total records from statistics
        total_records = self.statistics.get('total_records', 0)
        if total_records == 0:
            return False
        
        uniqueness_ratio = self.get_uniqueness_ratio(total_records)
        return uniqueness_ratio >= threshold_ratio
    
    def requires_masking(self) -> bool:
        """Check if element requires data masking based on classification."""
        return (
            self.classification and 
            self.classification.is_personal_data() and
            self.classification.requires_encryption()
        )