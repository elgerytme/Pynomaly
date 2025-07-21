"""Data classification value objects."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import Field, validator
from packages.core.domain.abstractions.base_value_object import BaseValueObject


class DataSensitivityLevel(str, Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataComplianceTag(str, Enum):
    """Data compliance and regulatory tags."""
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOX = "sox"  # Sarbanes-Oxley
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act


class DataQualityDimension(str, Enum):
    """Data quality assessment dimensions."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class DataClassification(BaseValueObject):
    """Comprehensive data classification and metadata."""
    
    sensitivity_level: DataSensitivityLevel = Field(..., description="Data sensitivity level")
    compliance_tags: List[DataComplianceTag] = Field(default_factory=list, description="Regulatory compliance tags")
    business_category: Optional[str] = Field(None, description="Business domain category")
    data_domain: Optional[str] = Field(None, description="Data domain classification")
    retention_period_days: Optional[int] = Field(None, ge=0, description="Data retention period in days")
    encryption_required: bool = Field(default=False, description="Whether encryption is required")
    access_restrictions: List[str] = Field(default_factory=list, description="Access restriction rules")
    quality_requirements: Dict[DataQualityDimension, float] = Field(default_factory=dict, description="Quality thresholds")
    tags: List[str] = Field(default_factory=list, description="Custom classification tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('quality_requirements')
    def validate_quality_thresholds(cls, v: Dict[DataQualityDimension, float]) -> Dict[DataQualityDimension, float]:
        """Validate quality thresholds are between 0 and 1."""
        for dimension, threshold in v.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"Quality threshold for {dimension} must be between 0 and 1")
        return v
    
    def requires_encryption(self) -> bool:
        """Check if data requires encryption based on classification."""
        if self.encryption_required:
            return True
        
        # High sensitivity levels require encryption
        if self.sensitivity_level in [DataSensitivityLevel.CONFIDENTIAL, DataSensitivityLevel.RESTRICTED]:
            return True
            
        # Certain compliance tags require encryption
        encryption_required_tags = {
            DataComplianceTag.PII,
            DataComplianceTag.PHI, 
            DataComplianceTag.PCI,
            DataComplianceTag.HIPAA
        }
        return bool(set(self.compliance_tags) & encryption_required_tags)
    
    def requires_access_controls(self) -> bool:
        """Check if data requires special access controls."""
        return (
            self.sensitivity_level != DataSensitivityLevel.PUBLIC or
            bool(self.compliance_tags) or
            bool(self.access_restrictions)
        )
    
    def get_retention_policy(self) -> Optional[str]:
        """Get retention policy description based on compliance tags."""
        if DataComplianceTag.GDPR in self.compliance_tags:
            return "GDPR: Right to be forgotten applies"
        elif DataComplianceTag.CCPA in self.compliance_tags:
            return "CCPA: Consumer deletion rights apply"
        elif DataComplianceTag.HIPAA in self.compliance_tags:
            return "HIPAA: 6-year minimum retention required"
        elif self.retention_period_days:
            return f"Retain for {self.retention_period_days} days"
        return None
    
    def is_personal_data(self) -> bool:
        """Check if this is classified as personal data."""
        personal_data_tags = {DataComplianceTag.PII, DataComplianceTag.PHI, DataComplianceTag.GDPR, DataComplianceTag.CCPA}
        return bool(set(self.compliance_tags) & personal_data_tags)
    
    def get_quality_score(self, quality_metrics: Dict[DataQualityDimension, float]) -> float:
        """Calculate overall quality score based on requirements and metrics."""
        if not self.quality_requirements:
            return 1.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, required_threshold in self.quality_requirements.items():
            actual_metric = quality_metrics.get(dimension, 0.0)
            # Score is 1.0 if meets requirement, proportional if below
            score = min(1.0, actual_metric / required_threshold) if required_threshold > 0 else 1.0
            total_score += score
            total_weight += 1.0
        
        return total_score / total_weight if total_weight > 0 else 1.0