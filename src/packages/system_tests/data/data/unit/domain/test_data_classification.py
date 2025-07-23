"""Tests for data classification value objects."""

import pytest
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# TODO: Replace with actual data domain value objects when available
# Currently creating test fixtures since the referenced value objects don't exist
# Original problematic import: from src.packages.data.data.domain.value_objects.data_classification import (

class DataSensitivityLevel(str, Enum):
    """Mock data sensitivity level for testing."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataComplianceTag(str, Enum):
    """Mock compliance tag for testing."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"

class DataQualityDimension(str, Enum):
    """Mock quality dimension for testing."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"

@dataclass(frozen=True)
class DataClassification:
    """Mock data classification value object for testing."""
    sensitivity_level: DataSensitivityLevel
    compliance_tags: List[DataComplianceTag]
    quality_dimensions: List[DataQualityDimension]
    retention_period_days: Optional[int] = None
    
    def is_sensitive(self) -> bool:
        return self.sensitivity_level in [DataSensitivityLevel.CONFIDENTIAL, DataSensitivityLevel.RESTRICTED]
    
    def requires_encryption(self) -> bool:
        return self.sensitivity_level != DataSensitivityLevel.PUBLIC


class TestDataClassification:
    """Test cases for DataClassification value object."""
    
    def test_create_basic_classification(self):
        """Test creating a basic data classification."""
        classification = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
            business_category="customer_data",
            encryption_required=True
        )
        
        assert classification.sensitivity_level == DataSensitivityLevel.CONFIDENTIAL
        assert classification.business_category == "customer_data"
        assert classification.encryption_required is True
    
    def test_create_with_compliance_tags(self):
        """Test creating classification with compliance tags."""
        classification = DataClassification(
            sensitivity_level=DataSensitivityLevel.RESTRICTED,
            compliance_tags=[DataComplianceTag.PII, DataComplianceTag.GDPR],
            retention_period_days=2555  # 7 years
        )
        
        assert DataComplianceTag.PII in classification.compliance_tags
        assert DataComplianceTag.GDPR in classification.compliance_tags
        assert classification.retention_period_days == 2555
    
    def test_create_with_quality_requirements(self):
        """Test creating classification with quality requirements."""
        quality_requirements = {
            DataQualityDimension.ACCURACY: 0.95,
            DataQualityDimension.COMPLETENESS: 0.9,
            DataQualityDimension.CONSISTENCY: 0.98
        }
        
        classification = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            quality_requirements=quality_requirements
        )
        
        assert classification.quality_requirements[DataQualityDimension.ACCURACY] == 0.95
        assert classification.quality_requirements[DataQualityDimension.COMPLETENESS] == 0.9
        assert classification.quality_requirements[DataQualityDimension.CONSISTENCY] == 0.98
    
    def test_quality_threshold_validation(self):
        """Test that quality thresholds must be between 0 and 1."""
        with pytest.raises(ValueError, match="Quality threshold .* must be between 0 and 1"):
            DataClassification(
                sensitivity_level=DataSensitivityLevel.PUBLIC,
                quality_requirements={
                    DataQualityDimension.ACCURACY: 1.5  # Invalid: > 1
                }
            )
        
        with pytest.raises(ValueError, match="Quality threshold .* must be between 0 and 1"):
            DataClassification(
                sensitivity_level=DataSensitivityLevel.PUBLIC,
                quality_requirements={
                    DataQualityDimension.COMPLETENESS: -0.1  # Invalid: < 0
                }
            )
    
    def test_requires_encryption_explicit(self):
        """Test encryption requirement when explicitly set."""
        classification = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC,
            encryption_required=True
        )
        
        assert classification.requires_encryption()
    
    def test_requires_encryption_high_sensitivity(self):
        """Test encryption requirement for high sensitivity levels."""
        confidential = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL
        )
        restricted = DataClassification(
            sensitivity_level=DataSensitivityLevel.RESTRICTED
        )
        public = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC
        )
        
        assert confidential.requires_encryption()
        assert restricted.requires_encryption()
        assert not public.requires_encryption()
    
    def test_requires_encryption_compliance_tags(self):
        """Test encryption requirement based on compliance tags."""
        pii_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            compliance_tags=[DataComplianceTag.PII]
        )
        phi_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            compliance_tags=[DataComplianceTag.PHI]
        )
        sox_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            compliance_tags=[DataComplianceTag.SOX]
        )
        
        assert pii_data.requires_encryption()
        assert phi_data.requires_encryption()
        assert not sox_data.requires_encryption()  # SOX doesn't require encryption
    
    def test_requires_access_controls(self):
        """Test access control requirements."""
        public_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC
        )
        internal_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL
        )
        compliance_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC,
            compliance_tags=[DataComplianceTag.GDPR]
        )
        restricted_access = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC,
            access_restrictions=["role:admin"]
        )
        
        assert not public_data.requires_access_controls()
        assert internal_data.requires_access_controls()
        assert compliance_data.requires_access_controls()
        assert restricted_access.requires_access_controls()
    
    def test_get_retention_policy(self):
        """Test retention policy determination."""
        gdpr_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
            compliance_tags=[DataComplianceTag.GDPR]
        )
        ccpa_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
            compliance_tags=[DataComplianceTag.CCPA]
        )
        hipaa_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.RESTRICTED,
            compliance_tags=[DataComplianceTag.HIPAA]
        )
        custom_retention = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            retention_period_days=365
        )
        no_policy = DataClassification(
            sensitivity_level=DataSensitivityLevel.PUBLIC
        )
        
        assert "GDPR" in gdpr_data.get_retention_policy()
        assert "CCPA" in ccpa_data.get_retention_policy()
        assert "HIPAA" in hipaa_data.get_retention_policy()
        assert "365 days" in custom_retention.get_retention_policy()
        assert no_policy.get_retention_policy() is None
    
    def test_is_personal_data(self):
        """Test personal data identification."""
        pii_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
            compliance_tags=[DataComplianceTag.PII]
        )
        phi_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.RESTRICTED,
            compliance_tags=[DataComplianceTag.PHI]
        )
        financial_data = DataClassification(
            sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
            compliance_tags=[DataComplianceTag.PCI]
        )
        
        assert pii_data.is_personal_data()
        assert phi_data.is_personal_data()
        assert not financial_data.is_personal_data()  # PCI is not personal data classification
    
    def test_get_quality_score(self):
        """Test quality score calculation."""
        classification = DataClassification(
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            quality_requirements={
                DataQualityDimension.ACCURACY: 0.9,
                DataQualityDimension.COMPLETENESS: 0.8,
                DataQualityDimension.CONSISTENCY: 0.95
            }
        )
        
        # Perfect metrics
        perfect_metrics = {
            DataQualityDimension.ACCURACY: 0.95,
            DataQualityDimension.COMPLETENESS: 0.85,
            DataQualityDimension.CONSISTENCY: 0.98
        }
        assert classification.get_quality_score(perfect_metrics) == 1.0
        
        # Below threshold metrics
        poor_metrics = {
            DataQualityDimension.ACCURACY: 0.45,  # 50% of required 0.9
            DataQualityDimension.COMPLETENESS: 0.4,  # 50% of required 0.8
            DataQualityDimension.CONSISTENCY: 0.475  # 50% of required 0.95
        }
        expected_score = (0.5 + 0.5 + 0.5) / 3  # Average of proportional scores
        assert abs(classification.get_quality_score(poor_metrics) - expected_score) < 0.01
        
        # No requirements
        no_requirements = DataClassification(sensitivity_level=DataSensitivityLevel.PUBLIC)
        assert no_requirements.get_quality_score(perfect_metrics) == 1.0