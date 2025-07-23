"""Shared interfaces for data quality services."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class DataQualityInterface(ABC):
    """Interface for data quality assessment services."""
    
    @abstractmethod
    def assess_quality(
        self, 
        schema_profile: Optional[Dict[str, Any]], 
        data: pd.DataFrame
    ) -> Any:  # QualityAssessmentResult
        """Assess data quality based on profile and data."""
        pass


class ValidationEngineInterface(ABC):
    """Interface for data validation services."""
    
    @abstractmethod
    def validate_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset against rules."""
        pass


class DataCleansingInterface(ABC):
    """Interface for data cleansing services."""
    
    @abstractmethod
    def cleanse_dataset(
        self, 
        data: pd.DataFrame, 
        quality_assessment: Any
    ) -> pd.DataFrame:
        """Cleanse dataset based on quality assessment."""
        pass


class QualityMonitoringInterface(ABC):
    """Interface for quality monitoring services."""
    
    @abstractmethod
    async def monitor_data_quality(
        self,
        data_source: str,
        quality_profile: Any
    ) -> Dict[str, Any]:
        """Monitor data quality continuously."""
        pass


class RemediationInterface(ABC):
    """Interface for automated remediation services."""
    
    @abstractmethod
    async def remediate_quality_issues(
        self,
        issues: list,
        remediation_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Automatically remediate quality issues."""
        pass