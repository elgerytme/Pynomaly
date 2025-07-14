"""
Advanced Anomaly Pattern Detector for Data Profiling

Comprehensive anomaly pattern detection system specifically designed for data profiling operations.
Detects statistical, structural, semantic, and behavioral anomalies in datasets to support
data quality assessment and anomaly detection pipeline development.

This enhances Issue #144: Phase 2.3: Data Profiling Package - Advanced Pattern Discovery
with sophisticated anomaly detection capabilities for data profiling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import re
from scipy import stats

# Optional ML dependencies with graceful fallback
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.stats import zscore, iqr, jarque_bera, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    
    # Statistical anomalies
    STATISTICAL_OUTLIER = "statistical_outlier"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    VARIANCE_ANOMALY = "variance_anomaly"
    CORRELATION_ANOMALY = "correlation_anomaly"
    
    # Structural anomalies
    FORMAT_INCONSISTENCY = "format_inconsistency"
    LENGTH_ANOMALY = "length_anomaly"
    PATTERN_VIOLATION = "pattern_violation"
    ENCODING_ISSUE = "encoding_issue"
    
    # Semantic anomalies
    TYPE_MISMATCH = "type_mismatch"
    DOMAIN_VIOLATION = "domain_violation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    REFERENCE_INTEGRITY = "reference_integrity"
    
    # Behavioral anomalies
    FREQUENCY_ANOMALY = "frequency_anomaly"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    DEPENDENCY_ANOMALY = "dependency_anomaly"
    
    # Data quality anomalies
    COMPLETENESS_ANOMALY = "completeness_anomaly"
    CONSISTENCY_ANOMALY = "consistency_anomaly"
    ACCURACY_ANOMALY = "accuracy_anomaly"
    TIMELINESS_ANOMALY = "timeliness_anomaly"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    
    CRITICAL = "critical"  # Data corruption, major quality issues
    HIGH = "high"         # Significant data quality problems
    MEDIUM = "medium"     # Moderate issues that should be investigated
    LOW = "low"          # Minor inconsistencies or edge cases
    INFO = "info"        # Informational findings


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    affected_values: List[Any]
    affected_indices: List[int]
    description: str
    statistical_evidence: Dict[str, Any]
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    
    # Statistical thresholds
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.1
    
    # Detection toggles
    enable_statistical_detection: bool = True
    enable_structural_detection: bool = True
    enable_semantic_detection: bool = True
    enable_behavioral_detection: bool = True
    enable_quality_detection: bool = True
    
    # Performance settings
    max_sample_size: int = 10000
    min_sample_size: int = 100
    detection_timeout_seconds: int = 300
    
    # Domain-specific settings
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    domain_constraints: Dict[str, Any] = field(default_factory=dict)
    expected_distributions: Dict[str, str] = field(default_factory=dict)


class AnomalyPatternDetector:
    """Advanced anomaly pattern detector for data profiling operations."""
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """Initialize the anomaly pattern detector."""
        
        self.config = config or AnomalyDetectionConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize detection strategies
        self._init_detection_strategies()
        
        # Statistical test cache
        self._statistical_cache = {}
        
        # Pattern libraries
        self._pattern_library = self._load_pattern_library()
        
    def _init_detection_strategies(self) -> None:
        """Initialize detection strategy mappings."""
        
        self.statistical_detectors = {
            'outliers': self._detect_statistical_outliers,
            'distribution': self._detect_distribution_anomalies,
            'variance': self._detect_variance_anomalies,
            'correlation': self._detect_correlation_anomalies
        }
        
        self.structural_detectors = {
            'format': self._detect_format_inconsistencies,
            'length': self._detect_length_anomalies,
            'pattern': self._detect_pattern_violations,
            'encoding': self._detect_encoding_issues
        }
        
        self.semantic_detectors = {
            'type': self._detect_type_mismatches,
            'domain': self._detect_domain_violations,
            'constraint': self._detect_constraint_violations,
            'reference': self._detect_reference_integrity_issues
        }
        
        self.behavioral_detectors = {
            'frequency': self._detect_frequency_anomalies,
            'temporal': self._detect_temporal_anomalies,
            'sequence': self._detect_sequence_anomalies,
            'dependency': self._detect_dependency_anomalies
        }
        
        self.quality_detectors = {
            'completeness': self._detect_completeness_anomalies,
            'consistency': self._detect_consistency_anomalies,
            'accuracy': self._detect_accuracy_anomalies,
            'timeliness': self._detect_timeliness_anomalies
        }
    
    def _load_pattern_library(self) -> Dict[str, Any]:
        """Load comprehensive pattern library for anomaly detection."""
        
        return {
            'common_formats': {
                'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'phone': r'^[\+]?[\d\s\-\(\)]{10,}$',
                'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'ip_address': r'^(\d{1,3}\.){3}\d{1,3}$',
                'url': r'^https?://[^\s]+$',
                'date_iso': r'^\d{4}-\d{2}-\d{2}$',
                'currency': r'^\$?\d+(\.\d{2})?$'
            },
            'anomaly_indicators': {
                'null_variants': ['null', 'none', 'n/a', 'na', 'nil', 'undefined', 'missing'],
                'error_indicators': ['error', 'err', 'invalid', 'corrupt', 'malformed'],
                'test_data': ['test', 'dummy', 'sample', 'fake', 'mock']
            },
            'encoding_issues': {
                'replacement_char': '\ufffd',
                'null_byte': '\x00',
                'bom_markers': ['\ufeff', '\ufffe']
            }
        }
    
    async def detect_anomalies(
        self,
        df: pd.DataFrame,
        column_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[AnomalyDetectionResult]]:
        """Detect anomalies in dataset or specific column."""
        
        start_time = time.time()
        self.logger.info(f"Starting anomaly detection for {len(df.columns)} columns")
        
        # Determine columns to analyze
        columns_to_analyze = [column_name] if column_name else df.columns.tolist()
        
        results = {}
        
        for col in columns_to_analyze:
            if col not in df.columns:
                continue
                
            self.logger.debug(f"Detecting anomalies in column: {col}")
            
            try:
                # Sample data if too large
                column_data = self._sample_column_data(df[col])
                
                # Detect anomalies using all enabled strategies
                column_anomalies = await self._detect_column_anomalies(
                    column_data, col, context
                )
                
                results[col] = column_anomalies
                
            except Exception as e:
                self.logger.error(f"Anomaly detection failed for column {col}: {e}")
                results[col] = []
        
        # Cross-column anomaly detection
        if len(columns_to_analyze) > 1:
            cross_column_anomalies = await self._detect_cross_column_anomalies(
                df[columns_to_analyze], context
            )
            results['_cross_column'] = cross_column_anomalies
        
        execution_time = time.time() - start_time
        self.logger.info(f"Anomaly detection completed in {execution_time:.2f} seconds")
        
        return results
    
    def _sample_column_data(self, series: pd.Series) -> pd.Series:
        """Sample column data for performance optimization."""
        
        if len(series) <= self.config.max_sample_size:
            return series
        
        # Stratified sampling to preserve distribution
        if series.dtype == 'object' or series.dtype.name == 'category':
            # For categorical data, sample proportionally
            value_counts = series.value_counts()
            sample_size = min(self.config.max_sample_size, len(series))
            
            sampled_indices = []
            for value, count in value_counts.items():
                value_indices = series[series == value].index.tolist()
                sample_count = max(1, int(count / len(series) * sample_size))
                sampled_indices.extend(np.random.choice(
                    value_indices, 
                    size=min(sample_count, len(value_indices)), 
                    replace=False
                ))
            
            return series.loc[sampled_indices]
        else:
            # For numerical data, random sampling
            return series.sample(n=self.config.max_sample_size, random_state=42)
    
    async def _detect_column_anomalies(
        self,
        series: pd.Series,
        column_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies in a single column."""
        
        anomalies = []
        
        # Statistical anomalies
        if self.config.enable_statistical_detection:
            statistical_anomalies = await self._detect_statistical_anomalies(series, column_name)
            anomalies.extend(statistical_anomalies)
        
        # Structural anomalies
        if self.config.enable_structural_detection:
            structural_anomalies = await self._detect_structural_anomalies(series, column_name)
            anomalies.extend(structural_anomalies)
        
        # Semantic anomalies
        if self.config.enable_semantic_detection:
            semantic_anomalies = await self._detect_semantic_anomalies(series, column_name, context)
            anomalies.extend(semantic_anomalies)
        
        # Behavioral anomalies
        if self.config.enable_behavioral_detection:
            behavioral_anomalies = await self._detect_behavioral_anomalies(series, column_name)
            anomalies.extend(behavioral_anomalies)
        
        # Data quality anomalies
        if self.config.enable_quality_detection:
            quality_anomalies = await self._detect_quality_anomalies(series, column_name)
            anomalies.extend(quality_anomalies)
        
        # Rank anomalies by severity and confidence
        anomalies.sort(key=lambda x: (x.severity.value, -x.confidence))
        
        return anomalies
    
    async def _detect_statistical_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect statistical anomalies in the data."""
        
        anomalies = []
        
        # Only process numerical data for statistical analysis
        if not pd.api.types.is_numeric_dtype(series):
            return anomalies
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) < self.config.min_sample_size:
            return anomalies
        
        try:
            # Statistical outlier detection
            outlier_anomalies = await self.statistical_detectors['outliers'](
                numeric_series, column_name
            )
            anomalies.extend(outlier_anomalies)
            
            # Distribution anomaly detection
            distribution_anomalies = await self.statistical_detectors['distribution'](
                numeric_series, column_name
            )
            anomalies.extend(distribution_anomalies)
            
            # Variance anomaly detection
            variance_anomalies = await self.statistical_detectors['variance'](
                numeric_series, column_name
            )
            anomalies.extend(variance_anomalies)
            
        except Exception as e:
            self.logger.error(f"Statistical anomaly detection failed for {column_name}: {e}")
        
        return anomalies
    
    async def _detect_statistical_outliers(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect statistical outliers using multiple methods."""
        
        anomalies = []
        
        # Z-score method
        if SCIPY_AVAILABLE:
            z_scores = np.abs(zscore(series))
            outlier_mask = z_scores > self.config.z_score_threshold
            
            if outlier_mask.any():
                outlier_indices = series.index[outlier_mask].tolist()
                outlier_values = series[outlier_mask].tolist()
                
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=0.8,
                    affected_values=outlier_values[:20],  # Limit examples
                    affected_indices=outlier_indices,
                    description=f"Z-score outliers detected (threshold: {self.config.z_score_threshold})",
                    statistical_evidence={
                        'method': 'z_score',
                        'threshold': self.config.z_score_threshold,
                        'outlier_count': len(outlier_values),
                        'outlier_percentage': (len(outlier_values) / len(series)) * 100,
                        'max_z_score': float(np.max(z_scores[outlier_mask]))
                    },
                    recommended_actions=[
                        "Review outlier values for data entry errors",
                        "Consider data transformation if outliers are legitimate",
                        "Investigate data source for systematic issues"
                    ]
                ))
        
        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr_value = q3 - q1
        lower_bound = q1 - self.config.iqr_multiplier * iqr_value
        upper_bound = q3 + self.config.iqr_multiplier * iqr_value
        
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        if not iqr_outliers.empty:
            anomalies.append(AnomalyDetectionResult(
                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.75,
                affected_values=iqr_outliers.tolist()[:20],
                affected_indices=iqr_outliers.index.tolist(),
                description=f"IQR outliers detected (multiplier: {self.config.iqr_multiplier})",
                statistical_evidence={
                    'method': 'iqr',
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr_value),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': len(iqr_outliers),
                    'outlier_percentage': (len(iqr_outliers) / len(series)) * 100
                },
                recommended_actions=[
                    "Validate outlier values against business rules",
                    "Consider robust statistical methods",
                    "Review data collection process"
                ]
            ))
        
        # Isolation Forest method (if sklearn available)
        if SKLEARN_AVAILABLE and len(series) >= 100:
            try:
                isolation_forest = IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=42
                )
                outlier_labels = isolation_forest.fit_predict(series.values.reshape(-1, 1))
                isolation_outliers = series[outlier_labels == -1]
                
                if not isolation_outliers.empty:
                    anomalies.append(AnomalyDetectionResult(
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=AnomalySeverity.MEDIUM,
                        confidence=0.7,
                        affected_values=isolation_outliers.tolist()[:20],
                        affected_indices=isolation_outliers.index.tolist(),
                        description="Isolation Forest outliers detected",
                        statistical_evidence={
                            'method': 'isolation_forest',
                            'contamination': self.config.isolation_forest_contamination,
                            'outlier_count': len(isolation_outliers),
                            'outlier_percentage': (len(isolation_outliers) / len(series)) * 100
                        },
                        recommended_actions=[
                            "Cross-validate with domain expertise",
                            "Consider ensemble outlier detection methods",
                            "Investigate anomaly patterns"
                        ]
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Isolation Forest failed for {column_name}: {e}")
        
        return anomalies
    
    async def _detect_distribution_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect distribution anomalies."""
        
        anomalies = []
        
        if not SCIPY_AVAILABLE:
            return anomalies
        
        try:
            # Test for normality
            if len(series) >= 20:  # Minimum sample size for normality tests
                # Jarque-Bera test
                jb_stat, jb_pvalue = jarque_bera(series)
                
                # Shapiro-Wilk test (for smaller samples)
                if len(series) <= 5000:
                    sw_stat, sw_pvalue = stats.shapiro(series)
                else:
                    sw_stat, sw_pvalue = None, None
                
                # Anderson-Darling test
                ad_result = stats.anderson(series, dist='norm')
                
                # If expected to be normal but tests suggest otherwise
                expected_dist = self.config.expected_distributions.get(column_name, '')
                if expected_dist.lower() == 'normal':
                    if jb_pvalue < 0.05:  # Reject normality
                        anomalies.append(AnomalyDetectionResult(
                            anomaly_type=AnomalyType.DISTRIBUTION_ANOMALY,
                            severity=AnomalySeverity.HIGH,
                            confidence=0.85,
                            affected_values=[],
                            affected_indices=[],
                            description=f"Distribution deviates from expected normal distribution",
                            statistical_evidence={
                                'jarque_bera_stat': float(jb_stat),
                                'jarque_bera_pvalue': float(jb_pvalue),
                                'shapiro_wilk_stat': float(sw_stat) if sw_stat else None,
                                'shapiro_wilk_pvalue': float(sw_pvalue) if sw_pvalue else None,
                                'anderson_darling_stat': float(ad_result.statistic),
                                'skewness': float(stats.skew(series)),
                                'kurtosis': float(stats.kurtosis(series))
                            },
                            recommended_actions=[
                                "Investigate data transformation options",
                                "Consider non-parametric analysis methods",
                                "Review data generation process"
                            ]
                        ))
            
            # Detect extreme skewness
            skewness = stats.skew(series)
            if abs(skewness) > 2:  # Highly skewed
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.DISTRIBUTION_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=0.8,
                    affected_values=[],
                    affected_indices=[],
                    description=f"Extreme skewness detected (skewness: {skewness:.2f})",
                    statistical_evidence={
                        'skewness': float(skewness),
                        'kurtosis': float(stats.kurtosis(series)),
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'std': float(series.std())
                    },
                    recommended_actions=[
                        "Consider log transformation for positive skew",
                        "Investigate causes of distribution asymmetry",
                        "Use robust statistical measures"
                    ]
                ))
            
            # Detect extreme kurtosis
            kurt = stats.kurtosis(series)
            if abs(kurt) > 3:  # Heavy or light tails
                severity = AnomalySeverity.HIGH if abs(kurt) > 10 else AnomalySeverity.MEDIUM
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.DISTRIBUTION_ANOMALY,
                    severity=severity,
                    confidence=0.75,
                    affected_values=[],
                    affected_indices=[],
                    description=f"Extreme kurtosis detected (kurtosis: {kurt:.2f})",
                    statistical_evidence={
                        'kurtosis': float(kurt),
                        'skewness': float(skewness),
                        'interpretation': 'heavy tails' if kurt > 0 else 'light tails'
                    },
                    recommended_actions=[
                        "Investigate extreme values",
                        "Consider robust estimation methods",
                        "Check for data quality issues"
                    ]
                ))
                
        except Exception as e:
            self.logger.warning(f"Distribution analysis failed for {column_name}: {e}")
        
        return anomalies
    
    async def _detect_variance_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect variance-related anomalies."""
        
        anomalies = []
        
        try:
            variance = series.var()
            std_dev = series.std()
            mean_val = series.mean()
            
            # Coefficient of variation
            cv = std_dev / abs(mean_val) if mean_val != 0 else float('inf')
            
            # Extremely high coefficient of variation
            if cv > 2:  # Very high variability
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.VARIANCE_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=0.8,
                    affected_values=[],
                    affected_indices=[],
                    description=f"Extremely high coefficient of variation (CV: {cv:.2f})",
                    statistical_evidence={
                        'coefficient_of_variation': float(cv),
                        'variance': float(variance),
                        'standard_deviation': float(std_dev),
                        'mean': float(mean_val),
                        'range': float(series.max() - series.min())
                    },
                    recommended_actions=[
                        "Investigate causes of high variability",
                        "Consider data standardization",
                        "Check for measurement errors"
                    ]
                ))
            
            # Near-zero variance (indicating potential constant or near-constant values)
            if std_dev < 1e-6 and len(series.unique()) > 1:
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.VARIANCE_ANOMALY,
                    severity=AnomalySeverity.LOW,
                    confidence=0.9,
                    affected_values=[],
                    affected_indices=[],
                    description="Near-zero variance detected with multiple unique values",
                    statistical_evidence={
                        'variance': float(variance),
                        'standard_deviation': float(std_dev),
                        'unique_count': int(series.nunique()),
                        'value_range': float(series.max() - series.min())
                    },
                    recommended_actions=[
                        "Verify data precision requirements",
                        "Check for rounding or truncation issues",
                        "Consider measurement scale appropriateness"
                    ]
                ))
                
        except Exception as e:
            self.logger.warning(f"Variance analysis failed for {column_name}: {e}")
        
        return anomalies
    
    async def _detect_correlation_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect correlation anomalies (placeholder for cross-column analysis)."""
        
        # This will be implemented in cross-column anomaly detection
        return []
    
    async def _detect_structural_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect structural anomalies in the data."""
        
        anomalies = []
        
        try:
            # Format inconsistencies
            format_anomalies = await self.structural_detectors['format'](series, column_name)
            anomalies.extend(format_anomalies)
            
            # Length anomalies
            length_anomalies = await self.structural_detectors['length'](series, column_name)
            anomalies.extend(length_anomalies)
            
            # Pattern violations
            pattern_anomalies = await self.structural_detectors['pattern'](series, column_name)
            anomalies.extend(pattern_anomalies)
            
            # Encoding issues
            encoding_anomalies = await self.structural_detectors['encoding'](series, column_name)
            anomalies.extend(encoding_anomalies)
            
        except Exception as e:
            self.logger.error(f"Structural anomaly detection failed for {column_name}: {e}")
        
        return anomalies
    
    async def _detect_format_inconsistencies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect format inconsistencies in string data."""
        
        anomalies = []
        
        if series.dtype != 'object':
            return anomalies
        
        # Convert to string and analyze formats
        string_series = series.astype(str)
        
        # Extract format templates
        templates = {}
        for value in string_series.unique()[:1000]:  # Limit for performance
            template = self._string_to_template(value)
            if template not in templates:
                templates[template] = []
            templates[template].append(value)
        
        if len(templates) < 2:
            return anomalies
        
        # Find dominant template
        template_counts = {t: len(values) for t, values in templates.items()}
        sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        
        dominant_template, dominant_count = sorted_templates[0]
        total_values = len(string_series.unique())
        
        # Detect significant format deviations
        if dominant_count / total_values < 0.8:  # Less than 80% follow dominant pattern
            deviation_examples = []
            for template, values in templates.items():
                if template != dominant_template:
                    deviation_examples.extend(values[:5])
            
            anomalies.append(AnomalyDetectionResult(
                anomaly_type=AnomalyType.FORMAT_INCONSISTENCY,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.8,
                affected_values=deviation_examples[:20],
                affected_indices=[],
                description=f"Format inconsistencies detected. Dominant pattern covers {(dominant_count/total_values)*100:.1f}% of values",
                statistical_evidence={
                    'dominant_template': dominant_template,
                    'dominant_percentage': (dominant_count / total_values) * 100,
                    'template_count': len(templates),
                    'deviation_count': total_values - dominant_count
                },
                recommended_actions=[
                    "Standardize data format across all values",
                    "Implement data validation rules",
                    "Review data input processes"
                ]
            ))
        
        return anomalies
    
    async def _detect_length_anomalies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect length anomalies in string data."""
        
        anomalies = []
        
        if series.dtype != 'object':
            return anomalies
        
        # Calculate string lengths
        lengths = series.astype(str).str.len()
        
        if SCIPY_AVAILABLE and len(lengths) > 0:
            # Statistical outlier detection on lengths
            z_scores = np.abs(zscore(lengths))
            outlier_mask = z_scores > 2.5  # Slightly lower threshold for lengths
            
            if outlier_mask.any():
                outlier_indices = lengths.index[outlier_mask].tolist()
                outlier_values = series[outlier_mask].tolist()
                outlier_lengths = lengths[outlier_mask].tolist()
                
                anomalies.append(AnomalyDetectionResult(
                    anomaly_type=AnomalyType.LENGTH_ANOMALY,
                    severity=AnomalySeverity.LOW,
                    confidence=0.7,
                    affected_values=outlier_values[:10],
                    affected_indices=outlier_indices,
                    description="Unusual string lengths detected",
                    statistical_evidence={
                        'mean_length': float(lengths.mean()),
                        'std_length': float(lengths.std()),
                        'min_length': int(lengths.min()),
                        'max_length': int(lengths.max()),
                        'outlier_lengths': outlier_lengths[:10],
                        'outlier_count': len(outlier_values)
                    },
                    recommended_actions=[
                        "Review values with unusual lengths",
                        "Check for truncation or padding issues",
                        "Validate against business rules"
                    ]
                ))
        
        return anomalies
    
    async def _detect_pattern_violations(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect pattern violations using predefined patterns."""
        
        anomalies = []
        
        if series.dtype != 'object':
            return anomalies
        
        # Check against common format patterns
        for format_name, pattern in self._pattern_library['common_formats'].items():
            if format_name.lower() in column_name.lower():
                # Column name suggests it should follow this pattern
                string_series = series.astype(str)
                matches = string_series.str.match(pattern, case=False)
                violation_count = (~matches).sum()
                
                if violation_count > 0:
                    violation_values = string_series[~matches].tolist()
                    
                    anomalies.append(AnomalyDetectionResult(
                        anomaly_type=AnomalyType.PATTERN_VIOLATION,
                        severity=AnomalySeverity.HIGH,
                        confidence=0.9,
                        affected_values=violation_values[:10],
                        affected_indices=series[~matches].index.tolist(),
                        description=f"Values violating expected {format_name} pattern",
                        statistical_evidence={
                            'expected_pattern': pattern,
                            'pattern_type': format_name,
                            'violation_count': int(violation_count),
                            'violation_percentage': (violation_count / len(series)) * 100,
                            'match_count': int(matches.sum())
                        },
                        recommended_actions=[
                            f"Validate {format_name} format compliance",
                            "Implement input validation",
                            "Clean invalid entries"
                        ]
                    ))
                break
        
        return anomalies
    
    async def _detect_encoding_issues(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyDetectionResult]:
        """Detect encoding issues in string data."""
        
        anomalies = []
        
        if series.dtype != 'object':
            return anomalies
        
        encoding_issues = []
        
        for value in series.unique()[:500]:  # Limit for performance
            str_value = str(value)
            
            # Check for encoding issue indicators
            for issue_char in self._pattern_library['encoding_issues']['replacement_char']:
                if issue_char in str_value:
                    encoding_issues.append(str_value)
                    break
            
            # Check for null bytes
            if '\x00' in str_value:
                encoding_issues.append(str_value)
            
            # Check for BOM markers
            for bom in self._pattern_library['encoding_issues']['bom_markers']:
                if str_value.startswith(bom):
                    encoding_issues.append(str_value)
                    break
        
        if encoding_issues:
            anomalies.append(AnomalyDetectionResult(
                anomaly_type=AnomalyType.ENCODING_ISSUE,
                severity=AnomalySeverity.HIGH,
                confidence=0.95,
                affected_values=encoding_issues[:10],
                affected_indices=[],
                description=f"Encoding issues detected in {len(encoding_issues)} values",
                statistical_evidence={
                    'encoding_issue_count': len(encoding_issues),
                    'encoding_issue_percentage': (len(encoding_issues) / len(series)) * 100
                },
                recommended_actions=[
                    "Check source data encoding",
                    "Implement proper encoding handling",
                    "Clean corrupted characters"
                ]
            ))
        
        return anomalies
    
    def _string_to_template(self, s: str) -> str:
        """Convert string to format template."""
        template = []
        for char in s:
            if char.isdigit():
                template.append('D')
            elif char.isalpha():
                template.append('U' if char.isupper() else 'L')
            elif char.isspace():
                template.append('S')
            else:
                template.append(char)
        return ''.join(template)
    
    # Placeholder methods for other detection types
    async def _detect_semantic_anomalies(self, series: pd.Series, column_name: str, context: Optional[Dict[str, Any]] = None) -> List[AnomalyDetectionResult]:
        """Detect semantic anomalies (to be implemented)."""
        return []
    
    async def _detect_behavioral_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        """Detect behavioral anomalies (to be implemented)."""
        return []
    
    async def _detect_quality_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        """Detect data quality anomalies (to be implemented)."""
        return []
    
    async def _detect_cross_column_anomalies(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> List[AnomalyDetectionResult]:
        """Detect cross-column anomalies (to be implemented)."""
        return []
    
    # Additional detection methods will be implemented as needed
    async def _detect_type_mismatches(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_domain_violations(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_constraint_violations(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_reference_integrity_issues(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_frequency_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_temporal_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_sequence_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_dependency_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_completeness_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_consistency_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_accuracy_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []
    
    async def _detect_timeliness_anomalies(self, series: pd.Series, column_name: str) -> List[AnomalyDetectionResult]:
        return []