"""
Advanced Pattern Discovery Service for Data Profiling

Enhanced pattern discovery with ML-powered data understanding, semantic type classification,
and intelligent pattern recognition capabilities. This builds foundational capabilities for
anomaly detection systems by understanding normal data patterns.

This addresses Issue #144: Phase 2.3: Data Profiling Package - Advanced Pattern Discovery
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path

# Optional ML dependencies with graceful fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.stats import entropy, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

from ...domain.entities.data_profile import Pattern, PatternType, DataType
from .pattern_discovery_service import PatternDiscoveryService


class SemanticType(Enum):
    """Enhanced semantic type classification."""
    
    # Personal Information
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    
    # Identifiers
    UUID = "uuid"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    
    # Geographic
    POSTAL_CODE = "postal_code"
    COUNTRY_CODE = "country_code"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    
    # Temporal
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    DATETIME = "datetime"
    
    # Financial
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    STOCK_SYMBOL = "stock_symbol"
    
    # Technical
    URL = "url"
    DOMAIN = "domain"
    FILE_PATH = "file_path"
    VERSION = "version"
    
    # Textual
    DESCRIPTION = "description"
    COMMENT = "comment"
    CATEGORY = "category"
    TAG = "tag"
    
    # Numerical
    SCORE = "score"
    RATING = "rating"
    COUNT = "count"
    MEASUREMENT = "measurement"
    
    # Unknown/Other
    UNKNOWN = "unknown"


class AnomalyPatternType(Enum):
    """Types of anomaly patterns detected in data."""
    
    STATISTICAL_OUTLIER = "statistical_outlier"
    LENGTH_ANOMALY = "length_anomaly"
    FORMAT_DEVIATION = "format_deviation"
    ENCODING_ANOMALY = "encoding_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    RELATIONSHIP_ANOMALY = "relationship_anomaly"


@dataclass
class SemanticClassification:
    """Semantic type classification result."""
    
    semantic_type: SemanticType
    confidence: float
    reasoning: List[str]
    evidence: Dict[str, Any]
    alternative_types: List[Tuple[SemanticType, float]]


@dataclass
class AnomalyPattern:
    """Detected anomaly pattern in data."""
    
    pattern_type: AnomalyPatternType
    affected_values: List[str]
    confidence: float
    severity: str  # "low", "medium", "high", "critical"
    description: str
    statistical_evidence: Dict[str, float]
    recommended_action: Optional[str] = None


@dataclass
class TemporalPatternAnalysis:
    """Temporal pattern analysis results."""
    
    has_temporal_structure: bool
    seasonality_detected: bool
    trend_detected: bool
    cycle_length: Optional[int]
    temporal_anomalies: List[Dict[str, Any]]
    forecasting_accuracy: Optional[float] = None


@dataclass
class IntelligentTypeInference:
    """Intelligent data type inference with ML enhancement."""
    
    inferred_type: DataType
    confidence: float
    semantic_type: Optional[SemanticType]
    type_evolution: List[Tuple[DataType, float]]  # How type confidence evolves with more data
    mixed_type_analysis: Optional[Dict[str, Any]] = None


class AdvancedPatternDiscoveryService:
    """Advanced pattern discovery service with ML-powered capabilities."""
    
    def __init__(self, base_service: Optional[PatternDiscoveryService] = None):
        self.logger = logging.getLogger(__name__)
        self.base_service = base_service or PatternDiscoveryService()
        
        # ML models cache
        self._ml_models_cache = {}
        
        # Semantic classification rules
        self._semantic_rules = self._initialize_semantic_rules()
        
        # Pattern libraries
        self._pattern_library = self._load_pattern_library()
        
        self.logger.info("Advanced Pattern Discovery Service initialized")
    
    def _initialize_semantic_rules(self) -> Dict[SemanticType, Dict[str, Any]]:
        """Initialize semantic classification rules and patterns."""
        
        return {
            SemanticType.EMAIL: {
                'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'keywords': ['email', 'mail', 'e_mail'],
                'validation': self._validate_email,
                'confidence_boost': 0.3
            },
            SemanticType.PHONE: {
                'regex': r'^[\+]?[\d\s\-\(\)]{10,}$',
                'keywords': ['phone', 'tel', 'mobile', 'contact'],
                'validation': self._validate_phone,
                'confidence_boost': 0.25
            },
            SemanticType.UUID: {
                'regex': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'keywords': ['uuid', 'guid', 'id'],
                'validation': self._validate_uuid,
                'confidence_boost': 0.4
            },
            SemanticType.SSN: {
                'regex': r'^\d{3}-\d{2}-\d{4}$',
                'keywords': ['ssn', 'social', 'security'],
                'validation': self._validate_ssn,
                'confidence_boost': 0.35
            },
            SemanticType.CREDIT_CARD: {
                'regex': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
                'keywords': ['credit', 'card', 'payment'],
                'validation': self._validate_credit_card,
                'confidence_boost': 0.3
            },
            SemanticType.IP_ADDRESS: {
                'regex': r'^(\d{1,3}\.){3}\d{1,3}$',
                'keywords': ['ip', 'address', 'host'],
                'validation': self._validate_ip,
                'confidence_boost': 0.35
            },
            SemanticType.URL: {
                'regex': r'^https?://[^\s]+$',
                'keywords': ['url', 'link', 'website'],
                'validation': self._validate_url,
                'confidence_boost': 0.25
            },
            SemanticType.CURRENCY: {
                'regex': r'^\$?\d+(\.\d{2})?$',
                'keywords': ['price', 'cost', 'amount', 'currency'],
                'validation': self._validate_currency,
                'confidence_boost': 0.2
            },
            SemanticType.POSTAL_CODE: {
                'regex': r'^\d{5}(-\d{4})?$',
                'keywords': ['zip', 'postal', 'code'],
                'validation': self._validate_postal_code,
                'confidence_boost': 0.25
            },
            SemanticType.DATE: {
                'regex': r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$|^\d{2}-\d{2}-\d{4}$',
                'keywords': ['date', 'created', 'updated'],
                'validation': self._validate_date,
                'confidence_boost': 0.3
            }
        }
    
    def _load_pattern_library(self) -> Dict[str, List[str]]:
        """Load comprehensive pattern library for different domains."""
        
        return {
            'anomaly_indicators': [
                r'null', r'n/a', r'unknown', r'missing', r'error',
                r'invalid', r'corrupt', r'malformed', r'undefined'
            ],
            'temporal_patterns': [
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
                r'\d{2}/\d{2}/\d{4}',  # US date format
                r'\d{2}-\d{2}-\d{4}',  # EU date format
                r'\d{2}:\d{2}:\d{2}',  # Time format
            ],
            'identifier_patterns': [
                r'[A-Z]{2,3}\d{6,}',  # Common ID format
                r'\d{8,}',  # Numeric ID
                r'[A-Z0-9]{8,}',  # Alphanumeric ID
            ],
            'measurement_patterns': [
                r'\d+\.?\d*\s?(kg|lb|g|mg)',  # Weight
                r'\d+\.?\d*\s?(m|ft|cm|mm)',  # Length
                r'\d+\.?\d*\s?(C|F|K)',  # Temperature
            ]
        }
    
    async def discover_advanced_patterns(
        self,
        df: pd.DataFrame,
        include_ml_analysis: bool = True,
        include_anomaly_detection: bool = True,
        include_temporal_analysis: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Discover advanced patterns with ML-powered analysis."""
        
        self.logger.info(f"Starting advanced pattern discovery for {len(df.columns)} columns")
        start_time = time.time()
        
        results = {}
        
        for column in df.columns:
            self.logger.debug(f"Analyzing column: {column}")
            
            column_analysis = {
                'basic_patterns': [],
                'semantic_classification': None,
                'anomaly_patterns': [],
                'intelligent_type_inference': None,
                'temporal_analysis': None,
                'ml_insights': {}
            }
            
            try:
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                
                # 1. Basic pattern discovery (from base service)
                basic_patterns = self.base_service._discover_column_patterns(series, column)
                column_analysis['basic_patterns'] = basic_patterns
                
                # 2. Enhanced semantic classification
                semantic_classification = await self._classify_semantic_type_enhanced(series, column)
                column_analysis['semantic_classification'] = semantic_classification
                
                # 3. Intelligent type inference
                type_inference = await self._infer_intelligent_type(series, column)
                column_analysis['intelligent_type_inference'] = type_inference
                
                # 4. Anomaly pattern detection
                if include_anomaly_detection:
                    anomaly_patterns = await self._detect_anomaly_patterns(series, column)
                    column_analysis['anomaly_patterns'] = anomaly_patterns
                
                # 5. Temporal analysis for datetime-like columns
                if include_temporal_analysis and self._is_temporal_candidate(series):
                    temporal_analysis = await self._analyze_temporal_patterns(series, column)
                    column_analysis['temporal_analysis'] = temporal_analysis
                
                # 6. ML-powered insights
                if include_ml_analysis and SKLEARN_AVAILABLE:
                    ml_insights = await self._extract_ml_insights(series, column)
                    column_analysis['ml_insights'] = ml_insights
                
                results[column] = column_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing column {column}: {e}")
                continue
        
        analysis_time = time.time() - start_time
        self.logger.info(f"Advanced pattern discovery completed in {analysis_time:.2f}s")
        
        # Cross-column analysis
        cross_column_insights = await self._analyze_cross_column_relationships(df, results)
        results['_cross_column_analysis'] = cross_column_insights
        
        return results
    
    async def _classify_semantic_type_enhanced(
        self,
        series: pd.Series,
        column_name: str
    ) -> SemanticClassification:
        """Enhanced semantic type classification with ML and heuristics."""
        
        string_series = series.astype(str)
        unique_values = string_series.unique()[:1000]  # Limit for performance
        
        # Score each semantic type
        type_scores = {}
        reasoning = []
        evidence = {}
        
        for semantic_type, rules in self._semantic_rules.items():
            score = 0.0
            type_reasoning = []
            
            # Regex matching
            regex_matches = 0
            for value in unique_values:
                if re.match(rules['regex'], value, re.IGNORECASE):
                    regex_matches += 1
            
            regex_percentage = (regex_matches / len(unique_values)) * 100
            if regex_percentage > 0:
                score += regex_percentage / 100 * 0.5
                type_reasoning.append(f"Regex match: {regex_percentage:.1f}%")
            
            # Column name keywords
            col_name_lower = column_name.lower()
            for keyword in rules['keywords']:
                if keyword in col_name_lower:
                    score += rules['confidence_boost']
                    type_reasoning.append(f"Column name contains '{keyword}'")
                    break
            
            # Validation function
            if 'validation' in rules and regex_matches > 0:
                try:
                    validation_score = rules['validation'](unique_values[:100])
                    score += validation_score * 0.3
                    type_reasoning.append(f"Validation score: {validation_score:.2f}")
                except Exception:
                    pass
            
            if score > 0:
                type_scores[semantic_type] = score
                reasoning.extend(type_reasoning)
                evidence[semantic_type.value] = {
                    'regex_matches': regex_matches,
                    'regex_percentage': regex_percentage,
                    'total_score': score
                }
        
        # Determine best semantic type
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda x: type_scores[x])
            confidence = min(type_scores[best_type], 1.0)
            
            # Alternative types (sorted by score)
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            alternative_types = [(t, s) for t, s in sorted_types[1:6]]  # Top 5 alternatives
            
        else:
            best_type = SemanticType.UNKNOWN
            confidence = 0.0
            alternative_types = []
        
        return SemanticClassification(
            semantic_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            alternative_types=alternative_types
        )
    
    async def _detect_anomaly_patterns(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyPattern]:
        """Detect anomaly patterns in data that could indicate data quality issues."""
        
        anomaly_patterns = []
        string_series = series.astype(str)
        
        # 1. Statistical outliers in string lengths
        length_anomalies = await self._detect_length_anomalies(string_series)
        anomaly_patterns.extend(length_anomalies)
        
        # 2. Format deviations
        format_anomalies = await self._detect_format_deviations(string_series)
        anomaly_patterns.extend(format_anomalies)
        
        # 3. Encoding anomalies
        encoding_anomalies = await self._detect_encoding_anomalies(string_series)
        anomaly_patterns.extend(encoding_anomalies)
        
        # 4. Frequency anomalies
        frequency_anomalies = await self._detect_frequency_anomalies(string_series)
        anomaly_patterns.extend(frequency_anomalies)
        
        # 5. Semantic inconsistencies
        semantic_anomalies = await self._detect_semantic_inconsistencies(string_series, column_name)
        anomaly_patterns.extend(semantic_anomalies)
        
        return anomaly_patterns
    
    async def _detect_length_anomalies(self, series: pd.Series) -> List[AnomalyPattern]:
        """Detect anomalies in string lengths using statistical methods."""
        
        anomalies = []
        lengths = [len(str(val)) for val in series]
        
        if len(set(lengths)) < 2:  # All same length
            return anomalies
        
        # Use statistical methods to detect outliers
        if SCIPY_AVAILABLE:
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Z-score based detection
            z_scores = np.abs((lengths - mean_length) / std_length) if std_length > 0 else np.zeros_like(lengths)
            outlier_indices = np.where(z_scores > 3)[0]  # 3-sigma rule
            
            if len(outlier_indices) > 0:
                outlier_values = [str(series.iloc[i]) for i in outlier_indices]
                outlier_lengths = [lengths[i] for i in outlier_indices]
                
                severity = "high" if len(outlier_indices) / len(series) > 0.1 else "medium"
                
                anomalies.append(AnomalyPattern(
                    pattern_type=AnomalyPatternType.LENGTH_ANOMALY,
                    affected_values=outlier_values[:10],  # Limit examples
                    confidence=0.8,
                    severity=severity,
                    description=f"Found {len(outlier_indices)} values with unusual lengths (mean: {mean_length:.1f}, std: {std_length:.1f})",
                    statistical_evidence={
                        'mean_length': mean_length,
                        'std_length': std_length,
                        'outlier_count': len(outlier_indices),
                        'outlier_percentage': (len(outlier_indices) / len(series)) * 100,
                        'outlier_lengths': outlier_lengths[:10]
                    },
                    recommended_action="Review values with unusual lengths for data quality issues"
                ))
        
        return anomalies
    
    async def _detect_format_deviations(self, series: pd.Series) -> List[AnomalyPattern]:
        """Detect format deviations using pattern analysis."""
        
        anomalies = []
        
        # Extract format templates
        templates = {}
        for value in series.unique()[:500]:  # Limit for performance
            template = self._string_to_template(str(value))
            if template not in templates:
                templates[template] = []
            templates[template].append(str(value))
        
        # Find dominant template
        if len(templates) < 2:
            return anomalies
        
        template_counts = {t: len(values) for t, values in templates.items()}
        sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        
        dominant_template, dominant_count = sorted_templates[0]
        total_values = len(series)
        
        # If dominant template covers less than 70%, we might have format deviations
        if dominant_count / total_values < 0.7:
            deviation_values = []
            for template, values in templates.items():
                if template != dominant_template and len(values) < dominant_count * 0.1:
                    deviation_values.extend(values[:5])  # Limit examples
            
            if deviation_values:
                anomalies.append(AnomalyPattern(
                    pattern_type=AnomalyPatternType.FORMAT_DEVIATION,
                    affected_values=deviation_values[:10],
                    confidence=0.7,
                    severity="medium",
                    description=f"Format inconsistencies detected. Dominant pattern covers {(dominant_count/total_values)*100:.1f}% of values",
                    statistical_evidence={
                        'dominant_template': dominant_template,
                        'dominant_percentage': (dominant_count / total_values) * 100,
                        'template_count': len(templates),
                        'deviation_count': len(deviation_values)
                    },
                    recommended_action="Standardize data formats or investigate source of inconsistencies"
                ))
        
        return anomalies
    
    async def _detect_encoding_anomalies(self, series: pd.Series) -> List[AnomalyPattern]:
        """Detect encoding anomalies and special characters."""
        
        anomalies = []
        encoding_issues = []
        
        for value in series.unique()[:500]:
            str_value = str(value)
            
            # Check for common encoding issues
            if any(char in str_value for char in ['�', '\ufffd', '\x00']):
                encoding_issues.append(str_value)
            
            # Check for mixed encodings
            try:
                str_value.encode('ascii')
            except UnicodeEncodeError:
                # Contains non-ASCII characters - could be legitimate or encoding issue
                if len([c for c in str_value if ord(c) > 127]) / len(str_value) > 0.5:
                    encoding_issues.append(str_value)
        
        if encoding_issues:
            anomalies.append(AnomalyPattern(
                pattern_type=AnomalyPatternType.ENCODING_ANOMALY,
                affected_values=encoding_issues[:10],
                confidence=0.9,
                severity="high",
                description=f"Found {len(encoding_issues)} values with potential encoding issues",
                statistical_evidence={
                    'encoding_issue_count': len(encoding_issues),
                    'encoding_issue_percentage': (len(encoding_issues) / len(series)) * 100
                },
                recommended_action="Check data source encoding and conversion processes"
            ))
        
        return anomalies
    
    async def _detect_frequency_anomalies(self, series: pd.Series) -> List[AnomalyPattern]:
        """Detect frequency anomalies using statistical analysis."""
        
        anomalies = []
        value_counts = series.value_counts()
        
        if len(value_counts) < 2:
            return anomalies
        
        # Use isolation forest to detect frequency outliers
        if SKLEARN_AVAILABLE and len(value_counts) > 5:
            try:
                frequencies = value_counts.values.reshape(-1, 1)
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(frequencies)
                
                outlier_indices = np.where(outlier_labels == -1)[0]
                if len(outlier_indices) > 0:
                    outlier_values = [value_counts.index[i] for i in outlier_indices]
                    outlier_frequencies = [value_counts.iloc[i] for i in outlier_indices]
                    
                    anomalies.append(AnomalyPattern(
                        pattern_type=AnomalyPatternType.FREQUENCY_ANOMALY,
                        affected_values=[str(v) for v in outlier_values[:10]],
                        confidence=0.6,
                        severity="low",
                        description=f"Found {len(outlier_indices)} values with unusual frequencies",
                        statistical_evidence={
                            'outlier_count': len(outlier_indices),
                            'outlier_frequencies': outlier_frequencies[:10],
                            'mean_frequency': np.mean(frequencies),
                            'std_frequency': np.std(frequencies)
                        },
                        recommended_action="Review unusually frequent or rare values for data quality"
                    ))
            except Exception as e:
                self.logger.debug(f"Frequency anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_semantic_inconsistencies(
        self,
        series: pd.Series,
        column_name: str
    ) -> List[AnomalyPattern]:
        """Detect semantic inconsistencies based on expected patterns."""
        
        anomalies = []
        
        # Check against anomaly indicators
        anomaly_indicators = self._pattern_library['anomaly_indicators']
        suspicious_values = []
        
        for value in series.unique()[:500]:
            str_value = str(value).lower()
            if any(re.search(pattern, str_value) for pattern in anomaly_indicators):
                suspicious_values.append(str(value))
        
        if suspicious_values:
            anomalies.append(AnomalyPattern(
                pattern_type=AnomalyPatternType.SEMANTIC_INCONSISTENCY,
                affected_values=suspicious_values[:10],
                confidence=0.8,
                severity="medium",
                description=f"Found {len(suspicious_values)} values indicating potential data quality issues",
                statistical_evidence={
                    'suspicious_count': len(suspicious_values),
                    'suspicious_percentage': (len(suspicious_values) / len(series)) * 100
                },
                recommended_action="Investigate and clean suspicious values"
            ))
        
        return anomalies
    
    async def _analyze_temporal_patterns(
        self,
        series: pd.Series,
        column_name: str
    ) -> TemporalPatternAnalysis:
        """Analyze temporal patterns for time-series anomaly detection."""
        
        # Try to convert to datetime
        try:
            datetime_series = pd.to_datetime(series, errors='coerce')
            valid_dates = datetime_series.dropna()
            
            if len(valid_dates) < 10:  # Need minimum data for analysis
                return TemporalPatternAnalysis(
                    has_temporal_structure=False,
                    seasonality_detected=False,
                    trend_detected=False,
                    cycle_length=None,
                    temporal_anomalies=[]
                )
            
            # Sort by datetime
            sorted_dates = valid_dates.sort_values()
            
            # Detect patterns
            has_temporal_structure = True
            seasonality_detected = False
            trend_detected = False
            cycle_length = None
            temporal_anomalies = []
            
            # Basic trend detection
            if len(sorted_dates) > 30:
                # Check for monotonic trend
                diffs = sorted_dates.diff().dt.total_seconds().dropna()
                if len(diffs) > 0:
                    positive_diffs = (diffs > 0).sum()
                    negative_diffs = (diffs < 0).sum()
                    
                    if positive_diffs / len(diffs) > 0.8:
                        trend_detected = True
                    elif negative_diffs / len(diffs) > 0.8:
                        trend_detected = True
            
            # Gap analysis for temporal anomalies
            if len(sorted_dates) > 2:
                gaps = sorted_dates.diff().dt.total_seconds().dropna()
                if len(gaps) > 0:
                    median_gap = np.median(gaps)
                    large_gaps = gaps[gaps > median_gap * 5]  # Gaps 5x larger than median
                    
                    for gap_idx in large_gaps.index:
                        temporal_anomalies.append({
                            'type': 'large_gap',
                            'timestamp': sorted_dates.iloc[gap_idx-1],
                            'gap_duration_seconds': gaps.iloc[gap_idx-1],
                            'expected_duration_seconds': median_gap
                        })
            
            # Simple seasonality detection (for daily/weekly patterns)
            if len(sorted_dates) > 50:
                # Check for weekly patterns
                day_of_week_counts = sorted_dates.dt.dayofweek.value_counts()
                if day_of_week_counts.std() / day_of_week_counts.mean() < 0.3:  # Low variation
                    seasonality_detected = True
                    cycle_length = 7  # Weekly cycle
            
            return TemporalPatternAnalysis(
                has_temporal_structure=has_temporal_structure,
                seasonality_detected=seasonality_detected,
                trend_detected=trend_detected,
                cycle_length=cycle_length,
                temporal_anomalies=temporal_anomalies
            )
            
        except Exception as e:
            self.logger.debug(f"Temporal analysis failed: {e}")
            return TemporalPatternAnalysis(
                has_temporal_structure=False,
                seasonality_detected=False,
                trend_detected=False,
                cycle_length=None,
                temporal_anomalies=[]
            )
    
    async def _infer_intelligent_type(
        self,
        series: pd.Series,
        column_name: str
    ) -> IntelligentTypeInference:
        """Intelligent data type inference with confidence scoring."""
        
        # Initial type inference
        type_evidence = {}
        
        # Check for different data types
        numeric_count = 0
        date_count = 0
        boolean_count = 0
        
        for value in series.dropna().unique()[:500]:
            str_value = str(value)
            
            # Numeric check
            try:
                float(str_value)
                numeric_count += 1
            except ValueError:
                pass
            
            # Date check
            try:
                pd.to_datetime(str_value)
                date_count += 1
            except (ValueError, TypeError):
                pass
            
            # Boolean check
            if str_value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                boolean_count += 1
        
        total_unique = len(series.dropna().unique())
        if total_unique == 0:
            total_unique = 1
        
        # Calculate type probabilities
        numeric_prob = numeric_count / total_unique
        date_prob = date_count / total_unique
        boolean_prob = boolean_count / total_unique
        string_prob = 1.0 - max(numeric_prob, date_prob, boolean_prob)
        
        # Type evolution simulation (how confidence changes with more data)
        type_evolution = [
            (DataType.FLOAT, numeric_prob),
            (DataType.DATETIME, date_prob),
            (DataType.BOOLEAN, boolean_prob),
            (DataType.STRING, string_prob)
        ]
        type_evolution.sort(key=lambda x: x[1], reverse=True)
        
        # Determine primary type
        inferred_type = type_evolution[0][0]
        confidence = type_evolution[0][1]
        
        # Mixed type analysis
        mixed_type_analysis = None
        if confidence < 0.8:  # Mixed types detected
            mixed_type_analysis = {
                'is_mixed': True,
                'type_distribution': {
                    'numeric': numeric_prob,
                    'date': date_prob,
                    'boolean': boolean_prob,
                    'string': string_prob
                },
                'recommendation': 'Consider data cleaning or type conversion'
            }
        
        return IntelligentTypeInference(
            inferred_type=inferred_type,
            confidence=confidence,
            semantic_type=None,  # Will be filled from semantic classification
            type_evolution=type_evolution,
            mixed_type_analysis=mixed_type_analysis
        )
    
    async def _extract_ml_insights(
        self,
        series: pd.Series,
        column_name: str
    ) -> Dict[str, Any]:
        """Extract ML-powered insights from the data."""
        
        insights = {}
        
        if not SKLEARN_AVAILABLE:
            return insights
        
        try:
            string_series = series.astype(str)
            unique_values = string_series.unique()
            
            if len(unique_values) < 5:
                return insights
            
            # Feature extraction for clustering
            features = self._extract_advanced_string_features(unique_values)
            
            if features.shape[0] > 3 and features.shape[1] > 0:
                # Clustering analysis
                cluster_insights = await self._perform_clustering_analysis(features, unique_values)
                insights['clustering'] = cluster_insights
                
                # Dimensionality reduction insights
                if features.shape[1] > 2:
                    dim_reduction_insights = await self._perform_dimensionality_analysis(features)
                    insights['dimensionality'] = dim_reduction_insights
                
                # Anomaly detection using ML
                anomaly_insights = await self._ml_anomaly_detection(features, unique_values)
                insights['ml_anomalies'] = anomaly_insights
        
        except Exception as e:
            self.logger.debug(f"ML insights extraction failed: {e}")
        
        return insights
    
    def _extract_advanced_string_features(self, unique_values: np.ndarray) -> np.ndarray:
        """Extract advanced features from strings for ML analysis."""
        
        features = []
        
        for value in unique_values:
            str_value = str(value)
            feature_vector = [
                len(str_value),  # Length
                sum(c.isdigit() for c in str_value),  # Digit count
                sum(c.isalpha() for c in str_value),  # Letter count
                sum(c.isupper() for c in str_value),  # Uppercase count
                sum(c.islower() for c in str_value),  # Lowercase count
                sum(not c.isalnum() and not c.isspace() for c in str_value),  # Special char count
                len(set(str_value)),  # Unique character count
                str_value.count(' '),  # Space count
                str_value.count('-'),  # Hyphen count
                str_value.count('.'),  # Dot count
                str_value.count('_'),  # Underscore count
                str_value.count('@'),  # At symbol count
                len(str_value.split()),  # Word count
            ]
            
            # Add text complexity features if available
            if TEXTSTAT_AVAILABLE:
                try:
                    feature_vector.extend([
                        textstat.flesch_reading_ease(str_value) if len(str_value) > 10 else 0,
                        textstat.flesch_kincaid_grade(str_value) if len(str_value) > 10 else 0,
                    ])
                except Exception:
                    feature_vector.extend([0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _perform_clustering_analysis(
        self,
        features: np.ndarray,
        unique_values: np.ndarray
    ) -> Dict[str, Any]:
        """Perform clustering analysis on string features."""
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal cluster count
            n_clusters = min(max(2, len(unique_values) // 5), 8)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            # Analyze clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(str(unique_values[i]))
            
            return {
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'cluster_sizes': [len(values) for values in clusters.values()],
                'cluster_examples': {
                    f'cluster_{k}': v[:3] for k, v in clusters.items()
                }
            }
        
        except Exception as e:
            self.logger.debug(f"Clustering analysis failed: {e}")
            return {}
    
    async def _perform_dimensionality_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform dimensionality reduction analysis."""
        
        try:
            # PCA analysis
            pca = PCA()
            pca.fit(features)
            
            # Find number of components for 95% variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            return {
                'original_dimensions': features.shape[1],
                'components_for_95_variance': int(n_components_95),
                'explained_variance_ratio': pca.explained_variance_ratio_[:5].tolist(),
                'intrinsic_dimensionality': int(n_components_95)
            }
        
        except Exception as e:
            self.logger.debug(f"Dimensionality analysis failed: {e}")
            return {}
    
    async def _ml_anomaly_detection(
        self,
        features: np.ndarray,
        unique_values: np.ndarray
    ) -> Dict[str, Any]:
        """ML-based anomaly detection on string features."""
        
        try:
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            anomaly_values = [str(unique_values[i]) for i in anomaly_indices]
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(unique_values)) * 100,
                'anomaly_examples': anomaly_values[:5],
                'model_type': 'IsolationForest'
            }
        
        except Exception as e:
            self.logger.debug(f"ML anomaly detection failed: {e}")
            return {}
    
    async def _analyze_cross_column_relationships(
        self,
        df: pd.DataFrame,
        column_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships and patterns across multiple columns."""
        
        cross_analysis = {
            'semantic_relationships': [],
            'pattern_correlations': [],
            'anomaly_clusters': [],
            'data_quality_score': 0.0
        }
        
        try:
            # Semantic relationship analysis
            semantic_types = {}
            for col, results in column_results.items():
                if col.startswith('_'):  # Skip metadata columns
                    continue
                
                semantic_classification = results.get('semantic_classification')
                if semantic_classification:
                    semantic_type = semantic_classification.semantic_type
                    if semantic_type not in semantic_types:
                        semantic_types[semantic_type] = []
                    semantic_types[semantic_type].append(col)
            
            cross_analysis['semantic_relationships'] = [
                {'semantic_type': st.value, 'columns': cols}
                for st, cols in semantic_types.items() if len(cols) > 1
            ]
            
            # Pattern correlation analysis
            pattern_correlations = []
            columns = [col for col in column_results.keys() if not col.startswith('_')]
            
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    patterns1 = column_results[col1].get('basic_patterns', [])
                    patterns2 = column_results[col2].get('basic_patterns', [])
                    
                    if patterns1 and patterns2:
                        # Check for similar patterns
                        pattern_types1 = {p.pattern_type for p in patterns1}
                        pattern_types2 = {p.pattern_type for p in patterns2}
                        
                        common_patterns = pattern_types1.intersection(pattern_types2)
                        if common_patterns:
                            pattern_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'common_patterns': [pt.value for pt in common_patterns],
                                'correlation_strength': len(common_patterns) / max(len(pattern_types1), len(pattern_types2))
                            })
            
            cross_analysis['pattern_correlations'] = pattern_correlations
            
            # Anomaly clustering
            columns_with_anomalies = []
            for col, results in column_results.items():
                if col.startswith('_'):
                    continue
                    
                anomaly_patterns = results.get('anomaly_patterns', [])
                if anomaly_patterns:
                    columns_with_anomalies.append({
                        'column': col,
                        'anomaly_count': len(anomaly_patterns),
                        'severity_distribution': {
                            ap.severity: 1 for ap in anomaly_patterns
                        }
                    })
            
            cross_analysis['anomaly_clusters'] = columns_with_anomalies
            
            # Overall data quality score
            total_columns = len([col for col in column_results.keys() if not col.startswith('_')])
            if total_columns > 0:
                anomaly_score = len(columns_with_anomalies) / total_columns
                quality_score = max(0.0, 1.0 - anomaly_score)
                cross_analysis['data_quality_score'] = quality_score
        
        except Exception as e:
            self.logger.error(f"Cross-column analysis failed: {e}")
        
        return cross_analysis
    
    # Validation functions for semantic types
    def _validate_email(self, values: List[str]) -> float:
        """Validate email addresses."""
        valid_count = 0
        for value in values[:50]:  # Limit for performance
            if '@' in value and '.' in value.split('@')[-1]:
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_phone(self, values: List[str]) -> float:
        """Validate phone numbers."""
        valid_count = 0
        for value in values[:50]:
            digits = re.sub(r'[^\d]', '', value)
            if 10 <= len(digits) <= 15:  # Reasonable phone number length
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_uuid(self, values: List[str]) -> float:
        """Validate UUIDs."""
        valid_count = 0
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        for value in values[:50]:
            if re.match(uuid_pattern, value, re.IGNORECASE):
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_ssn(self, values: List[str]) -> float:
        """Validate SSN format."""
        valid_count = 0
        for value in values[:50]:
            if re.match(r'^\d{3}-\d{2}-\d{4}$', value):
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_credit_card(self, values: List[str]) -> float:
        """Validate credit card format."""
        valid_count = 0
        for value in values[:50]:
            digits = re.sub(r'[^\d]', '', value)
            if len(digits) in [13, 14, 15, 16]:  # Common credit card lengths
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_ip(self, values: List[str]) -> float:
        """Validate IP addresses."""
        valid_count = 0
        for value in values[:50]:
            parts = value.split('.')
            if len(parts) == 4:
                try:
                    if all(0 <= int(part) <= 255 for part in parts):
                        valid_count += 1
                except ValueError:
                    pass
        return valid_count / min(len(values), 50)
    
    def _validate_url(self, values: List[str]) -> float:
        """Validate URLs."""
        valid_count = 0
        for value in values[:50]:
            if value.startswith(('http://', 'https://')) and '.' in value:
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_currency(self, values: List[str]) -> float:
        """Validate currency format."""
        valid_count = 0
        for value in values[:50]:
            if re.match(r'^\$?\d+(\.\d{2})?$', value):
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_postal_code(self, values: List[str]) -> float:
        """Validate postal codes."""
        valid_count = 0
        for value in values[:50]:
            if re.match(r'^\d{5}(-\d{4})?$', value):
                valid_count += 1
        return valid_count / min(len(values), 50)
    
    def _validate_date(self, values: List[str]) -> float:
        """Validate date formats."""
        valid_count = 0
        for value in values[:50]:
            try:
                pd.to_datetime(value)
                valid_count += 1
            except (ValueError, TypeError):
                pass
        return valid_count / min(len(values), 50)
    
    def _is_temporal_candidate(self, series: pd.Series) -> bool:
        """Check if a series is a candidate for temporal analysis."""
        # Check if values look like dates
        sample_values = series.dropna().astype(str).unique()[:10]
        date_like_count = 0
        
        for value in sample_values:
            try:
                pd.to_datetime(value)
                date_like_count += 1
            except (ValueError, TypeError):
                pass
        
        return date_like_count / len(sample_values) > 0.5 if sample_values.size > 0 else False
    
    def _string_to_template(self, s: str) -> str:
        """Convert string to format template for pattern analysis."""
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
    
    async def generate_pattern_summary_report(
        self,
        pattern_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive pattern discovery summary report."""
        
        report = {
            'summary': {},
            'recommendations': [],
            'data_quality_assessment': {},
            'anomaly_summary': {},
            'semantic_insights': {}
        }
        
        try:
            columns_analyzed = len([k for k in pattern_results.keys() if not k.startswith('_')])
            
            # Summary statistics
            total_patterns = 0
            total_anomalies = 0
            semantic_types_found = set()
            
            for col, results in pattern_results.items():
                if col.startswith('_'):
                    continue
                
                basic_patterns = results.get('basic_patterns', [])
                total_patterns += len(basic_patterns)
                
                anomaly_patterns = results.get('anomaly_patterns', [])
                total_anomalies += len(anomaly_patterns)
                
                semantic_classification = results.get('semantic_classification')
                if semantic_classification:
                    semantic_types_found.add(semantic_classification.semantic_type)
            
            report['summary'] = {
                'columns_analyzed': columns_analyzed,
                'total_patterns_discovered': total_patterns,
                'total_anomalies_detected': total_anomalies,
                'unique_semantic_types': len(semantic_types_found),
                'average_patterns_per_column': total_patterns / columns_analyzed if columns_analyzed > 0 else 0,
                'anomaly_rate': total_anomalies / columns_analyzed if columns_analyzed > 0 else 0
            }
            
            # Generate recommendations
            if total_anomalies > columns_analyzed * 0.3:  # High anomaly rate
                report['recommendations'].append("High anomaly rate detected - recommend comprehensive data quality review")
            
            if len(semantic_types_found) > columns_analyzed * 0.5:  # Rich semantic diversity
                report['recommendations'].append("Rich semantic diversity detected - consider implementing data governance policies")
            
            cross_column = pattern_results.get('_cross_column_analysis', {})
            if cross_column.get('data_quality_score', 1.0) < 0.7:
                report['recommendations'].append("Data quality score below threshold - implement data quality monitoring")
            
            report['data_quality_assessment'] = {
                'overall_score': cross_column.get('data_quality_score', 1.0),
                'semantic_consistency': len(cross_column.get('semantic_relationships', [])),
                'pattern_correlations': len(cross_column.get('pattern_correlations', []))
            }
            
            # Anomaly summary
            anomaly_types = defaultdict(int)
            severity_distribution = defaultdict(int)
            
            for col, results in pattern_results.items():
                if col.startswith('_'):
                    continue
                
                for anomaly in results.get('anomaly_patterns', []):
                    anomaly_types[anomaly.pattern_type.value] += 1
                    severity_distribution[anomaly.severity] += 1
            
            report['anomaly_summary'] = {
                'types': dict(anomaly_types),
                'severity_distribution': dict(severity_distribution)
            }
            
            # Semantic insights
            semantic_distribution = defaultdict(int)
            for col, results in pattern_results.items():
                if col.startswith('_'):
                    continue
                
                semantic_classification = results.get('semantic_classification')
                if semantic_classification:
                    semantic_distribution[semantic_classification.semantic_type.value] += 1
            
            report['semantic_insights'] = {
                'type_distribution': dict(semantic_distribution),
                'most_common_type': max(semantic_distribution.items(), key=lambda x: x[1])[0] if semantic_distribution else None
            }
        
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
        
        return report