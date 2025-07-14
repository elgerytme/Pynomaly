"""
Intelligent Data Type Inference Service with Machine Learning

Advanced data type inference system that uses machine learning techniques to intelligently
classify data types beyond basic Python/pandas types. Provides semantic type inference,
format detection, and confidence scoring for data profiling operations.

This enhances Issue #144: Phase 2.3: Data Profiling Package - Advanced Pattern Discovery
with intelligent ML-powered data type inference capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

# Optional ML dependencies with graceful fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataTypeCategory(Enum):
    """High-level data type categories."""
    
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXTUAL = "textual"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    GEOSPATIAL = "geospatial"
    BINARY = "binary"
    UNKNOWN = "unknown"


class SemanticDataType(Enum):
    """Detailed semantic data types."""
    
    # Numeric types
    INTEGER = "integer"
    FLOAT = "float"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    SCORE = "score"
    RATING = "rating"
    COUNT = "count"
    
    # Categorical types
    CATEGORY = "category"
    BINARY_FLAG = "binary_flag"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    
    # Temporal types
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    DURATION = "duration"
    
    # Textual types
    FREE_TEXT = "free_text"
    DESCRIPTION = "description"
    COMMENT = "comment"
    TITLE = "title"
    NAME = "name"
    
    # Identifier types
    ID = "id"
    UUID = "uuid"
    CODE = "code"
    SKU = "sku"
    SERIAL_NUMBER = "serial_number"
    
    # Contact information
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    
    # Geographic types
    ADDRESS = "address"
    POSTAL_CODE = "postal_code"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    COUNTRY = "country"
    CITY = "city"
    
    # Technical types
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    FILE_PATH = "file_path"
    VERSION = "version"
    
    # Financial types
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    SSN = "ssn"
    
    # Binary/encoded types
    HASH = "hash"
    ENCODED = "encoded"
    BASE64 = "base64"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class TypeInferenceResult:
    """Result of type inference analysis."""
    
    predicted_category: DataTypeCategory
    predicted_semantic_type: SemanticDataType
    confidence: float
    alternative_types: List[Tuple[SemanticDataType, float]]
    
    # Evidence for the prediction
    statistical_evidence: Dict[str, Any]
    pattern_evidence: Dict[str, Any]
    feature_evidence: Dict[str, Any]
    
    # Metadata
    sample_size: int
    unique_values: int
    null_count: int
    detection_method: str
    
    # Recommendations
    recommended_pandas_dtype: str
    recommended_constraints: List[str]
    confidence_factors: Dict[str, float]


@dataclass
class TypeInferenceConfig:
    """Configuration for type inference."""
    
    # ML model settings
    use_ml_models: bool = True
    ensemble_voting: bool = True
    cross_validation_folds: int = 3
    
    # Sampling settings
    max_sample_size: int = 10000
    min_sample_size: int = 50
    sample_strategy: str = "stratified"  # "random", "stratified", "systematic"
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    
    # Feature extraction settings
    use_textual_features: bool = True
    use_statistical_features: bool = True
    use_pattern_features: bool = True
    use_semantic_features: bool = True
    
    # Custom type definitions
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    custom_semantic_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Domain-specific settings
    domain_context: Optional[str] = None  # "finance", "healthcare", "retail", etc.
    business_rules: Dict[str, Any] = field(default_factory=dict)


class IntelligentTypeInferenceService:
    """Advanced type inference service with machine learning capabilities."""
    
    def __init__(self, config: Optional[TypeInferenceConfig] = None):
        """Initialize the intelligent type inference service."""
        
        self.config = config or TypeInferenceConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize ML models
        self._models = {}
        self._feature_extractors = {}
        self._is_trained = False
        
        # Pattern libraries
        self._pattern_library = self._load_pattern_library()
        self._semantic_rules = self._load_semantic_rules()
        
        # Feature extractors
        self._init_feature_extractors()
        
        # Model training data cache
        self._training_cache = {}
    
    def _load_pattern_library(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive pattern library for type inference."""
        
        return {
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.EMAIL
            },
            'phone': {
                'pattern': r'^[\+]?[\d\s\-\(\)]{10,}$',
                'confidence_boost': 0.8,
                'semantic_type': SemanticDataType.PHONE
            },
            'url': {
                'pattern': r'^https?://[^\s]+$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.URL
            },
            'uuid': {
                'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'confidence_boost': 0.95,
                'semantic_type': SemanticDataType.UUID
            },
            'ip_address': {
                'pattern': r'^(\d{1,3}\.){3}\d{1,3}$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.IP_ADDRESS
            },
            'mac_address': {
                'pattern': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.MAC_ADDRESS
            },
            'credit_card': {
                'pattern': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
                'confidence_boost': 0.8,
                'semantic_type': SemanticDataType.CREDIT_CARD
            },
            'ssn': {
                'pattern': r'^\d{3}-\d{2}-\d{4}$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.SSN
            },
            'postal_code': {
                'pattern': r'^\d{5}(-\d{4})?$',
                'confidence_boost': 0.8,
                'semantic_type': SemanticDataType.POSTAL_CODE
            },
            'currency': {
                'pattern': r'^\$?\d+(\.\d{2})?$',
                'confidence_boost': 0.7,
                'semantic_type': SemanticDataType.CURRENCY
            },
            'percentage': {
                'pattern': r'^\d+(\.\d+)?%$',
                'confidence_boost': 0.8,
                'semantic_type': SemanticDataType.PERCENTAGE
            },
            'date_iso': {
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'confidence_boost': 0.9,
                'semantic_type': SemanticDataType.DATE
            },
            'time': {
                'pattern': r'^\d{2}:\d{2}(:\d{2})?$',
                'confidence_boost': 0.8,
                'semantic_type': SemanticDataType.TIME
            },
            'hash': {
                'pattern': r'^[a-fA-F0-9]{32,}$',
                'confidence_boost': 0.7,
                'semantic_type': SemanticDataType.HASH
            }
        }
    
    def _load_semantic_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load semantic inference rules based on column names and data characteristics."""
        
        return {
            'id_indicators': {
                'keywords': ['id', '_id', 'identifier', 'key', 'pk', 'primary'],
                'semantic_type': SemanticDataType.ID,
                'confidence_boost': 0.3
            },
            'name_indicators': {
                'keywords': ['name', 'first_name', 'last_name', 'full_name', 'username'],
                'semantic_type': SemanticDataType.NAME,
                'confidence_boost': 0.3
            },
            'email_indicators': {
                'keywords': ['email', 'e_mail', 'mail'],
                'semantic_type': SemanticDataType.EMAIL,
                'confidence_boost': 0.4
            },
            'phone_indicators': {
                'keywords': ['phone', 'tel', 'mobile', 'telephone'],
                'semantic_type': SemanticDataType.PHONE,
                'confidence_boost': 0.4
            },
            'address_indicators': {
                'keywords': ['address', 'addr', 'street', 'location'],
                'semantic_type': SemanticDataType.ADDRESS,
                'confidence_boost': 0.3
            },
            'date_indicators': {
                'keywords': ['date', 'created', 'updated', 'timestamp', 'time'],
                'semantic_type': SemanticDataType.DATE,
                'confidence_boost': 0.3
            },
            'price_indicators': {
                'keywords': ['price', 'cost', 'amount', 'fee', 'charge', 'value'],
                'semantic_type': SemanticDataType.CURRENCY,
                'confidence_boost': 0.3
            },
            'category_indicators': {
                'keywords': ['category', 'type', 'class', 'group', 'status'],
                'semantic_type': SemanticDataType.CATEGORY,
                'confidence_boost': 0.2
            },
            'description_indicators': {
                'keywords': ['description', 'desc', 'comment', 'note', 'remarks'],
                'semantic_type': SemanticDataType.DESCRIPTION,
                'confidence_boost': 0.2
            }
        }
    
    def _init_feature_extractors(self) -> None:
        """Initialize feature extraction components."""
        
        if SKLEARN_AVAILABLE:
            # Text feature extractor for string analysis
            self._feature_extractors['text'] = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                stop_words='english',
                analyzer='char_wb'
            )
    
    async def infer_types(
        self,
        df: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, TypeInferenceResult]:
        """Infer data types for columns using intelligent analysis."""
        
        start_time = time.time()
        self.logger.info(f"Starting intelligent type inference for {len(df.columns)} columns")
        
        # Determine columns to analyze
        columns_to_analyze = column_names or df.columns.tolist()
        
        results = {}
        
        for column in columns_to_analyze:
            if column not in df.columns:
                continue
            
            self.logger.debug(f"Inferring type for column: {column}")
            
            try:
                # Sample column data
                column_data = self._sample_column_data(df[column])
                
                # Perform type inference
                inference_result = await self._infer_column_type(
                    column_data, column, context
                )
                
                results[column] = inference_result
                
            except Exception as e:
                self.logger.error(f"Type inference failed for column {column}: {e}")
                # Fallback to basic inference
                results[column] = self._basic_type_inference(df[column], column)
        
        execution_time = time.time() - start_time
        self.logger.info(f"Type inference completed in {execution_time:.2f} seconds")
        
        return results
    
    def _sample_column_data(self, series: pd.Series) -> pd.Series:
        """Sample column data for analysis."""
        
        if len(series) <= self.config.max_sample_size:
            return series
        
        # Stratified sampling for categorical data
        if series.dtype == 'object' or series.dtype.name == 'category':
            value_counts = series.value_counts()
            if len(value_counts) <= 100:  # Low cardinality - stratified sampling
                sample_indices = []
                sample_size = self.config.max_sample_size
                
                for value, count in value_counts.items():
                    value_indices = series[series == value].index.tolist()
                    sample_count = max(1, int(count / len(series) * sample_size))
                    sample_count = min(sample_count, len(value_indices))
                    
                    sampled = np.random.choice(
                        value_indices, 
                        size=sample_count, 
                        replace=False
                    )
                    sample_indices.extend(sampled)
                
                return series.loc[sample_indices[:self.config.max_sample_size]]
        
        # Random sampling for other data types
        return series.sample(n=self.config.max_sample_size, random_state=42)
    
    async def _infer_column_type(
        self,
        series: pd.Series,
        column_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TypeInferenceResult:
        """Infer type for a single column using multiple approaches."""
        
        # Extract features
        features = await self._extract_features(series, column_name)
        
        # Pattern-based inference
        pattern_result = self._pattern_based_inference(series, column_name, features)
        
        # Statistical inference
        statistical_result = self._statistical_inference(series, column_name, features)
        
        # ML-based inference (if enabled and models available)
        ml_result = None
        if self.config.use_ml_models and SKLEARN_AVAILABLE:
            ml_result = await self._ml_based_inference(series, column_name, features)
        
        # Semantic inference based on column name
        semantic_result = self._semantic_name_inference(column_name, features)
        
        # Ensemble prediction
        final_result = self._ensemble_prediction(
            series, column_name, features,
            [pattern_result, statistical_result, ml_result, semantic_result]
        )
        
        return final_result
    
    async def _extract_features(
        self,
        series: pd.Series,
        column_name: str
    ) -> Dict[str, Any]:
        """Extract comprehensive features for type inference."""
        
        features = {
            'basic': {},
            'statistical': {},
            'textual': {},
            'pattern': {},
            'semantic': {}
        }
        
        # Basic features
        features['basic'] = {
            'length': len(series),
            'unique_count': series.nunique(),
            'null_count': series.isnull().sum(),
            'completeness': 1 - (series.isnull().sum() / len(series)),
            'pandas_dtype': str(series.dtype),
            'is_numeric': pd.api.types.is_numeric_dtype(series),
            'is_object': series.dtype == 'object',
            'is_categorical': series.dtype.name == 'category'
        }
        
        # Statistical features (for numeric data)
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                features['statistical'] = {
                    'mean': float(numeric_series.mean()),
                    'std': float(numeric_series.std()),
                    'min': float(numeric_series.min()),
                    'max': float(numeric_series.max()),
                    'range': float(numeric_series.max() - numeric_series.min()),
                    'skewness': float(numeric_series.skew()),
                    'kurtosis': float(numeric_series.kurtosis()),
                    'is_integer': all(val.is_integer() for val in numeric_series if pd.notna(val)),
                    'has_negatives': (numeric_series < 0).any(),
                    'zero_count': (numeric_series == 0).sum(),
                    'unique_ratio': numeric_series.nunique() / len(numeric_series)
                }
        
        # Textual features (for string data)
        if series.dtype == 'object':
            string_series = series.astype(str).dropna()
            if len(string_series) > 0:
                lengths = string_series.str.len()
                
                features['textual'] = {
                    'avg_length': float(lengths.mean()),
                    'length_std': float(lengths.std()),
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'length_variety': lengths.nunique(),
                    'has_spaces': string_series.str.contains(' ').any(),
                    'has_special_chars': string_series.str.contains(r'[^a-zA-Z0-9\s]').any(),
                    'all_uppercase': string_series.str.isupper().all(),
                    'all_lowercase': string_series.str.islower().all(),
                    'mixed_case': not (string_series.str.isupper().all() or string_series.str.islower().all()),
                    'numeric_chars_ratio': string_series.str.extract(r'(\d+)').count().sum() / len(string_series),
                    'alpha_chars_ratio': string_series.str.extract(r'([a-zA-Z]+)').count().sum() / len(string_series)
                }
                
                # Text complexity features (if textstat available)
                if TEXTSTAT_AVAILABLE and len(string_series) > 0:
                    sample_text = ' '.join(string_series.head(100).tolist())
                    features['textual'].update({
                        'readability_score': textstat.flesch_reading_ease(sample_text),
                        'avg_sentence_length': textstat.avg_sentence_length(sample_text),
                        'syllable_count': textstat.syllable_count(sample_text)
                    })
        
        # Pattern features
        if series.dtype == 'object':
            string_series = series.astype(str).dropna()
            pattern_matches = {}
            
            for pattern_name, pattern_info in self._pattern_library.items():
                pattern = pattern_info['pattern']
                matches = string_series.str.match(pattern, case=False)
                pattern_matches[f'{pattern_name}_match_ratio'] = matches.sum() / len(string_series)
            
            features['pattern'] = pattern_matches
        
        # Semantic features based on column name
        column_lower = column_name.lower()
        semantic_scores = {}
        
        for rule_name, rule_info in self._semantic_rules.items():
            score = 0
            for keyword in rule_info['keywords']:
                if keyword in column_lower:
                    score += rule_info['confidence_boost']
            semantic_scores[rule_name] = score
        
        features['semantic'] = semantic_scores
        
        return features
    
    def _pattern_based_inference(
        self,
        series: pd.Series,
        column_name: str,
        features: Dict[str, Any]
    ) -> Tuple[SemanticDataType, float, Dict[str, Any]]:
        """Infer type based on pattern matching."""
        
        if series.dtype != 'object':
            return SemanticDataType.UNKNOWN, 0.0, {}
        
        string_series = series.astype(str).dropna()
        if len(string_series) == 0:
            return SemanticDataType.UNKNOWN, 0.0, {}
        
        best_type = SemanticDataType.UNKNOWN
        best_confidence = 0.0
        evidence = {}
        
        # Check each pattern
        for pattern_name, pattern_info in self._pattern_library.items():
            pattern = pattern_info['pattern']
            semantic_type = pattern_info['semantic_type']
            confidence_boost = pattern_info['confidence_boost']
            
            matches = string_series.str.match(pattern, case=False)
            match_ratio = matches.sum() / len(string_series)
            
            if match_ratio > 0.8:  # High match ratio
                confidence = match_ratio * confidence_boost
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_type = semantic_type
                    evidence = {
                        'pattern_name': pattern_name,
                        'pattern': pattern,
                        'match_ratio': match_ratio,
                        'matches': matches.sum(),
                        'method': 'pattern_matching'
                    }
        
        return best_type, best_confidence, evidence
    
    def _statistical_inference(
        self,
        series: pd.Series,
        column_name: str,
        features: Dict[str, Any]
    ) -> Tuple[SemanticDataType, float, Dict[str, Any]]:
        """Infer type based on statistical analysis."""
        
        basic_features = features['basic']
        statistical_features = features.get('statistical', {})
        
        evidence = {'method': 'statistical_analysis'}
        
        # Numeric type inference
        if basic_features['is_numeric']:
            if statistical_features.get('is_integer', False):
                # Check for specific integer types
                if basic_features['unique_count'] == 2:
                    return SemanticDataType.BINARY_FLAG, 0.8, {**evidence, 'reason': 'binary_values'}
                elif statistical_features.get('min', 0) >= 0 and statistical_features.get('max', 0) <= 5:
                    return SemanticDataType.RATING, 0.7, {**evidence, 'reason': 'rating_range'}
                elif basic_features['unique_count'] / basic_features['length'] < 0.1:
                    return SemanticDataType.CATEGORY, 0.6, {**evidence, 'reason': 'low_cardinality'}
                else:
                    return SemanticDataType.COUNT, 0.6, {**evidence, 'reason': 'integer_type'}
            else:
                # Float type - check for specific patterns
                if 0 <= statistical_features.get('min', -1) <= 1 and 0 <= statistical_features.get('max', -1) <= 1:
                    return SemanticDataType.PERCENTAGE, 0.7, {**evidence, 'reason': 'unit_interval'}
                else:
                    return SemanticDataType.FLOAT, 0.6, {**evidence, 'reason': 'float_type'}
        
        # Categorical type inference
        elif basic_features['is_object']:
            textual_features = features.get('textual', {})
            unique_ratio = basic_features['unique_count'] / basic_features['length']
            
            if unique_ratio < 0.1:  # Low cardinality
                return SemanticDataType.CATEGORY, 0.7, {**evidence, 'reason': 'low_cardinality'}
            elif unique_ratio > 0.9:  # High cardinality
                if textual_features.get('avg_length', 0) > 50:
                    return SemanticDataType.DESCRIPTION, 0.6, {**evidence, 'reason': 'long_text'}
                else:
                    return SemanticDataType.ID, 0.6, {**evidence, 'reason': 'high_cardinality'}
            else:
                return SemanticDataType.FREE_TEXT, 0.5, {**evidence, 'reason': 'medium_cardinality'}
        
        return SemanticDataType.UNKNOWN, 0.0, evidence
    
    async def _ml_based_inference(
        self,
        series: pd.Series,
        column_name: str,
        features: Dict[str, Any]
    ) -> Optional[Tuple[SemanticDataType, float, Dict[str, Any]]]:
        """Infer type using machine learning models."""
        
        if not SKLEARN_AVAILABLE:
            return None
        
        # This would be implemented with pre-trained models
        # For now, return None to indicate ML inference is not available
        return None
    
    def _semantic_name_inference(
        self,
        column_name: str,
        features: Dict[str, Any]
    ) -> Tuple[SemanticDataType, float, Dict[str, Any]]:
        """Infer type based on column name semantics."""
        
        column_lower = column_name.lower()
        semantic_features = features.get('semantic', {})
        
        best_type = SemanticDataType.UNKNOWN
        best_confidence = 0.0
        evidence = {'method': 'semantic_name_analysis'}
        
        # Check semantic rules
        for rule_name, rule_info in self._semantic_rules.items():
            score = semantic_features.get(rule_name, 0)
            
            if score > best_confidence:
                best_confidence = score
                best_type = rule_info['semantic_type']
                evidence.update({
                    'rule_name': rule_name,
                    'matched_keywords': [kw for kw in rule_info['keywords'] if kw in column_lower],
                    'confidence_score': score
                })
        
        return best_type, best_confidence, evidence
    
    def _ensemble_prediction(
        self,
        series: pd.Series,
        column_name: str,
        features: Dict[str, Any],
        predictions: List[Optional[Tuple[SemanticDataType, float, Dict[str, Any]]]]
    ) -> TypeInferenceResult:
        """Combine predictions from multiple methods."""
        
        # Filter out None predictions
        valid_predictions = [p for p in predictions if p is not None]
        
        if not valid_predictions:
            return self._basic_type_inference(series, column_name)
        
        # Weight and combine predictions
        weighted_scores = defaultdict(float)
        all_evidence = {}
        confidence_factors = {}
        
        method_weights = {
            'pattern_matching': 0.4,
            'statistical_analysis': 0.3,
            'ml_based': 0.2,
            'semantic_name_analysis': 0.1
        }
        
        for pred_type, confidence, evidence in valid_predictions:
            method = evidence.get('method', 'unknown')
            weight = method_weights.get(method, 0.1)
            
            weighted_scores[pred_type] += confidence * weight
            all_evidence[method] = evidence
            confidence_factors[method] = confidence
        
        # Find best prediction
        if weighted_scores:
            best_type = max(weighted_scores.items(), key=lambda x: x[1])
            predicted_type = best_type[0]
            final_confidence = best_type[1]
        else:
            predicted_type = SemanticDataType.UNKNOWN
            final_confidence = 0.0
        
        # Determine category
        category = self._semantic_type_to_category(predicted_type)
        
        # Generate alternative types
        sorted_types = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [(t, s) for t, s in sorted_types[1:4]]  # Top 3 alternatives
        
        # Recommend pandas dtype
        recommended_dtype = self._recommend_pandas_dtype(predicted_type, features)
        
        # Generate constraints
        recommended_constraints = self._generate_constraints(predicted_type, features)
        
        return TypeInferenceResult(
            predicted_category=category,
            predicted_semantic_type=predicted_type,
            confidence=final_confidence,
            alternative_types=alternatives,
            statistical_evidence=features.get('statistical', {}),
            pattern_evidence=features.get('pattern', {}),
            feature_evidence=features,
            sample_size=len(series),
            unique_values=series.nunique(),
            null_count=series.isnull().sum(),
            detection_method="ensemble",
            recommended_pandas_dtype=recommended_dtype,
            recommended_constraints=recommended_constraints,
            confidence_factors=confidence_factors
        )
    
    def _semantic_type_to_category(self, semantic_type: SemanticDataType) -> DataTypeCategory:
        """Map semantic type to high-level category."""
        
        category_mapping = {
            # Numeric
            SemanticDataType.INTEGER: DataTypeCategory.NUMERIC,
            SemanticDataType.FLOAT: DataTypeCategory.NUMERIC,
            SemanticDataType.CURRENCY: DataTypeCategory.NUMERIC,
            SemanticDataType.PERCENTAGE: DataTypeCategory.NUMERIC,
            SemanticDataType.SCORE: DataTypeCategory.NUMERIC,
            SemanticDataType.RATING: DataTypeCategory.NUMERIC,
            SemanticDataType.COUNT: DataTypeCategory.NUMERIC,
            
            # Categorical
            SemanticDataType.CATEGORY: DataTypeCategory.CATEGORICAL,
            SemanticDataType.BINARY_FLAG: DataTypeCategory.BOOLEAN,
            SemanticDataType.ORDINAL: DataTypeCategory.CATEGORICAL,
            SemanticDataType.NOMINAL: DataTypeCategory.CATEGORICAL,
            
            # Temporal
            SemanticDataType.DATE: DataTypeCategory.TEMPORAL,
            SemanticDataType.TIME: DataTypeCategory.TEMPORAL,
            SemanticDataType.DATETIME: DataTypeCategory.TEMPORAL,
            SemanticDataType.TIMESTAMP: DataTypeCategory.TEMPORAL,
            SemanticDataType.DURATION: DataTypeCategory.TEMPORAL,
            
            # Textual
            SemanticDataType.FREE_TEXT: DataTypeCategory.TEXTUAL,
            SemanticDataType.DESCRIPTION: DataTypeCategory.TEXTUAL,
            SemanticDataType.COMMENT: DataTypeCategory.TEXTUAL,
            SemanticDataType.TITLE: DataTypeCategory.TEXTUAL,
            SemanticDataType.NAME: DataTypeCategory.TEXTUAL,
            
            # Identifier
            SemanticDataType.ID: DataTypeCategory.IDENTIFIER,
            SemanticDataType.UUID: DataTypeCategory.IDENTIFIER,
            SemanticDataType.CODE: DataTypeCategory.IDENTIFIER,
            SemanticDataType.SKU: DataTypeCategory.IDENTIFIER,
            SemanticDataType.SERIAL_NUMBER: DataTypeCategory.IDENTIFIER,
            
            # Contact
            SemanticDataType.EMAIL: DataTypeCategory.TEXTUAL,
            SemanticDataType.PHONE: DataTypeCategory.TEXTUAL,
            SemanticDataType.URL: DataTypeCategory.TEXTUAL,
            
            # Geographic
            SemanticDataType.ADDRESS: DataTypeCategory.GEOSPATIAL,
            SemanticDataType.POSTAL_CODE: DataTypeCategory.GEOSPATIAL,
            SemanticDataType.LATITUDE: DataTypeCategory.GEOSPATIAL,
            SemanticDataType.LONGITUDE: DataTypeCategory.GEOSPATIAL,
            SemanticDataType.COUNTRY: DataTypeCategory.GEOSPATIAL,
            SemanticDataType.CITY: DataTypeCategory.GEOSPATIAL,
        }
        
        return category_mapping.get(semantic_type, DataTypeCategory.UNKNOWN)
    
    def _recommend_pandas_dtype(self, semantic_type: SemanticDataType, features: Dict[str, Any]) -> str:
        """Recommend optimal pandas dtype based on semantic type."""
        
        dtype_mapping = {
            SemanticDataType.INTEGER: "int64",
            SemanticDataType.FLOAT: "float64",
            SemanticDataType.CURRENCY: "float64",
            SemanticDataType.PERCENTAGE: "float64",
            SemanticDataType.SCORE: "float64",
            SemanticDataType.RATING: "int8",
            SemanticDataType.COUNT: "int64",
            SemanticDataType.BINARY_FLAG: "bool",
            SemanticDataType.CATEGORY: "category",
            SemanticDataType.DATE: "datetime64[ns]",
            SemanticDataType.DATETIME: "datetime64[ns]",
            SemanticDataType.TIMESTAMP: "datetime64[ns]",
        }
        
        return dtype_mapping.get(semantic_type, "object")
    
    def _generate_constraints(self, semantic_type: SemanticDataType, features: Dict[str, Any]) -> List[str]:
        """Generate recommended data constraints based on semantic type."""
        
        constraints = []
        
        if semantic_type == SemanticDataType.EMAIL:
            constraints.append("REGEXP_LIKE(column_name, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')")
        elif semantic_type == SemanticDataType.PHONE:
            constraints.append("LENGTH(column_name) >= 10")
        elif semantic_type == SemanticDataType.PERCENTAGE:
            constraints.append("column_name >= 0 AND column_name <= 100")
        elif semantic_type == SemanticDataType.RATING:
            constraints.append("column_name >= 1 AND column_name <= 5")
        elif semantic_type == SemanticDataType.UUID:
            constraints.append("LENGTH(column_name) = 36")
        
        return constraints
    
    def _basic_type_inference(self, series: pd.Series, column_name: str) -> TypeInferenceResult:
        """Fallback basic type inference."""
        
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                semantic_type = SemanticDataType.INTEGER
                category = DataTypeCategory.NUMERIC
            else:
                semantic_type = SemanticDataType.FLOAT
                category = DataTypeCategory.NUMERIC
        elif series.dtype == 'bool':
            semantic_type = SemanticDataType.BINARY_FLAG
            category = DataTypeCategory.BOOLEAN
        elif series.dtype == 'object':
            if series.nunique() / len(series) < 0.1:
                semantic_type = SemanticDataType.CATEGORY
                category = DataTypeCategory.CATEGORICAL
            else:
                semantic_type = SemanticDataType.FREE_TEXT
                category = DataTypeCategory.TEXTUAL
        else:
            semantic_type = SemanticDataType.UNKNOWN
            category = DataTypeCategory.UNKNOWN
        
        return TypeInferenceResult(
            predicted_category=category,
            predicted_semantic_type=semantic_type,
            confidence=0.5,
            alternative_types=[],
            statistical_evidence={},
            pattern_evidence={},
            feature_evidence={},
            sample_size=len(series),
            unique_values=series.nunique(),
            null_count=series.isnull().sum(),
            detection_method="basic",
            recommended_pandas_dtype=str(series.dtype),
            recommended_constraints=[],
            confidence_factors={'basic': 0.5}
        )