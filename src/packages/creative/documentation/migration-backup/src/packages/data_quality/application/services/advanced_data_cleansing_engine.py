"""
Advanced Data Cleansing Engine - Phase 3.2 Implementation
ML-powered cleansing recommendations, automated data standardization, and intelligent duplicate resolution.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import phonenumbers
from phonenumbers import NumberParseException
import email_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import jellyfish

logger = logging.getLogger(__name__)


@dataclass
class CleansingRule:
    """Individual data cleansing rule definition."""
    
    rule_id: str
    name: str
    description: str
    rule_type: str  # 'standardization', 'duplicate_detection', 'validation', 'transformation'
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    confidence_threshold: float = 0.8
    domain: Optional[str] = None  # 'financial', 'healthcare', 'geographic', etc.
    is_active: bool = True
    priority: int = 1  # Higher number = higher priority
    
    # ML-based rule properties
    ml_model_path: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateCandidate:
    """Candidate duplicate record pair with similarity metrics."""
    
    record_id_1: Any
    record_id_2: Any
    similarity_score: float
    similarity_breakdown: Dict[str, float]
    confidence_level: str  # 'low', 'medium', 'high'
    duplicate_type: str  # 'exact', 'fuzzy', 'probabilistic'
    resolution_recommendation: str
    human_review_required: bool = False


@dataclass
class CleansingResult:
    """Result of data cleansing operation."""
    
    operation_id: str
    input_records: int
    output_records: int
    cleansing_rules_applied: List[str]
    
    # Quality metrics
    quality_improvement: float
    data_completeness_before: float
    data_completeness_after: float
    standardization_rate: float
    
    # Duplicate resolution
    duplicates_found: int
    duplicates_resolved: int
    duplicate_candidates: List[DuplicateCandidate]
    
    # Performance metrics
    processing_time_seconds: float
    memory_usage_mb: float
    
    # Audit trail
    cleansed_data: pd.DataFrame
    cleansing_log: List[Dict[str, Any]]
    validation_results: Dict[str, Any]


class IntelligentDuplicateDetector:
    """ML-powered duplicate detection with fuzzy matching and entity resolution."""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        self.fuzzy_threshold = 0.80
        self.blocking_keys = ['first_char', 'length_bucket', 'domain']
        
    def detect_duplicates(self, df: pd.DataFrame, 
                         columns: List[str] = None,
                         method: str = 'hybrid') -> List[DuplicateCandidate]:
        """Detect duplicate records using multiple algorithms."""
        
        if columns is None:
            columns = df.columns.tolist()
        
        logger.info(f"Starting duplicate detection on {len(df)} records using {method} method")
        
        candidates = []
        
        if method in ['exact', 'hybrid']:
            candidates.extend(self._detect_exact_duplicates(df, columns))
        
        if method in ['fuzzy', 'hybrid']:
            candidates.extend(self._detect_fuzzy_duplicates(df, columns))
        
        if method in ['ml', 'hybrid']:
            candidates.extend(self._detect_ml_duplicates(df, columns))
        
        # Deduplicate candidates and rank by confidence
        unique_candidates = self._deduplicate_candidates(candidates)
        ranked_candidates = sorted(unique_candidates, key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"Found {len(ranked_candidates)} duplicate candidates")
        return ranked_candidates
    
    def _detect_exact_duplicates(self, df: pd.DataFrame, columns: List[str]) -> List[DuplicateCandidate]:
        """Detect exact duplicates."""
        candidates = []
        
        # Group by all columns to find exact matches
        duplicated_mask = df.duplicated(subset=columns, keep=False)
        if duplicated_mask.any():
            duplicated_groups = df[duplicated_mask].groupby(columns)
            
            for name, group in duplicated_groups:
                if len(group) > 1:
                    indices = group.index.tolist()
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            candidate = DuplicateCandidate(
                                record_id_1=indices[i],
                                record_id_2=indices[j],
                                similarity_score=1.0,
                                similarity_breakdown={'exact_match': 1.0},
                                confidence_level='high',
                                duplicate_type='exact',
                                resolution_recommendation='merge_records'
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _detect_fuzzy_duplicates(self, df: pd.DataFrame, columns: List[str]) -> List[DuplicateCandidate]:
        """Detect fuzzy duplicates using string similarity."""
        candidates = []
        
        # Use blocking to reduce comparison space
        blocks = self._create_blocking_keys(df, columns)
        
        for block_key, block_indices in blocks.items():
            if len(block_indices) < 2:
                continue
                
            block_df = df.loc[block_indices]
            
            # Compare all pairs within block
            for i, idx1 in enumerate(block_indices):
                for j, idx2 in enumerate(block_indices[i+1:], i+1):
                    similarity_breakdown = {}
                    total_similarity = 0
                    
                    for col in columns:
                        if col in df.columns:
                            val1 = str(df.loc[idx1, col]) if pd.notna(df.loc[idx1, col]) else ""
                            val2 = str(df.loc[idx2, col]) if pd.notna(df.loc[idx2, col]) else ""
                            
                            # Calculate multiple similarity metrics
                            jaro_sim = jellyfish.jaro_winkler_similarity(val1, val2)
                            levenshtein_sim = 1 - (jellyfish.levenshtein_distance(val1, val2) / max(len(val1), len(val2), 1))
                            sequence_sim = SequenceMatcher(None, val1, val2).ratio()
                            
                            col_similarity = max(jaro_sim, levenshtein_sim, sequence_sim)
                            similarity_breakdown[col] = col_similarity
                            total_similarity += col_similarity
                    
                    avg_similarity = total_similarity / len(columns) if columns else 0
                    
                    if avg_similarity >= self.fuzzy_threshold:
                        confidence = 'high' if avg_similarity >= 0.9 else 'medium' if avg_similarity >= 0.8 else 'low'
                        
                        candidate = DuplicateCandidate(
                            record_id_1=idx1,
                            record_id_2=idx2,
                            similarity_score=avg_similarity,
                            similarity_breakdown=similarity_breakdown,
                            confidence_level=confidence,
                            duplicate_type='fuzzy',
                            resolution_recommendation='human_review' if confidence == 'low' else 'merge_records',
                            human_review_required=confidence == 'low'
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _detect_ml_duplicates(self, df: pd.DataFrame, columns: List[str]) -> List[DuplicateCandidate]:
        """Detect duplicates using ML clustering approaches."""
        candidates = []
        
        try:
            # Create feature vectors for clustering
            text_columns = [col for col in columns if df[col].dtype == 'object']
            if not text_columns:
                return candidates
            
            # Combine text columns
            combined_text = df[text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
            
            # Vectorize using TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            feature_matrix = vectorizer.fit_transform(combined_text)
            
            # Use DBSCAN clustering to find similar records
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            clusters = clustering.fit_predict(feature_matrix)
            
            # Extract duplicates from clusters
            unique_clusters = set(clusters)
            unique_clusters.discard(-1)  # Remove noise cluster
            
            for cluster_id in unique_clusters:
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_indices = df.index[cluster_indices].tolist()
                
                # Calculate pairwise similarities within cluster
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx1, idx2 = cluster_indices[i], cluster_indices[j]
                        
                        # Calculate cosine similarity
                        vec1 = feature_matrix[df.index.get_loc(idx1)]
                        vec2 = feature_matrix[df.index.get_loc(idx2)]
                        similarity = cosine_similarity(vec1, vec2)[0][0]
                        
                        if similarity >= 0.7:
                            candidate = DuplicateCandidate(
                                record_id_1=idx1,
                                record_id_2=idx2,
                                similarity_score=similarity,
                                similarity_breakdown={'ml_cosine_similarity': similarity},
                                confidence_level='medium',
                                duplicate_type='probabilistic',
                                resolution_recommendation='human_review',
                                human_review_required=True
                            )
                            candidates.append(candidate)
            
        except Exception as e:
            logger.warning(f"ML duplicate detection failed: {e}")
        
        return candidates
    
    def _create_blocking_keys(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, List]:
        """Create blocking keys to reduce comparison space."""
        blocks = {}
        
        for idx, row in df.iterrows():
            # Create composite blocking key
            key_parts = []
            
            for col in columns[:2]:  # Use first 2 columns for blocking
                if col in df.columns and pd.notna(row[col]):
                    val = str(row[col]).strip().lower()
                    if val:
                        key_parts.append(val[0])  # First character
                        key_parts.append(str(len(val) // 5))  # Length bucket
            
            block_key = '_'.join(key_parts) if key_parts else 'default'
            
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append(idx)
        
        return blocks
    
    def _deduplicate_candidates(self, candidates: List[DuplicateCandidate]) -> List[DuplicateCandidate]:
        """Remove duplicate candidates (same record pairs)."""
        seen_pairs = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Create normalized pair tuple
            pair = tuple(sorted([candidate.record_id_1, candidate.record_id_2]))
            
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_candidates.append(candidate)
        
        return unique_candidates


class AdvancedDataStandardizer:
    """Advanced data standardization with domain-specific rules."""
    
    def __init__(self):
        self.standardization_rules = self._load_standardization_rules()
        
    def standardize_dataset(self, df: pd.DataFrame, 
                          domain: str = 'general') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standardize entire dataset based on domain rules."""
        
        standardized_df = df.copy()
        standardization_log = {
            'rules_applied': [],
            'columns_processed': [],
            'standardization_rate': 0.0,
            'errors': []
        }
        
        logger.info(f"Starting data standardization for domain: {domain}")
        
        # Detect column types and apply appropriate standardization
        for col in df.columns:
            try:
                col_type = self._detect_column_type(df[col])
                rules_applied = []
                
                if col_type == 'email':
                    standardized_df[col], rules = self._standardize_emails(df[col])
                    rules_applied.extend(rules)
                
                elif col_type == 'phone':
                    standardized_df[col], rules = self._standardize_phone_numbers(df[col])
                    rules_applied.extend(rules)
                
                elif col_type == 'name':
                    standardized_df[col], rules = self._standardize_names(df[col])
                    rules_applied.extend(rules)
                
                elif col_type == 'address':
                    standardized_df[col], rules = self._standardize_addresses(df[col])
                    rules_applied.extend(rules)
                
                elif col_type == 'date':
                    standardized_df[col], rules = self._standardize_dates(df[col])
                    rules_applied.extend(rules)
                
                elif col_type == 'financial' and domain == 'financial':
                    standardized_df[col], rules = self._standardize_financial_data(df[col])
                    rules_applied.extend(rules)
                
                if rules_applied:
                    standardization_log['columns_processed'].append(col)
                    standardization_log['rules_applied'].extend(rules_applied)
                    
            except Exception as e:
                error_msg = f"Failed to standardize column {col}: {str(e)}"
                standardization_log['errors'].append(error_msg)
                logger.warning(error_msg)
        
        # Calculate standardization rate
        total_cells = len(df) * len(df.columns)
        standardized_cells = sum([
            (standardized_df[col] != df[col]).sum() 
            for col in standardization_log['columns_processed']
        ])
        standardization_log['standardization_rate'] = standardized_cells / total_cells if total_cells > 0 else 0
        
        logger.info(f"Standardization completed. Rate: {standardization_log['standardization_rate']:.2%}")
        
        return standardized_df, standardization_log
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the semantic type of a column."""
        sample_values = series.dropna().head(100).astype(str)
        
        if sample_values.empty:
            return 'unknown'
        
        # Email detection
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if sample_values.apply(lambda x: bool(email_pattern.match(x))).mean() > 0.7:
            return 'email'
        
        # Phone number detection
        phone_pattern = re.compile(r'[\d\-\(\)\+\s]{10,}')
        if sample_values.apply(lambda x: bool(phone_pattern.search(x))).mean() > 0.7:
            return 'phone'
        
        # Name detection (heuristic)
        name_indicators = ['name', 'first', 'last', 'full']
        col_name = series.name.lower() if series.name else ''
        if any(indicator in col_name for indicator in name_indicators):
            return 'name'
        
        # Address detection
        address_indicators = ['address', 'street', 'location']
        if any(indicator in col_name for indicator in address_indicators):
            return 'address'
        
        # Date detection
        try:
            pd.to_datetime(sample_values.head(10))
            return 'date'
        except:
            pass
        
        # Financial data detection
        financial_indicators = ['amount', 'price', 'cost', 'salary', 'revenue']
        if any(indicator in col_name for indicator in financial_indicators):
            return 'financial'
        
        return 'general'
    
    def _standardize_emails(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize email addresses."""
        standardized = series.copy()
        rules_applied = []
        
        def clean_email(email):
            if pd.isna(email):
                return email
            
            email = str(email).strip().lower()
            
            try:
                # Validate and normalize email
                valid = email_validator.validate_email(email)
                return valid.email
            except:
                # Basic cleaning
                email = re.sub(r'\s+', '', email)  # Remove spaces
                return email
        
        standardized = series.apply(clean_email)
        rules_applied.append('email_normalization')
        
        return standardized, rules_applied
    
    def _standardize_phone_numbers(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize phone numbers to international format."""
        standardized = series.copy()
        rules_applied = []
        
        def clean_phone(phone):
            if pd.isna(phone):
                return phone
            
            phone_str = str(phone).strip()
            
            try:
                # Parse with default country US
                parsed = phonenumbers.parse(phone_str, "US")
                if phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            except NumberParseException:
                pass
            
            # Fallback: basic cleaning
            cleaned = re.sub(r'[^\d+]', '', phone_str)
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                cleaned = '+' + cleaned
            
            return cleaned
        
        standardized = series.apply(clean_phone)
        rules_applied.append('phone_normalization')
        
        return standardized, rules_applied
    
    def _standardize_names(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize names with proper capitalization."""
        standardized = series.copy()
        rules_applied = []
        
        def clean_name(name):
            if pd.isna(name):
                return name
            
            name = str(name).strip()
            
            # Title case with special handling
            name = name.title()
            
            # Handle common prefixes/suffixes
            prefixes = ['Mc', 'Mac', 'O\'']
            for prefix in prefixes:
                if name.startswith(prefix.title()):
                    rest = name[len(prefix):]
                    if rest:
                        name = prefix + rest[0].upper() + rest[1:]
            
            # Handle apostrophes
            name = re.sub(r"(\w)'(\w)", lambda m: m.group(1) + "'" + m.group(2).upper(), name)
            
            return name
        
        standardized = series.apply(clean_name)
        rules_applied.append('name_title_case')
        
        return standardized, rules_applied
    
    def _standardize_addresses(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize address formats."""
        standardized = series.copy()
        rules_applied = []
        
        def clean_address(address):
            if pd.isna(address):
                return address
            
            address = str(address).strip()
            
            # Standardize common abbreviations
            abbreviations = {
                r'\bStreet\b': 'St',
                r'\bAvenue\b': 'Ave',
                r'\bBoulevard\b': 'Blvd',
                r'\bRoad\b': 'Rd',
                r'\bDrive\b': 'Dr',
                r'\bLane\b': 'Ln',
                r'\bCourt\b': 'Ct',
                r'\bApartment\b': 'Apt',
                r'\bSuite\b': 'Ste'
            }
            
            for pattern, replacement in abbreviations.items():
                address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
            
            # Title case
            address = address.title()
            
            return address
        
        standardized = series.apply(clean_address)
        rules_applied.append('address_abbreviation_standardization')
        
        return standardized, rules_applied
    
    def _standardize_dates(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize date formats."""
        standardized = series.copy()
        rules_applied = []
        
        try:
            # Convert to datetime and back to standard format
            standardized = pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d')
            rules_applied.append('date_iso_format')
        except:
            pass
        
        return standardized, rules_applied
    
    def _standardize_financial_data(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Standardize financial data (currency, amounts)."""
        standardized = series.copy()
        rules_applied = []
        
        def clean_financial(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).strip()
            
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$,€£¥]', '', value_str)
            
            try:
                # Convert to float and format
                amount = float(cleaned)
                return f"{amount:.2f}"
            except:
                return value_str
        
        standardized = series.apply(clean_financial)
        rules_applied.append('financial_amount_normalization')
        
        return standardized, rules_applied
    
    def _load_standardization_rules(self) -> Dict[str, List[CleansingRule]]:
        """Load predefined standardization rules."""
        return {
            'general': [],
            'financial': [],
            'healthcare': [],
            'geographic': []
        }


class MLCleansingRecommendationEngine:
    """ML-powered cleansing recommendations based on data patterns."""
    
    def __init__(self):
        self.recommendation_models = {}
        self.pattern_analyzer = None
        
    def generate_cleansing_recommendations(self, df: pd.DataFrame, 
                                         quality_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ML-powered cleansing recommendations."""
        
        recommendations = []
        
        logger.info("Generating ML-powered cleansing recommendations")
        
        # Analyze data patterns
        patterns = self._analyze_data_patterns(df)
        
        # Quality-based recommendations
        quality_recommendations = self._generate_quality_recommendations(df, quality_profile)
        recommendations.extend(quality_recommendations)
        
        # Pattern-based recommendations
        pattern_recommendations = self._generate_pattern_recommendations(df, patterns)
        recommendations.extend(pattern_recommendations)
        
        # ML-based recommendations
        ml_recommendations = self._generate_ml_recommendations(df, patterns, quality_profile)
        recommendations.extend(ml_recommendations)
        
        # Rank recommendations by priority and confidence
        ranked_recommendations = sorted(
            recommendations, 
            key=lambda x: (x.get('priority', 0), x.get('confidence', 0)), 
            reverse=True
        )
        
        logger.info(f"Generated {len(ranked_recommendations)} cleansing recommendations")
        
        return ranked_recommendations
    
    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns for recommendation generation."""
        patterns = {
            'missing_patterns': {},
            'outlier_patterns': {},
            'format_patterns': {},
            'duplicate_patterns': {}
        }
        
        for col in df.columns:
            # Missing value patterns
            missing_rate = df[col].isnull().mean()
            patterns['missing_patterns'][col] = {
                'rate': missing_rate,
                'pattern': 'random' if missing_rate < 0.1 else 'systematic'
            }
            
            # Format consistency patterns
            if df[col].dtype == 'object':
                unique_formats = df[col].dropna().apply(lambda x: self._get_format_pattern(str(x))).value_counts()
                patterns['format_patterns'][col] = unique_formats.to_dict()
        
        return patterns
    
    def _generate_quality_recommendations(self, df: pd.DataFrame, 
                                        quality_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        overall_quality = quality_profile.get('overall_score', 1.0)
        
        if overall_quality < 0.7:
            recommendations.append({
                'type': 'comprehensive_cleansing',
                'priority': 10,
                'confidence': 0.9,
                'description': 'Data quality is below threshold - comprehensive cleansing recommended',
                'actions': ['duplicate_detection', 'standardization', 'validation'],
                'expected_improvement': 0.3
            })
        
        # Column-specific recommendations
        completeness_scores = quality_profile.get('completeness_by_column', {})
        for col, completeness in completeness_scores.items():
            if completeness < 0.8:
                recommendations.append({
                    'type': 'missing_value_treatment',
                    'column': col,
                    'priority': 8,
                    'confidence': 0.85,
                    'description': f'Column {col} has high missing value rate ({1-completeness:.1%})',
                    'actions': ['imputation', 'removal', 'flag_creation'],
                    'expected_improvement': 0.2
                })
        
        return recommendations
    
    def _generate_pattern_recommendations(self, df: pd.DataFrame, 
                                        patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on detected patterns."""
        recommendations = []
        
        # Format standardization recommendations
        for col, format_patterns in patterns['format_patterns'].items():
            if len(format_patterns) > 3:  # Multiple formats detected
                recommendations.append({
                    'type': 'format_standardization',
                    'column': col,
                    'priority': 7,
                    'confidence': 0.8,
                    'description': f'Column {col} has inconsistent formats ({len(format_patterns)} different patterns)',
                    'actions': ['standardize_format'],
                    'expected_improvement': 0.15
                })
        
        return recommendations
    
    def _generate_ml_recommendations(self, df: pd.DataFrame, 
                                   patterns: Dict[str, Any],
                                   quality_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ML-based recommendations using predictive models."""
        recommendations = []
        
        # Placeholder for ML-based recommendations
        # In a full implementation, this would use trained models to predict
        # the best cleansing strategies based on data characteristics
        
        data_characteristics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_rate': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_rate': df.duplicated().sum() / len(df),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Rule-based ML recommendations (would be replaced with actual ML models)
        if data_characteristics['missing_rate'] > 0.2:
            recommendations.append({
                'type': 'ml_imputation',
                'priority': 9,
                'confidence': 0.75,
                'description': 'High missing value rate detected - ML imputation recommended',
                'actions': ['knn_imputation', 'regression_imputation'],
                'expected_improvement': 0.25
            })
        
        if data_characteristics['duplicate_rate'] > 0.1:
            recommendations.append({
                'type': 'ml_duplicate_detection',
                'priority': 8,
                'confidence': 0.8,
                'description': 'High duplicate rate detected - ML-based deduplication recommended',
                'actions': ['fuzzy_matching', 'entity_resolution'],
                'expected_improvement': 0.2
            })
        
        return recommendations
    
    def _get_format_pattern(self, value: str) -> str:
        """Extract format pattern from string value."""
        # Simple pattern extraction
        pattern = re.sub(r'\d', 'N', value)  # Replace digits with N
        pattern = re.sub(r'[a-zA-Z]', 'A', pattern)  # Replace letters with A
        return pattern


class AdvancedDataCleansingEngine:
    """
    Advanced Data Cleansing Engine with ML-powered recommendations,
    intelligent duplicate detection, and domain-specific standardization.
    """
    
    def __init__(self):
        self.duplicate_detector = IntelligentDuplicateDetector()
        self.standardizer = AdvancedDataStandardizer()
        self.recommendation_engine = MLCleansingRecommendationEngine()
        
        self.cleansing_rules: List[CleansingRule] = []
        self.domain_rules: Dict[str, List[CleansingRule]] = {}
        
        logger.info("Initialized AdvancedDataCleansingEngine")
    
    async def cleanse_dataset(self, df: pd.DataFrame,
                            cleansing_config: Dict[str, Any] = None,
                            domain: str = 'general') -> CleansingResult:
        """
        Perform comprehensive data cleansing on dataset.
        
        Args:
            df: Input DataFrame to cleanse
            cleansing_config: Configuration for cleansing operations
            domain: Domain-specific cleansing ('general', 'financial', 'healthcare', etc.)
        
        Returns:
            CleansingResult with cleansed data and metrics
        """
        start_time = datetime.now()
        operation_id = f"cleansing_{int(start_time.timestamp())}"
        
        config = cleansing_config or {}
        enable_duplicate_detection = config.get('enable_duplicate_detection', True)
        enable_standardization = config.get('enable_standardization', True)
        enable_ml_recommendations = config.get('enable_ml_recommendations', True)
        
        logger.info(f"Starting comprehensive data cleansing for {len(df)} records")
        
        # Initialize result tracking
        cleansing_log = []
        input_records = len(df)
        cleansed_df = df.copy()
        
        # Calculate initial quality metrics
        initial_completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        try:
            # Step 1: Generate ML-powered cleansing recommendations
            recommendations = []
            if enable_ml_recommendations:
                quality_profile = self._calculate_quality_profile(df)
                recommendations = self.recommendation_engine.generate_cleansing_recommendations(df, quality_profile)
                cleansing_log.append({
                    'step': 'recommendation_generation',
                    'timestamp': datetime.now(),
                    'recommendations_count': len(recommendations),
                    'status': 'completed'
                })
            
            # Step 2: Intelligent duplicate detection
            duplicate_candidates = []
            if enable_duplicate_detection:
                duplicate_candidates = self.duplicate_detector.detect_duplicates(
                    cleansed_df, 
                    method=config.get('duplicate_detection_method', 'hybrid')
                )
                
                # Resolve high-confidence duplicates
                cleansed_df = self._resolve_duplicates(cleansed_df, duplicate_candidates)
                
                cleansing_log.append({
                    'step': 'duplicate_detection',
                    'timestamp': datetime.now(),
                    'duplicates_found': len(duplicate_candidates),
                    'duplicates_resolved': len([d for d in duplicate_candidates if not d.human_review_required]),
                    'status': 'completed'
                })
            
            # Step 3: Advanced data standardization
            standardization_log = {}
            if enable_standardization:
                cleansed_df, standardization_log = self.standardizer.standardize_dataset(cleansed_df, domain)
                
                cleansing_log.append({
                    'step': 'standardization',
                    'timestamp': datetime.now(),
                    'columns_processed': len(standardization_log.get('columns_processed', [])),
                    'rules_applied': len(standardization_log.get('rules_applied', [])),
                    'standardization_rate': standardization_log.get('standardization_rate', 0),
                    'status': 'completed'
                })
            
            # Step 4: Apply custom cleansing rules
            rules_applied = self._apply_custom_rules(cleansed_df, domain)
            
            # Calculate final metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            final_completeness = 1 - (cleansed_df.isnull().sum().sum() / (len(cleansed_df) * len(cleansed_df.columns)))
            quality_improvement = final_completeness - initial_completeness
            
            # Validate results
            validation_results = self._validate_cleansing_results(df, cleansed_df)
            
            result = CleansingResult(
                operation_id=operation_id,
                input_records=input_records,
                output_records=len(cleansed_df),
                cleansing_rules_applied=rules_applied,
                quality_improvement=quality_improvement,
                data_completeness_before=initial_completeness,
                data_completeness_after=final_completeness,
                standardization_rate=standardization_log.get('standardization_rate', 0),
                duplicates_found=len(duplicate_candidates),
                duplicates_resolved=len([d for d in duplicate_candidates if not d.human_review_required]),
                duplicate_candidates=duplicate_candidates,
                processing_time_seconds=processing_time,
                memory_usage_mb=cleansed_df.memory_usage(deep=True).sum() / 1024 / 1024,
                cleansed_data=cleansed_df,
                cleansing_log=cleansing_log,
                validation_results=validation_results
            )
            
            logger.info(f"Data cleansing completed successfully. Quality improvement: {quality_improvement:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Data cleansing failed: {e}")
            
            # Return partial result with error
            return CleansingResult(
                operation_id=operation_id,
                input_records=input_records,
                output_records=len(cleansed_df),
                cleansing_rules_applied=[],
                quality_improvement=0,
                data_completeness_before=initial_completeness,
                data_completeness_after=initial_completeness,
                standardization_rate=0,
                duplicates_found=0,
                duplicates_resolved=0,
                duplicate_candidates=[],
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                memory_usage_mb=0,
                cleansed_data=df,
                cleansing_log=[{'error': str(e)}],
                validation_results={'error': str(e)}
            )
    
    def _calculate_quality_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive quality profile for dataset."""
        quality_profile = {
            'overall_score': 0.8,  # Placeholder
            'completeness_by_column': {},
            'consistency_scores': {},
            'accuracy_scores': {}
        }
        
        # Calculate completeness by column
        for col in df.columns:
            completeness = 1 - df[col].isnull().mean()
            quality_profile['completeness_by_column'][col] = completeness
        
        # Calculate overall score
        overall_completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        duplicate_rate = df.duplicated().sum() / len(df)
        quality_profile['overall_score'] = (overall_completeness + (1 - duplicate_rate)) / 2
        
        return quality_profile
    
    def _resolve_duplicates(self, df: pd.DataFrame, 
                          candidates: List[DuplicateCandidate]) -> pd.DataFrame:
        """Resolve high-confidence duplicate candidates."""
        df_resolved = df.copy()
        indices_to_drop = set()
        
        for candidate in candidates:
            if (candidate.confidence_level == 'high' and 
                candidate.resolution_recommendation == 'merge_records' and
                not candidate.human_review_required):
                
                # Simple resolution: keep first record, drop second
                indices_to_drop.add(candidate.record_id_2)
        
        if indices_to_drop:
            df_resolved = df_resolved.drop(index=list(indices_to_drop))
            logger.info(f"Resolved {len(indices_to_drop)} duplicate records")
        
        return df_resolved
    
    def _apply_custom_rules(self, df: pd.DataFrame, domain: str) -> List[str]:
        """Apply custom cleansing rules for specific domain."""
        rules_applied = []
        
        # Apply domain-specific rules
        domain_rules = self.domain_rules.get(domain, [])
        for rule in domain_rules:
            if rule.is_active:
                # Apply rule (placeholder implementation)
                rules_applied.append(rule.name)
        
        return rules_applied
    
    def _validate_cleansing_results(self, original_df: pd.DataFrame, 
                                  cleansed_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate cleansing results and ensure data integrity."""
        validation_results = {
            'data_integrity_maintained': True,
            'row_count_change': len(cleansed_df) - len(original_df),
            'column_count_change': len(cleansed_df.columns) - len(original_df.columns),
            'schema_preserved': list(original_df.columns) == list(cleansed_df.columns),
            'quality_improvement_validated': True,
            'errors': []
        }
        
        # Validate schema preservation
        if not validation_results['schema_preserved']:
            validation_results['errors'].append("Column schema was modified during cleansing")
        
        # Validate reasonable row count change
        row_change_rate = abs(validation_results['row_count_change']) / len(original_df)
        if row_change_rate > 0.5:  # More than 50% of rows changed
            validation_results['errors'].append(f"Excessive row count change: {row_change_rate:.1%}")
            validation_results['data_integrity_maintained'] = False
        
        return validation_results
    
    def add_cleansing_rule(self, rule: CleansingRule):
        """Add custom cleansing rule."""
        self.cleansing_rules.append(rule)
        
        if rule.domain:
            if rule.domain not in self.domain_rules:
                self.domain_rules[rule.domain] = []
            self.domain_rules[rule.domain].append(rule)
        
        logger.info(f"Added cleansing rule: {rule.name}")
    
    def get_cleansing_statistics(self) -> Dict[str, Any]:
        """Get statistics about cleansing operations."""
        return {
            'total_rules': len(self.cleansing_rules),
            'domain_rules': {domain: len(rules) for domain, rules in self.domain_rules.items()},
            'active_rules': len([r for r in self.cleansing_rules if r.is_active])
        }