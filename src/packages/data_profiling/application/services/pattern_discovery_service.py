import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import hashlib
from ...domain.entities.data_profile import Pattern, PatternType


class PatternDiscoveryService:
    """Advanced service to discover patterns in data using ML and statistical techniques."""
    
    def __init__(self):
        # Predefined regex patterns
        self.patterns = {
            PatternType.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            PatternType.PHONE: r'^[\+]?[\d\s\-\(\)]{10,}$',
            PatternType.URL: r'^https?://[^\s]+$',
            PatternType.DATE: r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$|^\d{2}-\d{2}-\d{4}$',
            PatternType.TIME: r'^\d{2}:\d{2}(:\d{2})?(\s?(AM|PM))?$',
            PatternType.NUMERIC: r'^\d+(\.\d+)?$',
            PatternType.ALPHANUMERIC: r'^[a-zA-Z0-9]+$'
        }
        
        # Semantic type classifiers
        self.semantic_classifiers = {
            'credit_card': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'ip_address': r'^(\d{1,3}\.){3}\d{1,3}$',
            'mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'currency': r'^\$?\d+(\.\d{2})?$',
            'percentage': r'^\d+(\.\d+)?%$'
        }
    
    def discover(self, df: pd.DataFrame) -> Dict[str, List[Pattern]]:
        """Discover patterns across all columns using advanced techniques."""
        patterns_by_column = {}
        
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            
            column_patterns = self._discover_column_patterns(series, col)
            if column_patterns:
                patterns_by_column[col] = column_patterns
        
        return patterns_by_column
    
    def _discover_column_patterns(self, series: pd.Series, column_name: str) -> List[Pattern]:
        """Discover patterns in a single column using multiple techniques."""
        patterns = []
        
        # Convert to string for pattern analysis
        string_series = series.astype(str)
        unique_values = string_series.unique()
        
        # Skip if too few unique values for meaningful pattern analysis
        if len(unique_values) < 2:
            return patterns
        
        # 1. Predefined pattern matching
        predefined_patterns = self._match_predefined_patterns(string_series, unique_values)
        patterns.extend(predefined_patterns)
        
        # 2. Semantic type classification
        semantic_patterns = self._classify_semantic_types(string_series, unique_values, column_name)
        patterns.extend(semantic_patterns)
        
        # 3. Statistical pattern discovery
        statistical_patterns = self._discover_statistical_patterns(string_series, unique_values)
        patterns.extend(statistical_patterns)
        
        # 4. Format structure analysis
        format_patterns = self._analyze_format_structures(string_series, unique_values)
        patterns.extend(format_patterns)
        
        # 5. ML-based pattern clustering
        ml_patterns = self._discover_ml_patterns(string_series, unique_values)
        patterns.extend(ml_patterns)
        
        # Deduplicate and rank patterns by confidence
        patterns = self._deduplicate_and_rank_patterns(patterns)
        
        return patterns
    
    def _match_predefined_patterns(self, series: pd.Series, unique_values: np.ndarray) -> List[Pattern]:
        """Match against predefined regex patterns."""
        patterns = []
        
        for pattern_type, regex in self.patterns.items():
            matches = []
            for value in unique_values[:1000]:  # Limit for performance
                if re.match(regex, str(value), re.IGNORECASE):
                    matches.append(value)
            
            if matches:
                match_count = len(matches)
                total_count = len(unique_values)
                percentage = (match_count / total_count) * 100
                
                # Only include patterns with reasonable coverage
                if percentage >= 10:
                    confidence = min(percentage / 100, 1.0)
                    patterns.append(Pattern(
                        pattern_type=pattern_type,
                        regex=regex,
                        frequency=match_count,
                        percentage=percentage,
                        examples=matches[:5],
                        confidence=confidence
                    ))
        
        return patterns
    
    def _classify_semantic_types(self, series: pd.Series, unique_values: np.ndarray, column_name: str) -> List[Pattern]:
        """Classify semantic types using heuristics and column names."""
        patterns = []
        col_name_lower = column_name.lower()
        
        # Use column name hints for better classification
        semantic_hints = {
            'email': ['email', 'mail', 'e_mail'],
            'phone': ['phone', 'tel', 'mobile', 'contact'],
            'address': ['address', 'addr', 'street', 'location'],
            'name': ['name', 'first_name', 'last_name', 'full_name'],
            'id': ['id', '_id', 'identifier', 'key'],
            'date': ['date', 'created', 'updated', 'time'],
            'price': ['price', 'cost', 'amount', 'fee', 'charge']
        }
        
        # Check semantic classifiers
        for semantic_type, regex in self.semantic_classifiers.items():
            matches = []
            for value in unique_values[:1000]:
                if re.match(regex, str(value), re.IGNORECASE):
                    matches.append(value)
            
            if matches:
                match_count = len(matches)
                total_count = len(unique_values)
                percentage = (match_count / total_count) * 100
                
                # Boost confidence if column name matches semantic type
                confidence_boost = 0.0
                for hint_type, hint_keywords in semantic_hints.items():
                    if any(keyword in col_name_lower for keyword in hint_keywords):
                        if semantic_type in hint_type or hint_type in semantic_type:
                            confidence_boost = 0.2
                            break
                
                if percentage >= 5:  # Lower threshold for semantic types
                    confidence = min((percentage / 100) + confidence_boost, 1.0)
                    patterns.append(Pattern(
                        pattern_type=PatternType.CUSTOM,
                        regex=regex,
                        frequency=match_count,
                        percentage=percentage,
                        examples=matches[:5],
                        confidence=confidence
                    ))
        
        return patterns
    
    def _discover_statistical_patterns(self, series: pd.Series, unique_values: np.ndarray) -> List[Pattern]:
        """Discover patterns using statistical analysis."""
        patterns = []
        
        # Analyze string lengths
        lengths = [len(str(val)) for val in unique_values]
        length_counter = Counter(lengths)
        
        # If most strings have the same length, it might be a pattern
        most_common_length, most_common_count = length_counter.most_common(1)[0]
        if most_common_count / len(unique_values) > 0.7:
            # Generate a pattern for fixed-length strings
            examples_of_length = [val for val in unique_values if len(str(val)) == most_common_length][:5]
            
            # Analyze character patterns at each position
            fixed_length_pattern = self._analyze_fixed_length_pattern(examples_of_length, most_common_length)
            if fixed_length_pattern:
                patterns.append(Pattern(
                    pattern_type=PatternType.CUSTOM,
                    regex=fixed_length_pattern,
                    frequency=most_common_count,
                    percentage=(most_common_count / len(unique_values)) * 100,
                    examples=examples_of_length,
                    confidence=0.8
                ))
        
        # Analyze character composition
        char_patterns = self._analyze_character_composition(unique_values)
        patterns.extend(char_patterns)
        
        return patterns
    
    def _analyze_fixed_length_pattern(self, examples: List[str], length: int) -> Optional[str]:
        """Analyze fixed-length strings to create a regex pattern."""
        if not examples or length == 0:
            return None
        
        pattern_chars = []
        for i in range(length):
            chars_at_position = set()
            for example in examples:
                if i < len(example):
                    chars_at_position.add(example[i])
            
            # Classify the character type at this position
            if all(c.isdigit() for c in chars_at_position):
                pattern_chars.append(r'\d')
            elif all(c.isalpha() for c in chars_at_position):
                pattern_chars.append(r'[a-zA-Z]')
            elif all(c.isalnum() for c in chars_at_position):
                pattern_chars.append(r'[a-zA-Z0-9]')
            elif len(chars_at_position) == 1:
                # Fixed character
                char = list(chars_at_position)[0]
                if char in r'[](){}.*+?^$|\\':
                    pattern_chars.append(f'\\{char}')
                else:
                    pattern_chars.append(char)
            else:
                # Mixed characters
                pattern_chars.append(r'.')
        
        return '^' + ''.join(pattern_chars) + '$'
    
    def _analyze_character_composition(self, unique_values: np.ndarray) -> List[Pattern]:
        """Analyze character composition patterns."""
        patterns = []
        
        # Categorize strings by character composition
        composition_categories = {
            'only_digits': [],
            'only_letters': [],
            'only_uppercase': [],
            'only_lowercase': [],
            'alphanumeric': [],
            'with_special_chars': []
        }
        
        for value in unique_values[:1000]:
            val_str = str(value)
            if val_str.isdigit():
                composition_categories['only_digits'].append(value)
            elif val_str.isalpha():
                composition_categories['only_letters'].append(value)
                if val_str.isupper():
                    composition_categories['only_uppercase'].append(value)
                elif val_str.islower():
                    composition_categories['only_lowercase'].append(value)
            elif val_str.isalnum():
                composition_categories['alphanumeric'].append(value)
            else:
                composition_categories['with_special_chars'].append(value)
        
        # Create patterns for significant categories
        for category, values in composition_categories.items():
            if len(values) >= max(5, len(unique_values) * 0.1):  # At least 10% or 5 values
                percentage = (len(values) / len(unique_values)) * 100
                
                # Generate appropriate regex
                regex_mapping = {
                    'only_digits': r'^\d+$',
                    'only_letters': r'^[a-zA-Z]+$',
                    'only_uppercase': r'^[A-Z]+$',
                    'only_lowercase': r'^[a-z]+$',
                    'alphanumeric': r'^[a-zA-Z0-9]+$',
                    'with_special_chars': r'^.*[^a-zA-Z0-9].*$'
                }
                
                patterns.append(Pattern(
                    pattern_type=PatternType.CUSTOM,
                    regex=regex_mapping.get(category, r'.*'),
                    frequency=len(values),
                    percentage=percentage,
                    examples=values[:5],
                    confidence=0.7
                ))
        
        return patterns
    
    def _analyze_format_structures(self, series: pd.Series, unique_values: np.ndarray) -> List[Pattern]:
        """Analyze format structures using template extraction."""
        patterns = []
        
        # Extract format templates
        format_templates = self._extract_format_templates(unique_values)
        
        for template, examples in format_templates.items():
            if len(examples) >= max(3, len(unique_values) * 0.05):  # At least 5% coverage
                percentage = (len(examples) / len(unique_values)) * 100
                regex = self._template_to_regex(template)
                
                patterns.append(Pattern(
                    pattern_type=PatternType.CUSTOM,
                    regex=regex,
                    frequency=len(examples),
                    percentage=percentage,
                    examples=examples[:5],
                    confidence=0.6
                ))
        
        return patterns
    
    def _extract_format_templates(self, unique_values: np.ndarray) -> Dict[str, List[str]]:
        """Extract format templates from strings."""
        templates = {}
        
        for value in unique_values[:500]:  # Limit for performance
            val_str = str(value)
            template = self._string_to_template(val_str)
            
            if template not in templates:
                templates[template] = []
            templates[template].append(val_str)
        
        return templates
    
    def _string_to_template(self, s: str) -> str:
        """Convert a string to a format template."""
        template = []
        for char in s:
            if char.isdigit():
                template.append('D')
            elif char.isalpha():
                if char.isupper():
                    template.append('U')
                else:
                    template.append('L')
            elif char.isspace():
                template.append('S')
            else:
                template.append(char)  # Keep special characters as-is
        return ''.join(template)
    
    def _template_to_regex(self, template: str) -> str:
        """Convert a format template to regex."""
        regex_chars = []
        for char in template:
            if char == 'D':
                regex_chars.append(r'\d')
            elif char == 'U':
                regex_chars.append(r'[A-Z]')
            elif char == 'L':
                regex_chars.append(r'[a-z]')
            elif char == 'S':
                regex_chars.append(r'\s')
            else:
                # Escape special regex characters
                if char in r'[](){}.*+?^$|\\':
                    regex_chars.append(f'\\{char}')
                else:
                    regex_chars.append(char)
        
        return '^' + ''.join(regex_chars) + '$'
    
    def _discover_ml_patterns(self, series: pd.Series, unique_values: np.ndarray) -> List[Pattern]:
        """Discover patterns using ML clustering techniques."""
        patterns = []
        
        if len(unique_values) < 10:
            return patterns
        
        try:
            # Feature extraction for clustering
            features = self._extract_string_features(unique_values)
            
            # Perform clustering
            clusters = self._cluster_strings(features, unique_values)
            
            # Generate patterns for each cluster
            for cluster_id, cluster_strings in clusters.items():
                if len(cluster_strings) >= max(3, len(unique_values) * 0.05):
                    pattern = self._generate_cluster_pattern(cluster_strings)
                    if pattern:
                        percentage = (len(cluster_strings) / len(unique_values)) * 100
                        patterns.append(Pattern(
                            pattern_type=PatternType.CUSTOM,
                            regex=pattern,
                            frequency=len(cluster_strings),
                            percentage=percentage,
                            examples=cluster_strings[:5],
                            confidence=0.5
                        ))
        
        except Exception:
            # If ML clustering fails, fall back to simpler methods
            pass
        
        return patterns
    
    def _extract_string_features(self, unique_values: np.ndarray) -> np.ndarray:
        """Extract numerical features from strings for clustering."""
        features = []
        
        for value in unique_values:
            val_str = str(value)
            feature_vector = [
                len(val_str),  # Length
                sum(c.isdigit() for c in val_str),  # Number of digits
                sum(c.isalpha() for c in val_str),  # Number of letters
                sum(c.isupper() for c in val_str),  # Number of uppercase
                sum(c.islower() for c in val_str),  # Number of lowercase
                sum(not c.isalnum() for c in val_str),  # Number of special chars
                len(set(val_str)),  # Number of unique characters
                val_str.count(' '),  # Number of spaces
                val_str.count('-'),  # Number of hyphens
                val_str.count('.')   # Number of dots
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _cluster_strings(self, features: np.ndarray, unique_values: np.ndarray) -> Dict[int, List[str]]:
        """Cluster strings based on their features."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal number of clusters (simple heuristic)
            n_clusters = min(max(2, len(unique_values) // 10), 10)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Group strings by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(str(unique_values[i]))
            
            return clusters
            
        except ImportError:
            # If sklearn not available, use simple hashing-based clustering
            return self._simple_hash_clustering(unique_values)
    
    def _simple_hash_clustering(self, unique_values: np.ndarray) -> Dict[int, List[str]]:
        """Simple clustering based on string characteristics."""
        clusters = {}
        
        for value in unique_values:
            val_str = str(value)
            # Create a simple hash based on string characteristics
            char_signature = (
                len(val_str),
                sum(c.isdigit() for c in val_str),
                sum(c.isalpha() for c in val_str),
                bool(any(not c.isalnum() for c in val_str))
            )
            
            cluster_id = hash(char_signature) % 10  # Simple hash to cluster ID
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(val_str)
        
        return clusters
    
    def _generate_cluster_pattern(self, cluster_strings: List[str]) -> Optional[str]:
        """Generate a regex pattern for a cluster of similar strings."""
        if not cluster_strings:
            return None
        
        # Find common prefix and suffix
        common_prefix = self._find_common_prefix(cluster_strings)
        common_suffix = self._find_common_suffix(cluster_strings)
        
        # Extract middle parts
        middle_parts = []
        for s in cluster_strings:
            start = len(common_prefix)
            end = len(s) - len(common_suffix) if common_suffix else len(s)
            if start < end:
                middle_parts.append(s[start:end])
        
        # Generate pattern for middle part
        middle_pattern = self._generate_middle_pattern(middle_parts)
        
        # Escape special characters in prefix and suffix
        escaped_prefix = re.escape(common_prefix) if common_prefix else ''
        escaped_suffix = re.escape(common_suffix) if common_suffix else ''
        
        return f'^{escaped_prefix}{middle_pattern}{escaped_suffix}$'
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """Find common prefix among strings."""
        if not strings:
            return ''
        
        prefix = strings[0]
        for s in strings[1:]:
            while prefix and not s.startswith(prefix):
                prefix = prefix[:-1]
        
        return prefix
    
    def _find_common_suffix(self, strings: List[str]) -> str:
        """Find common suffix among strings."""
        if not strings:
            return ''
        
        suffix = strings[0]
        for s in strings[1:]:
            while suffix and not s.endswith(suffix):
                suffix = suffix[1:]
        
        return suffix
    
    def _generate_middle_pattern(self, middle_parts: List[str]) -> str:
        """Generate regex pattern for the variable middle parts."""
        if not middle_parts:
            return ''
        
        # Analyze character types in middle parts
        has_digits = any(any(c.isdigit() for c in part) for part in middle_parts)
        has_letters = any(any(c.isalpha() for c in part) for part in middle_parts)
        has_special = any(any(not c.isalnum() for c in part) for part in middle_parts)
        
        # Generate appropriate pattern
        if has_digits and has_letters and has_special:
            return r'.*'
        elif has_digits and has_letters:
            return r'[a-zA-Z0-9]+'
        elif has_digits:
            return r'\d+'
        elif has_letters:
            return r'[a-zA-Z]+'
        else:
            return r'.*'
    
    def _deduplicate_and_rank_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Remove duplicate patterns and rank by confidence."""
        # Remove patterns with very low confidence or coverage
        filtered_patterns = [p for p in patterns if p.confidence >= 0.3 and p.percentage >= 5]
        
        # Remove duplicate regex patterns
        seen_regexes = set()
        deduplicated = []
        for pattern in filtered_patterns:
            if pattern.regex not in seen_regexes:
                seen_regexes.add(pattern.regex)
                deduplicated.append(pattern)
        
        # Sort by confidence (descending) and percentage (descending)
        deduplicated.sort(key=lambda p: (p.confidence, p.percentage), reverse=True)
        
        return deduplicated[:10]  # Return top 10 patterns
    
    def analyze_data_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships and dependencies between columns."""
        relationships = {}
        
        # Cross-column pattern analysis
        relationships['cross_column_patterns'] = self._analyze_cross_column_patterns(df)
        
        # Functional dependencies
        relationships['functional_dependencies'] = self._detect_functional_dependencies(df)
        
        # Value correlations for categorical data
        relationships['categorical_correlations'] = self._analyze_categorical_correlations(df)
        
        return relationships
    
    def _analyze_cross_column_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns that span multiple columns."""
        cross_patterns = []
        
        string_columns = df.select_dtypes(include=['object']).columns
        
        for i, col1 in enumerate(string_columns):
            for col2 in string_columns[i+1:]:
                # Check if columns have related patterns
                series1 = df[col1].dropna().astype(str)
                series2 = df[col2].dropna().astype(str)
                
                # Find rows where both columns have values
                common_indices = series1.index.intersection(series2.index)
                if len(common_indices) < 10:
                    continue
                
                # Analyze combined patterns
                combined_values = series1[common_indices] + '|' + series2[common_indices]
                templates = self._extract_format_templates(combined_values.values)
                
                significant_templates = {
                    template: examples for template, examples in templates.items()
                    if len(examples) >= len(common_indices) * 0.1
                }
                
                if significant_templates:
                    cross_patterns.append({
                        'columns': [col1, col2],
                        'pattern_type': 'combined_format',
                        'templates': list(significant_templates.keys())[:3],
                        'coverage': max(len(examples) for examples in significant_templates.values()) / len(common_indices)
                    })
        
        return cross_patterns
    
    def _detect_functional_dependencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect functional dependencies between columns."""
        dependencies = []
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    # Check if col1 -> col2 (col1 functionally determines col2)
                    dependency_strength = self._calculate_functional_dependency(df[col1], df[col2])
                    
                    if dependency_strength > 0.9:  # Strong dependency
                        dependencies.append({
                            'determinant': col1,
                            'dependent': col2,
                            'strength': dependency_strength,
                            'type': 'functional_dependency'
                        })
        
        return dependencies
    
    def _calculate_functional_dependency(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate functional dependency strength between two series."""
        # Remove rows where either value is null
        combined = pd.DataFrame({'col1': series1, 'col2': series2}).dropna()
        
        if len(combined) == 0:
            return 0.0
        
        # Group by col1 and check if col2 values are consistent
        grouped = combined.groupby('col1')['col2'].nunique()
        
        # Functional dependency exists if each value in col1 maps to exactly one value in col2
        perfect_mappings = (grouped == 1).sum()
        total_unique_col1 = len(grouped)
        
        return perfect_mappings / total_unique_col1 if total_unique_col1 > 0 else 0.0
    
    def _analyze_categorical_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze correlations between categorical columns."""
        correlations = []
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                # Calculate categorical correlation using Cramér's V
                correlation = self._calculate_cramers_v(df[col1], df[col2])
                
                if correlation > 0.3:  # Moderate to strong correlation
                    correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': correlation,
                        'type': 'categorical_correlation'
                    })
        
        return correlations
    
    def _calculate_cramers_v(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate Cramér's V for categorical correlation."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(series1, series2)
            
            # Calculate chi-square statistic
            from scipy.stats import chi2_contingency
            chi2, _, _, _ = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return min(cramers_v, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0