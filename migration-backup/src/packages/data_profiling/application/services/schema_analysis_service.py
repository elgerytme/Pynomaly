import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
from ...domain.entities.data_profile import (
    SchemaProfile, ColumnProfile, ValueDistribution, StatisticalSummary,
    DataType, CardinalityLevel, SemanticType, TableRelationship, Constraint,
    IndexInfo, SizeMetrics, SchemaEvolution, Pattern
)


class SchemaAnalysisService:
    """Advanced service to perform comprehensive schema analysis on a pandas DataFrame."""
    
    def __init__(self):
        self.type_inference_mapping = {
            'int64': DataType.INTEGER,
            'int32': DataType.INTEGER,
            'float64': DataType.FLOAT,
            'float32': DataType.FLOAT,
            'bool': DataType.BOOLEAN,
            'datetime64[ns]': DataType.DATETIME,
            'object': DataType.STRING
        }
        
        # PII detection patterns
        self.pii_patterns = {
            SemanticType.PII_EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            SemanticType.PII_PHONE: re.compile(r'^\+?1?[-\s\.]?\(?[0-9]{3}\)?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4}$'),
            SemanticType.PII_SSN: re.compile(r'^\d{3}-\d{2}-\d{4}$'),
            SemanticType.URL: re.compile(r'^https?:\/\/[\w\-\.]+\.[a-zA-Z]{2,}([\/\w\-\._~:/?#[\]@!$&\'()*+,;=]*)?$'),
            SemanticType.IP_ADDRESS: re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
        }
        
        # Semantic type indicators
        self.semantic_indicators = {
            SemanticType.FINANCIAL_AMOUNT: ['amount', 'price', 'cost', 'value', 'salary', 'revenue', 'profit'],
            SemanticType.GEOGRAPHIC_LOCATION: ['country', 'state', 'city', 'zip', 'postal', 'region'],
            SemanticType.IDENTIFIER: ['id', 'key', 'code', 'number', 'ref'],
            SemanticType.TIMESTAMP: ['created', 'updated', 'modified', 'timestamp', 'date', 'time'],
            SemanticType.PII_NAME: ['name', 'firstname', 'lastname', 'fullname', 'username'],
            SemanticType.PII_ADDRESS: ['address', 'street', 'location', 'residence']
        }
    
    def infer(self, df: pd.DataFrame) -> SchemaProfile:
        """Infer comprehensive schema profile from DataFrame."""
        columns = []
        total_rows = len(df)
        
        for col in df.columns:
            column_profile = self._analyze_column(df[col], col, total_rows)
            columns.append(column_profile)
        
        # Detect relationships and constraints
        primary_keys = self._detect_primary_keys(df, columns)
        foreign_keys = self._detect_foreign_keys(df, columns)
        relationships = self._detect_table_relationships(df, columns)
        constraints = self._detect_constraints(df, columns)
        indexes = self._recommend_indexes(df, columns)
        
        # Calculate comprehensive size metrics
        size_metrics = self._calculate_size_metrics(df)
        
        # Create schema evolution tracking
        schema_evolution = SchemaEvolution(
            current_version="1.0.0",
            changes_detected=[],
            breaking_changes=[],
            last_change_date=datetime.now()
        )
        
        schema_profile = SchemaProfile(
            total_tables=1,
            total_columns=len(columns),
            total_rows=total_rows,
            columns=columns,
            relationships=relationships,
            constraints=constraints,
            indexes=indexes,
            size_metrics=size_metrics,
            schema_evolution=schema_evolution,
            primary_keys=primary_keys,
            foreign_keys=[f"{k}:{v}" for k, v in foreign_keys.items()],
            estimated_size_bytes=size_metrics.estimated_size_bytes if size_metrics else None
        )
        return schema_profile
    
    def detect_advanced_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced relationships between columns including functional dependencies."""
        relationships = {
            'functional_dependencies': self._detect_functional_dependencies(df),
            'correlation_relationships': self._detect_correlation_relationships(df),
            'hierarchical_relationships': self._detect_hierarchical_relationships(df),
            'categorical_relationships': self._detect_categorical_relationships(df),
            'temporal_relationships': self._detect_temporal_relationships(df),
            'inclusion_dependencies': self._detect_inclusion_dependencies(df)
        }
        return relationships
    
    def _detect_functional_dependencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect functional dependencies (A -> B where A determines B)."""
        functional_deps = []
        column_names = list(df.columns)
        
        for i, col_a in enumerate(column_names):
            for j, col_b in enumerate(column_names):
                if i != j:
                    # Check if col_a functionally determines col_b
                    dependency_strength = self._calculate_functional_dependency_strength(df, col_a, col_b)
                    
                    if dependency_strength >= 0.95:  # Strong functional dependency
                        functional_deps.append({
                            'determinant': col_a,
                            'dependent': col_b,
                            'strength': dependency_strength,
                            'type': 'strong' if dependency_strength >= 0.99 else 'weak'
                        })
        
        return functional_deps
    
    def _calculate_functional_dependency_strength(self, df: pd.DataFrame, col_a: str, col_b: str) -> float:
        """Calculate the strength of functional dependency A -> B."""
        # Remove rows where either column is null
        clean_df = df[[col_a, col_b]].dropna()
        
        if len(clean_df) == 0:
            return 0.0
        
        # Group by col_a and check if col_b is consistent
        grouped = clean_df.groupby(col_a)[col_b].nunique()
        
        # Count how many groups have exactly one unique value in col_b
        single_value_groups = (grouped == 1).sum()
        total_groups = len(grouped)
        
        if total_groups == 0:
            return 0.0
        
        return single_value_groups / total_groups
    
    def _detect_correlation_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect correlation relationships between numeric columns."""
        correlations = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            corr_matrix = df[numeric_columns].corr()
            
            for i, col_a in enumerate(numeric_columns):
                for j, col_b in enumerate(numeric_columns):
                    if i < j:  # Avoid duplicates and self-correlation
                        correlation = corr_matrix.loc[col_a, col_b]
                        
                        if abs(correlation) >= 0.7:  # Strong correlation
                            correlations.append({
                                'column_a': col_a,
                                'column_b': col_b,
                                'correlation': correlation,
                                'strength': 'strong' if abs(correlation) >= 0.9 else 'moderate',
                                'direction': 'positive' if correlation > 0 else 'negative'
                            })
        
        return correlations
    
    def _detect_hierarchical_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect hierarchical relationships (parent-child) in data."""
        hierarchical_rels = []
        
        # Look for columns that might represent hierarchical data
        potential_hierarchy_patterns = [
            ('category', 'subcategory'),
            ('parent', 'child'),
            ('group', 'item'),
            ('department', 'section'),
            ('country', 'state', 'city'),
            ('year', 'month', 'day')
        ]
        
        column_names = [col.lower() for col in df.columns]
        
        for pattern in potential_hierarchy_patterns:
            if len(pattern) == 2:
                parent_pattern, child_pattern = pattern
                
                # Find columns matching the pattern
                parent_cols = [col for col in df.columns if parent_pattern in col.lower()]
                child_cols = [col for col in df.columns if child_pattern in col.lower()]
                
                for parent_col in parent_cols:
                    for child_col in child_cols:
                        # Check if there's a hierarchical relationship
                        hierarchy_strength = self._calculate_hierarchy_strength(df, parent_col, child_col)
                        
                        if hierarchy_strength >= 0.8:
                            hierarchical_rels.append({
                                'parent_column': parent_col,
                                'child_column': child_col,
                                'strength': hierarchy_strength,
                                'type': 'hierarchical'
                            })
        
        return hierarchical_rels
    
    def _calculate_hierarchy_strength(self, df: pd.DataFrame, parent_col: str, child_col: str) -> float:
        """Calculate the strength of hierarchical relationship between two columns."""
        # Remove null values
        clean_df = df[[parent_col, child_col]].dropna()
        
        if len(clean_df) == 0:
            return 0.0
        
        # For each parent value, count unique child values
        child_counts = clean_df.groupby(parent_col)[child_col].nunique()
        
        # A strong hierarchy has consistent parent-child relationships
        # Calculate consistency score
        total_parent_values = len(child_counts)
        avg_children_per_parent = child_counts.mean()
        std_children_per_parent = child_counts.std()
        
        if total_parent_values == 0 or avg_children_per_parent == 0:
            return 0.0
        
        # Lower standard deviation relative to mean indicates stronger hierarchy
        coefficient_of_variation = std_children_per_parent / avg_children_per_parent if avg_children_per_parent > 0 else float('inf')
        
        # Convert to strength score (lower CV = higher strength)
        if coefficient_of_variation == 0:
            return 1.0
        elif coefficient_of_variation < 0.5:
            return 0.9
        elif coefficient_of_variation < 1.0:
            return 0.8
        elif coefficient_of_variation < 2.0:
            return 0.6
        else:
            return 0.3
    
    def _detect_categorical_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect relationships between categorical columns."""
        categorical_rels = []
        
        # Get categorical columns (strings and low-cardinality numerics)
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                if unique_count <= 20 or unique_count / total_count <= 0.1:
                    categorical_cols.append(col)
        
        # Check relationships between categorical columns
        for i, col_a in enumerate(categorical_cols):
            for j, col_b in enumerate(categorical_cols):
                if i < j:
                    relationship_type, strength = self._analyze_categorical_relationship(df, col_a, col_b)
                    
                    if strength >= 0.7:
                        categorical_rels.append({
                            'column_a': col_a,
                            'column_b': col_b,
                            'relationship_type': relationship_type,
                            'strength': strength
                        })
        
        return categorical_rels
    
    def _analyze_categorical_relationship(self, df: pd.DataFrame, col_a: str, col_b: str) -> Tuple[str, float]:
        """Analyze the relationship between two categorical columns."""
        # Create contingency table
        contingency_table = pd.crosstab(df[col_a], df[col_b])
        
        # Calculate various relationship metrics
        
        # 1. Check for one-to-one mapping
        if self._is_one_to_one_mapping(contingency_table):
            return 'one_to_one', 0.95
        
        # 2. Check for one-to-many mapping
        if self._is_one_to_many_mapping(contingency_table):
            return 'one_to_many', 0.85
        
        # 3. Calculate mutual information / association strength
        total_observations = contingency_table.sum().sum()
        if total_observations == 0:
            return 'no_relationship', 0.0
        
        # Calculate Cramér's V (measures association strength)
        chi2 = 0
        for i in range(contingency_table.shape[0]):
            for j in range(contingency_table.shape[1]):
                observed = contingency_table.iloc[i, j]
                expected = (contingency_table.iloc[i, :].sum() * contingency_table.iloc[:, j].sum()) / total_observations
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        
        cramers_v = np.sqrt(chi2 / (total_observations * (min(contingency_table.shape) - 1)))
        
        if cramers_v >= 0.7:
            return 'strong_association', cramers_v
        elif cramers_v >= 0.3:
            return 'moderate_association', cramers_v
        else:
            return 'weak_association', cramers_v
    
    def _is_one_to_one_mapping(self, contingency_table: pd.DataFrame) -> bool:
        """Check if contingency table represents one-to-one mapping."""
        # One-to-one: each row and column should have exactly one non-zero value
        return ((contingency_table > 0).sum(axis=1) == 1).all() and ((contingency_table > 0).sum(axis=0) == 1).all()
    
    def _is_one_to_many_mapping(self, contingency_table: pd.DataFrame) -> bool:
        """Check if contingency table represents one-to-many mapping."""
        # One-to-many: each row should have values in multiple columns, but each column should have values in only one row
        row_coverage = (contingency_table > 0).sum(axis=1)
        col_coverage = (contingency_table > 0).sum(axis=0)
        
        # Check if it's one-to-many from rows to columns
        if (col_coverage == 1).all() and (row_coverage > 1).any():
            return True
        
        # Check if it's one-to-many from columns to rows
        if (row_coverage == 1).all() and (col_coverage > 1).any():
            return True
        
        return False
    
    def _detect_temporal_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect temporal relationships between date/time columns."""
        temporal_rels = []
        
        # Find date/time columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    sample = df[col].dropna().head(10)
                    pd.to_datetime(sample)
                    date_cols.append(col)
                except:
                    pass
        
        # Analyze relationships between date columns
        for i, col_a in enumerate(date_cols):
            for j, col_b in enumerate(date_cols):
                if i < j:
                    relationship = self._analyze_temporal_relationship(df, col_a, col_b)
                    if relationship:
                        temporal_rels.append(relationship)
        
        return temporal_rels
    
    def _analyze_temporal_relationship(self, df: pd.DataFrame, col_a: str, col_b: str) -> Optional[Dict[str, Any]]:
        """Analyze temporal relationship between two date columns."""
        try:
            # Convert to datetime
            date_a = pd.to_datetime(df[col_a], errors='coerce')
            date_b = pd.to_datetime(df[col_b], errors='coerce')
            
            # Remove rows with null dates
            clean_df = pd.DataFrame({'date_a': date_a, 'date_b': date_b}).dropna()
            
            if len(clean_df) < 10:
                return None
            
            # Calculate time differences
            time_diff = (clean_df['date_b'] - clean_df['date_a']).dt.days
            
            # Analyze the relationship
            if (time_diff >= 0).all():
                relationship_type = 'sequential'
            elif (time_diff <= 0).all():
                relationship_type = 'reverse_sequential'
            else:
                relationship_type = 'overlapping'
            
            # Calculate consistency
            if relationship_type in ['sequential', 'reverse_sequential']:
                consistency = 1.0
            else:
                positive_diff = (time_diff >= 0).sum()
                consistency = max(positive_diff, len(time_diff) - positive_diff) / len(time_diff)
            
            return {
                'column_a': col_a,
                'column_b': col_b,
                'relationship_type': relationship_type,
                'consistency': consistency,
                'avg_time_diff_days': float(time_diff.mean()),
                'median_time_diff_days': float(time_diff.median())
            }
        except Exception:
            return None
    
    def _detect_inclusion_dependencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect inclusion dependencies (column A values are subset of column B values)."""
        inclusion_deps = []
        
        # Only analyze string/object columns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for i, col_a in enumerate(text_cols):
            for j, col_b in enumerate(text_cols):
                if i != j:
                    # Check if col_a values are subset of col_b values
                    values_a = set(df[col_a].dropna().astype(str))
                    values_b = set(df[col_b].dropna().astype(str))
                    
                    if values_a and values_b:
                        intersection = values_a.intersection(values_b)
                        inclusion_ratio = len(intersection) / len(values_a)
                        
                        if inclusion_ratio >= 0.8:  # Strong inclusion dependency
                            inclusion_deps.append({
                                'subset_column': col_a,
                                'superset_column': col_b,
                                'inclusion_ratio': inclusion_ratio,
                                'common_values': len(intersection),
                                'type': 'inclusion_dependency'
                            })
        
        return inclusion_deps
    
    def _analyze_column(self, series: pd.Series, column_name: str, total_rows: int) -> ColumnProfile:
        """Analyze a single column comprehensively."""
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique(dropna=True))
        completeness_ratio = (total_rows - null_count) / total_rows if total_rows > 0 else 0.0
        
        # Infer data types
        pandas_dtype = str(series.dtype)
        data_type = self._infer_data_type(series, pandas_dtype)
        
        # Calculate cardinality
        cardinality = self._calculate_cardinality(unique_count, total_rows)
        
        # Create value distribution
        distribution = self._create_value_distribution(series, null_count, total_rows)
        
        # Discover patterns
        patterns = self._discover_column_patterns(series, column_name)
        
        # Infer semantic type
        semantic_type = self._infer_semantic_type(series, column_name)
        
        # Infer constraints
        inferred_constraints = self._infer_column_constraints(series, column_name)
        
        # Quality assessment
        quality_score = self._calculate_quality_score(series, semantic_type)
        
        return ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            nullable=null_count > 0,
            unique_count=unique_count,
            null_count=null_count,
            total_count=total_rows,
            completeness_ratio=completeness_ratio,
            cardinality=cardinality,
            distribution=distribution,
            patterns=patterns,
            semantic_type=semantic_type,
            inferred_constraints=inferred_constraints,
            quality_score=quality_score
        )
    
    def _infer_data_type(self, series: pd.Series, pandas_dtype: str) -> DataType:
        """Infer the appropriate DataType enum value."""
        if pandas_dtype in self.type_inference_mapping:
            return self.type_inference_mapping[pandas_dtype]
        
        # Additional inference for object columns
        if pandas_dtype == 'object':
            sample_values = series.dropna().head(100)
            if sample_values.empty:
                return DataType.STRING
            
            # Try to infer if it's actually numeric
            try:
                pd.to_numeric(sample_values)
                return DataType.FLOAT
            except (ValueError, TypeError):
                pass
            
            # Try to infer if it's datetime
            try:
                pd.to_datetime(sample_values)
                return DataType.DATETIME
            except (ValueError, TypeError):
                pass
        
        return DataType.UNKNOWN
    
    def _infer_semantic_type(self, series: pd.Series, column_name: str) -> Optional[SemanticType]:
        """Infer semantic type using patterns and column name analysis."""
        # Clean and get sample values
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return SemanticType.UNKNOWN
        
        sample_values = non_null_values.head(100).astype(str)
        
        # Check PII patterns
        for semantic_type, pattern in self.pii_patterns.items():
            if sum(1 for val in sample_values if pattern.match(val)) > len(sample_values) * 0.7:
                return semantic_type
        
        # Check column name indicators
        col_lower = column_name.lower()
        for semantic_type, indicators in self.semantic_indicators.items():
            if any(indicator in col_lower for indicator in indicators):
                return semantic_type
        
        # Check data characteristics
        if series.dtype in ['int64', 'float64']:
            if col_lower in ['count', 'quantity', 'number', 'total']:
                return SemanticType.COUNT
            elif col_lower in ['amount', 'price', 'cost', 'value', 'salary']:
                return SemanticType.FINANCIAL_AMOUNT
            elif col_lower in ['lat', 'latitude', 'lng', 'longitude']:
                return SemanticType.GEOGRAPHIC_COORDINATE
            else:
                return SemanticType.MEASUREMENT
        
        # Check for categorical data
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # Low cardinality suggests categorical
                return SemanticType.CATEGORICAL
        
        return SemanticType.UNKNOWN
    
    def _discover_column_patterns(self, series: pd.Series, column_name: str) -> List[Pattern]:
        """Discover patterns in column data."""
        patterns = []
        non_null_values = series.dropna()
        
        if len(non_null_values) == 0:
            return patterns
        
        # Convert to string for pattern analysis
        string_values = non_null_values.astype(str)
        unique_values = string_values.unique()[:100]  # Sample for pattern analysis
        
        # Common patterns to detect
        pattern_definitions = [
            ("UUID", r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            ("Email", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            ("Phone", r'^\+?1?[-\s\.]?\(?[0-9]{3}\)?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4}$'),
            ("Credit Card", r'^[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}$'),
            ("Date YYYY-MM-DD", r'^\d{4}-\d{2}-\d{2}$'),
            ("Time HH:MM:SS", r'^\d{2}:\d{2}:\d{2}$'),
            ("IPv4", r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
            ("URL", r'^https?://[^\s/$.?#].[^\s]*$'),
            ("Alphanumeric Code", r'^[A-Z0-9]{3,20}$'),
            ("Numeric Code", r'^\d{3,20}$')
        ]
        
        for pattern_name, regex in pattern_definitions:
            compiled_pattern = re.compile(regex, re.IGNORECASE)
            matches = sum(1 for val in unique_values if compiled_pattern.match(val))
            
            if matches > 0:
                percentage = (matches / len(unique_values)) * 100
                confidence = min(percentage / 100, 1.0)
                
                if percentage >= 50:  # At least 50% of values match
                    examples = [val for val in unique_values[:5] if compiled_pattern.match(val)]
                    
                    patterns.append(Pattern(
                        pattern_type=pattern_name,
                        regex=regex,
                        frequency=matches,
                        percentage=percentage,
                        examples=examples,
                        confidence=confidence,
                        description=f"{pattern_name} pattern detected in {percentage:.1f}% of values"
                    ))
        
        return patterns
    
    def _infer_column_constraints(self, series: pd.Series, column_name: str) -> List[str]:
        """Infer constraints for a column."""
        constraints = []
        
        # Not null constraint
        if series.isnull().sum() == 0:
            constraints.append("NOT NULL")
        
        # Unique constraint
        if series.nunique() == len(series):
            constraints.append("UNIQUE")
        
        # Check constraint for numeric ranges
        if pd.api.types.is_numeric_dtype(series):
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                min_val = non_null_values.min()
                max_val = non_null_values.max()
                
                # Positive values constraint
                if min_val >= 0:
                    constraints.append("CHECK (value >= 0)")
                
                # Range constraint if reasonable
                if min_val >= 0 and max_val <= 1:
                    constraints.append("CHECK (value BETWEEN 0 AND 1)")
                elif min_val >= 0 and max_val <= 100:
                    constraints.append("CHECK (value BETWEEN 0 AND 100)")
        
        # String length constraints
        elif series.dtype == 'object':
            non_null_values = series.dropna().astype(str)
            if len(non_null_values) > 0:
                max_length = non_null_values.str.len().max()
                min_length = non_null_values.str.len().min()
                
                if min_length == max_length:
                    constraints.append(f"CHECK (LENGTH(value) = {max_length})")
                elif min_length > 0:
                    constraints.append(f"CHECK (LENGTH(value) BETWEEN {min_length} AND {max_length})")
        
        return constraints
    
    def _calculate_quality_score(self, series: pd.Series, semantic_type: Optional[SemanticType]) -> float:
        """Calculate overall quality score for a column."""
        score = 1.0
        
        # Completeness factor
        completeness = 1 - (series.isnull().sum() / len(series))
        score *= completeness
        
        # Consistency factor (based on pattern adherence)
        if semantic_type and semantic_type in self.pii_patterns:
            pattern = self.pii_patterns[semantic_type]
            non_null_values = series.dropna().astype(str)
            if len(non_null_values) > 0:
                matches = sum(1 for val in non_null_values if pattern.match(val))
                consistency = matches / len(non_null_values)
                score *= consistency
        
        # Uniqueness factor (for identifier types)
        if semantic_type in [SemanticType.IDENTIFIER, SemanticType.PII_EMAIL]:
            uniqueness = series.nunique() / len(series.dropna()) if len(series.dropna()) > 0 else 0
            score *= uniqueness
        
        return max(0.0, min(1.0, score))
    
    def _detect_table_relationships(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[TableRelationship]:
        """Detect relationships between columns that suggest table relationships."""
        relationships = []
        
        # Look for foreign key relationships
        for column in columns:
            if column.semantic_type == SemanticType.IDENTIFIER or column.column_name.lower().endswith('_id'):
                # This could be a foreign key
                target_table = self._infer_target_table(column.column_name)
                if target_table:
                    confidence = self._calculate_relationship_confidence(df[column.column_name])
                    
                    relationships.append(TableRelationship(
                        source_table="current_table",
                        source_column=column.column_name,
                        target_table=target_table,
                        target_column="id",
                        relationship_type="foreign_key",
                        confidence=confidence,
                        cardinality="many_to_one"
                    ))
        
        return relationships
    
    def _infer_target_table(self, column_name: str) -> Optional[str]:
        """Infer target table from foreign key column name."""
        col_lower = column_name.lower()
        
        if col_lower.endswith('_id'):
            return col_lower[:-3]  # Remove '_id'
        elif col_lower.endswith('id') and len(col_lower) > 2:
            return col_lower[:-2]  # Remove 'id'
        
        return None
    
    def _calculate_relationship_confidence(self, series: pd.Series) -> float:
        """Calculate confidence level for a relationship."""
        # Base confidence on data characteristics
        confidence = 0.5
        
        # Higher confidence for integer IDs
        if pd.api.types.is_integer_dtype(series):
            confidence += 0.2
        
        # Higher confidence for reasonable cardinality
        uniqueness = series.nunique() / len(series.dropna()) if len(series.dropna()) > 0 else 0
        if 0.1 <= uniqueness <= 0.9:
            confidence += 0.2
        
        # Higher confidence for positive values
        if pd.api.types.is_numeric_dtype(series):
            non_null_values = series.dropna()
            if len(non_null_values) > 0 and (non_null_values > 0).all():
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _detect_constraints(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[Constraint]:
        """Detect database constraints."""
        constraints = []
        
        # Primary key constraints
        for column in columns:
            if (column.unique_count == column.total_count and 
                column.null_count == 0 and column.total_count > 0):
                constraints.append(Constraint(
                    constraint_type="primary_key",
                    table_name="current_table",
                    column_names=[column.column_name],
                    definition=f"PRIMARY KEY ({column.column_name})",
                    is_enforced=True
                ))
        
        # Unique constraints
        for column in columns:
            if (column.unique_count == column.total_count - column.null_count and 
                column.total_count > 0):
                constraints.append(Constraint(
                    constraint_type="unique",
                    table_name="current_table",
                    column_names=[column.column_name],
                    definition=f"UNIQUE ({column.column_name})",
                    is_enforced=True
                ))
        
        # Not null constraints
        for column in columns:
            if column.null_count == 0 and column.total_count > 0:
                constraints.append(Constraint(
                    constraint_type="not_null",
                    table_name="current_table",
                    column_names=[column.column_name],
                    definition=f"{column.column_name} NOT NULL",
                    is_enforced=True
                ))
        
        return constraints
    
    def _recommend_indexes(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[IndexInfo]:
        """Recommend indexes based on column characteristics."""
        indexes = []
        
        for column in columns:
            # Primary key index
            if (column.unique_count == column.total_count and 
                column.null_count == 0):
                indexes.append(IndexInfo(
                    index_name=f"pk_{column.column_name}",
                    table_name="current_table",
                    column_names=[column.column_name],
                    index_type="btree",
                    is_unique=True,
                    is_primary=True
                ))
            
            # Foreign key index
            elif column.semantic_type == SemanticType.IDENTIFIER:
                indexes.append(IndexInfo(
                    index_name=f"idx_{column.column_name}",
                    table_name="current_table",
                    column_names=[column.column_name],
                    index_type="btree",
                    is_unique=False,
                    is_primary=False
                ))
            
            # High cardinality columns that might benefit from indexing
            elif column.cardinality in [CardinalityLevel.HIGH, CardinalityLevel.UNIQUE]:
                indexes.append(IndexInfo(
                    index_name=f"idx_{column.column_name}",
                    table_name="current_table",
                    column_names=[column.column_name],
                    index_type="btree",
                    is_unique=column.cardinality == CardinalityLevel.UNIQUE,
                    is_primary=False
                ))
        
        return indexes
    
    def _calculate_size_metrics(self, df: pd.DataFrame) -> SizeMetrics:
        """Calculate comprehensive size metrics."""
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Calculate memory usage
        memory_usage = df.memory_usage(deep=True)
        total_bytes = memory_usage.sum()
        average_row_size = total_bytes / total_rows if total_rows > 0 else 0
        
        # Estimate compressed size (rough approximation)
        # Text columns typically compress better
        text_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        text_compression_ratio = 0.3  # Assume 70% compression for text
        numeric_compression_ratio = 0.8  # Assume 20% compression for numbers
        
        text_size = sum(memory_usage[col] for col in text_columns)
        numeric_size = sum(memory_usage[col] for col in numeric_columns)
        
        compressed_size = int(text_size * text_compression_ratio + numeric_size * numeric_compression_ratio)
        storage_efficiency = compressed_size / total_bytes if total_bytes > 0 else 0
        
        return SizeMetrics(
            total_rows=total_rows,
            total_columns=total_columns,
            estimated_size_bytes=int(total_bytes),
            compressed_size_bytes=compressed_size,
            average_row_size_bytes=average_row_size,
            storage_efficiency=storage_efficiency
        )
    
    def _calculate_cardinality(self, unique_count: int, total_count: int) -> CardinalityLevel:
        """Calculate cardinality level based on unique/total ratio."""
        if total_count == 0:
            return CardinalityLevel.LOW
        
        ratio = unique_count / total_count
        if unique_count < 10:
            return CardinalityLevel.LOW
        elif unique_count < 100:
            return CardinalityLevel.MEDIUM
        elif unique_count < 1000:
            return CardinalityLevel.HIGH
        else:
            return CardinalityLevel.VERY_HIGH
    
    def _create_value_distribution(self, series: pd.Series, null_count: int, total_count: int) -> ValueDistribution:
        """Create comprehensive value distribution."""
        unique_count = int(series.nunique(dropna=True))
        completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0.0
        
        # Get top values
        value_counts = series.value_counts(dropna=True)
        top_values = {str(k): int(v) for k, v in value_counts.head(10).items()}
        
        return ValueDistribution(
            unique_count=unique_count,
            null_count=null_count,
            total_count=total_count,
            completeness_ratio=completeness_ratio,
            top_values=top_values
        )
    
    def _create_statistical_summary(self, series: pd.Series) -> StatisticalSummary:
        """Create statistical summary for numeric columns."""
        clean_series = series.dropna()
        if clean_series.empty:
            return StatisticalSummary()
        
        quartiles = clean_series.quantile([0.25, 0.5, 0.75]).tolist()
        
        return StatisticalSummary(
            min_value=float(clean_series.min()),
            max_value=float(clean_series.max()),
            mean=float(clean_series.mean()),
            median=float(clean_series.median()),
            std_dev=float(clean_series.std()),
            quartiles=quartiles
        )
    
    def _assess_column_quality(self, series: pd.Series, column_name: str) -> Tuple[float, List[QualityIssue]]:
        """Assess column quality and identify issues."""
        issues = []
        quality_score = 1.0
        total_count = len(series)
        
        # Check for missing values
        null_count = series.isnull().sum()
        if null_count > 0:
            null_percentage = (null_count / total_count) * 100
            severity = "high" if null_percentage > 20 else "medium" if null_percentage > 5 else "low"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_VALUES,
                severity=severity,
                description=f"Column has {null_count} missing values ({null_percentage:.1f}%)",
                affected_rows=int(null_count),
                affected_percentage=null_percentage,
                examples=[],
                suggested_action="Consider imputation or removal of missing values"
            ))
            quality_score -= (null_percentage / 100) * 0.5
        
        # Check for duplicate values (if column should be unique)
        non_null_series = series.dropna()
        if not non_null_series.empty:
            duplicate_count = non_null_series.duplicated().sum()
            if duplicate_count > 0:
                duplicate_percentage = (duplicate_count / len(non_null_series)) * 100
                if duplicate_percentage > 10:  # Only flag if significant duplication
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.DUPLICATE_VALUES,
                        severity="medium",
                        description=f"Column has {duplicate_count} duplicate values ({duplicate_percentage:.1f}%)",
                        affected_rows=int(duplicate_count),
                        affected_percentage=duplicate_percentage,
                        examples=non_null_series[non_null_series.duplicated()].head(3).tolist(),
                        suggested_action="Review if duplicates are expected or need deduplication"
                    ))
                    quality_score -= (duplicate_percentage / 100) * 0.2
        
        # Check for outliers in numeric data
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 10:  # Need sufficient data for outlier detection
                Q1 = clean_series.quantile(0.25)
                Q3 = clean_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(clean_series)) * 100
                    severity = "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.OUTLIERS,
                        severity=severity,
                        description=f"Column has {len(outliers)} outliers ({outlier_percentage:.1f}%)",
                        affected_rows=len(outliers),
                        affected_percentage=outlier_percentage,
                        examples=outliers.head(3).tolist(),
                        suggested_action="Review outliers for data entry errors or legitimate extreme values"
                    ))
                    quality_score -= (outlier_percentage / 100) * 0.1
        
        return max(0.0, quality_score), issues
    
    def _detect_primary_keys(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[str]:
        """Detect potential primary key columns."""
        primary_keys = []
        for column in columns:
            # A primary key candidate should be unique and not null
            if (column.distribution.unique_count == column.distribution.total_count and 
                column.distribution.null_count == 0):
                primary_keys.append(column.column_name)
        return primary_keys
    
    def _detect_foreign_keys(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> Dict[str, str]:
        """Detect potential foreign key relationships using advanced heuristics."""
        foreign_keys = {}
        
        for column in columns:
            col_name = column.column_name
            series = df[col_name]
            
            # Advanced heuristic analysis for foreign key detection
            potential_fk_score = 0
            reference_table = None
            
            # 1. Naming convention analysis
            if (col_name.lower().endswith('_id') or 
                col_name.lower().endswith('id') or
                col_name.lower().endswith('_key') or
                col_name.lower().endswith('_ref')):
                potential_fk_score += 30
                
                # Try to infer referenced table from column name
                if col_name.lower().endswith('_id'):
                    reference_table = col_name[:-3]  # Remove '_id'
                elif col_name.lower().endswith('id'):
                    reference_table = col_name[:-2]  # Remove 'id'
                elif col_name.lower().endswith('_key'):
                    reference_table = col_name[:-4]  # Remove '_key'
                elif col_name.lower().endswith('_ref'):
                    reference_table = col_name[:-4]  # Remove '_ref'
            
            # 2. Data type analysis
            if column.data_type in [DataType.INTEGER, DataType.STRING]:
                potential_fk_score += 20
            
            # 3. Cardinality analysis
            uniqueness_ratio = column.distribution.unique_count / column.distribution.total_count
            if 0.1 <= uniqueness_ratio <= 0.9:  # Not too unique, not too repetitive
                potential_fk_score += 25
            
            # 4. Pattern analysis for IDs
            if column.data_type == DataType.INTEGER:
                # Check if values follow ID patterns (sequential, positive)
                non_null_values = series.dropna()
                if len(non_null_values) > 0:
                    if (non_null_values > 0).all():  # All positive
                        potential_fk_score += 15
                    
                    # Check for sequential-like patterns
                    sorted_values = sorted(non_null_values.unique())
                    if len(sorted_values) > 1:
                        gaps = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
                        avg_gap = sum(gaps) / len(gaps)
                        if 1 <= avg_gap <= 10:  # Reasonable sequential gaps
                            potential_fk_score += 10
            
            elif column.data_type == DataType.STRING:
                # Check for UUID patterns, structured codes, etc.
                non_null_values = series.dropna()
                if len(non_null_values) > 0:
                    sample_values = non_null_values.head(100)
                    
                    # UUID pattern detection
                    uuid_pattern_count = sum(1 for val in sample_values 
                                           if isinstance(val, str) and len(val) == 36 and val.count('-') == 4)
                    if uuid_pattern_count / len(sample_values) > 0.5:
                        potential_fk_score += 20
                    
                    # Code pattern detection (consistent length, alphanumeric)
                    lengths = [len(str(val)) for val in sample_values]
                    if len(set(lengths)) <= 2:  # Consistent length
                        potential_fk_score += 10
            
            # 5. Check for null values (FKs often allow nulls)
            if column.distribution.null_count > 0:
                null_ratio = column.distribution.null_count / column.distribution.total_count
                if 0.01 <= null_ratio <= 0.3:  # Some nulls but not too many
                    potential_fk_score += 5
            
            # 6. Value distribution analysis
            if column.distribution.unique_count < column.distribution.total_count * 0.8:
                # Values are repeated, which is common for FKs
                potential_fk_score += 15
            
            # Decision threshold
            if potential_fk_score >= 60:
                if reference_table:
                    foreign_keys[col_name] = f"{reference_table}.id"
                else:
                    foreign_keys[col_name] = "unknown_table.id"
        
        return foreign_keys
    
    def _detect_unique_constraints(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[List[str]]:
        """Detect unique constraints on single or multiple columns."""
        unique_constraints = []
        
        # Single column unique constraints
        for column in columns:
            if (column.distribution.unique_count == column.distribution.total_count - column.distribution.null_count
                and column.distribution.total_count > 0):
                unique_constraints.append([column.column_name])
        
        # Multi-column unique constraints detection
        self._detect_multi_column_unique_constraints(df, columns, unique_constraints)
        
        return unique_constraints
    
    def _detect_multi_column_unique_constraints(self, df: pd.DataFrame, columns: List[ColumnProfile], 
                                               unique_constraints: List[List[str]]) -> None:
        """Detect multi-column unique constraints."""
        column_names = [col.column_name for col in columns]
        
        # Check pairs of columns first
        for i in range(len(column_names)):
            for j in range(i + 1, len(column_names)):
                col1, col2 = column_names[i], column_names[j]
                
                # Skip if either column is already uniquely constrained individually
                if [col1] in unique_constraints or [col2] in unique_constraints:
                    continue
                
                # Check if combination is unique
                combination_df = df[[col1, col2]].dropna()
                if len(combination_df) > 0:
                    unique_combinations = combination_df.drop_duplicates()
                    if len(unique_combinations) == len(combination_df):
                        unique_constraints.append([col1, col2])
        
        # Check triplets for smaller datasets or if there are strong candidates
        if len(df) <= 10000:  # Only for reasonably sized datasets
            self._detect_triplet_unique_constraints(df, column_names, unique_constraints)
    
    def _detect_triplet_unique_constraints(self, df: pd.DataFrame, column_names: List[str], 
                                          unique_constraints: List[List[str]]) -> None:
        """Detect three-column unique constraints for smaller datasets."""
        # Look for triplets that might form unique constraints
        # Focus on columns that seem related (similar names, data types)
        
        potential_triplets = []
        
        # Group columns by potential relationships
        date_cols = [col for col in column_names if 'date' in col.lower() or 'time' in col.lower()]
        id_cols = [col for col in column_names if col.lower().endswith('id') or col.lower().endswith('_id')]
        name_cols = [col for col in column_names if 'name' in col.lower()]
        
        # Check meaningful combinations
        for group in [date_cols, id_cols, name_cols]:
            if len(group) >= 3:
                # Check first 3 columns in each semantic group
                potential_triplets.append(group[:3])
        
        # Also check mixed combinations that might make business sense
        if len(date_cols) >= 1 and len(id_cols) >= 1 and len(name_cols) >= 1:
            potential_triplets.append([date_cols[0], id_cols[0], name_cols[0]])
        
        for triplet in potential_triplets:
            if len(triplet) == 3:
                # Check if any pair is already uniquely constrained
                if (any([triplet[i], triplet[j]] in unique_constraints 
                       for i in range(3) for j in range(i+1, 3)) or
                    any([col] in unique_constraints for col in triplet)):
                    continue
                
                # Check if triplet combination is unique
                combination_df = df[triplet].dropna()
                if len(combination_df) > 0:
                    unique_combinations = combination_df.drop_duplicates()
                    if len(unique_combinations) == len(combination_df):
                        unique_constraints.append(triplet)