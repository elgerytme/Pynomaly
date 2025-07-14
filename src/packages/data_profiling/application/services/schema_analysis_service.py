import pandas as pd
from ...domain.entities.profiles import SchemaProfile, ColumnProfile, ValueDistribution

class SchemaAnalysisService:
    """Service to perform schema analysis on a pandas DataFrame."""
    def infer(self, df: pd.DataFrame) -> SchemaProfile:
        """Infer schema profile from DataFrame."""
        columns = []
        for col in df.columns:
            series = df[col]
            data_type = str(series.dtype)
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique(dropna=True))
            total = len(series)
            completeness_ratio = (total - null_count) / total if total > 0 else 0.0
            ratio = unique_count / total if total > 0 else 0.0
            if ratio < 0.1:
                cardinality = 'low'
            elif ratio < 0.5:
                cardinality = 'medium'
            else:
                cardinality = 'high'
            value_counts = series.value_counts(dropna=True).to_dict()
            top_values = list(value_counts.items())[:5]
            statistical_summary = {}
            if pd.api.types.is_numeric_dtype(series):
                statistical_summary = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std())
                }
                distribution_type = 'numeric'
            else:
                distribution_type = 'categorical'
            distribution = ValueDistribution(
                value_counts=value_counts,
                top_values=top_values,
                histogram=None,
                statistical_summary=statistical_summary,
                distribution_type=distribution_type
            )
            col_profile = ColumnProfile(
                column_name=col,
                data_type=data_type,
                inferred_type=None,
                nullable=null_count > 0,
                unique_count=unique_count,
                null_count=null_count,
                completeness_ratio=completeness_ratio,
                cardinality=cardinality,
                distribution=distribution,
                patterns=[],
                quality_score=None,
                semantic_type=None
            )
            columns.append(col_profile)
        schema_profile = SchemaProfile(
            table_count=1,
            column_count=len(columns),
            columns=columns,
            relationships=[],
            constraints=[],
            indexes=[],
            size_metrics={'row_count': len(df), 'memory_usage': int(df.memory_usage(deep=True).sum())},
            schema_evolution=None
        )
        return schema_profile