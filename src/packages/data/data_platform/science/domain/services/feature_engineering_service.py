"""Feature Engineering Service for automated feature generation and transformation."""

from __future__ import annotations

from typing import Any, Optional, Union
import logging

# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.feature_importance import FeatureImportance
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.correlation_matrix import CorrelationMatrix


logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Domain service for feature engineering operations.
    
    This service provides comprehensive feature engineering capabilities
    including automated feature generation, transformation, selection,
    and optimization for machine learning pipelines.
    """
    
    def __init__(self) -> None:
        """Initialize the feature engineering service."""
        self._logger = logger
    
    def generate_polynomial_features(self, data: Any, degree: int = 2,
                                   feature_names: Optional[list[str]] = None,
                                   include_bias: bool = False) -> dict[str, Any]:
        """Generate polynomial features from input data.
        
        Args:
            data: Input data (2D array or DataFrame)
            degree: Polynomial degree
            feature_names: Names of input features
            include_bias: Whether to include bias column
            
        Returns:
            Dictionary with transformed data and feature names
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import PolynomialFeatures
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if feature_names is not None:
                if len(feature_names) != len(data.columns):
                    raise ValueError("Number of feature names must match number of columns")
                data.columns = feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(len(data.columns))]
                data.columns = feature_names
            
            # Generate polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            transformed_data = poly.fit_transform(data)
            
            # Generate feature names
            generated_feature_names = poly.get_feature_names_out(feature_names)
            
            # Create result DataFrame
            result_df = pd.DataFrame(transformed_data, columns=generated_feature_names)
            
            return {
                "transformed_data": result_df,
                "feature_names": list(generated_feature_names),
                "original_features": len(feature_names),
                "generated_features": len(generated_feature_names),
                "degree": degree,
                "include_bias": include_bias,
                "transformer": poly
            }
            
        except ImportError:
            raise ImportError("scikit-learn and pandas are required for polynomial features")
        except Exception as e:
            self._logger.error(f"Polynomial feature generation failed: {e}")
            raise
    
    def create_interaction_features(self, data: Any, 
                                  feature_names: Optional[list[str]] = None,
                                  max_interactions: int = 2) -> dict[str, Any]:
        """Create interaction features between variables.
        
        Args:
            data: Input data (2D array or DataFrame)
            feature_names: Names of input features
            max_interactions: Maximum number of features to interact
            
        Returns:
            Dictionary with interaction features and metadata
        """
        try:
            import numpy as np
            import pandas as pd
            from itertools import combinations
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if feature_names is not None:
                if len(feature_names) != len(data.columns):
                    raise ValueError("Number of feature names must match number of columns")
                data.columns = feature_names
            else:
                feature_names = list(data.columns)
            
            interaction_data = data.copy()
            interaction_features = []
            
            # Generate pairwise interactions
            if max_interactions >= 2:
                for feat1, feat2 in combinations(feature_names, 2):
                    interaction_name = f"{feat1}_x_{feat2}"
                    interaction_data[interaction_name] = data[feat1] * data[feat2]
                    interaction_features.append({
                        "name": interaction_name,
                        "type": "multiplication",
                        "features": [feat1, feat2]
                    })
            
            # Generate three-way interactions if requested
            if max_interactions >= 3 and len(feature_names) >= 3:
                for feat1, feat2, feat3 in combinations(feature_names, 3):
                    interaction_name = f"{feat1}_x_{feat2}_x_{feat3}"
                    interaction_data[interaction_name] = data[feat1] * data[feat2] * data[feat3]
                    interaction_features.append({
                        "name": interaction_name,
                        "type": "multiplication",
                        "features": [feat1, feat2, feat3]
                    })
            
            return {
                "transformed_data": interaction_data,
                "interaction_features": interaction_features,
                "original_features": len(feature_names),
                "total_features": len(interaction_data.columns),
                "new_features_count": len(interaction_features)
            }
            
        except Exception as e:
            self._logger.error(f"Interaction feature creation failed: {e}")
            raise
    
    def create_temporal_features(self, data: Any, 
                               datetime_columns: list[str],
                               feature_types: Optional[list[str]] = None) -> dict[str, Any]:
        """Create temporal features from datetime columns.
        
        Args:
            data: Input data with datetime columns
            datetime_columns: Names of datetime columns
            feature_types: Types of temporal features to create
            
        Returns:
            Dictionary with temporal features and metadata
        """
        if feature_types is None:
            feature_types = [
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'dayofweek', 'dayofyear', 'quarter', 'weekofyear',
                'is_weekend', 'is_month_end', 'is_month_start'
            ]
        
        try:
            import pandas as pd
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            temporal_data = data.copy()
            temporal_features = []
            
            for datetime_col in datetime_columns:
                if datetime_col not in data.columns:
                    self._logger.warning(f"Datetime column '{datetime_col}' not found")
                    continue
                
                # Ensure column is datetime
                if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
                    temporal_data[datetime_col] = pd.to_datetime(data[datetime_col])
                
                dt_series = temporal_data[datetime_col]
                
                # Generate temporal features
                for feature_type in feature_types:
                    feature_name = f"{datetime_col}_{feature_type}"
                    
                    if feature_type == 'year':
                        temporal_data[feature_name] = dt_series.dt.year
                    elif feature_type == 'month':
                        temporal_data[feature_name] = dt_series.dt.month
                    elif feature_type == 'day':
                        temporal_data[feature_name] = dt_series.dt.day
                    elif feature_type == 'hour':
                        temporal_data[feature_name] = dt_series.dt.hour
                    elif feature_type == 'minute':
                        temporal_data[feature_name] = dt_series.dt.minute
                    elif feature_type == 'second':
                        temporal_data[feature_name] = dt_series.dt.second
                    elif feature_type == 'dayofweek':
                        temporal_data[feature_name] = dt_series.dt.dayofweek
                    elif feature_type == 'dayofyear':
                        temporal_data[feature_name] = dt_series.dt.dayofyear
                    elif feature_type == 'quarter':
                        temporal_data[feature_name] = dt_series.dt.quarter
                    elif feature_type == 'weekofyear':
                        temporal_data[feature_name] = dt_series.dt.isocalendar().week
                    elif feature_type == 'is_weekend':
                        temporal_data[feature_name] = (dt_series.dt.dayofweek >= 5).astype(int)
                    elif feature_type == 'is_month_end':
                        temporal_data[feature_name] = dt_series.dt.is_month_end.astype(int)
                    elif feature_type == 'is_month_start':
                        temporal_data[feature_name] = dt_series.dt.is_month_start.astype(int)
                    
                    temporal_features.append({
                        "name": feature_name,
                        "type": feature_type,
                        "source_column": datetime_col
                    })
            
            return {
                "transformed_data": temporal_data,
                "temporal_features": temporal_features,
                "original_features": len(data.columns),
                "total_features": len(temporal_data.columns),
                "new_features_count": len(temporal_features)
            }
            
        except Exception as e:
            self._logger.error(f"Temporal feature creation failed: {e}")
            raise
    
    def create_binning_features(self, data: Any, 
                              numerical_columns: list[str],
                              binning_strategy: str = "quantile",
                              n_bins: int = 5) -> dict[str, Any]:
        """Create binning features from numerical columns.
        
        Args:
            data: Input data
            numerical_columns: Names of numerical columns to bin
            binning_strategy: Binning strategy ('quantile', 'uniform', 'kmeans')
            n_bins: Number of bins
            
        Returns:
            Dictionary with binned features and metadata
        """
        try:
            import pandas as pd
            from sklearn.preprocessing import KBinsDiscretizer
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            binned_data = data.copy()
            binning_features = []
            binning_info = {}
            
            for col in numerical_columns:
                if col not in data.columns:
                    self._logger.warning(f"Column '{col}' not found")
                    continue
                
                # Skip non-numerical columns
                if not pd.api.types.is_numeric_dtype(data[col]):
                    self._logger.warning(f"Column '{col}' is not numerical")
                    continue
                
                # Remove missing values for binning
                col_data = data[col].dropna().values.reshape(-1, 1)
                
                if len(col_data) == 0:
                    continue
                
                # Create discretizer
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, 
                    encode='ordinal', 
                    strategy=binning_strategy
                )
                
                # Fit and transform
                discretizer.fit(col_data)
                binned_values = discretizer.transform(data[col].values.reshape(-1, 1)).ravel()
                
                # Create binned feature
                binned_feature_name = f"{col}_binned"
                binned_data[binned_feature_name] = binned_values
                
                # Store binning information
                binning_info[col] = {
                    "bin_edges": discretizer.bin_edges_[0].tolist(),
                    "n_bins": n_bins,
                    "strategy": binning_strategy
                }
                
                binning_features.append({
                    "name": binned_feature_name,
                    "source_column": col,
                    "n_bins": n_bins,
                    "strategy": binning_strategy
                })
            
            return {
                "transformed_data": binned_data,
                "binning_features": binning_features,
                "binning_info": binning_info,
                "original_features": len(data.columns),
                "total_features": len(binned_data.columns),
                "new_features_count": len(binning_features)
            }
            
        except ImportError:
            raise ImportError("scikit-learn is required for binning features")
        except Exception as e:
            self._logger.error(f"Binning feature creation failed: {e}")
            raise
    
    def create_aggregation_features(self, data: Any, 
                                  group_by_columns: list[str],
                                  aggregation_columns: list[str],
                                  aggregation_functions: Optional[list[str]] = None) -> dict[str, Any]:
        """Create aggregation features grouped by categorical columns.
        
        Args:
            data: Input data
            group_by_columns: Columns to group by
            aggregation_columns: Columns to aggregate
            aggregation_functions: Aggregation functions to apply
            
        Returns:
            Dictionary with aggregation features and metadata
        """
        if aggregation_functions is None:
            aggregation_functions = ['mean', 'std', 'min', 'max', 'count']
        
        try:
            import pandas as pd
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            aggregated_data = data.copy()
            aggregation_features = []
            
            for group_col in group_by_columns:
                if group_col not in data.columns:
                    self._logger.warning(f"Group column '{group_col}' not found")
                    continue
                
                for agg_col in aggregation_columns:
                    if agg_col not in data.columns:
                        self._logger.warning(f"Aggregation column '{agg_col}' not found")
                        continue
                    
                    # Skip non-numerical columns for most aggregations
                    if not pd.api.types.is_numeric_dtype(data[agg_col]):
                        continue
                    
                    for agg_func in aggregation_functions:
                        try:
                            # Calculate aggregation
                            if agg_func == 'count':
                                agg_result = data.groupby(group_col)[agg_col].count()
                            elif agg_func == 'mean':
                                agg_result = data.groupby(group_col)[agg_col].mean()
                            elif agg_func == 'std':
                                agg_result = data.groupby(group_col)[agg_col].std()
                            elif agg_func == 'min':
                                agg_result = data.groupby(group_col)[agg_col].min()
                            elif agg_func == 'max':
                                agg_result = data.groupby(group_col)[agg_col].max()
                            elif agg_func == 'median':
                                agg_result = data.groupby(group_col)[agg_col].median()
                            elif agg_func == 'sum':
                                agg_result = data.groupby(group_col)[agg_col].sum()
                            else:
                                continue
                            
                            # Create feature name
                            feature_name = f"{group_col}_{agg_col}_{agg_func}"
                            
                            # Map back to original data
                            aggregated_data[feature_name] = data[group_col].map(agg_result)
                            
                            aggregation_features.append({
                                "name": feature_name,
                                "group_by": group_col,
                                "aggregation_column": agg_col,
                                "aggregation_function": agg_func
                            })
                            
                        except Exception as e:
                            self._logger.warning(f"Failed to create aggregation {agg_func} for {agg_col} by {group_col}: {e}")
                            continue
            
            return {
                "transformed_data": aggregated_data,
                "aggregation_features": aggregation_features,
                "original_features": len(data.columns),
                "total_features": len(aggregated_data.columns),
                "new_features_count": len(aggregation_features)
            }
            
        except Exception as e:
            self._logger.error(f"Aggregation feature creation failed: {e}")
            raise
    
    def select_features_by_importance(self, data: Any, target: Any,
                                    feature_importance: FeatureImportance,
                                    selection_method: str = "threshold",
                                    threshold: float = 0.01,
                                    k_best: int = 10) -> dict[str, Any]:
        """Select features based on importance scores.
        
        Args:
            data: Input features
            target: Target variable
            feature_importance: Feature importance object
            selection_method: Selection method ('threshold', 'k_best', 'percentile')
            threshold: Importance threshold for selection
            k_best: Number of best features to select
            
        Returns:
            Dictionary with selected features and metadata
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            feature_names = list(data.columns)
            
            # Get importance scores
            importance_scores = [
                feature_importance.get_importance_score(name) or 0 
                for name in feature_names
            ]
            
            # Select features based on method
            if selection_method == "threshold":
                selected_indices = [
                    i for i, score in enumerate(importance_scores) 
                    if score >= threshold
                ]
            elif selection_method == "k_best":
                # Get indices of k best features
                sorted_indices = np.argsort(importance_scores)[::-1]
                selected_indices = sorted_indices[:k_best].tolist()
            elif selection_method == "percentile":
                # Use threshold as percentile
                percentile_threshold = np.percentile(importance_scores, 100 - threshold)
                selected_indices = [
                    i for i, score in enumerate(importance_scores) 
                    if score >= percentile_threshold
                ]
            else:
                raise ValueError(f"Unsupported selection method: {selection_method}")
            
            # Create selected dataset
            selected_feature_names = [feature_names[i] for i in selected_indices]
            selected_data = data[selected_feature_names]
            
            # Calculate selection statistics
            selection_stats = {
                "original_features": len(feature_names),
                "selected_features": len(selected_feature_names),
                "selection_ratio": len(selected_feature_names) / len(feature_names),
                "min_importance": min([importance_scores[i] for i in selected_indices]) if selected_indices else 0,
                "max_importance": max([importance_scores[i] for i in selected_indices]) if selected_indices else 0,
                "mean_importance": np.mean([importance_scores[i] for i in selected_indices]) if selected_indices else 0
            }
            
            return {
                "selected_data": selected_data,
                "selected_features": selected_feature_names,
                "selected_indices": selected_indices,
                "importance_scores": importance_scores,
                "selection_method": selection_method,
                "threshold": threshold,
                "selection_stats": selection_stats
            }
            
        except Exception as e:
            self._logger.error(f"Feature selection failed: {e}")
            raise
    
    def create_target_encoding(self, data: Any, 
                             categorical_columns: list[str],
                             target: Any,
                             smoothing: float = 1.0) -> dict[str, Any]:
        """Create target encoding for categorical variables.
        
        Args:
            data: Input data
            categorical_columns: Categorical columns to encode
            target: Target variable
            smoothing: Smoothing parameter for regularization
            
        Returns:
            Dictionary with target encoded features
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if not isinstance(target, pd.Series):
                target = pd.Series(target)
            
            encoded_data = data.copy()
            encoding_features = []
            encoding_mappings = {}
            
            # Calculate global target mean
            global_mean = target.mean()
            
            for col in categorical_columns:
                if col not in data.columns:
                    self._logger.warning(f"Column '{col}' not found")
                    continue
                
                # Calculate target statistics by category
                category_stats = pd.DataFrame({
                    'target_sum': target.groupby(data[col]).sum(),
                    'count': target.groupby(data[col]).count()
                })
                
                # Apply smoothing
                category_stats['target_mean'] = (
                    (category_stats['target_sum'] + smoothing * global_mean) /
                    (category_stats['count'] + smoothing)
                )
                
                # Create encoded feature
                encoded_feature_name = f"{col}_target_encoded"
                encoded_data[encoded_feature_name] = data[col].map(category_stats['target_mean'])
                
                # Fill missing mappings with global mean
                encoded_data[encoded_feature_name].fillna(global_mean, inplace=True)
                
                # Store encoding mapping
                encoding_mappings[col] = category_stats['target_mean'].to_dict()
                
                encoding_features.append({
                    "name": encoded_feature_name,
                    "source_column": col,
                    "encoding_type": "target_encoding",
                    "smoothing": smoothing
                })
            
            return {
                "transformed_data": encoded_data,
                "encoding_features": encoding_features,
                "encoding_mappings": encoding_mappings,
                "global_mean": global_mean,
                "smoothing": smoothing,
                "original_features": len(data.columns),
                "total_features": len(encoded_data.columns),
                "new_features_count": len(encoding_features)
            }
            
        except Exception as e:
            self._logger.error(f"Target encoding failed: {e}")
            raise
    
    def create_frequency_encoding(self, data: Any, 
                                categorical_columns: list[str]) -> dict[str, Any]:
        """Create frequency encoding for categorical variables.
        
        Args:
            data: Input data
            categorical_columns: Categorical columns to encode
            
        Returns:
            Dictionary with frequency encoded features
        """
        try:
            import pandas as pd
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            encoded_data = data.copy()
            encoding_features = []
            frequency_mappings = {}
            
            for col in categorical_columns:
                if col not in data.columns:
                    self._logger.warning(f"Column '{col}' not found")
                    continue
                
                # Calculate frequency mapping
                frequency_map = data[col].value_counts().to_dict()
                
                # Create encoded feature
                encoded_feature_name = f"{col}_frequency"
                encoded_data[encoded_feature_name] = data[col].map(frequency_map)
                
                # Store frequency mapping
                frequency_mappings[col] = frequency_map
                
                encoding_features.append({
                    "name": encoded_feature_name,
                    "source_column": col,
                    "encoding_type": "frequency_encoding",
                    "unique_values": len(frequency_map)
                })
            
            return {
                "transformed_data": encoded_data,
                "encoding_features": encoding_features,
                "frequency_mappings": frequency_mappings,
                "original_features": len(data.columns),
                "total_features": len(encoded_data.columns),
                "new_features_count": len(encoding_features)
            }
            
        except Exception as e:
            self._logger.error(f"Frequency encoding failed: {e}")
            raise
    
    def create_automated_features(self, data: Any,
                                target: Optional[Any] = None,
                                feature_types: Optional[list[str]] = None) -> dict[str, Any]:
        """Create automated features using multiple techniques.
        
        Args:
            data: Input data
            target: Target variable (optional)
            feature_types: Types of features to create
            
        Returns:
            Dictionary with all generated features
        """
        if feature_types is None:
            feature_types = [
                'polynomial', 'interactions', 'binning', 'aggregations'
            ]
        
        try:
            import pandas as pd
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            automated_data = data.copy()
            all_features = []
            feature_generation_log = []
            
            # Get column types
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
            
            # Generate polynomial features
            if 'polynomial' in feature_types and len(numerical_cols) > 0:
                try:
                    poly_result = self.generate_polynomial_features(
                        data[numerical_cols], degree=2, include_bias=False
                    )
                    
                    # Add only new features
                    new_cols = [col for col in poly_result["feature_names"] 
                              if col not in automated_data.columns]
                    if new_cols:
                        for col in new_cols:
                            automated_data[col] = poly_result["transformed_data"][col]
                        
                        all_features.extend([{
                            "name": col,
                            "type": "polynomial",
                            "source": "automated"
                        } for col in new_cols])
                        
                        feature_generation_log.append({
                            "type": "polynomial",
                            "features_added": len(new_cols),
                            "status": "success"
                        })
                
                except Exception as e:
                    feature_generation_log.append({
                        "type": "polynomial",
                        "features_added": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Generate interaction features
            if 'interactions' in feature_types and len(numerical_cols) >= 2:
                try:
                    # Limit to top features to avoid explosion
                    top_cols = numerical_cols[:5]
                    interaction_result = self.create_interaction_features(
                        data[top_cols], max_interactions=2
                    )
                    
                    # Add only new features
                    new_cols = [col for col in interaction_result["transformed_data"].columns 
                              if col not in automated_data.columns]
                    if new_cols:
                        for col in new_cols:
                            automated_data[col] = interaction_result["transformed_data"][col]
                        
                        all_features.extend([{
                            "name": col,
                            "type": "interaction",
                            "source": "automated"
                        } for col in new_cols])
                        
                        feature_generation_log.append({
                            "type": "interactions",
                            "features_added": len(new_cols),
                            "status": "success"
                        })
                
                except Exception as e:
                    feature_generation_log.append({
                        "type": "interactions",
                        "features_added": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Generate binning features
            if 'binning' in feature_types and len(numerical_cols) > 0:
                try:
                    binning_result = self.create_binning_features(
                        data, numerical_cols, n_bins=5
                    )
                    
                    # Add only new features
                    new_cols = [col for col in binning_result["transformed_data"].columns 
                              if col not in automated_data.columns]
                    if new_cols:
                        for col in new_cols:
                            automated_data[col] = binning_result["transformed_data"][col]
                        
                        all_features.extend([{
                            "name": col,
                            "type": "binning",
                            "source": "automated"
                        } for col in new_cols])
                        
                        feature_generation_log.append({
                            "type": "binning",
                            "features_added": len(new_cols),
                            "status": "success"
                        })
                
                except Exception as e:
                    feature_generation_log.append({
                        "type": "binning",
                        "features_added": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Generate temporal features
            if datetime_cols:
                try:
                    temporal_result = self.create_temporal_features(
                        data, datetime_cols
                    )
                    
                    # Add only new features
                    new_cols = [col for col in temporal_result["transformed_data"].columns 
                              if col not in automated_data.columns]
                    if new_cols:
                        for col in new_cols:
                            automated_data[col] = temporal_result["transformed_data"][col]
                        
                        all_features.extend([{
                            "name": col,
                            "type": "temporal",
                            "source": "automated"
                        } for col in new_cols])
                        
                        feature_generation_log.append({
                            "type": "temporal",
                            "features_added": len(new_cols),
                            "status": "success"
                        })
                
                except Exception as e:
                    feature_generation_log.append({
                        "type": "temporal",
                        "features_added": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            return {
                "transformed_data": automated_data,
                "generated_features": all_features,
                "feature_generation_log": feature_generation_log,
                "original_features": len(data.columns),
                "total_features": len(automated_data.columns),
                "new_features_count": len(all_features),
                "column_types": {
                    "numerical": len(numerical_cols),
                    "categorical": len(categorical_cols),
                    "datetime": len(datetime_cols)
                }
            }
            
        except Exception as e:
            self._logger.error(f"Automated feature generation failed: {e}")
            raise