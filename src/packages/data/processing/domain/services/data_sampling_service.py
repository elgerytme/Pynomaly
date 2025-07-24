"""Data sampling service for generating representative samples from datasets."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import structlog

from ..entities.dataset import Dataset, DatasetType

logger = structlog.get_logger()


class SamplingMethod(Enum):
    """Available sampling methods."""
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    CLUSTER = "cluster"
    WEIGHTED = "weighted"
    BOOTSTRAP = "bootstrap"
    RESERVOIR = "reservoir"
    TEMPORAL = "temporal"


class SamplingObjective(Enum):
    """Sampling objectives."""
    REPRESENTATIVENESS = "representativeness"
    ANOMALY_DETECTION = "anomaly_detection"
    BALANCED_CLASSES = "balanced_classes"
    TEMPORAL_COVERAGE = "temporal_coverage"
    FEATURE_DIVERSITY = "feature_diversity"


@dataclass
class SamplingConfig:
    """Configuration for data sampling."""
    method: SamplingMethod
    sample_size: Optional[int] = None
    sample_ratio: Optional[float] = None
    objective: SamplingObjective = SamplingObjective.REPRESENTATIVENESS
    stratify_column: Optional[str] = None
    temporal_column: Optional[str] = None
    weight_column: Optional[str] = None
    cluster_columns: Optional[List[str]] = None
    random_seed: int = 42
    min_class_samples: int = 1
    preserve_distribution: bool = True
    bootstrap_iterations: int = 1000


@dataclass
class SamplingResult:
    """Result of sampling operation."""
    sample_data: Dataset
    original_size: int
    sample_size: int
    sampling_ratio: float
    method_used: SamplingMethod
    sampling_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    warnings: List[str]


class DataSamplingService:
    """Service for intelligent data sampling strategies."""
    
    def __init__(self):
        self.sampling_history: List[SamplingResult] = []
        self.random_generator = np.random.RandomState(42)
    
    async def sample_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        config: SamplingConfig
    ) -> SamplingResult:
        """Generate a sample from the dataset using specified configuration."""
        
        start_time = datetime.now()
        warnings = []
        
        try:
            # Convert to DataFrame for processing
            if isinstance(dataset, Dataset):
                df = dataset.to_dataframe()
                original_dataset = dataset
            else:
                df = dataset
                original_dataset = Dataset.from_dataframe(df)
            
            original_size = len(df)
            
            # Determine actual sample size
            if config.sample_size:
                target_size = min(config.sample_size, original_size)
            elif config.sample_ratio:
                target_size = int(original_size * config.sample_ratio)
            else:
                target_size = min(1000, original_size)  # Default
            
            if target_size >= original_size:
                warnings.append("Sample size equals or exceeds original size - returning full dataset")
                sample_df = df.copy()
            else:
                # Set random seed
                self.random_generator = np.random.RandomState(config.random_seed)
                np.random.seed(config.random_seed)
                
                logger.info("Starting data sampling",
                           original_size=original_size,
                           target_size=target_size,
                           method=config.method.value)
                
                # Apply sampling method
                sample_df = await self._apply_sampling_method(df, target_size, config, warnings)
            
            # Create result dataset
            sample_dataset = Dataset.from_dataframe(sample_df)
            sample_dataset.dataset_type = original_dataset.dataset_type
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(df, sample_df, config)
            
            # Create result
            sampling_time = (datetime.now() - start_time).total_seconds()
            
            result = SamplingResult(
                sample_data=sample_dataset,
                original_size=original_size,
                sample_size=len(sample_df),
                sampling_ratio=len(sample_df) / original_size,
                method_used=config.method,
                sampling_time=sampling_time,
                quality_metrics=quality_metrics,
                metadata={
                    "config": config.__dict__,
                    "timestamp": datetime.now().isoformat()
                },
                warnings=warnings
            )
            
            self.sampling_history.append(result)
            
            logger.info("Data sampling completed",
                       original_size=original_size,
                       sample_size=len(sample_df),
                       method=config.method.value,
                       sampling_time=sampling_time)
            
            return result
            
        except Exception as e:
            logger.error("Data sampling failed",
                        method=config.method.value,
                        error=str(e))
            raise
    
    async def _apply_sampling_method(
        self,
        df: pd.DataFrame,
        target_size: int,
        config: SamplingConfig,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Apply the specified sampling method."""
        
        if config.method == SamplingMethod.RANDOM:
            return self._random_sampling(df, target_size)
        
        elif config.method == SamplingMethod.SYSTEMATIC:
            return self._systematic_sampling(df, target_size)
        
        elif config.method == SamplingMethod.STRATIFIED:
            return await self._stratified_sampling(df, target_size, config, warnings)
        
        elif config.method == SamplingMethod.CLUSTER:
            return await self._cluster_sampling(df, target_size, config, warnings)
        
        elif config.method == SamplingMethod.WEIGHTED:
            return await self._weighted_sampling(df, target_size, config, warnings)
        
        elif config.method == SamplingMethod.BOOTSTRAP:
            return self._bootstrap_sampling(df, target_size, config)
        
        elif config.method == SamplingMethod.RESERVOIR:
            return self._reservoir_sampling(df, target_size)
        
        elif config.method == SamplingMethod.TEMPORAL:
            return await self._temporal_sampling(df, target_size, config, warnings)
        
        else:
            warnings.append(f"Unknown sampling method {config.method}, using random sampling")
            return self._random_sampling(df, target_size)
    
    def _random_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Simple random sampling."""
        return df.sample(n=target_size, random_state=self.random_generator.randint(0, 10000))
    
    def _systematic_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Systematic sampling with fixed interval."""
        if target_size >= len(df):
            return df.copy()
        
        # Calculate sampling interval
        interval = len(df) / target_size
        
        # Generate systematic indices
        start_idx = self.random_generator.randint(0, int(interval))
        indices = [int(start_idx + i * interval) for i in range(target_size)]
        indices = [min(idx, len(df) - 1) for idx in indices]  # Ensure within bounds
        
        return df.iloc[indices].copy()
    
    async def _stratified_sampling(
        self,
        df: pd.DataFrame,
        target_size: int,
        config: SamplingConfig,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Stratified sampling to preserve class distributions."""
        
        if not config.stratify_column or config.stratify_column not in df.columns:
            warnings.append("Stratify column not found, using random sampling")
            return self._random_sampling(df, target_size)
        
        strat_col = config.stratify_column
        
        # Get class distributions
        class_counts = df[strat_col].value_counts()
        
        # Calculate samples per class
        samples_per_class = {}
        remaining_samples = target_size
        
        if config.preserve_distribution:
            # Proportional sampling
            for class_val, count in class_counts.items():
                proportion = count / len(df)
                class_samples = max(config.min_class_samples, int(target_size * proportion))
                class_samples = min(class_samples, count)  # Can't sample more than available
                samples_per_class[class_val] = class_samples
                remaining_samples -= class_samples
        else:
            # Equal sampling per class
            samples_per_class_equal = target_size // len(class_counts)
            for class_val, count in class_counts.items():
                class_samples = min(samples_per_class_equal, count)
                samples_per_class[class_val] = class_samples
                remaining_samples -= class_samples
        
        # Distribute remaining samples proportionally
        if remaining_samples > 0:
            for class_val in class_counts.index[:remaining_samples]:
                if samples_per_class[class_val] < class_counts[class_val]:
                    samples_per_class[class_val] += 1
        
        # Sample from each class
        sampled_dfs = []
        for class_val, sample_count in samples_per_class.items():
            if sample_count > 0:
                class_df = df[df[strat_col] == class_val]
                if len(class_df) >= sample_count:
                    sampled_class = class_df.sample(n=sample_count, random_state=self.random_generator.randint(0, 10000))
                else:
                    sampled_class = class_df.copy()
                sampled_dfs.append(sampled_class)
        
        # Combine and shuffle
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        return result_df.sample(frac=1, random_state=self.random_generator.randint(0, 10000))  # Shuffle
    
    async def _cluster_sampling(
        self,
        df: pd.DataFrame,
        target_size: int,
        config: SamplingConfig,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Cluster-based sampling."""
        
        if not config.cluster_columns:
            warnings.append("No cluster columns specified, using random sampling")
            return self._random_sampling(df, target_size)
        
        # Select numeric columns for clustering
        cluster_cols = [col for col in config.cluster_columns if col in df.columns]
        numeric_cols = df[cluster_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            warnings.append("No numeric cluster columns found, using random sampling")
            return self._random_sampling(df, target_size)
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for clustering
            cluster_data = df[numeric_cols].dropna()
            if len(cluster_data) < target_size:
                warnings.append("Insufficient data for clustering, using available data")
                return cluster_data.copy()
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Determine number of clusters (heuristic)
            n_clusters = min(int(np.sqrt(target_size)), len(cluster_data) // 10, 50)
            n_clusters = max(n_clusters, 2)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to dataframe
            cluster_df = cluster_data.copy()
            cluster_df['_cluster'] = cluster_labels
            
            # Sample proportionally from each cluster
            samples_per_cluster = target_size // n_clusters
            remainder = target_size % n_clusters
            
            sampled_dfs = []
            for cluster_id in range(n_clusters):
                cluster_data_subset = cluster_df[cluster_df['_cluster'] == cluster_id]
                
                # Determine sample size for this cluster
                cluster_sample_size = samples_per_cluster
                if cluster_id < remainder:
                    cluster_sample_size += 1
                
                cluster_sample_size = min(cluster_sample_size, len(cluster_data_subset))
                
                if cluster_sample_size > 0:
                    sampled_cluster = cluster_data_subset.sample(
                        n=cluster_sample_size,
                        random_state=self.random_generator.randint(0, 10000)
                    )
                    sampled_dfs.append(sampled_cluster.drop('_cluster', axis=1))
            
            if sampled_dfs:
                result_df = pd.concat(sampled_dfs, ignore_index=True)
                return result_df.sample(frac=1, random_state=self.random_generator.randint(0, 10000))
            else:
                warnings.append("Cluster sampling failed, using random sampling")
                return self._random_sampling(df, target_size)
                
        except ImportError:
            warnings.append("Scikit-learn not available, using random sampling")
            return self._random_sampling(df, target_size)
        except Exception as e:
            warnings.append(f"Cluster sampling failed: {str(e)}, using random sampling")
            return self._random_sampling(df, target_size)
    
    async def _weighted_sampling(
        self,
        df: pd.DataFrame,
        target_size: int,
        config: SamplingConfig,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Weighted sampling based on specified weight column."""
        
        if not config.weight_column or config.weight_column not in df.columns:
            warnings.append("Weight column not found, using random sampling")
            return self._random_sampling(df, target_size)
        
        weight_col = config.weight_column
        
        # Ensure weights are numeric and positive
        weights = pd.to_numeric(df[weight_col], errors='coerce')
        weights = weights.fillna(0)
        weights = np.maximum(weights, 0)  # Ensure non-negative
        
        if weights.sum() == 0:
            warnings.append("All weights are zero, using random sampling")
            return self._random_sampling(df, target_size)
        
        # Normalize weights
        probabilities = weights / weights.sum()
        
        # Sample with replacement using probabilities
        sampled_indices = self.random_generator.choice(
            len(df),
            size=target_size,
            replace=True,
            p=probabilities
        )
        
        return df.iloc[sampled_indices].copy()
    
    def _bootstrap_sampling(self, df: pd.DataFrame, target_size: int, config: SamplingConfig) -> pd.DataFrame:
        """Bootstrap sampling (sampling with replacement)."""
        # Bootstrap sampling allows replacement and can oversample
        sampled_indices = self.random_generator.choice(
            len(df),
            size=target_size,
            replace=True
        )
        
        return df.iloc[sampled_indices].copy()
    
    def _reservoir_sampling(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Reservoir sampling for streaming data scenarios."""
        
        if target_size >= len(df):
            return df.copy()
        
        # Initialize reservoir with first k elements
        reservoir_indices = list(range(target_size))
        
        # Process remaining elements
        for i in range(target_size, len(df)):
            # Generate random index
            j = self.random_generator.randint(0, i + 1)
            
            # If j is in first k elements, replace reservoir[j] with current element
            if j < target_size:
                reservoir_indices[j] = i
        
        return df.iloc[reservoir_indices].copy()
    
    async def _temporal_sampling(
        self,
        df: pd.DataFrame,
        target_size: int,
        config: SamplingConfig,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Temporal sampling to ensure time coverage."""
        
        if not config.temporal_column or config.temporal_column not in df.columns:
            warnings.append("Temporal column not found, using random sampling")
            return self._random_sampling(df, target_size)
        
        temp_col = config.temporal_column
        
        try:
            # Convert to datetime if not already
            df_temp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp[temp_col]):
                df_temp[temp_col] = pd.to_datetime(df_temp[temp_col], errors='coerce')
            
            # Remove rows with invalid dates
            df_temp = df_temp.dropna(subset=[temp_col])
            
            if len(df_temp) == 0:
                warnings.append("No valid temporal data found, using random sampling")
                return self._random_sampling(df, target_size)
            
            # Sort by temporal column
            df_temp = df_temp.sort_values(temp_col)
            
            # Divide into time buckets
            n_buckets = min(target_size, 100)  # Maximum 100 time buckets
            bucket_size = len(df_temp) // n_buckets
            samples_per_bucket = target_size // n_buckets
            remainder = target_size % n_buckets
            
            sampled_dfs = []
            
            for i in range(n_buckets):
                start_idx = i * bucket_size
                end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(df_temp)
                
                bucket_df = df_temp.iloc[start_idx:end_idx]
                
                # Determine sample size for this bucket
                bucket_sample_size = samples_per_bucket
                if i < remainder:
                    bucket_sample_size += 1
                
                bucket_sample_size = min(bucket_sample_size, len(bucket_df))
                
                if bucket_sample_size > 0 and len(bucket_df) > 0:
                    if bucket_sample_size >= len(bucket_df):
                        sampled_bucket = bucket_df.copy()
                    else:
                        sampled_bucket = bucket_df.sample(
                            n=bucket_sample_size,
                            random_state=self.random_generator.randint(0, 10000)
                        )
                    sampled_dfs.append(sampled_bucket)
            
            if sampled_dfs:
                result_df = pd.concat(sampled_dfs, ignore_index=True)
                return result_df.sample(frac=1, random_state=self.random_generator.randint(0, 10000))
            else:
                warnings.append("Temporal sampling failed, using random sampling")
                return self._random_sampling(df, target_size)
                
        except Exception as e:
            warnings.append(f"Temporal sampling failed: {str(e)}, using random sampling")
            return self._random_sampling(df, target_size)
    
    async def _calculate_quality_metrics(
        self,
        original_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        config: SamplingConfig
    ) -> Dict[str, float]:
        """Calculate quality metrics for the sample."""
        
        metrics = {
            "sampling_ratio": len(sample_df) / len(original_df),
            "size_efficiency": min(1.0, len(sample_df) / max(1, config.sample_size or 1000))
        }
        
        try:
            # Distribution similarity for numeric columns
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                distribution_scores = []
                
                for col in numeric_cols:
                    if col in sample_df.columns:
                        orig_mean = original_df[col].mean()
                        sample_mean = sample_df[col].mean()
                        orig_std = original_df[col].std()
                        sample_std = sample_df[col].std()
                        
                        # Calculate relative differences
                        mean_diff = abs(orig_mean - sample_mean) / (abs(orig_mean) + 1e-8)
                        std_diff = abs(orig_std - sample_std) / (abs(orig_std) + 1e-8)
                        
                        # Distribution similarity score (higher is better)
                        dist_score = 1.0 / (1.0 + mean_diff + std_diff)
                        distribution_scores.append(dist_score)
                
                if distribution_scores:
                    metrics["distribution_similarity"] = np.mean(distribution_scores)
            
            # Coverage metrics for categorical columns
            categorical_cols = original_df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                coverage_scores = []
                
                for col in categorical_cols:
                    if col in sample_df.columns:
                        orig_unique = set(original_df[col].dropna().unique())
                        sample_unique = set(sample_df[col].dropna().unique())
                        
                        if len(orig_unique) > 0:
                            coverage = len(sample_unique & orig_unique) / len(orig_unique)
                            coverage_scores.append(coverage)
                
                if coverage_scores:
                    metrics["category_coverage"] = np.mean(coverage_scores)
            
            # Diversity metric (based on unique values)
            orig_total_unique = sum(original_df[col].nunique() for col in original_df.columns)
            sample_total_unique = sum(sample_df[col].nunique() for col in sample_df.columns if col in original_df.columns)
            
            if orig_total_unique > 0:
                metrics["diversity_preservation"] = sample_total_unique / orig_total_unique
            
        except Exception as e:
            logger.warning("Quality metrics calculation failed", error=str(e))
        
        return metrics
    
    async def evaluate_sampling_quality(self, result: SamplingResult) -> Dict[str, Any]:
        """Evaluate the quality of a sampling result."""
        
        quality_assessment = {
            "overall_score": 0.0,
            "detailed_scores": result.quality_metrics.copy(),
            "recommendations": [],
            "strengths": [],
            "weaknesses": []
        }
        
        # Calculate overall score
        score_components = []
        
        if "distribution_similarity" in result.quality_metrics:
            dist_sim = result.quality_metrics["distribution_similarity"]
            score_components.append(dist_sim * 0.4)  # 40% weight
            
            if dist_sim > 0.9:
                quality_assessment["strengths"].append("Excellent distribution preservation")
            elif dist_sim < 0.7:
                quality_assessment["weaknesses"].append("Poor distribution preservation")
                quality_assessment["recommendations"].append("Consider stratified or cluster sampling")
        
        if "category_coverage" in result.quality_metrics:
            cat_cov = result.quality_metrics["category_coverage"]
            score_components.append(cat_cov * 0.3)  # 30% weight
            
            if cat_cov > 0.8:
                quality_assessment["strengths"].append("Good categorical coverage")
            elif cat_cov < 0.5:
                quality_assessment["weaknesses"].append("Low categorical coverage")
                quality_assessment["recommendations"].append("Increase sample size or use stratified sampling")
        
        if "diversity_preservation" in result.quality_metrics:
            diversity = result.quality_metrics["diversity_preservation"]
            score_components.append(diversity * 0.2)  # 20% weight
            
            if diversity > 0.8:
                quality_assessment["strengths"].append("High diversity preservation")
            elif diversity < 0.5:
                quality_assessment["weaknesses"].append("Low diversity in sample")
        
        # Size efficiency
        size_eff = result.quality_metrics.get("size_efficiency", 1.0)
        score_components.append(size_eff * 0.1)  # 10% weight
        
        # Calculate overall score
        if score_components:
            quality_assessment["overall_score"] = sum(score_components)
        
        # General recommendations
        if result.sampling_ratio < 0.1:
            quality_assessment["recommendations"].append("Very small sample - consider increasing sample size")
        
        if len(result.warnings) > 0:
            quality_assessment["recommendations"].append("Address sampling warnings for better quality")
        
        return quality_assessment
    
    def get_sampling_recommendations(
        self,
        data_profile: Dict[str, Any],
        target_size: int,
        objective: SamplingObjective
    ) -> SamplingConfig:
        """Get recommended sampling configuration based on data profile and objective."""
        
        # Default configuration
        config = SamplingConfig(
            method=SamplingMethod.RANDOM,
            sample_size=target_size,
            objective=objective
        )
        
        total_rows = data_profile.get("total_rows", 0)
        total_columns = data_profile.get("total_columns", 0)
        
        # Adjust based on objective
        if objective == SamplingObjective.BALANCED_CLASSES:
            config.method = SamplingMethod.STRATIFIED
            config.preserve_distribution = False
            
        elif objective == SamplingObjective.REPRESENTATIVENESS:
            # Choose method based on data characteristics
            if total_rows > 100000:
                config.method = SamplingMethod.SYSTEMATIC
            elif "categorical_columns" in data_profile and data_profile["categorical_columns"] > 0:
                config.method = SamplingMethod.STRATIFIED
            else:
                config.method = SamplingMethod.RANDOM
                
        elif objective == SamplingObjective.ANOMALY_DETECTION:
            # Use bootstrap or weighted sampling to potentially oversample anomalies
            config.method = SamplingMethod.BOOTSTRAP
            config.sample_size = min(target_size * 2, total_rows)  # Oversample
            
        elif objective == SamplingObjective.TEMPORAL_COVERAGE:
            config.method = SamplingMethod.TEMPORAL
            
        elif objective == SamplingObjective.FEATURE_DIVERSITY:
            config.method = SamplingMethod.CLUSTER
        
        # Adjust sample size based on data size
        if total_rows < target_size:
            config.sample_size = total_rows
            config.sample_ratio = None
        elif total_rows > 1000000:
            # For very large datasets, use ratio-based sampling
            config.sample_ratio = target_size / total_rows
            config.sample_size = None
        
        return config
    
    def get_sampling_history(self) -> List[Dict[str, Any]]:
        """Get history of sampling operations."""
        
        return [
            {
                "timestamp": result.metadata.get("timestamp"),
                "method": result.method_used.value,
                "original_size": result.original_size,
                "sample_size": result.sample_size,
                "sampling_ratio": result.sampling_ratio,
                "sampling_time": result.sampling_time,
                "quality_score": result.quality_metrics.get("distribution_similarity", 0.0)
            }
            for result in self.sampling_history
        ]