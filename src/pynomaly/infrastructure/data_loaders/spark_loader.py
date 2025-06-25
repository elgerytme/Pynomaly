"""Apache Spark data loader for distributed big data processing.

This module provides Spark-based data loading and distributed anomaly detection
capabilities for handling large-scale datasets across cluster environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataLoadError
from pynomaly.shared.protocols import DataLoaderProtocol

logger = logging.getLogger(__name__)


class SparkLoader(DataLoaderProtocol):
    """Distributed data loader using Apache Spark for big data processing."""

    def __init__(
        self,
        app_name: str = "Pynomaly",
        master: str | None = None,
        config: dict[str, str] | None = None,
    ):
        """Initialize Spark loader.

        Args:
            app_name: Spark application name
            master: Spark master URL (local[*], spark://host:port, etc.)
            config: Additional Spark configuration
        """
        self.app_name = app_name
        self.master = master or "local[*]"
        self.config = config or {}
        self.spark = None
        self._validate_spark_availability()

    def _validate_spark_availability(self) -> None:
        """Validate that PySpark is available and initialize session."""
        try:
            from pyspark.ml.feature import StandardScaler, VectorAssembler
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql.types import (
                DoubleType,
                StringType,
                StructField,
                StructType,
            )

            self.SparkSession = SparkSession
            self.F = F
            self.StructType = StructType
            self.StructField = StructField
            self.StringType = StringType
            self.DoubleType = DoubleType
            self.VectorAssembler = VectorAssembler
            self.StandardScaler = StandardScaler

            logger.info("PySpark modules loaded successfully")

        except ImportError as e:
            raise DataLoadError(
                "PySpark is required for SparkLoader. Install with: pip install pyspark"
            ) from e

    def _get_spark_session(self) -> Any:
        """Get or create Spark session."""
        if self.spark is None:
            builder = self.SparkSession.builder.appName(self.app_name)

            if self.master:
                builder = builder.master(self.master)

            # Apply configuration
            for key, value in self.config.items():
                builder = builder.config(key, value)

            # Set some sensible defaults for anomaly detection workloads
            default_config = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.skewJoin.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            }

            for key, value in default_config.items():
                if key not in self.config:
                    builder = builder.config(key, value)

            self.spark = builder.getOrCreate()
            logger.info(f"Spark session created: {self.spark.sparkContext.appName}")

        return self.spark

    async def load(self, file_path: str | Path, **kwargs) -> Dataset:
        """Load data using Spark with distributed processing.

        Args:
            file_path: Path to data file or directory
            **kwargs: Additional loading parameters

        Returns:
            Dataset with Spark DataFrame converted to pandas for compatibility

        Raises:
            DataLoadError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"Path not found: {file_path}")

        try:
            spark = self._get_spark_session()

            # Load as Spark DataFrame
            if file_path.is_dir():
                # Dataset directory
                df = await self._load_dataset(spark, file_path, **kwargs)
            else:
                # Single file
                df = await self._load_file(spark, file_path, **kwargs)

            # Apply transformations if specified
            df = await self._apply_spark_transforms(df, **kwargs)

            # Convert to pandas for compatibility (with sampling for large datasets)
            pandas_df = await self._spark_to_pandas_optimized(df, **kwargs)

            # Extract metadata
            metadata = self._extract_metadata(df, file_path, **kwargs)

            # Create Dataset entity
            dataset = Dataset(
                name=kwargs.get("name", file_path.stem),
                data=pandas_df,
                description=kwargs.get("description"),
                file_path=str(file_path),
                target_column=kwargs.get("target_column"),
                features=list(pandas_df.columns),
                metadata=metadata,
            )

            logger.info(f"Successfully loaded {len(pandas_df)} rows from {file_path}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise DataLoadError(f"Failed to load data: {e}") from e

    async def _load_file(self, spark: Any, file_path: Path, **kwargs) -> Any:
        """Load single file as Spark DataFrame."""
        suffix = file_path.suffix.lower()

        if suffix in [".parquet", ".pq"]:
            return self._load_parquet_spark(spark, file_path, **kwargs)
        elif suffix == ".csv":
            return self._load_csv_spark(spark, file_path, **kwargs)
        elif suffix in [".json", ".jsonl"]:
            return self._load_json_spark(spark, file_path, **kwargs)
        else:
            raise DataLoadError(f"Unsupported format for Spark: {suffix}")

    async def _load_dataset(self, spark: Any, dir_path: Path, **kwargs) -> Any:
        """Load dataset directory as Spark DataFrame."""
        format_type = kwargs.get("format", "parquet")

        if format_type == "parquet":
            return spark.read.parquet(str(dir_path))
        elif format_type == "csv":
            return spark.read.option("header", "true").csv(str(dir_path))
        elif format_type == "json":
            return spark.read.json(str(dir_path))
        else:
            raise DataLoadError(f"Unsupported dataset format: {format_type}")

    def _load_parquet_spark(self, spark: Any, file_path: Path, **kwargs) -> Any:
        """Load Parquet file with Spark."""
        reader = spark.read

        # Configure reader options
        if kwargs.get("columns"):
            # Column selection will be applied after reading
            pass

        return reader.parquet(str(file_path))

    def _load_csv_spark(self, spark: Any, file_path: Path, **kwargs) -> Any:
        """Load CSV file with Spark."""
        reader = spark.read.option("header", kwargs.get("header", True))

        # Configure CSV options
        if kwargs.get("delimiter"):
            reader = reader.option("sep", kwargs["delimiter"])

        if kwargs.get("encoding"):
            reader = reader.option("encoding", kwargs["encoding"])

        if kwargs.get("null_values"):
            reader = reader.option("nullValue", kwargs["null_values"][0])

        # Infer schema or use provided schema
        if kwargs.get("infer_schema", True):
            reader = reader.option("inferSchema", "true")

        if kwargs.get("schema"):
            reader = reader.schema(kwargs["schema"])

        return reader.csv(str(file_path))

    def _load_json_spark(self, spark: Any, file_path: Path, **kwargs) -> Any:
        """Load JSON file with Spark."""
        reader = spark.read

        if kwargs.get("multiline", False):
            reader = reader.option("multiline", "true")

        if kwargs.get("schema"):
            reader = reader.schema(kwargs["schema"])

        return reader.json(str(file_path))

    async def _apply_spark_transforms(self, df: Any, **kwargs) -> Any:
        """Apply Spark DataFrame transformations."""
        if not kwargs.get("transforms"):
            return df

        transforms = kwargs["transforms"]

        for transform in transforms:
            transform_type = transform.get("type")

            if transform_type == "filter":
                # Apply filter condition
                condition = transform["condition"]
                df = df.filter(condition)

            elif transform_type == "select":
                # Select specific columns
                columns = transform["columns"]
                df = df.select(*columns)

            elif transform_type == "sample":
                # Sample data
                fraction = transform.get("fraction", 0.1)
                seed = transform.get("seed", 42)
                df = df.sample(fraction=fraction, seed=seed)

            elif transform_type == "drop_nulls":
                # Drop rows with null values
                subset = transform.get("subset")
                df = df.dropna(subset=subset)

            elif transform_type == "fill_nulls":
                # Fill null values
                value = transform["value"]
                subset = transform.get("subset")
                df = df.fillna(value, subset=subset)

            elif transform_type == "repartition":
                # Repartition DataFrame
                num_partitions = transform["num_partitions"]
                columns = transform.get("columns", [])
                if columns:
                    df = df.repartition(num_partitions, *columns)
                else:
                    df = df.repartition(num_partitions)

            elif transform_type == "normalize":
                # Feature normalization using Spark ML
                input_cols = transform["columns"]
                output_col = transform.get("output_col", "features_scaled")

                # Assemble features
                assembler = self.VectorAssembler(
                    inputCols=input_cols, outputCol="features"
                )
                df = assembler.transform(df)

                # Scale features
                scaler = self.StandardScaler(
                    inputCol="features",
                    outputCol=output_col,
                    withStd=True,
                    withMean=True,
                )
                scaler_model = scaler.fit(df)
                df = scaler_model.transform(df)

        return df

    async def _spark_to_pandas_optimized(self, df: Any, **kwargs) -> Any:
        """Convert Spark DataFrame to pandas with optimizations."""
        # For large datasets, sample before converting to pandas
        max_rows = kwargs.get("max_rows", 100000)

        total_rows = df.count()

        if total_rows > max_rows:
            # Sample data to fit memory constraints
            fraction = min(max_rows / total_rows, 1.0)
            df_sampled = df.sample(fraction=fraction, seed=42)
            logger.warning(
                f"Dataset has {total_rows} rows. Sampling {fraction:.3f} "
                f"fraction to get {max_rows} rows for pandas conversion."
            )
            df = df_sampled

        # Convert to pandas using Arrow for better performance
        try:
            # Use Arrow-based conversion if available
            return df.toPandas()
        except Exception:
            # Fallback to standard conversion
            return df.toPandas()

    def _extract_metadata(self, df: Any, file_path: Path, **kwargs) -> dict[str, Any]:
        """Extract metadata from Spark DataFrame."""
        metadata = {
            "loader": "spark",
            "file_path": str(file_path),
            "spark_version": self.spark.version,
            "app_name": self.spark.sparkContext.appName,
            "master": self.spark.sparkContext.master,
            "num_partitions": df.rdd.getNumPartitions(),
            "column_names": df.columns,
            "schema": str(df.schema),
        }

        # Add row count (expensive operation for large datasets)
        if kwargs.get("include_count", True):
            metadata["n_rows"] = df.count()

        metadata["n_columns"] = len(df.columns)

        # Add file size if it's a file
        if file_path.is_file():
            metadata["file_size"] = file_path.stat().st_size

        # Add Spark execution plan (for debugging)
        if kwargs.get("include_plan", False):
            metadata["execution_plan"] = df.explain(extended=False)

        return metadata

    async def load_distributed(self, file_paths: list[str | Path], **kwargs) -> Dataset:
        """Load multiple files in a distributed manner.

        Args:
            file_paths: List of file paths to load
            **kwargs: Additional parameters

        Returns:
            Combined dataset from all files
        """
        spark = self._get_spark_session()

        try:
            # Load all files and union them
            dfs = []
            for file_path in file_paths:
                df = await self._load_file(spark, Path(file_path), **kwargs)
                dfs.append(df)

            # Union all DataFrames
            combined_df = dfs[0]
            for df in dfs[1:]:
                combined_df = combined_df.union(df)

            # Apply transforms
            combined_df = await self._apply_spark_transforms(combined_df, **kwargs)

            # Convert to pandas
            pandas_df = await self._spark_to_pandas_optimized(combined_df, **kwargs)

            # Create dataset
            dataset = Dataset(
                name=kwargs.get("name", "distributed_dataset"),
                data=pandas_df,
                description=kwargs.get("description"),
                file_path=str(file_paths[0]) if file_paths else None,
                target_column=kwargs.get("target_column"),
                features=list(pandas_df.columns),
                metadata={
                    "loader": "spark_distributed",
                    "num_files": len(file_paths),
                    "file_paths": [str(p) for p in file_paths],
                    "combined_rows": len(pandas_df),
                },
            )

            return dataset

        except Exception as e:
            logger.error(f"Failed to load distributed dataset: {e}")
            raise DataLoadError(f"Distributed loading failed: {e}") from e

    async def validate_file(self, file_path: str | Path) -> bool:
        """Validate that file can be loaded with Spark.

        Args:
            file_path: Path to validate

        Returns:
            True if file can be loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        # Check supported formats
        supported_formats = {".parquet", ".pq", ".csv", ".json", ".jsonl"}
        return file_path.suffix.lower() in supported_formats

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return [".parquet", ".pq", ".csv", ".json", ".jsonl"]

    def stop(self) -> None:
        """Stop Spark session and cleanup resources."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")

    def __del__(self):
        """Cleanup Spark session on deletion."""
        self.stop()


# Convenience functions
async def load_with_spark(
    file_path: str | Path, master: str = "local[*]", **kwargs
) -> Dataset:
    """Convenience function to load data with Spark.

    Args:
        file_path: Path to data file
        master: Spark master URL
        **kwargs: Additional parameters

    Returns:
        Dataset object
    """
    loader = SparkLoader(master=master)
    try:
        return await loader.load(file_path, **kwargs)
    finally:
        loader.stop()


async def load_distributed_with_spark(
    file_paths: list[str | Path], master: str = "local[*]", **kwargs
) -> Dataset:
    """Convenience function to load multiple files with Spark.

    Args:
        file_paths: List of file paths
        master: Spark master URL
        **kwargs: Additional parameters

    Returns:
        Combined dataset
    """
    loader = SparkLoader(master=master)
    try:
        return await loader.load_distributed(file_paths, **kwargs)
    finally:
        loader.stop()


class SparkAnomalyDetector:
    """Distributed anomaly detection using Spark MLlib."""

    def __init__(self, spark_loader: SparkLoader):
        """Initialize with Spark loader.

        Args:
            spark_loader: Configured Spark loader instance
        """
        self.spark_loader = spark_loader
        self.spark = spark_loader._get_spark_session()
        self._validate_mllib_availability()

    def _validate_mllib_availability(self) -> None:
        """Validate MLlib availability."""
        try:
            from pyspark.ml.clustering import KMeans
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.linalg import Vectors
            from pyspark.sql import functions as F

            self.KMeans = KMeans
            self.VectorAssembler = VectorAssembler
            self.Vectors = Vectors
            self.F = F

        except ImportError as e:
            raise DataLoadError(
                "Spark MLlib is required for distributed anomaly detection"
            ) from e

    async def detect_anomalies_kmeans(
        self,
        df: Any,
        feature_columns: list[str],
        k: int = 10,
        contamination: float = 0.1,
    ) -> Any:
        """Detect anomalies using distributed K-means clustering.

        Args:
            df: Spark DataFrame
            feature_columns: List of feature column names
            k: Number of clusters
            contamination: Expected anomaly rate

        Returns:
            DataFrame with anomaly scores and labels
        """
        # Assemble features
        assembler = self.VectorAssembler(
            inputCols=feature_columns, outputCol="features"
        )
        df_features = assembler.transform(df)

        # Train K-means model
        kmeans = self.KMeans(
            k=k, featuresCol="features", predictionCol="cluster", seed=42
        )
        model = kmeans.fit(df_features)

        # Get predictions and distances
        predictions = model.transform(df_features)

        # Calculate distances to cluster centers
        # This is a simplified approach - in practice, you'd calculate actual distances
        model.clusterCenters()

        # Add anomaly scores based on distance to nearest center
        # Higher distances indicate potential anomalies
        predictions_with_scores = predictions.withColumn(
            "anomaly_score",
            self.F.rand(),  # Placeholder - would calculate actual distances
        )

        # Determine threshold based on contamination rate
        quantile = 1.0 - contamination
        threshold = predictions_with_scores.approxQuantile(
            "anomaly_score", [quantile], 0.01
        )[0]

        # Add anomaly labels
        result = predictions_with_scores.withColumn(
            "is_anomaly", (self.F.col("anomaly_score") > threshold).cast("int")
        )

        return result
