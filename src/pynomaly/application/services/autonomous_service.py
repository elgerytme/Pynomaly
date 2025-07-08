"""Autonomous anomaly detection service with self-configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.application.services.algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
)
from pynomaly.application.services.autonomous_preprocessing import (
    AutonomousPreprocessingOrchestrator,
    DataQualityReport,
)
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import (
    DataLoaderProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


@dataclass
class AutonomousConfig:
    """Configuration for autonomous detection."""

    max_samples_analysis: int = 10000
    confidence_threshold: float = 0.8
    max_algorithms: int = 5
    auto_tune_hyperparams: bool = True
    save_results: bool = True
    export_results: bool = False
    export_format: str = "csv"
    verbose: bool = False

    # Preprocessing configuration
    enable_preprocessing: bool = True
    quality_threshold: float = 0.8
    max_preprocessing_time: float = 300.0  # 5 minutes
    preprocessing_strategy: str = "auto"  # auto, aggressive, conservative, minimal

    # Explainability configuration
    enable_explainability: bool = True
    explain_algorithm_choices: bool = True
    explain_anomalies: bool = True
    explanation_method: str = "auto"  # auto, shap, lime, permutation


@dataclass
class AlgorithmExplanation:
    """Explanation for why an algorithm was selected or rejected."""

    algorithm: str
    selected: bool
    confidence: float
    reasoning: str
    data_characteristics: dict[str, Any]
    decision_factors: dict[str, float]
    alternatives_considered: list[str]
    performance_prediction: float
    computational_complexity: str
    memory_requirements: str
    interpretability_score: float


@dataclass
class AnomalyExplanation:
    """Explanation for detected anomalies."""

    sample_id: int
    anomaly_score: float
    contributing_features: dict[str, float]
    feature_importances: dict[str, float]
    normal_range_deviations: dict[str, float]
    similar_normal_samples: list[int]
    explanation_confidence: float
    explanation_method: str


@dataclass
class AutonomousExplanationReport:
    """Comprehensive explanation report for autonomous detection."""

    dataset_profile: DataProfile
    algorithm_explanations: list[AlgorithmExplanation]
    selected_algorithms: list[str]
    rejected_algorithms: list[str]
    ensemble_explanation: str | None
    anomaly_explanations: list[AnomalyExplanation]
    processing_explanation: str
    recommendations: list[str]
    decision_tree: dict[str, Any]


@dataclass
class DataProfile:
    """Data profiling results."""

    n_samples: int
    n_features: int
    numeric_features: int
    categorical_features: int
    temporal_features: int
    missing_values_ratio: float
    data_types: dict[str, str]
    correlation_score: float
    sparsity_ratio: float
    outlier_ratio_estimate: float
    seasonality_detected: bool
    trend_detected: bool
    recommended_contamination: float
    complexity_score: float

    # Preprocessing-related fields
    quality_score: float = 1.0
    quality_report: DataQualityReport | None = None
    preprocessing_recommended: bool = False
    preprocessing_applied: bool = False
    preprocessing_metadata: dict[str, Any] | None = None


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with confidence."""

    algorithm: str
    confidence: float
    reasoning: str
    hyperparams: dict[str, Any]
    expected_performance: float


class AutonomousDetectionService:
    """Service for fully autonomous anomaly detection."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
        data_loaders: dict[str, DataLoaderProtocol],
        adapter_registry: AlgorithmAdapterRegistry | None = None,
    ):
        """Initialize autonomous service.

        Args:
            detector_repository: Repository for detectors
            result_repository: Repository for results
            data_loaders: Data loaders by format
            adapter_registry: Registry for algorithm adapters
        """
        self.detector_repository = detector_repository
        self.result_repository = result_repository
        self.data_loaders = data_loaders
        self.adapter_registry = adapter_registry or AlgorithmAdapterRegistry()
        self.logger = logging.getLogger(__name__)

        # Initialize preprocessing orchestrator
        self.preprocessing_orchestrator = AutonomousPreprocessingOrchestrator()

    async def detect_autonomous(
        self,
        data_source: str | Path | pd.DataFrame,
        config: AutonomousConfig | None = None,
    ) -> dict[str, Any]:
        """Run fully autonomous anomaly detection.

        Args:
            data_source: Path to data file, connection string, or DataFrame
            config: Configuration options

        Returns:
            Complete detection results with metadata
        """
        config = config or AutonomousConfig()

        if config.verbose:
            self.logger.info("Starting autonomous anomaly detection")

        # Step 1: Auto-detect data source and load
        dataset = await self._auto_load_data(data_source, config)

        # Step 2: Assess data quality and preprocess if needed
        dataset, profile = await self._assess_and_preprocess_data(dataset, config)

        # Step 3: Profile the processed data
        profile = await self._profile_data(dataset, config, profile)

        # Step 4: Recommend algorithms
        recommendations = await self._recommend_algorithms(profile, config)

        # Step 5: Auto-tune and run detection
        results = await self._run_detection_pipeline(dataset, recommendations, config)

        # Step 6: Post-process and export
        final_results = await self._finalize_results(
            dataset, profile, recommendations, results, config
        )

        if config.verbose:
            self.logger.info("Autonomous detection completed")

        return final_results

    async def _auto_load_data(
        self, data_source: str | Path | pd.DataFrame, config: AutonomousConfig
    ) -> Dataset:
        """Automatically detect and load data source."""

        if isinstance(data_source, pd.DataFrame):
            # Direct DataFrame
            return Dataset(
                name="autonomous_data",
                data=data_source,
                metadata={"source": "dataframe", "loader": "direct"},
            )

        source_path = Path(data_source)

        # Detect data format
        format_type = self._detect_data_format(source_path)

        if format_type not in self.data_loaders:
            raise DataValidationError(f"Unsupported data format: {format_type}")

        loader = self.data_loaders[format_type]

        # Auto-configure loader options
        load_options = self._auto_configure_loader(source_path, format_type)

        if config.verbose:
            self.logger.info(f"Loading {format_type} data from {source_path}")

        return loader.load(source_path, name="autonomous_data", **load_options)

    def _detect_data_format(self, source_path: Path) -> str:
        """Detect data format from file extension and content."""

        extension = source_path.suffix.lower()

        # Extension-based detection
        format_map = {
            ".csv": "csv",
            ".tsv": "csv",
            ".txt": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".arrow": "arrow",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".jsonl": "json",
        }

        if extension in format_map:
            return format_map[extension]

        # Content-based detection for ambiguous files
        try:
            with open(source_path, encoding="utf-8") as f:
                first_line = f.readline()

                # Check for CSV-like delimiters
                if any(delimiter in first_line for delimiter in [",", "\t", ";", "|"]):
                    return "csv"

                # Check for JSON
                if first_line.strip().startswith("{") or first_line.strip().startswith(
                    "["
                ):
                    return "json"

        except (UnicodeDecodeError, OSError):
            pass

        # Default to CSV for unknown text files
        return "csv"

    def _auto_configure_loader(
        self, source_path: Path, format_type: str
    ) -> dict[str, Any]:
        """Auto-configure data loader options."""

        options = {}

        if format_type == "csv":
            # Auto-detect delimiter
            try:
                with open(source_path, encoding="utf-8") as f:
                    first_line = f.readline()

                    # Count delimiter candidates
                    delimiters = [",", "\t", ";", "|"]
                    delimiter_counts = {d: first_line.count(d) for d in delimiters}

                    # Choose most common delimiter
                    best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                    if delimiter_counts[best_delimiter] > 0:
                        options["delimiter"] = best_delimiter

            except (OSError, UnicodeDecodeError):
                pass

            # Auto-detect encoding
            try:
                import chardet

                with open(source_path, "rb") as f:
                    raw_data = f.read(10000)
                    encoding_result = chardet.detect(raw_data)
                    if encoding_result["confidence"] > 0.8:
                        options["encoding"] = encoding_result["encoding"]
            except (ImportError, OSError):
                pass

        return options

    async def _assess_and_preprocess_data(
        self, dataset: Dataset, config: AutonomousConfig
    ) -> tuple[Dataset, DataProfile]:
        """Assess data quality and apply preprocessing if needed.

        Args:
            dataset: Original dataset
            config: Autonomous configuration

        Returns:
            Tuple of (processed_dataset, initial_profile_with_quality_info)
        """
        if config.verbose:
            self.logger.info("Assessing data quality for preprocessing needs")

        # Initialize partial profile for quality assessment
        initial_profile = DataProfile(
            n_samples=len(dataset.data),
            n_features=len(dataset.data.columns),
            numeric_features=0,  # Will be filled later
            categorical_features=0,  # Will be filled later
            temporal_features=0,  # Will be filled later
            missing_values_ratio=0.0,  # Will be filled later
            data_types={},  # Will be filled later
            correlation_score=0.0,  # Will be filled later
            sparsity_ratio=0.0,  # Will be filled later
            outlier_ratio_estimate=0.0,  # Will be filled later
            seasonality_detected=False,  # Will be filled later
            trend_detected=False,  # Will be filled later
            recommended_contamination=0.1,  # Will be filled later
            complexity_score=0.0,  # Will be filled later
        )

        if not config.enable_preprocessing:
            if config.verbose:
                self.logger.info("Preprocessing disabled, skipping quality assessment")
            initial_profile.preprocessing_applied = False
            return dataset, initial_profile

        # Assess data quality
        should_preprocess, quality_report = (
            self.preprocessing_orchestrator.should_preprocess(
                dataset, config.quality_threshold
            )
        )

        # Update profile with quality information
        initial_profile.quality_score = quality_report.overall_score
        initial_profile.quality_report = quality_report
        initial_profile.preprocessing_recommended = should_preprocess

        if config.verbose:
            self.logger.info(f"Data quality score: {quality_report.overall_score:.2f}")
            if quality_report.issues:
                self.logger.info(
                    f"Found {len(quality_report.issues)} data quality issues"
                )

        if not should_preprocess:
            if config.verbose:
                self.logger.info("Data quality sufficient, skipping preprocessing")
            initial_profile.preprocessing_applied = False
            return dataset, initial_profile

        # Apply preprocessing
        if config.verbose:
            self.logger.info(
                "Applying intelligent preprocessing to improve data quality"
            )

        processed_dataset, preprocessing_metadata = (
            self.preprocessing_orchestrator.preprocess_for_autonomous_detection(
                dataset, quality_report, config.max_preprocessing_time
            )
        )

        # Update profile with preprocessing results
        initial_profile.preprocessing_applied = preprocessing_metadata.get(
            "preprocessing_applied", False
        )
        initial_profile.preprocessing_metadata = preprocessing_metadata

        if config.verbose:
            if initial_profile.preprocessing_applied:
                original_shape = preprocessing_metadata.get("original_shape", (0, 0))
                final_shape = preprocessing_metadata.get("final_shape", (0, 0))
                self.logger.info(
                    f"Preprocessing completed: {original_shape[0]:,}×{original_shape[1]} → {final_shape[0]:,}×{final_shape[1]}"
                )

                applied_steps = preprocessing_metadata.get("applied_steps", [])
                if applied_steps:
                    self.logger.info(
                        f"Applied {len(applied_steps)} preprocessing steps"
                    )
            else:
                reason = preprocessing_metadata.get("reason", "Unknown")
                self.logger.info(f"Preprocessing skipped: {reason}")

        return processed_dataset, initial_profile

    async def _profile_data(
        self,
        dataset: Dataset,
        config: AutonomousConfig,
        initial_profile: DataProfile | None = None,
    ) -> DataProfile:
        """Profile dataset to understand its characteristics."""

        df = dataset.data
        n_samples, n_features = df.shape

        # Sample data if too large
        if n_samples > config.max_samples_analysis:
            sample_df = df.sample(n=config.max_samples_analysis, random_state=42)
        else:
            sample_df = df

        # Basic statistics
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        categorical_cols = sample_df.select_dtypes(
            include=["object", "category"]
        ).columns

        numeric_features = len(numeric_cols)
        categorical_features = len(categorical_cols)

        # Detect temporal features
        temporal_features = 0
        for col in sample_df.columns:
            if (
                sample_df[col].dtype == "datetime64[ns]"
                or "date" in col.lower()
                or "time" in col.lower()
            ):
                temporal_features += 1

        # Missing values
        missing_ratio = sample_df.isnull().sum().sum() / (
            sample_df.shape[0] * sample_df.shape[1]
        )

        # Data types
        data_types = {col: str(dtype) for col, dtype in sample_df.dtypes.items()}

        # Correlation analysis (numeric features only)
        correlation_score = 0.0
        if numeric_features > 1:
            corr_matrix = sample_df[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            correlation_score = upper_triangle.stack().mean()

        # Sparsity analysis
        sparsity_ratio = 0.0
        if numeric_features > 0:
            numeric_data = sample_df[numeric_cols].values
            sparsity_ratio = np.count_nonzero(numeric_data == 0) / numeric_data.size

        # Rough outlier estimation using IQR
        outlier_ratio_estimate = 0.0
        if numeric_features > 0:
            outlier_counts = []
            for col in numeric_cols:
                Q1 = sample_df[col].quantile(0.25)
                Q3 = sample_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (
                    (sample_df[col] < lower_bound) | (sample_df[col] > upper_bound)
                ).sum()
                outlier_counts.append(outliers)

            outlier_ratio_estimate = (
                max(outlier_counts) / len(sample_df) if outlier_counts else 0.0
            )

        # Time series analysis
        seasonality_detected = False
        trend_detected = False

        if temporal_features > 0 and numeric_features > 0:
            # Simple trend detection
            for col in numeric_cols:
                values = sample_df[col].dropna().values
                if len(values) > 10:
                    trend_coef = np.corrcoef(np.arange(len(values)), values)[0, 1]
                    if abs(trend_coef) > 0.3:
                        trend_detected = True
                        break

        # Recommended contamination rate
        recommended_contamination = min(0.1, max(0.01, outlier_ratio_estimate * 1.5))

        # Complexity score (0-1, higher = more complex)
        complexity_factors = [
            n_features / 100,  # Feature count
            categorical_features / max(1, n_features),  # Categorical ratio
            correlation_score,  # Correlation complexity
            missing_ratio,  # Missing data complexity
            sparsity_ratio,  # Sparsity complexity
        ]
        complexity_score = min(1.0, np.mean(complexity_factors))

        # Create full profile with preprocessing information if available
        profile = DataProfile(
            n_samples=n_samples,
            n_features=n_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            temporal_features=temporal_features,
            missing_values_ratio=missing_ratio,
            data_types=data_types,
            correlation_score=correlation_score,
            sparsity_ratio=sparsity_ratio,
            outlier_ratio_estimate=outlier_ratio_estimate,
            seasonality_detected=seasonality_detected,
            trend_detected=trend_detected,
            recommended_contamination=recommended_contamination,
            complexity_score=complexity_score,
        )

        # Copy preprocessing information from initial profile if available
        if initial_profile:
            profile.quality_score = initial_profile.quality_score
            profile.quality_report = initial_profile.quality_report
            profile.preprocessing_recommended = (
                initial_profile.preprocessing_recommended
            )
            profile.preprocessing_applied = initial_profile.preprocessing_applied
            profile.preprocessing_metadata = initial_profile.preprocessing_metadata

        return profile

    async def _recommend_algorithms(
        self, profile: DataProfile, config: AutonomousConfig
    ) -> list[AlgorithmRecommendation]:
        """Recommend algorithms based on data profile."""

        recommendations = []

        # Algorithm selection logic based on data characteristics

        # 1. Isolation Forest - good general purpose
        iso_confidence = 0.8
        iso_reasoning = "General purpose algorithm, works well with mixed data types"
        iso_hyperparams = {
            "n_estimators": min(200, max(50, profile.n_samples // 100)),
            "contamination": profile.recommended_contamination,
            "random_state": 42,
        }

        if profile.n_features > 20:
            iso_confidence += 0.1
            iso_reasoning += ", handles high-dimensional data well"

        recommendations.append(
            AlgorithmRecommendation(
                algorithm="IsolationForest",
                confidence=iso_confidence,
                reasoning=iso_reasoning,
                hyperparams=iso_hyperparams,
                expected_performance=0.75,
            )
        )

        # 2. Local Outlier Factor - good for density-based anomalies
        if profile.numeric_features >= profile.n_features * 0.7:  # Mostly numeric
            lof_confidence = 0.75
            lof_reasoning = "Good for density-based anomalies in numeric data"
            lof_hyperparams = {
                "n_neighbors": min(30, max(5, profile.n_samples // 100)),
                "contamination": profile.recommended_contamination,
            }

            if profile.n_samples < 10000:
                lof_confidence += 0.1
                lof_reasoning += ", efficient for smaller datasets"

            recommendations.append(
                AlgorithmRecommendation(
                    algorithm="LOF",
                    confidence=lof_confidence,
                    reasoning=lof_reasoning,
                    hyperparams=lof_hyperparams,
                    expected_performance=0.72,
                )
            )

        # 3. One-Class SVM - good for complex decision boundaries
        if profile.n_samples < 50000 and profile.complexity_score > 0.5:
            svm_confidence = 0.7
            svm_reasoning = "Handles complex decision boundaries well"
            svm_hyperparams = {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": profile.recommended_contamination,
            }

            recommendations.append(
                AlgorithmRecommendation(
                    algorithm="OneClassSVM",
                    confidence=svm_confidence,
                    reasoning=svm_reasoning,
                    hyperparams=svm_hyperparams,
                    expected_performance=0.68,
                )
            )

        # 4. Elliptic Envelope - good for Gaussian-distributed data
        if profile.correlation_score < 0.8 and profile.numeric_features > 2:
            ee_confidence = 0.65
            ee_reasoning = "Good for Gaussian-distributed data with low correlation"
            ee_hyperparams = {
                "contamination": profile.recommended_contamination,
                "random_state": 42,
            }

            recommendations.append(
                AlgorithmRecommendation(
                    algorithm="EllipticEnvelope",
                    confidence=ee_confidence,
                    reasoning=ee_reasoning,
                    hyperparams=ee_hyperparams,
                    expected_performance=0.65,
                )
            )

        # 5. Deep learning approach for complex/large datasets
        if profile.n_samples > 10000 and profile.complexity_score > 0.6:
            ae_confidence = 0.75
            ae_reasoning = "Deep learning approach for complex, large datasets"
            ae_hyperparams = {
                "hidden_sizes": [profile.n_features // 2, profile.n_features // 4],
                "epochs": 100,
                "batch_size": min(512, max(32, profile.n_samples // 100)),
                "contamination": profile.recommended_contamination,
            }

            recommendations.append(
                AlgorithmRecommendation(
                    algorithm="AutoEncoder",
                    confidence=ae_confidence,
                    reasoning=ae_reasoning,
                    hyperparams=ae_hyperparams,
                    expected_performance=0.78,
                )
            )

        # Sort by confidence and limit to max_algorithms
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[: config.max_algorithms]

    async def _run_detection_pipeline(
        self,
        dataset: Dataset,
        recommendations: list[AlgorithmRecommendation],
        config: AutonomousConfig,
    ) -> dict[str, DetectionResult]:
        """Run the detection pipeline with recommended algorithms."""

        results = {}

        for rec in recommendations:
            if config.verbose:
                self.logger.info(
                    f"Running {rec.algorithm} (confidence: {rec.confidence:.2f})"
                )

            try:
                if config.verbose:
                    self.logger.info(f"Creating detector for {rec.algorithm}")

                # Create detector
                detector = self._create_detector(rec, dataset)

                if config.verbose:
                    self.logger.info(f"Auto-tuning {rec.algorithm}")

                # Auto-tune if requested
                if config.auto_tune_hyperparams:
                    detector = await self._auto_tune_detector(detector, dataset, config)

                if config.verbose:
                    self.logger.info(f"Training {rec.algorithm}")

                # Train and detect using adapter registry
                self.adapter_registry.fit_detector(detector, dataset)
                detector.is_fitted = True
                detector.trained_at = pd.Timestamp.now().to_pydatetime()

                if config.verbose:
                    self.logger.info(f"Running detection with {rec.algorithm}")

                # Perform detection
                scores = self.adapter_registry.score_with_detector(detector, dataset)
                predictions = self.adapter_registry.predict_with_detector(
                    detector, dataset
                )

                if config.verbose:
                    self.logger.info(f"Processing results for {rec.algorithm}")

                # Create detection result
                from pynomaly.domain.entities import Anomaly

                anomalies = []
                for i, (score, pred) in enumerate(
                    zip(scores, predictions, strict=False)
                ):
                    if pred == 1:  # Is anomaly
                        anomaly = Anomaly(
                            score=score,
                            data_point=dataset.data.iloc[i].to_dict(),
                            detector_name=detector.name,
                        )
                        anomalies.append(anomaly)

                # Calculate threshold
                import numpy as np

                score_values = [s.value for s in scores]
                anomaly_score_values = [
                    score_values[i]
                    for i in range(len(predictions))
                    if predictions[i] == 1
                ]

                if anomaly_score_values:
                    threshold = min(anomaly_score_values)
                else:
                    threshold = np.percentile(score_values, 95)

                if config.verbose:
                    self.logger.info(
                        f"Creating DetectionResult for {rec.algorithm}: anomalies={len(anomalies)}, scores={len(scores)}, predictions={len(predictions)}"
                    )

                result = DetectionResult(
                    detector_id=detector.id,
                    dataset_id=dataset.id,
                    anomalies=anomalies,
                    scores=scores,
                    labels=predictions,
                    threshold=threshold,
                    metadata={
                        "algorithm": detector.algorithm_name,
                        "auto_generated": True,
                    },
                )

                if config.verbose:
                    self.logger.info(
                        f"DetectionResult created successfully for {rec.algorithm}"
                    )

                # Store result
                results[rec.algorithm] = result

                # Save detector and result if requested
                if config.save_results:
                    self.detector_repository.save(detector)
                    self.result_repository.save(result)

            except Exception as e:
                if config.verbose:
                    self.logger.warning(f"Failed to run {rec.algorithm}: {e}")
                    import traceback

                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                continue

        return results

    def _create_detector(
        self, recommendation: AlgorithmRecommendation, dataset: Dataset
    ) -> Detector:
        """Create detector instance from recommendation."""

        from pynomaly.domain.value_objects import ContaminationRate

        detector_id = uuid4()
        contamination_rate = ContaminationRate(
            recommendation.hyperparams.get("contamination", 0.05)
        )

        detector = Detector(
            id=detector_id,
            name=f"auto_{recommendation.algorithm.lower()}",
            algorithm_name=recommendation.algorithm,
            parameters=recommendation.hyperparams,
            contamination_rate=contamination_rate,
            metadata={
                "auto_generated": True,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
            },
        )

        return detector

    async def _auto_tune_detector(
        self, detector: Detector, dataset: Dataset, config: AutonomousConfig
    ) -> Detector:
        """Auto-tune detector hyperparameters."""

        # Simple grid search for key parameters
        # In a full implementation, this would use more sophisticated optimization

        if detector.algorithm_name == "IsolationForest":
            # Tune n_estimators and max_features
            best_score = -np.inf
            best_params = detector.parameters.copy()

            for n_est in [50, 100, 200]:
                for max_feat in [0.5, 0.7, 1.0]:
                    test_params = detector.parameters.copy()
                    test_params["n_estimators"] = n_est
                    test_params["max_features"] = max_feat

                    # Create test detector and evaluate
                    test_detector = Detector(
                        id=uuid4(),
                        name=detector.name,
                        algorithm_name=detector.algorithm_name,
                        parameters=test_params,
                        contamination_rate=detector.contamination_rate,
                    )

                    try:
                        self.adapter_registry.fit_detector(test_detector, dataset)
                        scores = self.adapter_registry.score_with_detector(
                            test_detector, dataset
                        )
                        score_values = [s.value for s in scores]

                        # Use variance of scores as a simple quality metric
                        score_quality = np.var(score_values)

                        if score_quality > best_score:
                            best_score = score_quality
                            best_params = test_params.copy()

                    except Exception:
                        continue

            detector.parameters = best_params

        return detector

    async def _finalize_results(
        self,
        dataset: Dataset,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        results: dict[str, DetectionResult],
        config: AutonomousConfig,
    ) -> dict[str, Any]:
        """Finalize and format results."""

        # Select best result based on multiple criteria
        best_algorithm = None
        best_result = None

        if results:
            # Score each result
            result_scores = {}

            for algo, result in results.items():
                score = 0.0

                # Factor 1: Anomaly detection rate should be reasonable
                rate_score = 1.0 - abs(
                    result.anomaly_rate - profile.recommended_contamination
                )
                score += rate_score * 0.3

                # Factor 2: Score distribution quality
                score_values = [s.value for s in result.scores]
                score_var = np.var(score_values)
                score_range = np.max(score_values) - np.min(score_values)
                distribution_score = min(1.0, score_var * score_range)
                score += distribution_score * 0.4

                # Factor 3: Algorithm confidence
                algo_rec = next(
                    (r for r in recommendations if r.algorithm == algo), None
                )
                if algo_rec:
                    score += algo_rec.confidence * 0.3

                result_scores[algo] = score

            # Select best
            best_algorithm = max(result_scores, key=result_scores.get)
            best_result = results[best_algorithm]

        # Create comprehensive output
        output = {
            "autonomous_detection_results": {
                "success": best_result is not None,
                "best_algorithm": best_algorithm,
                "data_profile": {
                    "samples": profile.n_samples,
                    "features": profile.n_features,
                    "numeric_features": profile.numeric_features,
                    "categorical_features": profile.categorical_features,
                    "missing_ratio": profile.missing_values_ratio,
                    "complexity_score": profile.complexity_score,
                    "recommended_contamination": profile.recommended_contamination,
                },
                "algorithm_recommendations": [
                    {
                        "algorithm": rec.algorithm,
                        "confidence": rec.confidence,
                        "reasoning": rec.reasoning,
                    }
                    for rec in recommendations
                ],
                "detection_results": {},
            }
        }

        # Add detailed results
        for algo, result in results.items():
            output["autonomous_detection_results"]["detection_results"][algo] = {
                "anomalies_found": result.n_anomalies,
                "anomaly_rate": result.anomaly_rate,
                "threshold": result.threshold,
                "execution_time_ms": result.execution_time_ms,
                "score_statistics": result.score_statistics,
            }

        # Add best result details
        if best_result:
            output["autonomous_detection_results"]["best_result"] = {
                "algorithm": best_algorithm,
                "anomalies": [
                    {
                        "index": i,
                        "score": anomaly.score.value,
                        "confidence": (
                            anomaly.score.confidence
                            if hasattr(anomaly.score, "confidence")
                            else None
                        ),
                    }
                    for i, anomaly in enumerate(best_result.anomalies)
                ],
                "summary": {
                    "total_anomalies": best_result.n_anomalies,
                    "anomaly_rate": f"{best_result.anomaly_rate:.2%}",
                    "threshold": best_result.threshold,
                    "confidence": "High" if best_result.n_anomalies > 0 else "Medium",
                },
            }

        # Export results if requested
        if config.export_results and best_result:
            await self._export_results(dataset, best_result, config)
            output["autonomous_detection_results"]["exported"] = True

        return output

    async def _export_results(
        self, dataset: Dataset, result: DetectionResult, config: AutonomousConfig
    ) -> None:
        """Export results to file."""

        # Create results DataFrame
        df = dataset.data.copy()

        # Add anomaly scores and labels
        scores = [s.value for s in result.scores]
        df["anomaly_score"] = scores
        df["is_anomaly"] = result.labels
        df["anomaly_rank"] = pd.Series(scores).rank(ascending=False)

        # Export based on format
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"autonomous_detection_results_{timestamp}"

        if config.export_format.lower() == "csv":
            df.to_csv(f"{filename}.csv", index=False)
        elif config.export_format.lower() == "parquet":
            df.to_parquet(f"{filename}.parquet", index=False)

    async def explain_algorithm_choices(
        self,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        config: AutonomousConfig,
    ) -> list[AlgorithmExplanation]:
        """Generate detailed explanations for algorithm selection decisions."""

        explanations = []
        all_algorithms = [
            "IsolationForest",
            "LOF",
            "OneClassSVM",
            "EllipticEnvelope",
            "AutoEncoder",
            "ECOD",
            "COPOD",
            "KNN",
            "ABOD",
        ]

        selected_algos = [rec.algorithm for rec in recommendations]

        for algo in all_algorithms:
            is_selected = algo in selected_algos

            if is_selected:
                rec = next(r for r in recommendations if r.algorithm == algo)
                confidence = rec.confidence
                reasoning = rec.reasoning
                prediction = rec.expected_performance
            else:
                confidence, reasoning, prediction = self._explain_rejection(
                    algo, profile
                )

            # Calculate decision factors
            decision_factors = self._calculate_decision_factors(algo, profile)

            explanation = AlgorithmExplanation(
                algorithm=algo,
                selected=is_selected,
                confidence=confidence,
                reasoning=reasoning,
                data_characteristics=self._extract_relevant_characteristics(
                    algo, profile
                ),
                decision_factors=decision_factors,
                alternatives_considered=self._get_algorithm_alternatives(algo),
                performance_prediction=prediction,
                computational_complexity=self._get_computational_complexity(
                    algo, profile
                ),
                memory_requirements=self._get_memory_requirements(algo, profile),
                interpretability_score=self._get_interpretability_score(algo),
            )

            explanations.append(explanation)

        return explanations

    def _explain_rejection(
        self, algorithm: str, profile: DataProfile
    ) -> tuple[float, str, float]:
        """Explain why an algorithm was rejected."""

        rejections = {
            "LOF": {
                "condition": profile.numeric_features < profile.n_features * 0.7,
                "reason": "Dataset contains too many categorical features for LOF (requires 70%+ numeric)",
                "confidence": 0.3,
                "performance": 0.45,
            },
            "OneClassSVM": {
                "condition": profile.n_samples >= 50000
                or profile.complexity_score <= 0.5,
                "reason": "Dataset too large for efficient SVM computation or insufficient complexity",
                "confidence": 0.35,
                "performance": 0.50,
            },
            "EllipticEnvelope": {
                "condition": profile.correlation_score >= 0.8
                or profile.numeric_features <= 2,
                "reason": "High feature correlation or insufficient numeric features for robust covariance estimation",
                "confidence": 0.25,
                "performance": 0.40,
            },
            "AutoEncoder": {
                "condition": profile.n_samples <= 10000
                or profile.complexity_score <= 0.6,
                "reason": "Insufficient data size for deep learning or low complexity doesn't justify neural approach",
                "confidence": 0.30,
                "performance": 0.55,
            },
        }

        if algorithm in rejections:
            rejection = rejections[algorithm]
            if rejection["condition"]:
                return (
                    rejection["confidence"],
                    rejection["reason"],
                    rejection["performance"],
                )

        return (
            0.20,
            f"Algorithm {algorithm} did not meet selection criteria for this dataset",
            0.35,
        )

    def _calculate_decision_factors(
        self, algorithm: str, profile: DataProfile
    ) -> dict[str, float]:
        """Calculate the decision factors that influenced algorithm selection."""

        factors = {
            "dataset_size_match": 0.0,
            "feature_type_compatibility": 0.0,
            "complexity_alignment": 0.0,
            "performance_expectation": 0.0,
            "computational_efficiency": 0.0,
            "interpretability_requirement": 0.0,
        }

        # Dataset size matching
        if algorithm == "IsolationForest":
            factors["dataset_size_match"] = min(1.0, profile.n_samples / 10000)
        elif algorithm == "LOF":
            factors["dataset_size_match"] = 1.0 - min(1.0, profile.n_samples / 50000)
        elif algorithm == "AutoEncoder":
            factors["dataset_size_match"] = min(1.0, profile.n_samples / 5000)

        # Feature type compatibility
        if algorithm in ["LOF", "OneClassSVM", "EllipticEnvelope"]:
            factors["feature_type_compatibility"] = (
                profile.numeric_features / profile.n_features
            )
        else:
            factors["feature_type_compatibility"] = 0.8  # Generally robust

        # Complexity alignment
        if algorithm in ["IsolationForest", "AutoEncoder"]:
            factors["complexity_alignment"] = profile.complexity_score
        elif algorithm in ["ECOD", "COPOD"]:
            factors["complexity_alignment"] = 1.0 - profile.complexity_score

        # Performance expectation (based on algorithm characteristics)
        performance_map = {
            "IsolationForest": 0.85,
            "LOF": 0.80,
            "AutoEncoder": 0.90,
            "OneClassSVM": 0.75,
            "EllipticEnvelope": 0.70,
            "ECOD": 0.75,
            "COPOD": 0.70,
        }
        factors["performance_expectation"] = performance_map.get(algorithm, 0.65)

        # Computational efficiency
        efficiency_map = {
            "IsolationForest": 0.90,
            "ECOD": 0.95,
            "COPOD": 0.90,
            "LOF": 0.60,
            "OneClassSVM": 0.40,
            "AutoEncoder": 0.30,
            "EllipticEnvelope": 0.85,
        }
        factors["computational_efficiency"] = efficiency_map.get(algorithm, 0.50)

        # Interpretability
        interpretability_map = {
            "IsolationForest": 0.70,
            "LOF": 0.85,
            "ECOD": 0.90,
            "COPOD": 0.85,
            "OneClassSVM": 0.40,
            "AutoEncoder": 0.20,
            "EllipticEnvelope": 0.80,
        }
        factors["interpretability_requirement"] = interpretability_map.get(
            algorithm, 0.50
        )

        return factors

    def _extract_relevant_characteristics(
        self, algorithm: str, profile: DataProfile
    ) -> dict[str, Any]:
        """Extract data characteristics relevant to the specific algorithm."""

        base_chars = {
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "numeric_ratio": profile.numeric_features / profile.n_features,
        }

        if algorithm in ["LOF", "OneClassSVM"]:
            base_chars.update(
                {
                    "sparsity": profile.sparsity_ratio,
                    "correlation_score": profile.correlation_score,
                }
            )

        if algorithm == "EllipticEnvelope":
            base_chars.update(
                {
                    "correlation_score": profile.correlation_score,
                    "outlier_ratio": profile.outlier_ratio,
                }
            )

        if algorithm == "AutoEncoder":
            base_chars.update(
                {
                    "complexity_score": profile.complexity_score,
                    "missing_ratio": profile.missing_ratio,
                }
            )

        return base_chars

    def _get_algorithm_alternatives(self, algorithm: str) -> list[str]:
        """Get alternative algorithms for the given algorithm."""

        alternatives = {
            "IsolationForest": ["LOF", "OneClassSVM", "ECOD"],
            "LOF": ["IsolationForest", "KNN", "ABOD"],
            "OneClassSVM": ["IsolationForest", "EllipticEnvelope", "LOF"],
            "EllipticEnvelope": ["OneClassSVM", "COPOD", "IsolationForest"],
            "AutoEncoder": ["IsolationForest", "VAE", "OneClassSVM"],
            "ECOD": ["IsolationForest", "COPOD", "LOF"],
            "COPOD": ["ECOD", "EllipticEnvelope", "IsolationForest"],
        }

        return alternatives.get(algorithm, ["IsolationForest"])

    def _get_computational_complexity(
        self, algorithm: str, profile: DataProfile
    ) -> str:
        """Get computational complexity description for the algorithm."""

        n, d = profile.n_samples, profile.n_features

        complexities = {
            "IsolationForest": f"O(n log n) ≈ O({n} log {n}) - Efficient for large datasets",
            "LOF": f"O(n²) ≈ O({n}²) - Quadratic complexity, may be slow for large datasets",
            "OneClassSVM": "O(n²) to O(n³) - Depends on kernel, can be very slow for large datasets",
            "EllipticEnvelope": f"O(nd²) ≈ O({n}×{d}²) - Depends on feature count",
            "AutoEncoder": "O(epochs × batch_count × layer_complexity) - Training intensive",
            "ECOD": f"O(nd) ≈ O({n}×{d}) - Linear complexity, very efficient",
            "COPOD": f"O(nd²) ≈ O({n}×{d}²) - Moderate complexity",
        }

        return complexities.get(algorithm, "O(n) - Linear complexity")

    def _get_memory_requirements(self, algorithm: str, profile: DataProfile) -> str:
        """Get memory requirements description for the algorithm."""

        n, d = profile.n_samples, profile.n_features

        requirements = {
            "IsolationForest": f"Low - O(n) storage for {n} samples",
            "LOF": f"High - O(n²) for distance matrix of {n}² elements",
            "OneClassSVM": "Medium - O(n×support_vectors) depends on complexity",
            "EllipticEnvelope": f"Low - O(d²) for covariance matrix of {d}×{d}",
            "AutoEncoder": "High - O(model_parameters + batch_size×features)",
            "ECOD": "Very Low - O(d) for feature statistics",
            "COPOD": "Low - O(d²) for copula computations",
        }

        return requirements.get(algorithm, "Medium - Standard memory usage")

    def _get_interpretability_score(self, algorithm: str) -> float:
        """Get interpretability score for the algorithm (0-1, higher is more interpretable)."""

        scores = {
            "IsolationForest": 0.7,  # Feature importance available, tree-based
            "LOF": 0.85,  # Local density intuitive
            "OneClassSVM": 0.4,  # Kernel methods less interpretable
            "EllipticEnvelope": 0.8,  # Statistical approach, clear assumptions
            "AutoEncoder": 0.2,  # Neural networks are black boxes
            "ECOD": 0.9,  # Statistical, very interpretable
            "COPOD": 0.85,  # Copula-based, statistical foundation
            "KNN": 0.9,  # Nearest neighbors very intuitive
            "ABOD": 0.75,  # Angle-based, geometrically interpretable
        }

        return scores.get(algorithm, 0.5)

    async def explain_anomalies(
        self,
        dataset: Dataset,
        result: DetectionResult,
        detector: Detector,
        config: AutonomousConfig,
        max_explanations: int = 10,
    ) -> list[AnomalyExplanation]:
        """Generate explanations for detected anomalies."""

        explanations = []

        # Get top anomalies by score
        anomaly_indices = []
        anomaly_scores = []

        for _i, anomaly in enumerate(result.anomalies[:max_explanations]):
            anomaly_indices.append(anomaly.index)
            anomaly_scores.append(anomaly.score.value)

        if not anomaly_indices:
            return explanations

        # Calculate feature-level explanations
        for _i, (idx, score) in enumerate(
            zip(anomaly_indices, anomaly_scores, strict=False)
        ):
            try:
                explanation = await self._explain_single_anomaly(
                    dataset, idx, score, detector, config
                )
                explanations.append(explanation)
            except Exception as e:
                self.logger.warning(f"Failed to explain anomaly at index {idx}: {e}")
                continue

        return explanations

    async def _explain_single_anomaly(
        self,
        dataset: Dataset,
        sample_idx: int,
        anomaly_score: float,
        detector: Detector,
        config: AutonomousConfig,
    ) -> AnomalyExplanation:
        """Explain a single anomaly."""

        sample = dataset.data.iloc[sample_idx]

        # Calculate feature contributions (basic implementation)
        feature_contributions = {}
        feature_importances = {}
        normal_range_deviations = {}

        # Get normal samples for comparison
        normal_indices = self._find_similar_normal_samples(dataset, sample_idx, sample)

        # Calculate feature-level explanations
        for col in dataset.data.columns:
            if pd.api.types.is_numeric_dtype(dataset.data[col]):
                # Calculate deviation from normal range
                normal_values = dataset.data.iloc[normal_indices][col]
                sample_value = sample[col]

                mean_normal = normal_values.mean()
                std_normal = normal_values.std()

                if std_normal > 0:
                    z_score = abs(sample_value - mean_normal) / std_normal
                    normal_range_deviations[col] = float(z_score)

                    # Feature contribution (simplified)
                    contribution = min(1.0, z_score / 3.0)  # Normalize by 3-sigma rule
                    feature_contributions[col] = float(contribution)

                    # Feature importance (how much this feature typically contributes)
                    feature_importances[col] = float(min(1.0, z_score * 0.1))

        return AnomalyExplanation(
            sample_id=sample_idx,
            anomaly_score=anomaly_score,
            contributing_features=feature_contributions,
            feature_importances=feature_importances,
            normal_range_deviations=normal_range_deviations,
            similar_normal_samples=normal_indices[:5],  # Top 5 similar normal samples
            explanation_confidence=0.75,  # Basic implementation confidence
            explanation_method="statistical_deviation",
        )

    def _find_similar_normal_samples(
        self,
        dataset: Dataset,
        anomaly_idx: int,
        anomaly_sample: pd.Series,
        n_similar: int = 20,
    ) -> list[int]:
        """Find normal samples similar to the anomaly for comparison."""

        # Simple implementation: find samples with smallest Euclidean distance
        numeric_cols = dataset.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return list(range(min(n_similar, len(dataset.data))))

        # Calculate distances
        distances = []
        anomaly_values = anomaly_sample[numeric_cols].values

        for i, row in dataset.data[numeric_cols].iterrows():
            if i != anomaly_idx:  # Exclude the anomaly itself
                distance = np.linalg.norm(row.values - anomaly_values)
                distances.append((i, distance))

        # Sort by distance and return indices
        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:n_similar]]

    async def generate_explanation_report(
        self,
        dataset: Dataset,
        profile: DataProfile,
        recommendations: list[AlgorithmRecommendation],
        results: dict[str, DetectionResult],
        config: AutonomousConfig,
    ) -> AutonomousExplanationReport:
        """Generate a comprehensive explanation report."""

        # Algorithm explanations
        algorithm_explanations = await self.explain_algorithm_choices(
            profile, recommendations, config
        )

        # Anomaly explanations for best result
        anomaly_explanations = []
        if results:
            best_result = max(results.values(), key=lambda r: len(r.anomalies))
            best_detector = None  # Would need to pass this in

            if config.explain_anomalies and best_result.anomalies:
                anomaly_explanations = await self.explain_anomalies(
                    dataset, best_result, best_detector, config
                )

        # Generate recommendations
        recommendations_text = self._generate_recommendations(
            profile, algorithm_explanations, results
        )

        # Build decision tree
        decision_tree = self._build_decision_tree(profile, recommendations)

        return AutonomousExplanationReport(
            dataset_profile=profile,
            algorithm_explanations=algorithm_explanations,
            selected_algorithms=[rec.algorithm for rec in recommendations],
            rejected_algorithms=[
                exp.algorithm for exp in algorithm_explanations if not exp.selected
            ],
            ensemble_explanation=self._generate_ensemble_explanation(recommendations),
            anomaly_explanations=anomaly_explanations,
            processing_explanation=self._generate_processing_explanation(
                profile, config
            ),
            recommendations=recommendations_text,
            decision_tree=decision_tree,
        )

    def _generate_recommendations(
        self,
        profile: DataProfile,
        algorithm_explanations: list[AlgorithmExplanation],
        results: dict[str, DetectionResult],
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Data quality recommendations
        if profile.missing_ratio > 0.1:
            recommendations.append(
                f"Consider improving data quality - {profile.missing_ratio:.1%} missing values detected"
            )

        if profile.outlier_ratio > 0.2:
            recommendations.append(
                "High outlier ratio detected - consider data cleaning before anomaly detection"
            )

        # Algorithm recommendations
        selected_algos = [exp for exp in algorithm_explanations if exp.selected]
        if len(selected_algos) < 3:
            recommendations.append(
                "Consider ensemble methods with additional algorithms for improved performance"
            )

        # Performance recommendations
        if results:
            best_result = max(results.values(), key=lambda r: len(r.anomalies))
            if len(best_result.anomalies) == 0:
                recommendations.append(
                    "No anomalies detected - consider lowering contamination parameter or trying different algorithms"
                )
            elif len(best_result.anomalies) > len(results) * 0.1:
                recommendations.append(
                    "High anomaly rate detected - verify results and consider increasing threshold"
                )

        return recommendations

    def _generate_ensemble_explanation(
        self, recommendations: list[AlgorithmRecommendation]
    ) -> str | None:
        """Generate explanation for ensemble approach."""

        if len(recommendations) < 2:
            return None

        return (
            f"Ensemble of {len(recommendations)} algorithms recommended for improved robustness. "
            f"Combining {', '.join(rec.algorithm for rec in recommendations)} "
            f"balances different detection approaches and reduces false positives."
        )

    def _generate_processing_explanation(
        self, profile: DataProfile, config: AutonomousConfig
    ) -> str:
        """Generate explanation for data processing decisions."""

        explanations = []

        if config.enable_preprocessing:
            explanations.append(
                f"Preprocessing enabled with {config.preprocessing_strategy} strategy"
            )

        if profile.missing_ratio > 0:
            explanations.append(
                f"Missing value handling required ({profile.missing_ratio:.1%} missing)"
            )

        if profile.categorical_features > 0:
            explanations.append(
                f"Categorical encoding needed for {profile.categorical_features} features"
            )

        return (
            "; ".join(explanations)
            if explanations
            else "Minimal preprocessing required"
        )

    def _build_decision_tree(
        self, profile: DataProfile, recommendations: list[AlgorithmRecommendation]
    ) -> dict[str, Any]:
        """Build a decision tree explaining the selection process."""

        tree = {
            "root": "Algorithm Selection Process",
            "dataset_analysis": {
                "sample_size": {
                    "value": profile.n_samples,
                    "category": (
                        "large"
                        if profile.n_samples > 10000
                        else "medium"
                        if profile.n_samples > 1000
                        else "small"
                    ),
                },
                "feature_count": {
                    "value": profile.n_features,
                    "category": (
                        "high"
                        if profile.n_features > 50
                        else "medium"
                        if profile.n_features > 10
                        else "low"
                    ),
                },
                "data_complexity": {
                    "value": profile.complexity_score,
                    "category": (
                        "high"
                        if profile.complexity_score > 0.7
                        else "medium"
                        if profile.complexity_score > 0.4
                        else "low"
                    ),
                },
            },
            "algorithm_decisions": [
                {
                    "algorithm": rec.algorithm,
                    "decision": "selected",
                    "confidence": rec.confidence,
                    "primary_reason": rec.reasoning.split(",")[
                        0
                    ],  # First part of reasoning
                }
                for rec in recommendations
            ],
            "selection_criteria": {
                "confidence_threshold": 0.8,
                "max_algorithms": len(recommendations),
                "primary_factors": [
                    "dataset_size",
                    "feature_types",
                    "complexity",
                    "computational_efficiency",
                ],
            },
        }

        return tree
