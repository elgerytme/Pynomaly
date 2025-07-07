"""Comprehensive tests for configuration DTO functions.

This module provides complete test coverage for the critical configuration
management functions that were previously untested (0% coverage).
"""

from datetime import datetime

import pytest

from pynomaly.application.dto.configuration_dto import (
    AlgorithmConfigurationDTO,
    ConfigurationMetadataDTO,
    DatasetConfigurationDTO,
    EvaluationConfigurationDTO,
    ExperimentConfigurationDTO,
    PreprocessingConfigurationDTO,
    merge_configurations,
    validate_configuration_compatibility,
)


class TestMergeConfigurations:
    """Test suite for merge_configurations function."""

    @pytest.fixture
    def base_config(self) -> ExperimentConfigurationDTO:
        """Create a base configuration for testing."""
        return ExperimentConfigurationDTO(
            id="base-config-1",
            name="Base Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                algorithm_type="ensemble",
                parameters={"contamination": 0.1, "n_estimators": 100},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/base_dataset.csv",
                dataset_format="csv",
                target_column="target",
                feature_columns=["feature1", "feature2", "feature3"],
                validation_split=0.2,
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",
                handle_missing="mean",
                feature_selection_method="none",
                outlier_removal=False,
            ),
            evaluation_config=EvaluationConfigurationDTO(
                metrics=["precision", "recall", "f1"],
                cross_validation_folds=5,
                test_size=0.2,
                random_state=42,
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=["baseline", "production"],
                description="Base configuration for testing",
                version="1.0.0",
            ),
        )

    @pytest.fixture
    def override_config(self) -> ExperimentConfigurationDTO:
        """Create an override configuration for testing."""
        return ExperimentConfigurationDTO(
            id="override-config-1",
            name="Override Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="LOF",
                algorithm_type="proximity",
                parameters={"contamination": 0.15, "n_neighbors": 20},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/override_dataset.csv",
                dataset_format="csv",
                target_column="target",
                feature_columns=["feature1", "feature2", "feature3", "feature4"],
                validation_split=0.3,
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="minmax",
                handle_missing="median",
                feature_selection_method="variance",
                outlier_removal=True,
            ),
            evaluation_config=EvaluationConfigurationDTO(
                metrics=["auc", "precision", "recall"],
                cross_validation_folds=10,
                test_size=0.3,
                random_state=123,
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=["experimental", "tuned"],
                description="Override configuration for testing",
                version="2.0.0",
            ),
        )

    def test_merge_configurations_success(
        self,
        base_config: ExperimentConfigurationDTO,
        override_config: ExperimentConfigurationDTO,
    ):
        """Test successful configuration merging with complete override."""
        # Execute merge
        merged = merge_configurations(base_config, override_config)

        # Verify basic properties
        assert merged is not base_config  # Should be a new instance
        assert merged is not override_config  # Should be a new instance
        assert merged.id == base_config.id  # Keeps base ID
        assert merged.name == base_config.name  # Keeps base name

        # Verify algorithm config was overridden
        assert merged.algorithm_config.algorithm_name == "LOF"
        assert merged.algorithm_config.algorithm_type == "proximity"
        assert merged.algorithm_config.parameters["contamination"] == 0.15
        assert merged.algorithm_config.parameters["n_neighbors"] == 20

        # Verify preprocessing config was overridden
        assert merged.preprocessing_config.scaling_method == "minmax"
        assert merged.preprocessing_config.handle_missing == "median"
        assert merged.preprocessing_config.feature_selection_method == "variance"
        assert merged.preprocessing_config.outlier_removal is True

        # Verify evaluation config was overridden
        assert merged.evaluation_config.metrics == ["auc", "precision", "recall"]
        assert merged.evaluation_config.cross_validation_folds == 10
        assert merged.evaluation_config.test_size == 0.3
        assert merged.evaluation_config.random_state == 123

        # Verify metadata updates
        assert merged.metadata.derived_from == ["base-config-1", "override-config-1"]
        assert merged.metadata.parent_id == "base-config-1"
        assert merged.metadata.version == "merged-1.0.0"

    def test_merge_configurations_partial_override(
        self, base_config: ExperimentConfigurationDTO
    ):
        """Test merging when override has None values for optional configs."""
        # Create override with minimal configuration
        minimal_override = ExperimentConfigurationDTO(
            id="minimal-override",
            name="Minimal Override",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="OCSVM",
                algorithm_type="linear",
                parameters={"contamination": 0.2},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/new_dataset.csv", dataset_format="csv"
            ),
            preprocessing_config=None,  # Will not override
            evaluation_config=None,  # Will not override
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        # Execute merge
        merged = merge_configurations(base_config, minimal_override)

        # Verify algorithm was overridden
        assert merged.algorithm_config.algorithm_name == "OCSVM"
        assert merged.algorithm_config.algorithm_type == "linear"
        assert merged.algorithm_config.parameters["contamination"] == 0.2

        # Verify original configs were preserved
        assert merged.preprocessing_config.scaling_method == "standard"
        assert merged.preprocessing_config.handle_missing == "mean"
        assert merged.evaluation_config.metrics == ["precision", "recall", "f1"]
        assert merged.evaluation_config.cross_validation_folds == 5

        # Verify metadata updates
        assert merged.metadata.derived_from == ["base-config-1", "minimal-override"]
        assert merged.metadata.parent_id == "base-config-1"
        assert merged.metadata.version == "merged-1.0.0"

    def test_merge_configurations_deep_copy(
        self,
        base_config: ExperimentConfigurationDTO,
        override_config: ExperimentConfigurationDTO,
    ):
        """Test that merge creates deep copies and doesn't modify originals."""
        # Store original values
        original_base_algorithm = base_config.algorithm_config.algorithm_name
        original_override_algorithm = override_config.algorithm_config.algorithm_name
        original_base_params = base_config.algorithm_config.parameters.copy()

        # Execute merge
        merged = merge_configurations(base_config, override_config)

        # Modify merged configuration
        merged.algorithm_config.algorithm_name = "ModifiedAlgorithm"
        merged.algorithm_config.parameters["new_param"] = "new_value"

        # Verify originals were not modified
        assert base_config.algorithm_config.algorithm_name == original_base_algorithm
        assert (
            override_config.algorithm_config.algorithm_name
            == original_override_algorithm
        )
        assert base_config.algorithm_config.parameters == original_base_params
        assert "new_param" not in base_config.algorithm_config.parameters
        assert "new_param" not in override_config.algorithm_config.parameters

    def test_merge_configurations_preserves_dataset_config(
        self,
        base_config: ExperimentConfigurationDTO,
        override_config: ExperimentConfigurationDTO,
    ):
        """Test that dataset configuration is preserved from base."""
        # Execute merge
        merged = merge_configurations(base_config, override_config)

        # Dataset config should remain from base, not override
        assert (
            merged.dataset_config.dataset_path
            == base_config.dataset_config.dataset_path
        )
        assert (
            merged.dataset_config.validation_split
            == base_config.dataset_config.validation_split
        )
        assert (
            merged.dataset_config.feature_columns
            == base_config.dataset_config.feature_columns
        )


class TestValidateConfigurationCompatibility:
    """Test suite for validate_configuration_compatibility function."""

    @pytest.fixture
    def config1(self) -> ExperimentConfigurationDTO:
        """Create first configuration for compatibility testing."""
        return ExperimentConfigurationDTO(
            id="config-1",
            name="Configuration 1",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset1.csv", dataset_format="csv"
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard", handle_missing="mean"
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

    @pytest.fixture
    def config2_compatible(self) -> ExperimentConfigurationDTO:
        """Create second configuration that's compatible with config1."""
        return ExperimentConfigurationDTO(
            id="config-2",
            name="Configuration 2",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm
                algorithm_type="ensemble",
                parameters={"contamination": 0.15},  # Different params OK
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset1.csv",  # Same dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",  # Same scaling method
                handle_missing="median",  # Different missing handling OK
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

    def test_validate_compatible_configurations(
        self,
        config1: ExperimentConfigurationDTO,
        config2_compatible: ExperimentConfigurationDTO,
    ):
        """Test validation of compatible configurations."""
        issues = validate_configuration_compatibility(config1, config2_compatible)

        # Should have no compatibility issues
        assert len(issues) == 0
        assert issues == []

    def test_validate_different_algorithms(self, config1: ExperimentConfigurationDTO):
        """Test validation with different algorithms."""
        config2_different_algo = ExperimentConfigurationDTO(
            id="config-different-algo",
            name="Different Algorithm Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="LOF",  # Different algorithm
                algorithm_type="proximity",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset1.csv",  # Same dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",  # Same scaling
                handle_missing="mean",
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(config1, config2_different_algo)

        # Should detect algorithm difference
        assert len(issues) == 1
        assert "Different algorithms specified" in issues[0]

    def test_validate_different_datasets(self, config1: ExperimentConfigurationDTO):
        """Test validation with different datasets."""
        config2_different_dataset = ExperimentConfigurationDTO(
            id="config-different-dataset",
            name="Different Dataset Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset2.csv",  # Different dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",  # Same scaling
                handle_missing="mean",
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(
            config1, config2_different_dataset
        )

        # Should detect dataset difference
        assert len(issues) == 1
        assert "Different datasets specified" in issues[0]

    def test_validate_different_preprocessing_scaling(
        self, config1: ExperimentConfigurationDTO
    ):
        """Test validation with different preprocessing scaling methods."""
        config2_different_scaling = ExperimentConfigurationDTO(
            id="config-different-scaling",
            name="Different Scaling Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset1.csv",  # Same dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="minmax",  # Different scaling method
                handle_missing="mean",
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(
            config1, config2_different_scaling
        )

        # Should detect preprocessing difference
        assert len(issues) == 1
        assert "Different preprocessing scaling methods" in issues[0]

    def test_validate_multiple_incompatibilities(
        self, config1: ExperimentConfigurationDTO
    ):
        """Test validation with multiple incompatibilities."""
        config2_multiple_issues = ExperimentConfigurationDTO(
            id="config-multiple-issues",
            name="Multiple Issues Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="OCSVM",  # Different algorithm
                algorithm_type="linear",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/different_dataset.csv",  # Different dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="robust",  # Different scaling
                handle_missing="mean",
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(config1, config2_multiple_issues)

        # Should detect all three issues
        assert len(issues) == 3
        assert "Different algorithms specified" in issues[0]
        assert "Different datasets specified" in issues[1]
        assert "Different preprocessing scaling methods" in issues[2]

    def test_validate_none_dataset_paths(self, config1: ExperimentConfigurationDTO):
        """Test validation when one or both configs have None dataset paths."""
        config2_none_dataset = ExperimentConfigurationDTO(
            id="config-none-dataset",
            name="None Dataset Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path=None,  # No dataset path
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",  # Same scaling
                handle_missing="mean",
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(config1, config2_none_dataset)

        # Should not report dataset issues when one path is None
        dataset_issues = [issue for issue in issues if "datasets" in issue]
        assert len(dataset_issues) == 0

    def test_validate_none_preprocessing_configs(
        self, config1: ExperimentConfigurationDTO
    ):
        """Test validation when preprocessing configs are None."""
        config2_none_preprocessing = ExperimentConfigurationDTO(
            id="config-none-preprocessing",
            name="None Preprocessing Config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/dataset1.csv",  # Same dataset
                dataset_format="csv",
            ),
            preprocessing_config=None,  # No preprocessing config
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        issues = validate_configuration_compatibility(
            config1, config2_none_preprocessing
        )

        # Should not report preprocessing issues when one config is None
        preprocessing_issues = [issue for issue in issues if "preprocessing" in issue]
        assert len(preprocessing_issues) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_merge_with_identical_configs(self):
        """Test merging identical configurations."""
        config = ExperimentConfigurationDTO(
            id="identical-config",
            name="Identical Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/test.csv", dataset_format="csv"
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        # Merge with itself
        merged = merge_configurations(config, config)

        # Should create new instance with updated metadata
        assert merged is not config
        assert (
            merged.algorithm_config.algorithm_name
            == config.algorithm_config.algorithm_name
        )
        assert merged.metadata.derived_from == ["identical-config", "identical-config"]
        assert merged.metadata.parent_id == "identical-config"
        assert merged.metadata.version == "merged-1.0.0"

    def test_validate_identical_configs(self):
        """Test validation of identical configurations."""
        config = ExperimentConfigurationDTO(
            id="identical-config",
            name="Identical Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                algorithm_type="ensemble",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/test.csv", dataset_format="csv"
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard", handle_missing="mean"
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        # Validate with itself
        issues = validate_configuration_compatibility(config, config)

        # Should have no issues
        assert len(issues) == 0
        assert issues == []


# Integration tests
class TestConfigurationIntegration:
    """Integration tests for configuration functions."""

    def test_merge_then_validate_workflow(self):
        """Test realistic workflow of merging configs then validating compatibility."""
        # Create base production config
        base_config = ExperimentConfigurationDTO(
            id="prod-config",
            name="Production Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                algorithm_type="ensemble",
                parameters={"contamination": 0.1, "n_estimators": 100},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/production.csv",
                dataset_format="csv",
                feature_columns=["feature1", "feature2"],
                validation_split=0.2,
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard", handle_missing="mean", outlier_removal=False
            ),
            evaluation_config=EvaluationConfigurationDTO(
                metrics=["precision", "recall", "f1"],
                cross_validation_folds=5,
                test_size=0.2,
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=["production", "stable"],
                version="1.0.0",
            ),
        )

        # Create experimental override
        experimental_override = ExperimentConfigurationDTO(
            id="exp-config",
            name="Experimental Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",  # Same algorithm for compatibility
                algorithm_type="ensemble",
                parameters={
                    "contamination": 0.15,
                    "n_estimators": 200,
                },  # Different params
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/production.csv",  # Same dataset
                dataset_format="csv",
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard",  # Same scaling for compatibility
                handle_missing="median",  # Different missing handling
                outlier_removal=True,  # Different outlier setting
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=["experimental", "tuned"],
                version="2.0.0",
            ),
        )

        # Step 1: Merge configurations
        merged_config = merge_configurations(base_config, experimental_override)

        # Verify merge worked correctly
        assert merged_config.algorithm_config.parameters["contamination"] == 0.15
        assert merged_config.algorithm_config.parameters["n_estimators"] == 200
        assert merged_config.preprocessing_config.handle_missing == "median"
        assert merged_config.preprocessing_config.outlier_removal is True

        # Step 2: Validate compatibility between original and merged
        issues = validate_configuration_compatibility(base_config, merged_config)

        # Should be compatible (same algorithm, dataset, and scaling method)
        assert len(issues) == 0

        # Step 3: Test with incompatible config
        incompatible_config = ExperimentConfigurationDTO(
            id="incompatible-config",
            name="Incompatible Configuration",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="LOF",  # Different algorithm
                algorithm_type="proximity",
                parameters={"contamination": 0.1},
            ),
            dataset_config=DatasetConfigurationDTO(
                dataset_path="/data/production.csv", dataset_format="csv"
            ),
            preprocessing_config=PreprocessingConfigurationDTO(
                scaling_method="standard", handle_missing="mean"
            ),
            metadata=ConfigurationMetadataDTO(
                created_at=datetime.now(), updated_at=datetime.now(), version="1.0.0"
            ),
        )

        # Should detect incompatibility
        incompatible_issues = validate_configuration_compatibility(
            merged_config, incompatible_config
        )
        assert len(incompatible_issues) == 1
        assert "Different algorithms specified" in incompatible_issues[0]


if __name__ == "__main__":
    pytest.main([__file__])
