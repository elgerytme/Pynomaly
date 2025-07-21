"""
Unit tests for Algorithm Configuration Value Object
"""

import pytest
from ...domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType


class TestAlgorithmConfig:
    """Test cases for AlgorithmConfig value object."""
    
    def test_create_valid_isolation_forest_config(self):
        """Test creating a valid Isolation Forest configuration."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100, "max_samples": "auto"},
            contamination=0.1,
            random_state=42
        )
        
        assert config.algorithm_type == AlgorithmType.ISOLATION_FOREST
        assert config.contamination == 0.1
        assert config.random_state == 42
        assert config.is_valid()
    
    def test_create_valid_lof_config(self):
        """Test creating a valid LOF configuration."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            parameters={"n_neighbors": 20},
            contamination=0.05
        )
        
        assert config.algorithm_type == AlgorithmType.LOCAL_OUTLIER_FACTOR
        assert config.contamination == 0.05
        assert config.is_valid()
    
    def test_invalid_contamination_rate(self):
        """Test that invalid contamination rates raise ValueError."""
        with pytest.raises(ValueError, match="Contamination must be between 0.0 and 0.5"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={},
                contamination=0.6  # Invalid: > 0.5
            )
        
        with pytest.raises(ValueError, match="Contamination must be between 0.0 and 0.5"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={},
                contamination=0.0  # Invalid: = 0.0
            )
    
    def test_invalid_random_state(self):
        """Test that negative random state raises ValueError."""
        with pytest.raises(ValueError, match="Random state must be non-negative"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={},
                contamination=0.1,
                random_state=-1
            )
    
    def test_invalid_isolation_forest_parameters(self):
        """Test validation of Isolation Forest parameters."""
        with pytest.raises(ValueError, match="Invalid parameter for Isolation Forest"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={"invalid_param": 123},
                contamination=0.1
            )
        
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                parameters={"n_estimators": 0},
                contamination=0.1
            )
    
    def test_invalid_lof_parameters(self):
        """Test validation of LOF parameters."""
        with pytest.raises(ValueError, match="Invalid parameter for LOF"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                parameters={"invalid_param": 123},
                contamination=0.1
            )
        
        with pytest.raises(ValueError, match="n_neighbors must be a positive integer"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                parameters={"n_neighbors": 0},
                contamination=0.1
            )
    
    def test_to_dict_conversion(self):
        """Test converting configuration to dictionary."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100},
            contamination=0.1,
            random_state=42
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["algorithm_type"] == "isolation_forest"
        assert config_dict["parameters"] == {"n_estimators": 100}
        assert config_dict["contamination"] == 0.1
        assert config_dict["random_state"] == 42
    
    def test_from_dict_creation(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "algorithm_type": "isolation_forest",
            "parameters": {"n_estimators": 100},
            "contamination": 0.1,
            "random_state": 42
        }
        
        config = AlgorithmConfig.from_dict(config_dict)
        
        assert config.algorithm_type == AlgorithmType.ISOLATION_FOREST
        assert config.parameters == {"n_estimators": 100}
        assert config.contamination == 0.1
        assert config.random_state == 42
    
    def test_with_contamination(self):
        """Test creating new config with different contamination."""
        original_config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100},
            contamination=0.1,
            random_state=42
        )
        
        new_config = original_config.with_contamination(0.2)
        
        # Original should be unchanged
        assert original_config.contamination == 0.1
        
        # New config should have updated contamination
        assert new_config.contamination == 0.2
        assert new_config.algorithm_type == AlgorithmType.ISOLATION_FOREST
        assert new_config.parameters == {"n_estimators": 100}
        assert new_config.random_state == 42
    
    def test_with_parameters(self):
        """Test creating new config with updated parameters."""
        original_config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100},
            contamination=0.1,
            random_state=42
        )
        
        new_config = original_config.with_parameters(
            n_estimators=200,
            max_samples=0.8
        )
        
        # Original should be unchanged
        assert original_config.parameters == {"n_estimators": 100}
        
        # New config should have updated parameters
        expected_params = {"n_estimators": 200, "max_samples": 0.8}
        assert new_config.parameters == expected_params
        assert new_config.contamination == 0.1
        assert new_config.random_state == 42
    
    def test_immutability(self):
        """Test that the value object is immutable."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100},
            contamination=0.1
        )
        
        # Should not be able to modify the object
        with pytest.raises(AttributeError):
            config.contamination = 0.2  # type: ignore
        
        with pytest.raises(AttributeError):
            config.algorithm_type = AlgorithmType.LOCAL_OUTLIER_FACTOR  # type: ignore