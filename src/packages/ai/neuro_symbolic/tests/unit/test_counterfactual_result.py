"""Unit tests for CounterfactualResult value objects in neuro-symbolic reasoning."""

import pytest
from datetime import datetime
from neuro_symbolic.domain.value_objects.counterfactual_result import (
    CounterfactualResult, CounterfactualScenario, FeatureChange,
    CounterfactualAnalysisResult, CounterfactualType, ChangeDirection
)


class TestFeatureChange:
    """Test cases for FeatureChange value object in neuro-symbolic reasoning."""
    
    def test_create_valid_feature_change(self):
        """Test creating a valid feature change."""
        change = FeatureChange(
            feature_name="temperature",
            original_value=85.5,
            counterfactual_value=75.0,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10.5,
            confidence=0.9,
            feasibility=0.8
        )
        
        assert change.feature_name == "temperature"
        assert change.original_value == 85.5
        assert change.counterfactual_value == 75.0
        assert change.change_direction == ChangeDirection.DECREASE
        assert change.change_magnitude == 10.5
        assert change.confidence == 0.9
        assert change.feasibility == 0.8
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            FeatureChange(
                feature_name="temp",
                original_value=85.5,
                counterfactual_value=75.0,
                change_direction=ChangeDirection.DECREASE,
                change_magnitude=10.5,
                confidence=1.5,
                feasibility=0.8
            )
    
    def test_invalid_feasibility_raises_error(self):
        """Test that invalid feasibility values raise ValueError."""
        with pytest.raises(ValueError, match="Feasibility must be between 0 and 1"):
            FeatureChange(
                feature_name="temp",
                original_value=85.5,
                counterfactual_value=75.0,
                change_direction=ChangeDirection.DECREASE,
                change_magnitude=10.5,
                confidence=0.9,
                feasibility=-0.1
            )
    
    def test_negative_change_magnitude_raises_error(self):
        """Test that negative change magnitude raises ValueError."""
        with pytest.raises(ValueError, match="Change magnitude must be non-negative"):
            FeatureChange(
                feature_name="temp",
                original_value=85.5,
                counterfactual_value=75.0,
                change_direction=ChangeDirection.DECREASE,
                change_magnitude=-10.5,
                confidence=0.9,
                feasibility=0.8
            )
    
    def test_feature_change_string_representation(self):
        """Test string representation of feature change."""
        change = FeatureChange(
            feature_name="temperature",
            original_value=85.5,
            counterfactual_value=75.0,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10.5,
            confidence=0.9,
            feasibility=0.8
        )
        
        expected = "temperature: 85.5 → 75.0 (decrease, magnitude: 10.500)"
        assert str(change) == expected


class TestCounterfactualScenario:
    """Test cases for CounterfactualScenario value object in reasoning contexts."""
    
    def test_create_valid_scenario(self):
        """Test creating a valid counterfactual scenario."""
        change1 = FeatureChange(
            feature_name="temperature",
            original_value=85.5,
            counterfactual_value=75.0,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10.5,
            confidence=0.9,
            feasibility=0.8
        )
        
        change2 = FeatureChange(
            feature_name="pressure",
            original_value=120,
            counterfactual_value=100,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=20,
            confidence=0.8,
            feasibility=0.9
        )
        
        scenario = CounterfactualScenario(
            id="scenario_1",
            name="Reduce Temperature and Pressure",
            changes=[change1, change2],
            original_prediction="positive_class",
            counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8,
            scenario_probability=0.7,
            explanation="Reducing both temperature and pressure changes the classification outcome"
        )
        
        assert scenario.id == "scenario_1"
        assert scenario.name == "Reduce Temperature and Pressure"
        assert len(scenario.changes) == 2
        assert scenario.original_prediction == "positive_class"
        assert scenario.counterfactual_prediction == "negative_class"
        assert scenario.prediction_change_magnitude == 0.8
        assert scenario.scenario_probability == 0.7
    
    def test_empty_changes_raises_error(self):
        """Test that empty changes list raises ValueError."""
        with pytest.raises(ValueError, match="Scenario must have at least one change"):
            CounterfactualScenario(
                id="scenario_1",
                name="Empty Scenario",
                changes=[],
                original_prediction="positive_class",
                counterfactual_prediction="negative_class",
                prediction_change_magnitude=0.8,
                scenario_probability=0.7,
                explanation="Empty reasoning scenario"
            )
    
    def test_scenario_properties(self):
        """Test calculated properties of scenario."""
        change1 = FeatureChange(
            feature_name="temp",
            original_value=85,
            counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10,
            confidence=0.9,
            feasibility=0.8
        )
        
        change2 = FeatureChange(
            feature_name="pressure",
            original_value=120,
            counterfactual_value=100,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=20,
            confidence=0.8,
            feasibility=0.6
        )
        
        scenario = CounterfactualScenario(
            id="scenario_1",
            name="Test Scenario",
            changes=[change1, change2],
            original_prediction="positive_class",
            counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8,
            scenario_probability=0.7,
            explanation="Test explanation"
        )
        
        assert scenario.total_change_magnitude == 30  # 10 + 20
        assert scenario.average_feasibility == 0.7  # (0.8 + 0.6) / 2
        assert scenario.changed_features == {"temp", "pressure"}
    
    def test_get_change_by_feature(self):
        """Test getting change for specific feature."""
        change = FeatureChange(
            feature_name="temperature",
            original_value=85,
            counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10,
            confidence=0.9,
            feasibility=0.8
        )
        
        scenario = CounterfactualScenario(
            id="scenario_1",
            name="Test Scenario",
            changes=[change],
            original_prediction="positive_class",
            counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8,
            scenario_probability=0.7,
            explanation="Test explanation"
        )
        
        retrieved_change = scenario.get_change_by_feature("temperature")
        assert retrieved_change == change
        
        missing_change = scenario.get_change_by_feature("missing_feature")
        assert missing_change is None


class TestCounterfactualResult:
    """Test cases for CounterfactualResult value object in neuro-symbolic reasoning."""
    
    def test_create_valid_counterfactual_result(self):
        """Test creating a valid counterfactual reasoning result."""
        # Create scenario
        change = FeatureChange(
            feature_name="temperature",
            original_value=85,
            counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE,
            change_magnitude=10,
            confidence=0.9,
            feasibility=0.8
        )
        
        scenario = CounterfactualScenario(
            id="scenario_1",
            name="Reduce Temperature",
            changes=[change],
            original_prediction="positive_class",
            counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8,
            scenario_probability=0.7,
            explanation="Reduce temperature to change classification"
        )
        
        # Create counterfactual result
        result = CounterfactualResult.create(
            query="What if temperature was lower?",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temperature": 85, "pressure": 100},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario],
            feature_importance_ranking=[("temperature", 0.8), ("pressure", 0.3)],
            stability_score=0.7,
            robustness_score=0.6,
            assumptions=["Temperature is controllable"],
            limitations=["Only considers single feature changes"]
        )
        
        assert result.query == "What if temperature was lower?"
        assert result.counterfactual_type == CounterfactualType.WHAT_IF
        assert result.original_prediction == "positive_class"
        assert result.original_confidence == 0.9
        assert len(result.scenarios) == 1
        assert result.best_scenario == scenario
        assert result.stability_score == 0.7
        assert result.robustness_score == 0.6
        assert isinstance(result.timestamp, datetime)
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence scores raise ValueError."""
        change = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario = CounterfactualScenario(
            id="s1", name="Scenario", changes=[change],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Test"
        )
        
        with pytest.raises(ValueError, match="Original confidence must be between 0 and 1"):
            CounterfactualResult.create(
                query="Test query",
                counterfactual_type=CounterfactualType.WHAT_IF,
                original_input={"temp": 85},
                original_prediction="positive_class",
                original_confidence=1.5,  # Invalid
                scenarios=[scenario]
            )
    
    def test_optimal_reasoning_scenario_selection(self):
        """Test that optimal reasoning scenario is correctly selected."""
        # Create two scenarios with different feasibility and change magnitude characteristics
        change1 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.9
        )
        scenario1 = CounterfactualScenario(
            id="s1", name="Scenario1", changes=[change1],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="High feasibility, minimal change required"
        )
        
        change2 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=50,
            change_direction=ChangeDirection.DECREASE, change_magnitude=35,
            confidence=0.9, feasibility=0.3
        )
        scenario2 = CounterfactualScenario(
            id="s2", name="Scenario2", changes=[change2],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.9, scenario_probability=0.8,
            explanation="Low feasibility, significant change required"
        )
        
        result = CounterfactualResult.create(
            query="Test query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario1, scenario2]
        )
        
        # Optimal scenario should be the one with higher feasibility and minimal change requirements
        assert result.best_scenario == scenario1
    
    def test_get_reasoning_scenarios_by_type(self):
        """Test filtering reasoning scenarios by change direction."""
        change1 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario1 = CounterfactualScenario(
            id="s1", name="Decrease", changes=[change1],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Decrease temperature for classification change"
        )
        
        change2 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=95,
            change_direction=ChangeDirection.INCREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario2 = CounterfactualScenario(
            id="s2", name="Increase", changes=[change2],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Increase temperature for classification change"
        )
        
        result = CounterfactualResult.create(
            query="Test query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario1, scenario2]
        )
        
        decrease_scenarios = result.get_scenarios_by_type(ChangeDirection.DECREASE)
        assert len(decrease_scenarios) == 1
        assert decrease_scenarios[0] == scenario1
        
        increase_scenarios = result.get_scenarios_by_type(ChangeDirection.INCREASE)
        assert len(increase_scenarios) == 1
        assert increase_scenarios[0] == scenario2
    
    def test_feature_sensitivity_reasoning_analysis(self):
        """Test feature sensitivity analysis in reasoning context."""
        # Create scenarios affecting different input features for sensitivity analysis
        change1 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario1 = CounterfactualScenario(
            id="s1", name="Temp Change", changes=[change1],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Modify temperature for reasoning outcome"
        )
        
        change2 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=70,
            change_direction=ChangeDirection.DECREASE, change_magnitude=15,
            confidence=0.8, feasibility=0.7
        )
        scenario2 = CounterfactualScenario(
            id="s2", name="Bigger Temp Change", changes=[change2],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.9, scenario_probability=0.6,
            explanation="Greater temperature modification for reasoning outcome"
        )
        
        result = CounterfactualResult.create(
            query="Test query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario1, scenario2]
        )
        
        sensitivity = result.analyze_feature_sensitivity()
        
        assert "temp" in sensitivity
        temp_stats = sensitivity["temp"]
        assert temp_stats["scenarios_count"] == 2
        assert temp_stats["average_change_magnitude"] == 12.5  # (10 + 15) / 2
        assert temp_stats["average_prediction_impact"] == 0.85  # (0.8 + 0.9) / 2
        assert temp_stats["average_feasibility"] == 0.75  # (0.8 + 0.7) / 2
    
    def test_classification_boundary_analysis(self):
        """Test classification boundary analysis in reasoning context."""
        # Create scenario that changes classification outcome
        change = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        flipping_scenario = CounterfactualScenario(
            id="s1", name="Flip Prediction", changes=[change],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Changes classification outcome"
        )
        
        result = CounterfactualResult.create(
            query="Test query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[flipping_scenario]
        )
        
        boundary_analysis = result.get_decision_boundary_analysis()
        
        assert boundary_analysis["boundary_distance"] == 10  # Total change magnitude
        assert boundary_analysis["is_near_boundary"] is True  # < 1.0
        assert boundary_analysis["minimal_flip_changes"] == 1  # Minimum changes for outcome flip
        assert boundary_analysis["flip_probability"] == 0.7  # Probability of outcome change
        assert boundary_analysis["total_flip_scenarios"] == 1  # Scenarios that change outcome
    
    def test_validate_consistency(self):
        """Test consistency validation in reasoning scenarios."""
        # Create scenario with no changes (should be invalid for reasoning analysis)
        empty_scenario = CounterfactualScenario(
            id="s1", name="Empty", changes=[],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Empty reasoning scenario"
        )
        
        # This will raise an error during scenario creation due to empty changes list
        # So we can't test this specific validation case here
        
        # Instead, test feature importance consistency in reasoning scenarios
        change = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario = CounterfactualScenario(
            id="s1", name="Temp Change", changes=[change],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Modify temperature for reasoning outcome"
        )
        
        result = CounterfactualResult.create(
            query="Test query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario],
            feature_importance_ranking=[("pressure", 0.8)]  # Feature not present in reasoning scenarios
        )
        
        issues = result.validate_consistency()
        assert len(issues) > 0
        assert any("Important features not found in scenarios" in issue for issue in issues)


class TestCounterfactualAnalysisResult:
    """Test cases for CounterfactualAnalysisResult in reasoning analysis."""
    
    def test_create_analysis_result(self):
        """Test creating counterfactual reasoning analysis result."""
        # Create two counterfactual reasoning analysis results
        change1 = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario1 = CounterfactualScenario(
            id="s1", name="Scenario1", changes=[change1],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Primary reasoning scenario"
        )
        
        result1 = CounterfactualResult.create(
            query="Query 1",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario1],
            stability_score=0.8,
            robustness_score=0.7
        )
        
        change2 = FeatureChange(
            feature_name="pressure", original_value=100, counterfactual_value=90,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.8, feasibility=0.9
        )
        scenario2 = CounterfactualScenario(
            id="s2", name="Scenario2", changes=[change2],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.7, scenario_probability=0.8,
            explanation="Alternative reasoning scenario"
        )
        
        result2 = CounterfactualResult.create(
            query="Query 2",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"pressure": 100},
            original_prediction="positive_class",
            original_confidence=0.8,
            scenarios=[scenario2],
            stability_score=0.6,
            robustness_score=0.9
        )
        
        analysis_result = CounterfactualAnalysisResult.create(
            counterfactual_results=[result1, result2],
            metadata={"analysis_type": "comprehensive_reasoning"}
        )
        
        assert len(analysis_result.counterfactual_results) == 2
        assert analysis_result.overall_stability == 0.7  # (0.8 + 0.6) / 2
        assert analysis_result.overall_robustness == 0.8  # (0.7 + 0.9) / 2
        assert analysis_result.metadata["analysis_type"] == "comprehensive_reasoning"
    
    def test_empty_results_raises_error(self):
        """Test that empty counterfactual reasoning results list raises ValueError."""
        with pytest.raises(ValueError, match="Must have at least one counterfactual result"):
            CounterfactualAnalysisResult.create(counterfactual_results=[])
    
    def test_get_reasoning_analysis_by_type(self):
        """Test filtering reasoning analysis results by counterfactual reasoning type."""
        # Create different types of counterfactual reasoning results
        change = FeatureChange(
            feature_name="temp", original_value=85, counterfactual_value=75,
            change_direction=ChangeDirection.DECREASE, change_magnitude=10,
            confidence=0.9, feasibility=0.8
        )
        scenario = CounterfactualScenario(
            id="s1", name="Scenario", changes=[change],
            original_prediction="positive_class", counterfactual_prediction="negative_class",
            prediction_change_magnitude=0.8, scenario_probability=0.7,
            explanation="Test reasoning scenario"
        )
        
        what_if_result = CounterfactualResult.create(
            query="What if query",
            counterfactual_type=CounterfactualType.WHAT_IF,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario]
        )
        
        minimal_result = CounterfactualResult.create(
            query="Minimal change query",
            counterfactual_type=CounterfactualType.MINIMAL_CHANGE,
            original_input={"temp": 85},
            original_prediction="positive_class",
            original_confidence=0.9,
            scenarios=[scenario]
        )
        
        analysis_result = CounterfactualAnalysisResult.create(
            counterfactual_results=[what_if_result, minimal_result]
        )
        
        what_if_results = analysis_result.get_analysis_by_type(CounterfactualType.WHAT_IF)
        assert len(what_if_results) == 1
        assert what_if_results[0] == what_if_result
        
        minimal_results = analysis_result.get_analysis_by_type(CounterfactualType.MINIMAL_CHANGE)
        assert len(minimal_results) == 1
        assert minimal_results[0] == minimal_result
        
        feature_results = analysis_result.get_analysis_by_type(CounterfactualType.FEATURE_ATTRIBUTION)
        assert len(feature_results) == 0