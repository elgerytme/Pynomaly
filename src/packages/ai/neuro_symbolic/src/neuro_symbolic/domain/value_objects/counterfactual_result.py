"""Counterfactual reasoning value objects for neuro-symbolic AI."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class CounterfactualType(Enum):
    """Types of counterfactual analysis."""
    WHAT_IF = "what_if"  # What if X were different?
    NECESSARY_CONDITION = "necessary_condition"  # What if X were absent?
    SUFFICIENT_CONDITION = "sufficient_condition"  # What if only X were present?
    MINIMAL_CHANGE = "minimal_change"  # Smallest change to flip outcome
    MAXIMAL_CHANGE = "maximal_change"  # Largest meaningful change
    FEATURE_ATTRIBUTION = "feature_attribution"  # Impact of individual features


class ChangeDirection(Enum):
    """Direction of counterfactual change."""
    INCREASE = "increase"
    DECREASE = "decrease"
    REMOVE = "remove"
    ADD = "add"
    REPLACE = "replace"
    NEGATE = "negate"


@dataclass(frozen=True)
class FeatureChange:
    """Represents a change to a feature in counterfactual reasoning."""
    feature_name: str
    original_value: Any
    counterfactual_value: Any
    change_direction: ChangeDirection
    change_magnitude: float
    confidence: float
    feasibility: float  # How feasible/realistic is this change
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
        if not (0.0 <= self.feasibility <= 1.0):
            raise ValueError("Feasibility must be between 0 and 1")
        if self.change_magnitude < 0:
            raise ValueError("Change magnitude must be non-negative")
    
    def __str__(self) -> str:
        return (f"{self.feature_name}: {self.original_value} → {self.counterfactual_value} "
                f"({self.change_direction.value}, magnitude: {self.change_magnitude:.3f})")


@dataclass(frozen=True)
class CounterfactualScenario:
    """Represents a complete counterfactual scenario with multiple feature changes."""
    id: str
    name: str
    changes: List[FeatureChange]
    original_prediction: Any
    counterfactual_prediction: Any
    prediction_change_magnitude: float
    scenario_probability: float  # Probability this scenario could occur
    explanation: str
    constraints_satisfied: bool = True
    
    def __post_init__(self):
        if not self.changes:
            raise ValueError("Scenario must have at least one change")
        if not (0.0 <= self.scenario_probability <= 1.0):
            raise ValueError("Scenario probability must be between 0 and 1")
        if self.prediction_change_magnitude < 0:
            raise ValueError("Prediction change magnitude must be non-negative")
    
    @property
    def total_change_magnitude(self) -> float:
        """Calculate total magnitude of all changes."""
        return sum(change.change_magnitude for change in self.changes)
    
    @property
    def average_feasibility(self) -> float:
        """Calculate average feasibility of all changes."""
        return sum(change.feasibility for change in self.changes) / len(self.changes)
    
    @property
    def changed_features(self) -> Set[str]:
        """Get set of all changed feature names."""
        return {change.feature_name for change in self.changes}
    
    def get_change_by_feature(self, feature_name: str) -> Optional[FeatureChange]:
        """Get the change for a specific feature."""
        for change in self.changes:
            if change.feature_name == feature_name:
                return change
        return None
    
    def __str__(self) -> str:
        return (f"Scenario '{self.name}': {len(self.changes)} changes, "
                f"prediction: {self.original_prediction} → {self.counterfactual_prediction}")


@dataclass(frozen=True)
class CounterfactualResult:
    """
    Immutable value object representing the result of counterfactual reasoning.
    Answers questions like "What would happen if X were different?"
    """
    
    id: str
    query: str  # The counterfactual question being answered
    counterfactual_type: CounterfactualType
    original_input: Dict[str, Any]
    original_prediction: Any
    original_confidence: float
    
    scenarios: List[CounterfactualScenario]
    best_scenario: CounterfactualScenario
    
    # Analysis results
    feature_importance_ranking: List[Tuple[str, float]]  # (feature_name, importance_score)
    stability_score: float  # How stable is the original prediction
    robustness_score: float  # How robust is the model to changes
    
    methodology: str = "neural_symbolic_counterfactual"
    timestamp: datetime = field(default_factory=datetime.now)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        if not (0.0 <= self.original_confidence <= 1.0):
            raise ValueError("Original confidence must be between 0 and 1")
        
        if not (0.0 <= self.stability_score <= 1.0):
            raise ValueError("Stability score must be between 0 and 1")
        
        if not (0.0 <= self.robustness_score <= 1.0):
            raise ValueError("Robustness score must be between 0 and 1")
        
        if not self.scenarios:
            raise ValueError("Must have at least one scenario")
        
        if self.best_scenario not in self.scenarios:
            raise ValueError("Best scenario must be in scenarios list")
    
    @classmethod
    def create(
        cls,
        query: str,
        counterfactual_type: CounterfactualType,
        original_input: Dict[str, Any],
        original_prediction: Any,
        original_confidence: float,
        scenarios: List[CounterfactualScenario],
        feature_importance_ranking: Optional[List[Tuple[str, float]]] = None,
        stability_score: float = 0.5,
        robustness_score: float = 0.5,
        methodology: str = "neural_symbolic_counterfactual",
        assumptions: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "CounterfactualResult":
        """Create a new counterfactual result."""
        if not scenarios:
            raise ValueError("Must provide at least one scenario")
        
        # Select best scenario (highest feasibility and lowest change magnitude)
        best_scenario = max(
            scenarios,
            key=lambda s: s.average_feasibility / (1 + s.total_change_magnitude)
        )
        
        return cls(
            id=str(uuid.uuid4()),
            query=query,
            counterfactual_type=counterfactual_type,
            original_input=original_input,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            scenarios=scenarios,
            best_scenario=best_scenario,
            feature_importance_ranking=feature_importance_ranking or [],
            stability_score=stability_score,
            robustness_score=robustness_score,
            methodology=methodology,
            assumptions=assumptions or [],
            limitations=limitations or [],
            metadata=metadata
        )
    
    def get_scenarios_by_type(self, change_direction: ChangeDirection) -> List[CounterfactualScenario]:
        """Get scenarios that involve a specific type of change."""
        matching_scenarios = []
        for scenario in self.scenarios:
            if any(change.change_direction == change_direction for change in scenario.changes):
                matching_scenarios.append(scenario)
        return matching_scenarios
    
    def get_most_impactful_features(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get the most impactful features for counterfactual changes."""
        return self.feature_importance_ranking[:top_k]
    
    def get_minimal_change_scenario(self) -> CounterfactualScenario:
        """Get the scenario with the smallest total change magnitude."""
        return min(self.scenarios, key=lambda s: s.total_change_magnitude)
    
    def get_most_feasible_scenario(self) -> CounterfactualScenario:
        """Get the scenario with the highest average feasibility."""
        return max(self.scenarios, key=lambda s: s.average_feasibility)
    
    def get_largest_impact_scenario(self) -> CounterfactualScenario:
        """Get the scenario with the largest prediction change."""
        return max(self.scenarios, key=lambda s: s.prediction_change_magnitude)
    
    def analyze_feature_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Analyze sensitivity of each feature across all scenarios."""
        feature_stats = {}
        
        for scenario in self.scenarios:
            for change in scenario.changes:
                feature_name = change.feature_name
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'change_magnitudes': [],
                        'prediction_impacts': [],
                        'feasibilities': []
                    }
                
                feature_stats[feature_name]['change_magnitudes'].append(change.change_magnitude)
                feature_stats[feature_name]['prediction_impacts'].append(scenario.prediction_change_magnitude)
                feature_stats[feature_name]['feasibilities'].append(change.feasibility)
        
        # Calculate statistics
        sensitivity_analysis = {}
        for feature_name, stats in feature_stats.items():
            sensitivity_analysis[feature_name] = {
                'average_change_magnitude': sum(stats['change_magnitudes']) / len(stats['change_magnitudes']),
                'average_prediction_impact': sum(stats['prediction_impacts']) / len(stats['prediction_impacts']),
                'average_feasibility': sum(stats['feasibilities']) / len(stats['feasibilities']),
                'scenarios_count': len(stats['change_magnitudes']),
                'sensitivity_score': (
                    sum(stats['prediction_impacts']) / len(stats['prediction_impacts'])
                ) / (
                    sum(stats['change_magnitudes']) / len(stats['change_magnitudes']) + 1e-6
                )
            }
        
        return sensitivity_analysis
    
    def get_decision_boundary_analysis(self) -> Dict[str, Any]:
        """Analyze how close the original input is to decision boundaries."""
        # Find scenarios that flip the prediction
        flipping_scenarios = [
            s for s in self.scenarios 
            if s.original_prediction != s.counterfactual_prediction
        ]
        
        if not flipping_scenarios:
            return {
                'boundary_distance': float('inf'),
                'is_near_boundary': False,
                'minimal_flip_changes': 0,
                'flip_probability': 0.0
            }
        
        # Find minimal change needed to flip
        minimal_flip_scenario = min(flipping_scenarios, key=lambda s: s.total_change_magnitude)
        
        return {
            'boundary_distance': minimal_flip_scenario.total_change_magnitude,
            'is_near_boundary': minimal_flip_scenario.total_change_magnitude < 1.0,
            'minimal_flip_changes': len(minimal_flip_scenario.changes),
            'flip_probability': minimal_flip_scenario.scenario_probability,
            'flip_scenario': minimal_flip_scenario.name,
            'total_flip_scenarios': len(flipping_scenarios)
        }
    
    def compare_scenarios(self, scenario1_name: str, scenario2_name: str) -> Dict[str, Any]:
        """Compare two scenarios in detail."""
        scenario1 = None
        scenario2 = None
        
        for scenario in self.scenarios:
            if scenario.name == scenario1_name:
                scenario1 = scenario
            elif scenario.name == scenario2_name:
                scenario2 = scenario
        
        if not scenario1 or not scenario2:
            raise ValueError("One or both scenarios not found")
        
        common_features = scenario1.changed_features.intersection(scenario2.changed_features)
        unique_to_1 = scenario1.changed_features.difference(scenario2.changed_features)
        unique_to_2 = scenario2.changed_features.difference(scenario1.changed_features)
        
        return {
            'scenario1': scenario1.name,
            'scenario2': scenario2.name,
            'common_changed_features': list(common_features),
            'unique_to_scenario1': list(unique_to_1),
            'unique_to_scenario2': list(unique_to_2),
            'change_magnitude_difference': abs(
                scenario1.total_change_magnitude - scenario2.total_change_magnitude
            ),
            'feasibility_difference': abs(
                scenario1.average_feasibility - scenario2.average_feasibility
            ),
            'prediction_impact_difference': abs(
                scenario1.prediction_change_magnitude - scenario2.prediction_change_magnitude
            ),
            'more_feasible': scenario1.name if scenario1.average_feasibility > scenario2.average_feasibility else scenario2.name,
            'more_impactful': scenario1.name if scenario1.prediction_change_magnitude > scenario2.prediction_change_magnitude else scenario2.name
        }
    
    def get_explanation_summary(self) -> str:
        """Get a human-readable summary of the counterfactual result."""
        summary_parts = [
            f"Query: {self.query}",
            f"Type: {self.counterfactual_type.value}",
            f"Scenarios: {len(self.scenarios)}",
            f"Best scenario: {self.best_scenario.name}",
            f"Stability: {self.stability_score:.3f}",
            f"Robustness: {self.robustness_score:.3f}"
        ]
        
        # Add info about most important features
        if self.feature_importance_ranking:
            top_features = [f"{name}({score:.2f})" for name, score in self.feature_importance_ranking[:3]]
            summary_parts.append(f"Key features: {', '.join(top_features)}")
        
        # Add decision boundary info
        boundary_analysis = self.get_decision_boundary_analysis()
        if boundary_analysis['is_near_boundary']:
            summary_parts.append("⚠️ Near decision boundary")
        
        return " | ".join(summary_parts)
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of the counterfactual result."""
        return {
            'id': self.id,
            'query': self.query,
            'counterfactual_type': self.counterfactual_type.value,
            'original_input_summary': {
                'num_features': len(self.original_input),
                'feature_names': list(self.original_input.keys())
            },
            'original_prediction': self.original_prediction,
            'original_confidence': self.original_confidence,
            'scenarios_summary': {
                'total_scenarios': len(self.scenarios),
                'best_scenario': self.best_scenario.name,
                'minimal_change_scenario': self.get_minimal_change_scenario().name,
                'most_feasible_scenario': self.get_most_feasible_scenario().name,
                'largest_impact_scenario': self.get_largest_impact_scenario().name
            },
            'feature_importance': dict(self.feature_importance_ranking),
            'stability_score': self.stability_score,
            'robustness_score': self.robustness_score,
            'feature_sensitivity': self.analyze_feature_sensitivity(),
            'decision_boundary_analysis': self.get_decision_boundary_analysis(),
            'scenarios_detail': [
                {
                    'name': scenario.name,
                    'changes_count': len(scenario.changes),
                    'total_change_magnitude': scenario.total_change_magnitude,
                    'average_feasibility': scenario.average_feasibility,
                    'prediction_change': scenario.prediction_change_magnitude,
                    'probability': scenario.scenario_probability,
                    'explanation': scenario.explanation,
                    'changes': [
                        {
                            'feature': change.feature_name,
                            'original': change.original_value,
                            'counterfactual': change.counterfactual_value,
                            'direction': change.change_direction.value,
                            'magnitude': change.change_magnitude,
                            'feasibility': change.feasibility
                        }
                        for change in scenario.changes
                    ]
                }
                for scenario in self.scenarios
            ],
            'methodology': self.methodology,
            'assumptions': self.assumptions,
            'limitations': self.limitations,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def validate_consistency(self) -> List[str]:
        """Validate internal consistency of the counterfactual result."""
        issues = []
        
        # Check if all scenarios have valid changes
        for scenario in self.scenarios:
            if not scenario.changes:
                issues.append(f"Scenario '{scenario.name}' has no changes")
            
            # Check for duplicate feature changes within scenario
            feature_names = [change.feature_name for change in scenario.changes]
            if len(feature_names) != len(set(feature_names)):
                issues.append(f"Scenario '{scenario.name}' has duplicate feature changes")
        
        # Check feature importance consistency
        importance_features = set(name for name, _ in self.feature_importance_ranking)
        scenario_features = set()
        for scenario in self.scenarios:
            scenario_features.update(scenario.changed_features)
        
        missing_important_features = importance_features - scenario_features
        if missing_important_features:
            issues.append(f"Important features not found in scenarios: {missing_important_features}")
        
        # Check score consistency
        if self.stability_score < 0.3 and self.robustness_score > 0.8:
            issues.append("Low stability with high robustness seems inconsistent")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.get_detailed_analysis()
    
    def __str__(self) -> str:
        return self.get_explanation_summary()


@dataclass(frozen=True)
class CounterfactualAnalysisResult:
    """Result of comprehensive counterfactual analysis."""
    
    counterfactual_results: List[CounterfactualResult]
    overall_stability: float
    overall_robustness: float
    consensus_important_features: List[Tuple[str, float]]
    analysis_summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.counterfactual_results:
            raise ValueError("Must have at least one counterfactual result")
        
        if not (0.0 <= self.overall_stability <= 1.0):
            raise ValueError("Overall stability must be between 0 and 1")
        
        if not (0.0 <= self.overall_robustness <= 1.0):
            raise ValueError("Overall robustness must be between 0 and 1")
    
    @classmethod
    def create(
        cls,
        counterfactual_results: List[CounterfactualResult],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "CounterfactualAnalysisResult":
        """Create comprehensive counterfactual analysis result."""
        if not counterfactual_results:
            raise ValueError("Must provide at least one counterfactual result")
        
        # Calculate overall metrics
        overall_stability = sum(r.stability_score for r in counterfactual_results) / len(counterfactual_results)
        overall_robustness = sum(r.robustness_score for r in counterfactual_results) / len(counterfactual_results)
        
        # Find consensus important features
        feature_importance_counts = {}
        for result in counterfactual_results:
            for feature_name, importance in result.feature_importance_ranking:
                if feature_name not in feature_importance_counts:
                    feature_importance_counts[feature_name] = []
                feature_importance_counts[feature_name].append(importance)
        
        consensus_features = []
        for feature_name, importances in feature_importance_counts.items():
            if len(importances) >= len(counterfactual_results) / 2:  # Appears in majority
                avg_importance = sum(importances) / len(importances)
                consensus_features.append((feature_name, avg_importance))
        
        consensus_features.sort(key=lambda x: x[1], reverse=True)
        
        # Create summary
        analysis_summary = (
            f"Analyzed {len(counterfactual_results)} counterfactual queries. "
            f"Overall stability: {overall_stability:.3f}, "
            f"robustness: {overall_robustness:.3f}. "
            f"Consensus features: {len(consensus_features)}"
        )
        
        return cls(
            counterfactual_results=counterfactual_results,
            overall_stability=overall_stability,
            overall_robustness=overall_robustness,
            consensus_important_features=consensus_features,
            analysis_summary=analysis_summary,
            metadata=metadata
        )
    
    def get_analysis_by_type(self, counterfactual_type: CounterfactualType) -> List[CounterfactualResult]:
        """Get all analyses of a specific counterfactual type."""
        return [r for r in self.counterfactual_results if r.counterfactual_type == counterfactual_type]