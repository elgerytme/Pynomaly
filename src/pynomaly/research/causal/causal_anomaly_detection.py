"""Causal anomaly detection with causal inference, structural models, and counterfactual analysis."""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CausalMethod(str, Enum):
    """Causal inference methods."""
    
    STRUCTURAL_CAUSAL_MODEL = "structural_causal_model"
    DIRECTED_ACYCLIC_GRAPH = "directed_acyclic_graph"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    CAUSAL_FORESTS = "causal_forests"
    DOUBLE_ML = "double_ml"


class CausalGraphType(str, Enum):
    """Types of causal graphs."""
    
    DAG = "dag"  # Directed Acyclic Graph
    MAG = "mag"  # Maximal Ancestral Graph
    PAG = "pag"  # Partial Ancestral Graph
    CPDAG = "cpdag"  # Completed Partially Directed Acyclic Graph


class AnomalyType(str, Enum):
    """Types of causal anomalies."""
    
    STRUCTURAL_BREAK = "structural_break"  # Change in causal structure
    EFFECT_ANOMALY = "effect_anomaly"  # Unusual causal effect
    CONFOUNDING_ANOMALY = "confounding_anomaly"  # Unexpected confounding
    MEDIATION_ANOMALY = "mediation_anomaly"  # Unusual mediation effect
    COLLIDER_ANOMALY = "collider_anomaly"  # Collider bias anomaly
    SELECTION_ANOMALY = "selection_anomaly"  # Selection bias anomaly


@dataclass
class CausalVariable:
    """Causal variable definition."""
    
    name: str
    var_type: str  # continuous, binary, categorical, ordinal
    description: str
    domain: Optional[Tuple[float, float]] = None  # For continuous variables
    categories: Optional[List[str]] = None  # For categorical variables
    is_treatment: bool = False
    is_outcome: bool = False
    is_confounder: bool = False
    is_mediator: bool = False
    is_instrument: bool = False


@dataclass
class CausalEdge:
    """Edge in causal graph."""
    
    source: str
    target: str
    edge_type: str  # directed, bidirected, undirected
    strength: float = 0.0  # Causal effect strength
    confidence: float = 0.0  # Confidence in edge existence
    mechanism: str = ""  # Description of causal mechanism
    delay: int = 0  # Time delay for temporal causation


@dataclass
class CausalGraph:
    """Causal graph representation."""
    
    variables: Dict[str, CausalVariable]
    edges: List[CausalEdge]
    graph_type: CausalGraphType
    temporal: bool = False
    confounders: Set[str] = field(default_factory=set)
    mediators: Set[str] = field(default_factory=set)
    instruments: Set[str] = field(default_factory=set)


@dataclass
class CounterfactualQuery:
    """Counterfactual query specification."""
    
    query_id: str
    treatment_variables: Dict[str, Any]  # Variable -> counterfactual value
    outcome_variables: List[str]
    conditioning_variables: Optional[Dict[str, Any]] = None
    intervention_type: str = "atomic"  # atomic, stochastic, dynamic
    time_horizon: Optional[int] = None


@dataclass
class CausalAnomalyResult:
    """Result of causal anomaly detection."""
    
    sample_id: str
    anomaly_type: AnomalyType
    anomaly_score: float
    causal_explanation: str
    affected_variables: List[str]
    counterfactual_analysis: Dict[str, Any]
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    evidence: Dict[str, Any] = field(default_factory=dict)


class CausalInferenceEngine:
    """Base class for causal inference methods."""
    
    def __init__(self, method: CausalMethod, config: Dict[str, Any]):
        self.method = method
        self.config = config
        self.causal_graph: Optional[CausalGraph] = None
        self.learned_parameters: Dict[str, Any] = {}
    
    async def learn_causal_structure(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Learn causal structure from data."""
        raise NotImplementedError
    
    async def estimate_causal_effects(
        self,
        data: np.ndarray,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Estimate causal effects."""
        raise NotImplementedError
    
    async def generate_counterfactuals(
        self,
        query: CounterfactualQuery,
        observed_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate counterfactual outcomes."""
        raise NotImplementedError


class StructuralCausalModel(CausalInferenceEngine):
    """Structural Causal Model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CausalMethod.STRUCTURAL_CAUSAL_MODEL, config)
        self.structural_equations: Dict[str, str] = {}
        self.noise_distributions: Dict[str, Dict[str, Any]] = {}
        self.parameters: Dict[str, np.ndarray] = {}
    
    async def learn_causal_structure(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Learn causal structure using SCM approach."""
        try:
            logger.info("Learning causal structure with SCM")
            
            # Create variables
            variables = {}
            for i, name in enumerate(variable_names):
                var_data = data[:, i]
                var_type = self._infer_variable_type(var_data)
                
                variables[name] = CausalVariable(
                    name=name,
                    var_type=var_type,
                    description=f"Variable {name}",
                    domain=self._get_variable_domain(var_data, var_type)
                )
            
            # Learn causal edges using correlation and conditional independence
            edges = await self._discover_causal_edges(data, variable_names)
            
            # Create causal graph
            self.causal_graph = CausalGraph(
                variables=variables,
                edges=edges,
                graph_type=CausalGraphType.DAG
            )
            
            # Learn structural equations
            await self._learn_structural_equations(data, variable_names)
            
            logger.info(f"Learned SCM with {len(edges)} causal edges")
            return self.causal_graph
            
        except Exception as e:
            logger.error(f"SCM structure learning failed: {e}")
            # Return empty graph as fallback
            return CausalGraph(
                variables={name: CausalVariable(name=name, var_type="continuous", description="") 
                          for name in variable_names},
                edges=[],
                graph_type=CausalGraphType.DAG
            )
    
    async def _discover_causal_edges(self, data: np.ndarray, variable_names: List[str]) -> List[CausalEdge]:
        """Discover causal edges using statistical methods."""
        edges = []
        n_vars = len(variable_names)
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(data, rowvar=False)
        
        # Use correlation threshold and conditional independence tests
        threshold = self.config.get("correlation_threshold", 0.3)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and abs(correlation_matrix[i, j]) > threshold:
                    # Simple heuristic: assume direction based on variable ordering
                    # In practice, this would use more sophisticated methods
                    
                    strength = abs(correlation_matrix[i, j])
                    confidence = await self._test_conditional_independence(data, i, j, list(range(n_vars)))
                    
                    if confidence > 0.5:  # Threshold for edge existence
                        edge = CausalEdge(
                            source=variable_names[i],
                            target=variable_names[j],
                            edge_type="directed",
                            strength=strength,
                            confidence=confidence,
                            mechanism=f"Linear relationship from {variable_names[i]} to {variable_names[j]}"
                        )
                        edges.append(edge)
        
        return edges
    
    async def _test_conditional_independence(
        self,
        data: np.ndarray,
        var1: int,
        var2: int,
        conditioning_set: List[int]
    ) -> float:
        """Test conditional independence between variables."""
        try:
            # Simplified conditional independence test
            # In practice, this would use proper CI tests like partial correlation
            
            conditioning_vars = [v for v in conditioning_set if v != var1 and v != var2]
            
            if not conditioning_vars:
                # Unconditional correlation
                corr = abs(np.corrcoef(data[:, var1], data[:, var2])[0, 1])
                return corr
            
            # Compute partial correlation (simplified)
            subset_data = data[:, [var1, var2] + conditioning_vars]
            partial_corr = self._compute_partial_correlation(subset_data, 0, 1, list(range(2, len(conditioning_vars) + 2)))
            
            return abs(partial_corr)
            
        except Exception as e:
            logger.warning(f"Conditional independence test failed: {e}")
            return 0.0
    
    def _compute_partial_correlation(self, data: np.ndarray, x: int, y: int, z: List[int]) -> float:
        """Compute partial correlation between x and y given z."""
        try:
            if not z:
                return np.corrcoef(data[:, x], data[:, y])[0, 1]
            
            # Linear regression approach
            from sklearn.linear_model import LinearRegression
            
            # Regress x on z
            reg_x = LinearRegression()
            reg_x.fit(data[:, z], data[:, x])
            residual_x = data[:, x] - reg_x.predict(data[:, z])
            
            # Regress y on z
            reg_y = LinearRegression()
            reg_y.fit(data[:, z], data[:, y])
            residual_y = data[:, y] - reg_y.predict(data[:, z])
            
            # Correlation of residuals
            return np.corrcoef(residual_x, residual_y)[0, 1]
            
        except Exception as e:
            logger.warning(f"Partial correlation computation failed: {e}")
            return 0.0
    
    async def _learn_structural_equations(self, data: np.ndarray, variable_names: List[str]) -> None:
        """Learn structural equations for SCM."""
        try:
            from sklearn.linear_model import LinearRegression
            
            for i, target_var in enumerate(variable_names):
                # Find parents of this variable
                parents = []
                for edge in self.causal_graph.edges:
                    if edge.target == target_var:
                        parent_idx = variable_names.index(edge.source)
                        parents.append(parent_idx)
                
                if parents:
                    # Learn linear structural equation
                    X = data[:, parents]
                    y = data[:, i]
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    # Store parameters
                    self.parameters[target_var] = {
                        "coefficients": reg.coef_,
                        "intercept": reg.intercept_,
                        "parents": [variable_names[p] for p in parents]
                    }
                    
                    # Create structural equation string
                    equation_parts = [f"{reg.intercept_:.3f}"]
                    for j, parent_idx in enumerate(parents):
                        coef = reg.coef_[j]
                        parent_name = variable_names[parent_idx]
                        equation_parts.append(f"{coef:.3f}*{parent_name}")
                    
                    self.structural_equations[target_var] = f"{target_var} = " + " + ".join(equation_parts) + " + noise"
                    
                    # Estimate noise distribution
                    predicted = reg.predict(X)
                    residuals = y - predicted
                    self.noise_distributions[target_var] = {
                        "type": "gaussian",
                        "mean": np.mean(residuals),
                        "std": np.std(residuals)
                    }
                
                else:
                    # Root node - just estimate marginal distribution
                    self.parameters[target_var] = {
                        "coefficients": np.array([]),
                        "intercept": np.mean(data[:, i]),
                        "parents": []
                    }
                    
                    self.structural_equations[target_var] = f"{target_var} = {np.mean(data[:, i]):.3f} + noise"
                    self.noise_distributions[target_var] = {
                        "type": "gaussian",
                        "mean": 0.0,
                        "std": np.std(data[:, i])
                    }
            
            logger.info("Learned structural equations for SCM")
            
        except Exception as e:
            logger.error(f"Structural equation learning failed: {e}")
    
    async def estimate_causal_effects(
        self,
        data: np.ndarray,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Estimate causal effects using SCM."""
        try:
            if not self.causal_graph:
                raise ValueError("Causal graph not learned")
            
            variable_names = list(self.causal_graph.variables.keys())
            treatment_idx = variable_names.index(treatment)
            outcome_idx = variable_names.index(outcome)
            
            # Direct causal effect from structural equation
            if outcome in self.parameters:
                params = self.parameters[outcome]
                if treatment in params.get("parents", []):
                    parent_idx = params["parents"].index(treatment)
                    direct_effect = params["coefficients"][parent_idx]
                else:
                    direct_effect = 0.0
            else:
                direct_effect = 0.0
            
            # Total causal effect via intervention
            total_effect = await self._compute_total_effect(treatment, outcome, data, variable_names)
            
            # Indirect effect
            indirect_effect = total_effect - direct_effect
            
            return {
                "direct_effect": direct_effect,
                "indirect_effect": indirect_effect,
                "total_effect": total_effect,
                "natural_direct_effect": direct_effect,  # Simplified
                "natural_indirect_effect": indirect_effect  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Causal effect estimation failed: {e}")
            return {"total_effect": 0.0}
    
    async def _compute_total_effect(
        self,
        treatment: str,
        outcome: str,
        data: np.ndarray,
        variable_names: List[str]
    ) -> float:
        """Compute total causal effect using intervention."""
        try:
            # Simulate intervention: set treatment to different values
            treatment_idx = variable_names.index(treatment)
            outcome_idx = variable_names.index(outcome)
            
            # Get current treatment values
            current_treatment = data[:, treatment_idx]
            treatment_range = np.ptp(current_treatment)
            
            if treatment_range == 0:
                return 0.0
            
            # Simulate intervention: increase treatment by one standard deviation
            intervention_value = np.mean(current_treatment) + np.std(current_treatment)
            
            # Generate counterfactual outcomes
            counterfactual_outcomes = []
            
            for sample in data:
                # Create intervened sample
                intervened_sample = sample.copy()
                intervened_sample[treatment_idx] = intervention_value
                
                # Compute outcome under intervention using structural equations
                outcome_value = await self._simulate_outcome(intervened_sample, outcome, variable_names)
                counterfactual_outcomes.append(outcome_value)
            
            # Compare with observed outcomes
            observed_outcomes = data[:, outcome_idx]
            total_effect = np.mean(counterfactual_outcomes) - np.mean(observed_outcomes)
            
            return total_effect
            
        except Exception as e:
            logger.error(f"Total effect computation failed: {e}")
            return 0.0
    
    async def _simulate_outcome(
        self,
        sample: np.ndarray,
        outcome_var: str,
        variable_names: List[str]
    ) -> float:
        """Simulate outcome value using structural equations."""
        try:
            if outcome_var not in self.parameters:
                outcome_idx = variable_names.index(outcome_var)
                return sample[outcome_idx]
            
            params = self.parameters[outcome_var]
            parent_names = params.get("parents", [])
            
            if not parent_names:
                return params["intercept"]
            
            # Get parent values
            parent_values = []
            for parent in parent_names:
                parent_idx = variable_names.index(parent)
                parent_values.append(sample[parent_idx])
            
            # Compute outcome using structural equation
            outcome = params["intercept"] + np.dot(params["coefficients"], parent_values)
            
            return outcome
            
        except Exception as e:
            logger.warning(f"Outcome simulation failed: {e}")
            return 0.0
    
    async def generate_counterfactuals(
        self,
        query: CounterfactualQuery,
        observed_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate counterfactual outcomes using SCM."""
        try:
            if not self.causal_graph:
                raise ValueError("Causal graph not learned")
            
            variable_names = list(self.causal_graph.variables.keys())
            n_samples, n_vars = observed_data.shape
            
            counterfactual_data = observed_data.copy()
            
            # Apply interventions
            for treatment_var, treatment_value in query.treatment_variables.items():
                if treatment_var in variable_names:
                    var_idx = variable_names.index(treatment_var)
                    counterfactual_data[:, var_idx] = treatment_value
            
            # Re-compute affected variables using structural equations
            affected_vars = await self._find_affected_variables(
                list(query.treatment_variables.keys()),
                variable_names
            )
            
            for var_name in affected_vars:
                if var_name not in query.treatment_variables:  # Don't overwrite interventions
                    for i in range(n_samples):
                        new_value = await self._simulate_outcome(counterfactual_data[i], var_name, variable_names)
                        var_idx = variable_names.index(var_name)
                        counterfactual_data[i, var_idx] = new_value
            
            # Extract outcome variables
            results = {}
            for outcome_var in query.outcome_variables:
                if outcome_var in variable_names:
                    var_idx = variable_names.index(outcome_var)
                    results[outcome_var] = counterfactual_data[:, var_idx]
            
            return results
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {}
    
    async def _find_affected_variables(
        self,
        intervention_vars: List[str],
        variable_names: List[str]
    ) -> List[str]:
        """Find variables affected by intervention."""
        affected = set()
        queue = intervention_vars.copy()
        
        while queue:
            current_var = queue.pop(0)
            
            # Find children of current variable
            for edge in self.causal_graph.edges:
                if edge.source == current_var and edge.target not in affected:
                    affected.add(edge.target)
                    queue.append(edge.target)
        
        return list(affected)
    
    def _infer_variable_type(self, data: np.ndarray) -> str:
        """Infer variable type from data."""
        unique_values = len(np.unique(data))
        
        if unique_values == 2:
            return "binary"
        elif unique_values <= 10 and np.all(data == data.astype(int)):
            return "categorical"
        else:
            return "continuous"
    
    def _get_variable_domain(self, data: np.ndarray, var_type: str) -> Optional[Tuple[float, float]]:
        """Get variable domain."""
        if var_type == "continuous":
            return (float(np.min(data)), float(np.max(data)))
        return None


class CausalAnomalyDetector:
    """Causal anomaly detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.causal_engine: Optional[CausalInferenceEngine] = None
        self.baseline_effects: Dict[str, float] = {}
        self.effect_thresholds: Dict[str, float] = {}
        self.anomaly_history: List[CausalAnomalyResult] = []
    
    async def fit(
        self,
        data: np.ndarray,
        variable_names: List[str],
        causal_method: CausalMethod = CausalMethod.STRUCTURAL_CAUSAL_MODEL
    ) -> None:
        """Fit causal anomaly detector."""
        try:
            logger.info(f"Fitting causal anomaly detector with {causal_method}")
            
            # Initialize causal inference engine
            if causal_method == CausalMethod.STRUCTURAL_CAUSAL_MODEL:
                self.causal_engine = StructuralCausalModel(self.config)
            else:
                # Fallback to SCM for other methods
                self.causal_engine = StructuralCausalModel(self.config)
            
            # Learn causal structure
            causal_graph = await self.causal_engine.learn_causal_structure(data, variable_names)
            
            # Estimate baseline causal effects
            await self._estimate_baseline_effects(data, variable_names)
            
            # Set anomaly detection thresholds
            await self._set_anomaly_thresholds()
            
            logger.info("Causal anomaly detector fitted successfully")
            
        except Exception as e:
            logger.error(f"Causal anomaly detector fitting failed: {e}")
            raise
    
    async def detect_anomalies(
        self,
        data: np.ndarray,
        variable_names: List[str],
        sample_ids: Optional[List[str]] = None
    ) -> List[CausalAnomalyResult]:
        """Detect causal anomalies in data."""
        try:
            if not self.causal_engine:
                raise ValueError("Detector must be fitted before detection")
            
            anomalies = []
            n_samples = len(data)
            
            if sample_ids is None:
                sample_ids = [f"sample_{i}" for i in range(n_samples)]
            
            for i, sample in enumerate(data):
                sample_id = sample_ids[i]
                
                # Check for different types of causal anomalies
                anomaly_results = await self._check_sample_for_anomalies(
                    sample, sample_id, variable_names
                )
                
                anomalies.extend(anomaly_results)
            
            # Store in history
            self.anomaly_history.extend(anomalies)
            
            logger.info(f"Detected {len(anomalies)} causal anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Causal anomaly detection failed: {e}")
            return []
    
    async def _check_sample_for_anomalies(
        self,
        sample: np.ndarray,
        sample_id: str,
        variable_names: List[str]
    ) -> List[CausalAnomalyResult]:
        """Check individual sample for causal anomalies."""
        anomalies = []
        
        try:
            # 1. Check for structural break anomalies
            structural_anomaly = await self._check_structural_anomaly(sample, sample_id, variable_names)
            if structural_anomaly:
                anomalies.append(structural_anomaly)
            
            # 2. Check for effect anomalies
            effect_anomaly = await self._check_effect_anomaly(sample, sample_id, variable_names)
            if effect_anomaly:
                anomalies.append(effect_anomaly)
            
            # 3. Check for confounding anomalies
            confounding_anomaly = await self._check_confounding_anomaly(sample, sample_id, variable_names)
            if confounding_anomaly:
                anomalies.append(confounding_anomaly)
            
            # 4. Check for mediation anomalies
            mediation_anomaly = await self._check_mediation_anomaly(sample, sample_id, variable_names)
            if mediation_anomaly:
                anomalies.append(mediation_anomaly)
            
        except Exception as e:
            logger.warning(f"Anomaly check failed for sample {sample_id}: {e}")
        
        return anomalies
    
    async def _check_structural_anomaly(
        self,
        sample: np.ndarray,
        sample_id: str,
        variable_names: List[str]
    ) -> Optional[CausalAnomalyResult]:
        """Check for structural break anomalies."""
        try:
            if not isinstance(self.causal_engine, StructuralCausalModel):
                return None
            
            # Check if sample fits learned structural equations
            anomaly_scores = []
            affected_vars = []
            
            for var_name in variable_names:
                if var_name in self.causal_engine.parameters:
                    var_idx = variable_names.index(var_name)
                    observed_value = sample[var_idx]
                    
                    # Predict value using structural equation
                    predicted_value = await self.causal_engine._simulate_outcome(
                        sample, var_name, variable_names
                    )
                    
                    # Compute prediction error
                    if var_name in self.causal_engine.noise_distributions:
                        noise_std = self.causal_engine.noise_distributions[var_name]["std"]
                        if noise_std > 0:
                            error = abs(observed_value - predicted_value) / noise_std
                            anomaly_scores.append(error)
                            
                            if error > 3.0:  # 3-sigma rule
                                affected_vars.append(var_name)
            
            if anomaly_scores and max(anomaly_scores) > 3.0:
                return CausalAnomalyResult(
                    sample_id=sample_id,
                    anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                    anomaly_score=max(anomaly_scores),
                    causal_explanation=f"Sample violates learned structural equations for variables: {', '.join(affected_vars)}",
                    affected_variables=affected_vars,
                    counterfactual_analysis={},
                    confidence=min(1.0, max(anomaly_scores) / 5.0),
                    evidence={"prediction_errors": anomaly_scores}
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Structural anomaly check failed: {e}")
            return None
    
    async def _check_effect_anomaly(
        self,
        sample: np.ndarray,
        sample_id: str,
        variable_names: List[str]
    ) -> Optional[CausalAnomalyResult]:
        """Check for unusual causal effects."""
        try:
            # Check if causal effects in this sample deviate from baseline
            effect_anomalies = []
            
            for treatment_var in variable_names:
                for outcome_var in variable_names:
                    if treatment_var != outcome_var:
                        effect_key = f"{treatment_var}_to_{outcome_var}"
                        
                        if effect_key in self.baseline_effects:
                            # Estimate local causal effect
                            local_effect = await self._estimate_local_effect(
                                sample, treatment_var, outcome_var, variable_names
                            )
                            
                            baseline_effect = self.baseline_effects[effect_key]
                            threshold = self.effect_thresholds.get(effect_key, 1.0)
                            
                            if abs(local_effect - baseline_effect) > threshold:
                                effect_anomalies.append({
                                    "treatment": treatment_var,
                                    "outcome": outcome_var,
                                    "local_effect": local_effect,
                                    "baseline_effect": baseline_effect,
                                    "deviation": abs(local_effect - baseline_effect)
                                })
            
            if effect_anomalies:
                max_deviation = max(a["deviation"] for a in effect_anomalies)
                affected_vars = list(set([a["treatment"] for a in effect_anomalies] + 
                                       [a["outcome"] for a in effect_anomalies]))
                
                return CausalAnomalyResult(
                    sample_id=sample_id,
                    anomaly_type=AnomalyType.EFFECT_ANOMALY,
                    anomaly_score=max_deviation,
                    causal_explanation=f"Unusual causal effects detected: {len(effect_anomalies)} effect anomalies",
                    affected_variables=affected_vars,
                    counterfactual_analysis={},
                    confidence=min(1.0, max_deviation / 2.0),
                    evidence={"effect_anomalies": effect_anomalies}
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Effect anomaly check failed: {e}")
            return None
    
    async def _check_confounding_anomaly(
        self,
        sample: np.ndarray,
        sample_id: str,
        variable_names: List[str]
    ) -> Optional[CausalAnomalyResult]:
        """Check for confounding anomalies."""
        try:
            # Check for unexpected confounding patterns
            # This is a simplified implementation
            
            if not self.causal_engine or not self.causal_engine.causal_graph:
                return None
            
            confounders = self.causal_engine.causal_graph.confounders
            
            if not confounders:
                return None
            
            # Check if confounders have unusual relationships
            confounder_anomalies = []
            
            for confounder in confounders:
                if confounder in variable_names:
                    conf_idx = variable_names.index(confounder)
                    conf_value = sample[conf_idx]
                    
                    # Check if confounder value is extreme
                    conf_mean = np.mean([self.causal_engine.parameters.get(confounder, {}).get("intercept", 0)])
                    conf_std = self.causal_engine.noise_distributions.get(confounder, {}).get("std", 1.0)
                    
                    if conf_std > 0:
                        z_score = abs(conf_value - conf_mean) / conf_std
                        if z_score > 2.5:
                            confounder_anomalies.append({
                                "confounder": confounder,
                                "value": conf_value,
                                "z_score": z_score
                            })
            
            if confounder_anomalies:
                max_z_score = max(a["z_score"] for a in confounder_anomalies)
                
                return CausalAnomalyResult(
                    sample_id=sample_id,
                    anomaly_type=AnomalyType.CONFOUNDING_ANOMALY,
                    anomaly_score=max_z_score,
                    causal_explanation=f"Unusual confounding pattern detected in {len(confounder_anomalies)} confounders",
                    affected_variables=[a["confounder"] for a in confounder_anomalies],
                    counterfactual_analysis={},
                    confidence=min(1.0, max_z_score / 3.0),
                    evidence={"confounder_anomalies": confounder_anomalies}
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Confounding anomaly check failed: {e}")
            return None
    
    async def _check_mediation_anomaly(
        self,
        sample: np.ndarray,
        sample_id: str,
        variable_names: List[str]
    ) -> Optional[CausalAnomalyResult]:
        """Check for mediation anomalies."""
        try:
            # Check for unusual mediation effects
            # This is a simplified implementation
            
            if not self.causal_engine or not self.causal_engine.causal_graph:
                return None
            
            mediators = self.causal_engine.causal_graph.mediators
            
            if not mediators:
                return None
            
            mediation_anomalies = []
            
            for mediator in mediators:
                if mediator in variable_names:
                    # Check if mediation pathway is disrupted
                    med_idx = variable_names.index(mediator)
                    med_value = sample[med_idx]
                    
                    # Simplified check: compare to expected mediation level
                    expected_med = await self.causal_engine._simulate_outcome(
                        sample, mediator, variable_names
                    )
                    
                    if abs(med_value - expected_med) > 2.0:  # Threshold
                        mediation_anomalies.append({
                            "mediator": mediator,
                            "observed": med_value,
                            "expected": expected_med,
                            "deviation": abs(med_value - expected_med)
                        })
            
            if mediation_anomalies:
                max_deviation = max(a["deviation"] for a in mediation_anomalies)
                
                return CausalAnomalyResult(
                    sample_id=sample_id,
                    anomaly_type=AnomalyType.MEDIATION_ANOMALY,
                    anomaly_score=max_deviation,
                    causal_explanation=f"Unusual mediation pattern detected in {len(mediation_anomalies)} mediators",
                    affected_variables=[a["mediator"] for a in mediation_anomalies],
                    counterfactual_analysis={},
                    confidence=min(1.0, max_deviation / 3.0),
                    evidence={"mediation_anomalies": mediation_anomalies}
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Mediation anomaly check failed: {e}")
            return None
    
    async def _estimate_baseline_effects(self, data: np.ndarray, variable_names: List[str]) -> None:
        """Estimate baseline causal effects."""
        try:
            for i, treatment in enumerate(variable_names):
                for j, outcome in enumerate(variable_names):
                    if i != j:
                        effect_key = f"{treatment}_to_{outcome}"
                        
                        effects = await self.causal_engine.estimate_causal_effects(
                            data, treatment, outcome
                        )
                        
                        self.baseline_effects[effect_key] = effects.get("total_effect", 0.0)
            
            logger.info(f"Estimated {len(self.baseline_effects)} baseline causal effects")
            
        except Exception as e:
            logger.error(f"Baseline effect estimation failed: {e}")
    
    async def _set_anomaly_thresholds(self) -> None:
        """Set thresholds for anomaly detection."""
        try:
            for effect_key, baseline_effect in self.baseline_effects.items():
                # Set threshold as fraction of baseline effect
                threshold = max(abs(baseline_effect) * 0.5, 0.1)  # At least 0.1
                self.effect_thresholds[effect_key] = threshold
            
            logger.info("Set anomaly detection thresholds")
            
        except Exception as e:
            logger.error(f"Threshold setting failed: {e}")
    
    async def _estimate_local_effect(
        self,
        sample: np.ndarray,
        treatment: str,
        outcome: str,
        variable_names: List[str]
    ) -> float:
        """Estimate local causal effect for a sample."""
        try:
            # Simplified local effect estimation
            # In practice, this would use more sophisticated methods
            
            treatment_idx = variable_names.index(treatment)
            outcome_idx = variable_names.index(outcome)
            
            # Use structural equation if available
            if isinstance(self.causal_engine, StructuralCausalModel):
                if outcome in self.causal_engine.parameters:
                    params = self.causal_engine.parameters[outcome]
                    if treatment in params.get("parents", []):
                        parent_idx = params["parents"].index(treatment)
                        return params["coefficients"][parent_idx]
            
            # Fallback: use correlation as proxy
            return 0.0
            
        except Exception as e:
            logger.warning(f"Local effect estimation failed: {e}")
            return 0.0
    
    async def generate_counterfactual_explanation(
        self,
        anomaly: CausalAnomalyResult,
        original_sample: np.ndarray,
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation for anomaly."""
        try:
            if not self.causal_engine:
                return {}
            
            explanations = {}
            
            # Generate counterfactuals for affected variables
            for var_name in anomaly.affected_variables:
                if var_name in variable_names:
                    var_idx = variable_names.index(var_name)
                    current_value = original_sample[var_idx]
                    
                    # Create counterfactual query
                    query = CounterfactualQuery(
                        query_id=f"explain_{anomaly.sample_id}_{var_name}",
                        treatment_variables={var_name: current_value * 0.8},  # 20% reduction
                        outcome_variables=[v for v in variable_names if v != var_name]
                    )
                    
                    # Generate counterfactuals
                    counterfactual_data = original_sample.reshape(1, -1)
                    counterfactual_outcomes = await self.causal_engine.generate_counterfactuals(
                        query, counterfactual_data
                    )
                    
                    explanations[var_name] = {
                        "original_value": current_value,
                        "counterfactual_value": current_value * 0.8,
                        "predicted_outcomes": {
                            outcome: outcomes[0] if len(outcomes) > 0 else 0.0
                            for outcome, outcomes in counterfactual_outcomes.items()
                        }
                    }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Counterfactual explanation generation failed: {e}")
            return {}
    
    async def get_causal_summary(self) -> Dict[str, Any]:
        """Get summary of causal model and detected anomalies."""
        try:
            summary = {
                "causal_model": {
                    "method": self.causal_engine.method.value if self.causal_engine else "none",
                    "variables": len(self.causal_engine.causal_graph.variables) if self.causal_engine and self.causal_engine.causal_graph else 0,
                    "edges": len(self.causal_engine.causal_graph.edges) if self.causal_engine and self.causal_engine.causal_graph else 0
                },
                "anomaly_detection": {
                    "total_anomalies": len(self.anomaly_history),
                    "anomaly_types": {},
                    "most_affected_variables": {}
                }
            }
            
            # Count anomalies by type
            for anomaly in self.anomaly_history:
                anomaly_type = anomaly.anomaly_type.value
                summary["anomaly_detection"]["anomaly_types"][anomaly_type] = \
                    summary["anomaly_detection"]["anomaly_types"].get(anomaly_type, 0) + 1
            
            # Count most affected variables
            for anomaly in self.anomaly_history:
                for var in anomaly.affected_variables:
                    summary["anomaly_detection"]["most_affected_variables"][var] = \
                        summary["anomaly_detection"]["most_affected_variables"].get(var, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Causal summary generation failed: {e}")
            return {}