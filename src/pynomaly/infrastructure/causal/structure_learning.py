"""Causal structure learning algorithms for discovering causal relationships."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats

from pynomaly.domain.models.causal import (
    CausalEdge,
    CausalGraph,
    CausalGraphType,
    CausalMethod,
    CausalRelationType,
)


class StructureLearner:
    """Base class for causal structure learning algorithms."""
    
    def __init__(self, method: CausalMethod, **kwargs):
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.params = kwargs
    
    async def learn_structure(
        self, 
        data: np.ndarray, 
        variable_names: List[str],
        **kwargs
    ) -> CausalGraph:
        """Learn causal structure from data."""
        raise NotImplementedError("Subclasses must implement learn_structure")
    
    def _validate_data(self, data: np.ndarray, variable_names: List[str]) -> None:
        """Validate input data."""
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")
        if data.shape[1] != len(variable_names):
            raise ValueError("Number of variables must match data dimensions")
        if data.shape[0] < 10:
            raise ValueError("Need at least 10 samples for structure learning")


class PCAlgorithmLearner(StructureLearner):
    """PC Algorithm for causal structure learning."""
    
    def __init__(self, alpha: float = 0.05, **kwargs):
        super().__init__(CausalMethod.PC_ALGORITHM, **kwargs)
        self.alpha = alpha
    
    async def learn_structure(
        self, 
        data: np.ndarray, 
        variable_names: List[str],
        **kwargs
    ) -> CausalGraph:
        """Learn causal structure using PC algorithm."""
        self._validate_data(data, variable_names)
        
        self.logger.info(f"Learning causal structure with PC algorithm (alpha={self.alpha})")
        
        n_vars = len(variable_names)
        
        # Step 1: Start with complete undirected graph
        adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Step 2: Remove edges using conditional independence tests
        await self._skeleton_phase(data, adjacency, variable_names)
        
        # Step 3: Orient edges
        oriented_edges = await self._orientation_phase(data, adjacency, variable_names)
        
        # Create causal graph
        graph = CausalGraph(
            graph_id=uuid4(),
            graph_type=CausalGraphType.CPDAG,
            variables=variable_names,
            edges=oriented_edges,
            metadata={
                "method": self.method.value,
                "alpha": self.alpha,
                "n_samples": data.shape[0],
            }
        )
        
        self.logger.info(f"PC algorithm completed: {len(oriented_edges)} edges found")
        
        return graph
    
    async def _skeleton_phase(
        self, 
        data: np.ndarray, 
        adjacency: np.ndarray,
        variable_names: List[str]
    ) -> None:
        """Skeleton phase: remove edges using conditional independence."""
        n_vars = len(variable_names)
        
        # Test conditional independence for increasing conditioning set sizes
        for cond_size in range(n_vars - 2):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adjacency[i, j] == 0:
                        continue
                    
                    # Find neighbors for conditioning
                    neighbors_i = [k for k in range(n_vars) if adjacency[i, k] == 1 and k != j]
                    neighbors_j = [k for k in range(n_vars) if adjacency[j, k] == 1 and k != i]
                    
                    potential_cond = list(set(neighbors_i + neighbors_j))
                    
                    if len(potential_cond) >= cond_size:
                        # Test conditional independence
                        for cond_set in self._combinations(potential_cond, cond_size):
                            if await self._conditional_independence_test(data, i, j, cond_set):
                                adjacency[i, j] = adjacency[j, i] = 0
                                break
    
    async def _orientation_phase(
        self, 
        data: np.ndarray, 
        adjacency: np.ndarray,
        variable_names: List[str]
    ) -> List[CausalEdge]:
        """Orient edges to create CPDAG."""
        n_vars = len(variable_names)
        edges = []
        
        # Simple orientation rules (simplified PC algorithm)
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] == 1:
                    # Calculate causal strength using correlation
                    correlation = np.corrcoef(data[:, i], data[:, j])[0, 1]
                    strength = abs(correlation)
                    confidence = 1 - self.alpha
                    
                    # Determine edge type based on simple heuristics
                    edge_type = CausalRelationType.DIRECT_CAUSE
                    
                    edge = CausalEdge(
                        source=variable_names[i],
                        target=variable_names[j],
                        edge_type=edge_type,
                        strength=strength,
                        confidence=confidence,
                    )
                    edges.append(edge)
        
        return edges
    
    async def _conditional_independence_test(
        self, 
        data: np.ndarray, 
        x: int, 
        y: int, 
        cond_set: List[int]
    ) -> bool:
        """Test conditional independence X ⊥ Y | Z."""
        if not cond_set:
            # Marginal independence test
            _, p_value = stats.pearsonr(data[:, x], data[:, y])
            return p_value > self.alpha
        
        # Partial correlation test (simplified)
        # In practice, would use proper partial correlation or mutual information
        
        # Remove linear effect of conditioning variables
        x_residual = data[:, x].copy()
        y_residual = data[:, y].copy()
        
        for z in cond_set:
            z_data = data[:, z]
            # Simple linear regression residuals
            x_coef = np.cov(x_residual, z_data)[0, 1] / np.var(z_data)
            y_coef = np.cov(y_residual, z_data)[0, 1] / np.var(z_data)
            
            x_residual = x_residual - x_coef * z_data
            y_residual = y_residual - y_coef * z_data
        
        # Test independence of residuals
        _, p_value = stats.pearsonr(x_residual, y_residual)
        return p_value > self.alpha
    
    def _combinations(self, items: List[int], k: int) -> List[List[int]]:
        """Generate combinations of k items."""
        if k == 0:
            return [[]]
        if k > len(items):
            return []
        
        combinations = []
        for i in range(len(items) - k + 1):
            for sub_comb in self._combinations(items[i+1:], k-1):
                combinations.append([items[i]] + sub_comb)
        
        return combinations


class GrangerCausalityLearner(StructureLearner):
    """Granger causality for time series causal discovery."""
    
    def __init__(self, max_lag: int = 5, alpha: float = 0.05, **kwargs):
        super().__init__(CausalMethod.GRANGER_CAUSALITY, **kwargs)
        self.max_lag = max_lag
        self.alpha = alpha
    
    async def learn_structure(
        self, 
        data: np.ndarray, 
        variable_names: List[str],
        **kwargs
    ) -> CausalGraph:
        """Learn causal structure using Granger causality."""
        self._validate_data(data, variable_names)
        
        self.logger.info(f"Learning causal structure with Granger causality (max_lag={self.max_lag})")
        
        n_vars = len(variable_names)
        edges = []
        
        # Test Granger causality for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    causality_result = await self._test_granger_causality(
                        data[:, i], data[:, j], variable_names[i], variable_names[j]
                    )
                    
                    if causality_result:
                        edges.append(causality_result)
        
        graph = CausalGraph(
            graph_id=uuid4(),
            graph_type=CausalGraphType.DAG,
            variables=variable_names,
            edges=edges,
            metadata={
                "method": self.method.value,
                "max_lag": self.max_lag,
                "alpha": self.alpha,
                "n_samples": data.shape[0],
            }
        )
        
        self.logger.info(f"Granger causality completed: {len(edges)} causal relationships found")
        
        return graph
    
    async def _test_granger_causality(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        x_name: str,
        y_name: str
    ) -> Optional[CausalEdge]:
        """Test if X Granger-causes Y."""
        
        # Prepare lagged data
        n_samples = len(x)
        if n_samples <= self.max_lag + 10:
            return None
        
        # Create lagged variables
        lagged_y = []
        lagged_x = []
        target_y = []
        
        for t in range(self.max_lag, n_samples):
            # Target variable (Y at time t)
            target_y.append(y[t])
            
            # Lagged Y values
            y_lags = [y[t-lag] for lag in range(1, self.max_lag + 1)]
            lagged_y.append(y_lags)
            
            # Lagged X values
            x_lags = [x[t-lag] for lag in range(1, self.max_lag + 1)]
            lagged_x.append(x_lags)
        
        target_y = np.array(target_y)
        lagged_y = np.array(lagged_y)
        lagged_x = np.array(lagged_x)
        
        # Fit restricted model: Y(t) ~ Y(t-1), ..., Y(t-p)
        restricted_sse = self._fit_linear_model(lagged_y, target_y)
        
        # Fit unrestricted model: Y(t) ~ Y(t-1), ..., Y(t-p), X(t-1), ..., X(t-p)
        full_predictors = np.hstack([lagged_y, lagged_x])
        unrestricted_sse = self._fit_linear_model(full_predictors, target_y)
        
        # F-test for Granger causality
        n_obs = len(target_y)
        n_restr = lagged_y.shape[1]
        n_unrestr = full_predictors.shape[1]
        
        if unrestricted_sse >= restricted_sse or unrestricted_sse == 0:
            return None
        
        f_stat = ((restricted_sse - unrestricted_sse) / (n_unrestr - n_restr)) / \
                 (unrestricted_sse / (n_obs - n_unrestr - 1))
        
        # Calculate p-value using F-distribution
        p_value = 1 - stats.f.cdf(f_stat, n_unrestr - n_restr, n_obs - n_unrestr - 1)
        
        if p_value < self.alpha:
            # Significant Granger causality
            strength = min(1.0, f_stat / 10.0)  # Normalize F-statistic
            confidence = 1 - p_value
            
            # Determine optimal lag
            best_lag = self._find_optimal_lag(lagged_x, target_y, lagged_y)
            
            return CausalEdge(
                source=x_name,
                target=y_name,
                edge_type=CausalRelationType.DIRECT_CAUSE,
                strength=strength,
                confidence=confidence,
                lag=best_lag,
                mechanism=f"Granger causality (F={f_stat:.3f}, p={p_value:.3f})",
            )
        
        return None
    
    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit linear model and return sum of squared errors."""
        if X.size == 0:
            return np.sum((y - np.mean(y)) ** 2)
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
            # Solve normal equations
            coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ coeffs
            sse = np.sum((y - y_pred) ** 2)
            return sse
        except np.linalg.LinAlgError:
            # Singular matrix, return large SSE
            return np.sum((y - np.mean(y)) ** 2) * 10
    
    def _find_optimal_lag(
        self, 
        lagged_x: np.ndarray, 
        target_y: np.ndarray,
        lagged_y: np.ndarray
    ) -> int:
        """Find optimal lag using AIC/BIC."""
        best_lag = 1
        best_aic = float('inf')
        
        for lag in range(1, min(self.max_lag + 1, lagged_x.shape[1] + 1)):
            # Use first 'lag' columns
            x_subset = lagged_x[:, :lag]
            predictors = np.hstack([lagged_y, x_subset])
            
            sse = self._fit_linear_model(predictors, target_y)
            n_obs = len(target_y)
            n_params = predictors.shape[1] + 1  # +1 for intercept
            
            # Calculate AIC
            aic = n_obs * np.log(sse / n_obs) + 2 * n_params
            
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        
        return best_lag


class TransferEntropyLearner(StructureLearner):
    """Transfer entropy for nonlinear causal discovery."""
    
    def __init__(self, k: int = 1, alpha: float = 0.05, num_bins: int = 10, **kwargs):
        super().__init__(CausalMethod.TRANSFER_ENTROPY, **kwargs)
        self.k = k  # History length
        self.alpha = alpha
        self.num_bins = num_bins
    
    async def learn_structure(
        self, 
        data: np.ndarray, 
        variable_names: List[str],
        **kwargs
    ) -> CausalGraph:
        """Learn causal structure using transfer entropy."""
        self._validate_data(data, variable_names)
        
        self.logger.info(f"Learning causal structure with Transfer Entropy (k={self.k})")
        
        n_vars = len(variable_names)
        edges = []
        
        # Test transfer entropy for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    te_result = await self._compute_transfer_entropy(
                        data[:, i], data[:, j], variable_names[i], variable_names[j]
                    )
                    
                    if te_result:
                        edges.append(te_result)
        
        graph = CausalGraph(
            graph_id=uuid4(),
            graph_type=CausalGraphType.DAG,
            variables=variable_names,
            edges=edges,
            metadata={
                "method": self.method.value,
                "k": self.k,
                "num_bins": self.num_bins,
                "alpha": self.alpha,
                "n_samples": data.shape[0],
            }
        )
        
        self.logger.info(f"Transfer entropy completed: {len(edges)} causal relationships found")
        
        return graph
    
    async def _compute_transfer_entropy(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        x_name: str,
        y_name: str
    ) -> Optional[CausalEdge]:
        """Compute transfer entropy from X to Y."""
        
        # Discretize data
        x_discrete = self._discretize(x, self.num_bins)
        y_discrete = self._discretize(y, self.num_bins)
        
        n_samples = len(x)
        if n_samples <= self.k + 10:
            return None
        
        # Prepare symbolic sequences
        y_future = y_discrete[self.k:]
        y_past = []
        x_past = []
        
        for t in range(self.k, n_samples):
            y_hist = tuple(y_discrete[t-self.k:t])
            x_hist = tuple(x_discrete[t-self.k:t])
            y_past.append(y_hist)
            x_past.append(x_hist)
        
        # Compute transfer entropy: TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        te_value = self._calculate_transfer_entropy(y_future, y_past, x_past)
        
        # Statistical significance test (permutation test)
        significance = await self._transfer_entropy_significance_test(x, y, te_value)
        
        if significance < self.alpha:
            # Significant transfer entropy
            strength = min(1.0, te_value * 5)  # Normalize TE value
            confidence = 1 - significance
            
            return CausalEdge(
                source=x_name,
                target=y_name,
                edge_type=CausalRelationType.DIRECT_CAUSE,
                strength=strength,
                confidence=confidence,
                lag=1,  # Default lag for TE
                mechanism=f"Transfer entropy (TE={te_value:.3f}, p={significance:.3f})",
            )
        
        return None
    
    def _discretize(self, data: np.ndarray, num_bins: int) -> np.ndarray:
        """Discretize continuous data into bins."""
        # Equal-frequency binning
        data_sorted = np.sort(data)
        bin_edges = []
        
        for i in range(num_bins + 1):
            idx = int(i * len(data_sorted) / num_bins)
            if idx >= len(data_sorted):
                idx = len(data_sorted) - 1
            bin_edges.append(data_sorted[idx])
        
        bin_edges = np.unique(bin_edges)
        discretized = np.digitize(data, bin_edges) - 1
        
        # Ensure values are in [0, num_bins-1]
        discretized = np.clip(discretized, 0, num_bins - 1)
        
        return discretized
    
    def _calculate_transfer_entropy(
        self, 
        y_future: np.ndarray, 
        y_past: List[Tuple], 
        x_past: List[Tuple]
    ) -> float:
        """Calculate transfer entropy value."""
        
        # Count occurrences
        joint_counts = {}  # (y_future, y_past, x_past)
        y_y_past_counts = {}  # (y_future, y_past)
        y_past_counts = {}  # (y_past,)
        y_past_x_past_counts = {}  # (y_past, x_past)
        
        for i in range(len(y_future)):
            yf = y_future[i]
            yp = y_past[i]
            xp = x_past[i]
            
            # Joint counts
            joint_key = (yf, yp, xp)
            joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
            
            # Marginal counts
            y_y_past_key = (yf, yp)
            y_y_past_counts[y_y_past_key] = y_y_past_counts.get(y_y_past_key, 0) + 1
            
            y_past_counts[yp] = y_past_counts.get(yp, 0) + 1
            
            y_past_x_past_key = (yp, xp)
            y_past_x_past_counts[y_past_x_past_key] = y_past_x_past_counts.get(y_past_x_past_key, 0) + 1
        
        n_total = len(y_future)
        te_value = 0.0
        
        # Calculate transfer entropy
        for joint_key, joint_count in joint_counts.items():
            yf, yp, xp = joint_key
            
            p_joint = joint_count / n_total
            p_y_given_y_past = y_y_past_counts.get((yf, yp), 0) / y_past_counts.get(yp, 1)
            p_y_given_both = joint_count / y_past_x_past_counts.get((yp, xp), 1)
            
            if p_y_given_both > 0 and p_y_given_y_past > 0:
                te_value += p_joint * np.log2(p_y_given_both / p_y_given_y_past)
        
        return max(0, te_value)  # TE should be non-negative
    
    async def _transfer_entropy_significance_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        observed_te: float,
        num_permutations: int = 100
    ) -> float:
        """Test significance using permutation test."""
        
        null_tes = []
        
        for _ in range(num_permutations):
            # Permute X to break causal relationship
            x_permuted = np.random.permutation(x)
            
            # Compute TE with permuted data
            x_discrete = self._discretize(x_permuted, self.num_bins)
            y_discrete = self._discretize(y, self.num_bins)
            
            n_samples = len(x)
            if n_samples <= self.k + 1:
                continue
            
            y_future = y_discrete[self.k:]
            y_past = []
            x_past = []
            
            for t in range(self.k, n_samples):
                y_hist = tuple(y_discrete[t-self.k:t])
                x_hist = tuple(x_discrete[t-self.k:t])
                y_past.append(y_hist)
                x_past.append(x_hist)
            
            null_te = self._calculate_transfer_entropy(y_future, y_past, x_past)
            null_tes.append(null_te)
        
        if not null_tes:
            return 1.0
        
        # Calculate p-value
        p_value = sum(1 for null_te in null_tes if null_te >= observed_te) / len(null_tes)
        
        return p_value


class StructureLearningService:
    """Service for causal structure learning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learners = {
            CausalMethod.PC_ALGORITHM: PCAlgorithmLearner,
            CausalMethod.GRANGER_CAUSALITY: GrangerCausalityLearner,
            CausalMethod.TRANSFER_ENTROPY: TransferEntropyLearner,
        }
    
    async def learn_causal_structure(
        self,
        data: np.ndarray,
        variable_names: List[str],
        method: CausalMethod,
        **method_params
    ) -> CausalGraph:
        """Learn causal structure using specified method."""
        
        if method not in self.learners:
            raise ValueError(f"Method {method} not supported")
        
        self.logger.info(f"Learning causal structure with {method.value}")
        
        # Create learner instance
        learner_class = self.learners[method]
        learner = learner_class(**method_params)
        
        # Learn structure
        start_time = asyncio.get_event_loop().time()
        graph = await learner.learn_structure(data, variable_names)
        learning_time = asyncio.get_event_loop().time() - start_time
        
        # Add timing information to metadata
        graph.metadata["learning_time_seconds"] = learning_time
        
        self.logger.info(
            f"Structure learning completed in {learning_time:.2f}s: "
            f"{len(graph.edges)} edges discovered"
        )
        
        return graph
    
    async def compare_methods(
        self,
        data: np.ndarray,
        variable_names: List[str],
        methods: List[CausalMethod],
        **method_params
    ) -> Dict[CausalMethod, CausalGraph]:
        """Compare multiple structure learning methods."""
        
        self.logger.info(f"Comparing {len(methods)} structure learning methods")
        
        results = {}
        
        # Run methods concurrently
        tasks = []
        for method in methods:
            task = self.learn_causal_structure(data, variable_names, method, **method_params)
            tasks.append((method, task))
        
        # Wait for all methods to complete
        for method, task in tasks:
            try:
                graph = await task
                results[method] = graph
            except Exception as e:
                self.logger.error(f"Method {method.value} failed: {e}")
                continue
        
        self.logger.info(f"Method comparison completed: {len(results)} methods succeeded")
        
        return results
    
    def get_supported_methods(self) -> List[CausalMethod]:
        """Get list of supported causal learning methods."""
        return list(self.learners.keys())