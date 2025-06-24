#!/usr/bin/env python3
"""
Ensemble Selection Template

This template provides a comprehensive framework for building and selecting
optimal ensemble combinations of anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Ensemble methods
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

# Statistical imports
from scipy import stats
from scipy.stats import rankdata
import itertools
import time
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleSelector:
    """
    Comprehensive ensemble selection framework for anomaly detection.
    
    Features:
    - Multiple ensemble strategies (voting, averaging, stacking, dynamic)
    - Automated ensemble composition
    - Diversity measurement and optimization
    - Performance-based selection
    - Meta-learning for ensemble weights
    - Real-time ensemble adaptation
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the ensemble selector.
        
        Args:
            config: Configuration dictionary for ensemble parameters
            random_state: Random seed for reproducibility
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.random_state = random_state
        self.verbose = verbose
        
        # Ensemble components
        self.base_algorithms = {}
        self.ensemble_strategies = {}
        self.selected_ensembles = {}
        
        # Results storage
        self.diversity_analysis = {}
        self.ensemble_performance = {}
        self.optimization_history = {}
        
        # Set random seeds
        np.random.seed(random_state)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ensemble selection."""
        return {
            'base_algorithms': {
                'IsolationForest': {
                    'class': IsolationForest,
                    'params': {'contamination': 0.1, 'random_state': 42, 'n_estimators': 100}
                },
                'LocalOutlierFactor': {
                    'class': LocalOutlierFactor,
                    'params': {'contamination': 0.1, 'n_neighbors': 20, 'novelty': True}
                },
                'OneClassSVM': {
                    'class': OneClassSVM,
                    'params': {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'scale'}
                },
                'EllipticEnvelope': {
                    'class': EllipticEnvelope,
                    'params': {'contamination': 0.1, 'random_state': 42}
                }
            },
            'ensemble_strategies': {
                'simple_voting': {
                    'method': 'majority_vote',
                    'weights': None
                },
                'weighted_voting': {
                    'method': 'weighted_vote',
                    'weight_strategy': 'performance_based'
                },
                'score_averaging': {
                    'method': 'average_scores',
                    'normalization': 'z_score'
                },
                'rank_averaging': {
                    'method': 'average_ranks',
                    'rank_strategy': 'dense'
                },
                'stacking': {
                    'method': 'stacking',
                    'meta_learner': 'logistic_regression',
                    'cv_folds': 5
                },
                'dynamic_selection': {
                    'method': 'dynamic',
                    'selection_strategy': 'competence_based',
                    'k_nearest': 5
                }
            },
            'selection_criteria': {
                'diversity_measures': ['disagreement', 'correlation', 'kappa'],
                'performance_metrics': ['roc_auc', 'f1', 'precision', 'recall'],
                'efficiency_metrics': ['prediction_time', 'memory_usage'],
                'weights': {
                    'performance': 0.6,
                    'diversity': 0.3,
                    'efficiency': 0.1
                }
            },
            'optimization': {
                'search_strategy': 'exhaustive',  # 'exhaustive', 'greedy', 'genetic'
                'max_ensemble_size': 5,
                'min_ensemble_size': 2,
                'diversity_threshold': 0.1,
                'performance_threshold': 0.7
            },
            'validation': {
                'cross_validation': True,
                'cv_folds': 5,
                'test_size': 0.2,
                'bootstrap_samples': 100
            }
        }
    
    def select_ensemble(self, 
                       X: np.ndarray, 
                       y: Optional[np.ndarray] = None,
                       dataset_name: str = "Unknown") -> Dict[str, Any]:
        """
        Select optimal ensemble combinations for the given dataset.
        
        Args:
            X: Feature matrix
            y: True labels (optional, for supervised evaluation)
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Comprehensive ensemble selection results
        """
        logger.info(f"Starting ensemble selection on dataset: {dataset_name}")
        
        # Step 1: Train base algorithms
        self._train_base_algorithms(X, y)
        
        # Step 2: Analyze diversity between algorithms
        diversity_results = self._analyze_algorithm_diversity(X, y)
        
        # Step 3: Generate ensemble candidates
        ensemble_candidates = self._generate_ensemble_candidates()
        
        # Step 4: Evaluate ensemble strategies
        strategy_results = self._evaluate_ensemble_strategies(
            ensemble_candidates, X, y
        )
        
        # Step 5: Optimize ensemble composition
        optimization_results = self._optimize_ensemble_composition(
            strategy_results, X, y
        )
        
        # Step 6: Select best ensembles
        selection_results = self._select_best_ensembles(
            optimization_results, diversity_results
        )
        
        # Step 7: Validate selected ensembles
        validation_results = self._validate_selected_ensembles(
            selection_results, X, y
        )
        
        # Compile final results
        final_results = {
            'dataset_name': dataset_name,
            'base_algorithms': list(self.base_algorithms.keys()),
            'diversity_analysis': diversity_results,
            'ensemble_candidates': len(ensemble_candidates),
            'strategy_evaluation': strategy_results,
            'optimization_results': optimization_results,
            'selected_ensembles': selection_results,
            'validation_results': validation_results,
            'recommendations': self._generate_ensemble_recommendations(
                selection_results, validation_results
            )
        }
        
        # Store results
        self.ensemble_performance[dataset_name] = final_results
        
        return final_results
    
    def _train_base_algorithms(self, X: np.ndarray, y: Optional[np.ndarray]):
        """Train all base algorithms and collect predictions/scores."""
        
        base_config = self.config['base_algorithms']
        
        for algo_name, algo_config in base_config.items():
            logger.info(f"Training base algorithm: {algo_name}")
            
            try:
                # Initialize and train algorithm
                algorithm_class = algo_config['class']
                params = algo_config['params'].copy()
                
                model = algorithm_class(**params)
                
                # Fit the model
                start_time = time.time()
                model.fit(X)
                fit_time = time.time() - start_time
                
                # Get predictions and scores
                predictions = model.predict(X)
                
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                else:
                    scores = None
                
                # Convert predictions to binary (1 for normal, -1 for anomaly -> 0 for normal, 1 for anomaly)
                binary_predictions = (predictions == -1).astype(int)
                
                # Calculate basic metrics if labels available
                metrics = {}
                if y is not None:
                    if scores is not None:
                        try:
                            metrics['roc_auc'] = roc_auc_score(y, -scores)
                        except:
                            metrics['roc_auc'] = 0.5
                    
                    metrics['precision'] = precision_score(y, binary_predictions, zero_division=0)
                    metrics['recall'] = recall_score(y, binary_predictions, zero_division=0)
                    metrics['f1'] = f1_score(y, binary_predictions, zero_division=0)
                
                self.base_algorithms[algo_name] = {
                    'model': model,
                    'predictions': binary_predictions,
                    'scores': scores,
                    'metrics': metrics,
                    'fit_time': fit_time,
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"Error training {algo_name}: {str(e)}")
                self.base_algorithms[algo_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
    
    def _analyze_algorithm_diversity(self, X: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze diversity between base algorithms."""
        
        successful_algorithms = {
            name: data for name, data in self.base_algorithms.items()
            if data.get('status') == 'success'
        }
        
        if len(successful_algorithms) < 2:
            return {'status': 'insufficient_algorithms'}
        
        diversity_measures = {}
        
        # Collect predictions for diversity analysis
        predictions_matrix = {}
        scores_matrix = {}
        
        for algo_name, algo_data in successful_algorithms.items():
            predictions_matrix[algo_name] = algo_data['predictions']
            if algo_data['scores'] is not None:
                scores_matrix[algo_name] = algo_data['scores']
        
        # Calculate pairwise diversity measures
        algo_names = list(predictions_matrix.keys())
        
        # Disagreement measure
        disagreement_matrix = np.zeros((len(algo_names), len(algo_names)))
        for i, algo1 in enumerate(algo_names):
            for j, algo2 in enumerate(algo_names):
                if i != j:
                    pred1 = predictions_matrix[algo1]
                    pred2 = predictions_matrix[algo2]
                    disagreement = np.mean(pred1 != pred2)
                    disagreement_matrix[i, j] = disagreement
                else:
                    disagreement_matrix[i, j] = 0
        
        diversity_measures['disagreement'] = {
            'matrix': disagreement_matrix,
            'algorithm_names': algo_names,
            'average_disagreement': np.mean(disagreement_matrix[np.triu_indices_from(disagreement_matrix, k=1)])
        }
        
        # Correlation measure (for scores)
        if scores_matrix:
            correlation_matrix = np.corrcoef([scores_matrix[algo] for algo in algo_names if algo in scores_matrix])
            
            diversity_measures['correlation'] = {
                'matrix': correlation_matrix,
                'algorithm_names': list(scores_matrix.keys()),
                'average_correlation': np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            }
        
        # Kappa statistic
        kappa_matrix = np.zeros((len(algo_names), len(algo_names)))
        for i, algo1 in enumerate(algo_names):
            for j, algo2 in enumerate(algo_names):
                if i != j:
                    pred1 = predictions_matrix[algo1]
                    pred2 = predictions_matrix[algo2]
                    
                    # Calculate Cohen's kappa
                    p_o = np.mean(pred1 == pred2)  # Observed agreement
                    p_e = (np.mean(pred1) * np.mean(pred2) + 
                           (1 - np.mean(pred1)) * (1 - np.mean(pred2)))  # Expected agreement
                    
                    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
                    kappa_matrix[i, j] = kappa
                else:
                    kappa_matrix[i, j] = 1
        
        diversity_measures['kappa'] = {
            'matrix': kappa_matrix,
            'algorithm_names': algo_names,
            'average_kappa': np.mean(kappa_matrix[np.triu_indices_from(kappa_matrix, k=1)])
        }
        
        # Overall diversity score
        avg_disagreement = diversity_measures['disagreement']['average_disagreement']
        avg_correlation = diversity_measures.get('correlation', {}).get('average_correlation', 0)
        avg_kappa = diversity_measures['kappa']['average_kappa']
        
        # Higher disagreement and lower correlation = higher diversity
        # Lower kappa = higher diversity
        diversity_score = (avg_disagreement + (1 - abs(avg_correlation)) + (1 - avg_kappa)) / 3
        
        diversity_measures['overall_diversity_score'] = diversity_score
        
        self.diversity_analysis = diversity_measures
        return diversity_measures
    
    def _generate_ensemble_candidates(self) -> List[List[str]]:
        """Generate candidate ensemble combinations."""
        
        successful_algorithms = [
            name for name, data in self.base_algorithms.items()
            if data.get('status') == 'success'
        ]
        
        optimization_config = self.config['optimization']
        min_size = optimization_config['min_ensemble_size']
        max_size = min(optimization_config['max_ensemble_size'], len(successful_algorithms))
        
        ensemble_candidates = []
        
        # Generate all possible combinations
        for size in range(min_size, max_size + 1):
            combinations = list(itertools.combinations(successful_algorithms, size))
            ensemble_candidates.extend([list(combo) for combo in combinations])
        
        logger.info(f"Generated {len(ensemble_candidates)} ensemble candidates")
        return ensemble_candidates
    
    def _evaluate_ensemble_strategies(self, 
                                    ensemble_candidates: List[List[str]], 
                                    X: np.ndarray, 
                                    y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Evaluate different ensemble strategies on candidate combinations."""
        
        strategy_config = self.config['ensemble_strategies']
        strategy_results = {}
        
        for strategy_name, strategy_config in strategy_config.items():
            logger.info(f"Evaluating ensemble strategy: {strategy_name}")
            
            strategy_results[strategy_name] = {
                'ensemble_performances': {},
                'best_ensemble': None,
                'average_performance': 0,
                'strategy_config': strategy_config
            }
            
            ensemble_performances = {}
            
            for i, ensemble_combo in enumerate(ensemble_candidates):
                try:
                    # Apply ensemble strategy
                    ensemble_result = self._apply_ensemble_strategy(
                        ensemble_combo, strategy_config, X, y
                    )
                    
                    ensemble_performances[f"ensemble_{i}"] = {
                        'algorithms': ensemble_combo,
                        'performance': ensemble_result
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate ensemble {ensemble_combo} with strategy {strategy_name}: {str(e)}")
            
            # Find best ensemble for this strategy
            if ensemble_performances:
                if y is not None:
                    # Use ROC-AUC as primary metric
                    best_ensemble = max(
                        ensemble_performances.items(),
                        key=lambda x: x[1]['performance'].get('roc_auc', 0)
                    )
                else:
                    # Use ensemble diversity as metric
                    best_ensemble = max(
                        ensemble_performances.items(),
                        key=lambda x: len(x[1]['algorithms'])  # Prefer larger ensembles
                    )
                
                strategy_results[strategy_name]['best_ensemble'] = best_ensemble
                strategy_results[strategy_name]['ensemble_performances'] = ensemble_performances
                
                # Calculate average performance
                if y is not None:
                    performances = [
                        perf['performance'].get('roc_auc', 0) 
                        for perf in ensemble_performances.values()
                    ]
                    strategy_results[strategy_name]['average_performance'] = np.mean(performances)
        
        return strategy_results
    
    def _apply_ensemble_strategy(self, 
                               ensemble_algorithms: List[str],
                               strategy_config: Dict[str, Any],
                               X: np.ndarray,
                               y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply a specific ensemble strategy to a combination of algorithms."""
        
        method = strategy_config['method']
        
        # Collect predictions and scores from ensemble members
        predictions = []
        scores = []
        
        for algo_name in ensemble_algorithms:
            algo_data = self.base_algorithms[algo_name]
            predictions.append(algo_data['predictions'])
            if algo_data['scores'] is not None:
                scores.append(algo_data['scores'])
        
        predictions = np.array(predictions)
        scores = np.array(scores) if scores else None
        
        # Apply ensemble method
        if method == 'majority_vote':
            ensemble_predictions = self._majority_vote(predictions)
            ensemble_scores = np.mean(scores, axis=0) if scores is not None else None
        
        elif method == 'weighted_vote':
            weights = self._calculate_ensemble_weights(
                ensemble_algorithms, strategy_config.get('weight_strategy', 'equal')
            )
            ensemble_predictions = self._weighted_vote(predictions, weights)
            ensemble_scores = np.average(scores, axis=0, weights=weights) if scores is not None else None
        
        elif method == 'average_scores':
            if scores is not None:
                # Normalize scores if specified
                if strategy_config.get('normalization') == 'z_score':
                    scores = stats.zscore(scores, axis=1)
                
                ensemble_scores = np.mean(scores, axis=0)
                ensemble_predictions = (ensemble_scores < np.percentile(ensemble_scores, 90)).astype(int)
            else:
                ensemble_predictions = self._majority_vote(predictions)
                ensemble_scores = None
        
        elif method == 'average_ranks':
            if scores is not None:
                # Convert scores to ranks
                ranked_scores = np.array([rankdata(-score) for score in scores])  # Negative for proper ranking
                ensemble_ranks = np.mean(ranked_scores, axis=0)
                ensemble_predictions = (ensemble_ranks <= np.percentile(ensemble_ranks, 10)).astype(int)
                ensemble_scores = ensemble_ranks
            else:
                ensemble_predictions = self._majority_vote(predictions)
                ensemble_scores = None
        
        elif method == 'stacking':
            ensemble_predictions, ensemble_scores = self._stacking_ensemble(
                ensemble_algorithms, strategy_config, X, y
            )
        
        elif method == 'dynamic':
            ensemble_predictions, ensemble_scores = self._dynamic_selection_ensemble(
                ensemble_algorithms, strategy_config, X, y
            )
        
        else:
            # Default to majority voting
            ensemble_predictions = self._majority_vote(predictions)
            ensemble_scores = np.mean(scores, axis=0) if scores is not None else None
        
        # Calculate ensemble performance metrics
        metrics = {}
        if y is not None:
            if ensemble_scores is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y, -ensemble_scores)
                except:
                    metrics['roc_auc'] = 0.5
            
            metrics['precision'] = precision_score(y, ensemble_predictions, zero_division=0)
            metrics['recall'] = recall_score(y, ensemble_predictions, zero_division=0)
            metrics['f1'] = f1_score(y, ensemble_predictions, zero_division=0)
            
            # Additional metrics
            tn, fp, fn, tp = confusion_matrix(y, ensemble_predictions).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'predictions': ensemble_predictions,
            'scores': ensemble_scores,
            'metrics': metrics,
            'ensemble_size': len(ensemble_algorithms),
            'method': method
        }
    
    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        """Apply majority voting to predictions."""
        return (np.sum(predictions, axis=0) > len(predictions) / 2).astype(int)
    
    def _weighted_vote(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply weighted voting to predictions."""
        weighted_sum = np.sum(predictions * weights.reshape(-1, 1), axis=0)
        return (weighted_sum > np.sum(weights) / 2).astype(int)
    
    def _calculate_ensemble_weights(self, 
                                  ensemble_algorithms: List[str], 
                                  weight_strategy: str) -> np.ndarray:
        """Calculate weights for ensemble members."""
        
        if weight_strategy == 'equal':
            return np.ones(len(ensemble_algorithms)) / len(ensemble_algorithms)
        
        elif weight_strategy == 'performance_based':
            # Weight by individual algorithm performance
            weights = []
            for algo_name in ensemble_algorithms:
                metrics = self.base_algorithms[algo_name].get('metrics', {})
                performance = metrics.get('roc_auc', 0.5)  # Default to random performance
                weights.append(performance)
            
            weights = np.array(weights)
            return weights / np.sum(weights)  # Normalize
        
        elif weight_strategy == 'inverse_error':
            # Weight by inverse of error rate
            weights = []
            for algo_name in ensemble_algorithms:
                metrics = self.base_algorithms[algo_name].get('metrics', {})
                error_rate = 1 - metrics.get('accuracy', 0.5)
                weights.append(1 / (error_rate + 1e-8))  # Add small epsilon to avoid division by zero
            
            weights = np.array(weights)
            return weights / np.sum(weights)
        
        else:
            # Default to equal weights
            return np.ones(len(ensemble_algorithms)) / len(ensemble_algorithms)
    
    def _stacking_ensemble(self, 
                          ensemble_algorithms: List[str],
                          strategy_config: Dict[str, Any],
                          X: np.ndarray,
                          y: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Implement stacking ensemble method."""
        
        if y is None:
            # Fallback to majority voting if no labels
            predictions = np.array([self.base_algorithms[algo]['predictions'] for algo in ensemble_algorithms])
            return self._majority_vote(predictions), None
        
        # Create meta-features from base algorithm predictions
        meta_features = []
        for algo_name in ensemble_algorithms:
            algo_data = self.base_algorithms[algo_name]
            if algo_data['scores'] is not None:
                meta_features.append(algo_data['scores'])
            else:
                meta_features.append(algo_data['predictions'].astype(float))
        
        meta_features = np.column_stack(meta_features)
        
        # Train meta-learner
        meta_learner_type = strategy_config.get('meta_learner', 'logistic_regression')
        
        if meta_learner_type == 'logistic_regression':
            meta_learner = LogisticRegression(random_state=self.random_state)
        elif meta_learner_type == 'decision_tree':
            meta_learner = DecisionTreeClassifier(random_state=self.random_state)
        else:
            meta_learner = LogisticRegression(random_state=self.random_state)
        
        # Use cross-validation to generate meta-features
        cv_folds = strategy_config.get('cv_folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        meta_predictions = np.zeros(len(X))
        
        for train_idx, val_idx in cv.split(meta_features, y):
            X_train_meta, X_val_meta = meta_features[train_idx], meta_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            meta_learner.fit(X_train_meta, y_train)
            meta_predictions[val_idx] = meta_learner.predict(X_val_meta)
        
        # Get prediction probabilities for scores
        meta_learner.fit(meta_features, y)
        if hasattr(meta_learner, 'predict_proba'):
            meta_scores = meta_learner.predict_proba(meta_features)[:, 1]
        else:
            meta_scores = meta_learner.decision_function(meta_features)
        
        return meta_predictions.astype(int), meta_scores
    
    def _dynamic_selection_ensemble(self, 
                                  ensemble_algorithms: List[str],
                                  strategy_config: Dict[str, Any],
                                  X: np.ndarray,
                                  y: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Implement dynamic selection ensemble method."""
        
        # For now, implement a simple competence-based selection
        # In practice, this would use more sophisticated dynamic selection methods
        
        predictions = np.array([self.base_algorithms[algo]['predictions'] for algo in ensemble_algorithms])
        
        if y is None:
            return self._majority_vote(predictions), None
        
        k_nearest = strategy_config.get('k_nearest', 5)
        ensemble_predictions = np.zeros(len(X))
        
        # For each instance, select the most competent algorithm(s)
        for i in range(len(X)):
            # Find k nearest neighbors (simplified - using random selection here)
            # In practice, you would use actual distance metrics
            neighbor_indices = np.random.choice(len(X), min(k_nearest, len(X)), replace=False)
            
            # Calculate competence of each algorithm on neighbors
            competences = []
            for j, algo_name in enumerate(ensemble_algorithms):
                algo_predictions = predictions[j]
                # Competence = accuracy on neighbors
                competence = np.mean(algo_predictions[neighbor_indices] == y[neighbor_indices])
                competences.append(competence)
            
            # Select best performing algorithm for this instance
            best_algo_idx = np.argmax(competences)
            ensemble_predictions[i] = predictions[best_algo_idx, i]
        
        return ensemble_predictions.astype(int), None
    
    def _optimize_ensemble_composition(self, 
                                     strategy_results: Dict[str, Any],
                                     X: np.ndarray,
                                     y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Optimize ensemble composition using specified search strategy."""
        
        optimization_config = self.config['optimization']
        search_strategy = optimization_config['search_strategy']
        
        if search_strategy == 'exhaustive':
            return self._exhaustive_search_optimization(strategy_results)
        elif search_strategy == 'greedy':
            return self._greedy_search_optimization(strategy_results, X, y)
        elif search_strategy == 'genetic':
            return self._genetic_algorithm_optimization(strategy_results, X, y)
        else:
            return self._exhaustive_search_optimization(strategy_results)
    
    def _exhaustive_search_optimization(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform exhaustive search for optimal ensemble composition."""
        
        best_ensembles = {}
        
        for strategy_name, results in strategy_results.items():
            ensemble_performances = results.get('ensemble_performances', {})
            
            if ensemble_performances:
                # Find best ensemble for each strategy
                best_performance = 0
                best_ensemble_key = None
                
                for ensemble_key, ensemble_data in ensemble_performances.items():
                    performance = ensemble_data['performance'].get('roc_auc', 0)
                    if performance > best_performance:
                        best_performance = performance
                        best_ensemble_key = ensemble_key
                
                if best_ensemble_key:
                    best_ensembles[strategy_name] = {
                        'ensemble': ensemble_performances[best_ensemble_key],
                        'performance': best_performance
                    }
        
        return {
            'optimization_method': 'exhaustive_search',
            'best_ensembles_per_strategy': best_ensembles,
            'global_best': max(
                best_ensembles.items(),
                key=lambda x: x[1]['performance']
            ) if best_ensembles else None
        }
    
    def _greedy_search_optimization(self, 
                                  strategy_results: Dict[str, Any],
                                  X: np.ndarray,
                                  y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform greedy search for ensemble optimization."""
        
        # Start with best individual algorithm
        successful_algorithms = [
            name for name, data in self.base_algorithms.items()
            if data.get('status') == 'success'
        ]
        
        if not successful_algorithms or y is None:
            return {'optimization_method': 'greedy_search', 'status': 'insufficient_data'}
        
        # Find best individual algorithm
        best_individual = max(
            successful_algorithms,
            key=lambda x: self.base_algorithms[x].get('metrics', {}).get('roc_auc', 0)
        )
        
        current_ensemble = [best_individual]
        current_performance = self.base_algorithms[best_individual].get('metrics', {}).get('roc_auc', 0)
        
        optimization_config = self.config['optimization']
        max_size = optimization_config['max_ensemble_size']
        
        # Greedily add algorithms that improve performance
        remaining_algorithms = [algo for algo in successful_algorithms if algo != best_individual]
        
        while len(current_ensemble) < max_size and remaining_algorithms:
            best_addition = None
            best_improvement = 0
            
            for candidate in remaining_algorithms:
                # Test adding this candidate
                test_ensemble = current_ensemble + [candidate]
                
                # Evaluate ensemble with majority voting (simplified)
                test_performance = self._evaluate_greedy_ensemble(test_ensemble, X, y)
                
                improvement = test_performance - current_performance
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_addition = candidate
            
            # Add best candidate if it improves performance
            if best_addition and best_improvement > 0.001:  # Small threshold for improvement
                current_ensemble.append(best_addition)
                current_performance += best_improvement
                remaining_algorithms.remove(best_addition)
            else:
                break  # No improvement found
        
        return {
            'optimization_method': 'greedy_search',
            'optimal_ensemble': current_ensemble,
            'final_performance': current_performance,
            'optimization_steps': len(current_ensemble)
        }
    
    def _evaluate_greedy_ensemble(self, 
                                ensemble_algorithms: List[str], 
                                X: np.ndarray, 
                                y: np.ndarray) -> float:
        """Evaluate ensemble performance for greedy search."""
        
        predictions = []
        for algo_name in ensemble_algorithms:
            predictions.append(self.base_algorithms[algo_name]['predictions'])
        
        predictions = np.array(predictions)
        ensemble_predictions = self._majority_vote(predictions)
        
        try:
            return roc_auc_score(y, ensemble_predictions)
        except:
            return f1_score(y, ensemble_predictions, zero_division=0)
    
    def _genetic_algorithm_optimization(self, 
                                      strategy_results: Dict[str, Any],
                                      X: np.ndarray,
                                      y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform genetic algorithm optimization for ensemble composition."""
        
        # Simplified genetic algorithm implementation
        if y is None:
            return {'optimization_method': 'genetic_algorithm', 'status': 'no_labels'}
        
        successful_algorithms = [
            name for name, data in self.base_algorithms.items()
            if data.get('status') == 'success'
        ]
        
        if len(successful_algorithms) < 2:
            return {'optimization_method': 'genetic_algorithm', 'status': 'insufficient_algorithms'}
        
        # GA parameters
        population_size = 20
        generations = 10
        mutation_rate = 0.1
        
        # Initialize population (binary encoding: 1 = include algorithm, 0 = exclude)
        population = []
        for _ in range(population_size):
            # Random binary vector
            individual = np.random.choice([0, 1], size=len(successful_algorithms))
            # Ensure at least 2 algorithms are selected
            if np.sum(individual) < 2:
                individual[np.random.choice(len(individual), 2, replace=False)] = 1
            population.append(individual)
        
        best_fitness = 0
        best_individual = None
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                selected_algos = [successful_algorithms[i] for i, selected in enumerate(individual) if selected]
                if len(selected_algos) >= 2:
                    fitness = self._evaluate_greedy_ensemble(selected_algos, X, y)
                else:
                    fitness = 0
                fitness_scores.append(fitness)
            
            # Track best individual
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Selection, crossover, mutation (simplified)
            new_population = []
            
            # Keep best individuals (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(population_size // 4):
                new_population.append(population[sorted_indices[i]])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Selection (tournament)
                parent1 = population[np.random.choice(len(population))]
                parent2 = population[np.random.choice(len(population))]
                
                # Crossover
                crossover_point = np.random.randint(1, len(parent1))
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                
                # Mutation
                if np.random.random() < mutation_rate:
                    mutation_point = np.random.randint(len(child))
                    child[mutation_point] = 1 - child[mutation_point]
                
                # Ensure at least 2 algorithms
                if np.sum(child) < 2:
                    child[np.random.choice(len(child), 2, replace=False)] = 1
                
                new_population.append(child)
            
            population = new_population
        
        # Extract best ensemble
        if best_individual is not None:
            optimal_ensemble = [
                successful_algorithms[i] for i, selected in enumerate(best_individual) if selected
            ]
        else:
            optimal_ensemble = successful_algorithms[:2]  # Fallback
        
        return {
            'optimization_method': 'genetic_algorithm',
            'optimal_ensemble': optimal_ensemble,
            'final_performance': best_fitness,
            'generations': generations,
            'population_size': population_size
        }
    
    def _select_best_ensembles(self, 
                             optimization_results: Dict[str, Any],
                             diversity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select best ensembles based on multiple criteria."""
        
        selection_criteria = self.config['selection_criteria']
        weights = selection_criteria['weights']
        
        # Collect ensemble candidates from optimization results
        ensemble_candidates = {}
        
        # From exhaustive search
        if optimization_results.get('optimization_method') == 'exhaustive_search':
            best_ensembles = optimization_results.get('best_ensembles_per_strategy', {})
            for strategy_name, strategy_data in best_ensembles.items():
                ensemble_candidates[f"{strategy_name}_best"] = {
                    'algorithms': strategy_data['ensemble']['algorithms'],
                    'performance': strategy_data['performance'],
                    'strategy': strategy_name
                }
        
        # From greedy/genetic search
        elif optimization_results.get('optimal_ensemble'):
            ensemble_candidates['optimized'] = {
                'algorithms': optimization_results['optimal_ensemble'],
                'performance': optimization_results.get('final_performance', 0),
                'strategy': 'optimized'
            }
        
        # Score ensembles based on multiple criteria
        scored_ensembles = {}
        
        for ensemble_name, ensemble_data in ensemble_candidates.items():
            algorithms = ensemble_data['algorithms']
            performance = ensemble_data['performance']
            
            # Performance score (normalized)
            performance_score = performance
            
            # Diversity score
            diversity_score = self._calculate_ensemble_diversity_score(algorithms, diversity_results)
            
            # Efficiency score (simplified - based on ensemble size)
            max_size = self.config['optimization']['max_ensemble_size']
            efficiency_score = 1 - (len(algorithms) - 2) / (max_size - 2) if max_size > 2 else 1
            
            # Combined score
            combined_score = (
                weights['performance'] * performance_score +
                weights['diversity'] * diversity_score +
                weights['efficiency'] * efficiency_score
            )
            
            scored_ensembles[ensemble_name] = {
                'ensemble_data': ensemble_data,
                'scores': {
                    'performance': performance_score,
                    'diversity': diversity_score,
                    'efficiency': efficiency_score,
                    'combined': combined_score
                }
            }
        
        # Select top ensembles
        sorted_ensembles = sorted(
            scored_ensembles.items(),
            key=lambda x: x[1]['scores']['combined'],
            reverse=True
        )
        
        return {
            'scored_ensembles': scored_ensembles,
            'ranked_ensembles': sorted_ensembles,
            'selected_best': sorted_ensembles[0] if sorted_ensembles else None,
            'selection_criteria_weights': weights
        }
    
    def _calculate_ensemble_diversity_score(self, 
                                          algorithms: List[str], 
                                          diversity_results: Dict[str, Any]) -> float:
        """Calculate diversity score for a specific ensemble."""
        
        if len(algorithms) < 2:
            return 0
        
        # Get pairwise diversity measures
        disagreement_data = diversity_results.get('disagreement', {})
        algo_names = disagreement_data.get('algorithm_names', [])
        disagreement_matrix = disagreement_data.get('matrix', np.zeros((0, 0)))
        
        if len(algo_names) == 0 or disagreement_matrix.size == 0:
            return 0.5  # Default diversity score
        
        # Calculate average pairwise diversity for this ensemble
        ensemble_indices = []
        for algo in algorithms:
            if algo in algo_names:
                ensemble_indices.append(algo_names.index(algo))
        
        if len(ensemble_indices) < 2:
            return 0.5
        
        # Get pairwise disagreements for ensemble members
        diversity_values = []
        for i in range(len(ensemble_indices)):
            for j in range(i + 1, len(ensemble_indices)):
                idx1, idx2 = ensemble_indices[i], ensemble_indices[j]
                if idx1 < disagreement_matrix.shape[0] and idx2 < disagreement_matrix.shape[1]:
                    diversity_values.append(disagreement_matrix[idx1, idx2])
        
        return np.mean(diversity_values) if diversity_values else 0.5
    
    def _validate_selected_ensembles(self, 
                                   selection_results: Dict[str, Any],
                                   X: np.ndarray,
                                   y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate selected ensembles using cross-validation."""
        
        if y is None:
            return {'status': 'no_labels_for_validation'}
        
        validation_config = self.config['validation']
        
        if not validation_config['cross_validation']:
            return {'status': 'validation_disabled'}
        
        selected_best = selection_results.get('selected_best')
        if not selected_best:
            return {'status': 'no_ensemble_selected'}
        
        ensemble_algorithms = selected_best[1]['ensemble_data']['algorithms']
        
        # Perform cross-validation
        cv_folds = validation_config['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train ensemble members on training set
            ensemble_predictions = []
            
            for algo_name in ensemble_algorithms:
                algo_config = self.config['base_algorithms'][algo_name]
                model = algo_config['class'](**algo_config['params'])
                model.fit(X_train)
                
                test_predictions = model.predict(X_test)
                binary_predictions = (test_predictions == -1).astype(int)
                ensemble_predictions.append(binary_predictions)
            
            # Combine predictions (majority vote)
            ensemble_predictions = np.array(ensemble_predictions)
            final_predictions = self._majority_vote(ensemble_predictions)
            
            # Calculate metrics
            cv_scores['precision'].append(precision_score(y_test, final_predictions, zero_division=0))
            cv_scores['recall'].append(recall_score(y_test, final_predictions, zero_division=0))
            cv_scores['f1'].append(f1_score(y_test, final_predictions, zero_division=0))
            
            try:
                cv_scores['roc_auc'].append(roc_auc_score(y_test, final_predictions))
            except:
                cv_scores['roc_auc'].append(0.5)
        
        # Calculate validation statistics
        validation_stats = {}
        for metric, scores in cv_scores.items():
            validation_stats[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return {
            'status': 'success',
            'validation_method': 'cross_validation',
            'cv_folds': cv_folds,
            'ensemble_algorithms': ensemble_algorithms,
            'validation_statistics': validation_stats,
            'cv_scores': cv_scores
        }
    
    def _generate_ensemble_recommendations(self, 
                                         selection_results: Dict[str, Any],
                                         validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ensemble recommendations based on analysis results."""
        
        recommendations = {
            'primary_recommendation': {},
            'alternative_recommendations': [],
            'use_case_specific': {},
            'implementation_notes': []
        }
        
        selected_best = selection_results.get('selected_best')
        
        if selected_best:
            ensemble_name, ensemble_info = selected_best
            algorithms = ensemble_info['ensemble_data']['algorithms']
            combined_score = ensemble_info['scores']['combined']
            
            recommendations['primary_recommendation'] = {
                'ensemble_name': ensemble_name,
                'algorithms': algorithms,
                'ensemble_size': len(algorithms),
                'combined_score': combined_score,
                'strategy': ensemble_info['ensemble_data']['strategy'],
                'reasons': self._generate_recommendation_reasons(ensemble_info)
            }
        
        # Alternative recommendations
        ranked_ensembles = selection_results.get('ranked_ensembles', [])
        for i, (ensemble_name, ensemble_info) in enumerate(ranked_ensembles[1:4]):  # Next 3 best
            recommendations['alternative_recommendations'].append({
                'rank': i + 2,
                'ensemble_name': ensemble_name,
                'algorithms': ensemble_info['ensemble_data']['algorithms'],
                'combined_score': ensemble_info['scores']['combined'],
                'use_case': self._suggest_use_case(ensemble_info)
            })
        
        # Use case specific recommendations
        recommendations['use_case_specific'] = self._generate_use_case_recommendations(
            selection_results
        )
        
        # Implementation notes
        recommendations['implementation_notes'] = self._generate_implementation_notes(
            selected_best, validation_results
        )
        
        return recommendations
    
    def _generate_recommendation_reasons(self, ensemble_info: Dict[str, Any]) -> List[str]:
        """Generate reasons for ensemble recommendation."""
        
        reasons = []
        scores = ensemble_info['scores']
        
        if scores['performance'] > 0.8:
            reasons.append(f"High performance score: {scores['performance']:.3f}")
        
        if scores['diversity'] > 0.6:
            reasons.append(f"Good algorithm diversity: {scores['diversity']:.3f}")
        
        if scores['efficiency'] > 0.7:
            reasons.append(f"Efficient ensemble size: {scores['efficiency']:.3f}")
        
        ensemble_size = len(ensemble_info['ensemble_data']['algorithms'])
        if ensemble_size >= 3:
            reasons.append(f"Robust ensemble with {ensemble_size} algorithms")
        
        if not reasons:
            reasons.append("Best overall combination of performance, diversity, and efficiency")
        
        return reasons
    
    def _suggest_use_case(self, ensemble_info: Dict[str, Any]) -> str:
        """Suggest appropriate use case for ensemble."""
        
        scores = ensemble_info['scores']
        ensemble_size = len(ensemble_info['ensemble_data']['algorithms'])
        
        if scores['performance'] > 0.9:
            return "High-accuracy applications"
        elif scores['diversity'] > 0.8:
            return "Robust detection with diverse patterns"
        elif scores['efficiency'] > 0.9:
            return "Real-time or resource-constrained environments"
        elif ensemble_size >= 4:
            return "Complex datasets with multiple anomaly types"
        else:
            return "General-purpose anomaly detection"
    
    def _generate_use_case_recommendations(self, selection_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate use case specific recommendations."""
        
        scored_ensembles = selection_results.get('scored_ensembles', {})
        use_cases = {}
        
        # Find best for each criterion
        best_performance = max(
            scored_ensembles.items(),
            key=lambda x: x[1]['scores']['performance'],
            default=None
        )
        
        best_diversity = max(
            scored_ensembles.items(),
            key=lambda x: x[1]['scores']['diversity'],
            default=None
        )
        
        best_efficiency = max(
            scored_ensembles.items(),
            key=lambda x: x[1]['scores']['efficiency'],
            default=None
        )
        
        if best_performance:
            algorithms = best_performance[1]['ensemble_data']['algorithms']
            use_cases['high_accuracy'] = f"For maximum accuracy: {', '.join(algorithms)}"
        
        if best_diversity:
            algorithms = best_diversity[1]['ensemble_data']['algorithms']
            use_cases['robust_detection'] = f"For robust detection: {', '.join(algorithms)}"
        
        if best_efficiency:
            algorithms = best_efficiency[1]['ensemble_data']['algorithms']
            use_cases['real_time'] = f"For real-time applications: {', '.join(algorithms)}"
        
        return use_cases
    
    def _generate_implementation_notes(self, 
                                     selected_best: Optional[Tuple],
                                     validation_results: Dict[str, Any]) -> List[str]:
        """Generate implementation notes for the selected ensemble."""
        
        notes = []
        
        if selected_best:
            algorithms = selected_best[1]['ensemble_data']['algorithms']
            
            # Algorithm-specific notes
            if 'OneClassSVM' in algorithms:
                notes.append("OneClassSVM requires feature scaling - ensure data is standardized")
            
            if 'LocalOutlierFactor' in algorithms:
                notes.append("LOF is sensitive to the choice of k neighbors - consider tuning")
            
            if len(algorithms) > 3:
                notes.append("Large ensemble - consider performance impact in production")
            
            # Validation-based notes
            validation_stats = validation_results.get('validation_statistics', {})
            if validation_stats:
                roc_auc_std = validation_stats.get('roc_auc', {}).get('std', 0)
                if roc_auc_std > 0.1:
                    notes.append("High performance variance - consider more stable ensemble")
                
                f1_mean = validation_stats.get('f1', {}).get('mean', 0)
                if f1_mean < 0.7:
                    notes.append("Consider threshold tuning to improve F1 score")
        
        # General implementation notes
        notes.extend([
            "Implement ensemble voting in production for real-time inference",
            "Monitor individual algorithm performance for ensemble maintenance",
            "Consider retraining ensemble periodically with new data"
        ])
        
        return notes
    
    def save_results(self, filepath: str):
        """Save ensemble selection results to file."""
        
        # Prepare serializable results
        serializable_results = {}
        
        for dataset_name, results in self.ensemble_performance.items():
            serializable_results[dataset_name] = {
                'diversity_analysis': results.get('diversity_analysis', {}),
                'ensemble_candidates': results.get('ensemble_candidates', 0),
                'strategy_evaluation': results.get('strategy_evaluation', {}),
                'optimization_results': results.get('optimization_results', {}),
                'selected_ensembles': results.get('selected_ensembles', {}),
                'validation_results': results.get('validation_results', {}),
                'recommendations': results.get('recommendations', {}),
                'timestamp': datetime.now().isoformat()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Ensemble selection results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load ensemble selection results from file."""
        
        with open(filepath, 'r') as f:
            loaded_results = json.load(f)
        
        self.ensemble_performance.update(loaded_results)
        logger.info(f"Ensemble selection results loaded from {filepath}")


def main():
    """Example usage of the Ensemble Selector."""
    
    # Generate sample anomaly detection dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    contamination = 0.1
    
    # Generate normal data
    X_normal = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * (1 - contamination))
    )
    
    # Generate anomalous data
    X_anomaly = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=int(n_samples * contamination)
    )
    
    # Combine and shuffle data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([
        np.zeros(len(X_normal)),
        np.ones(len(X_anomaly))
    ])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Contamination rate: {np.mean(y):.3f}")
    
    # Initialize ensemble selector
    ensemble_selector = EnsembleSelector(verbose=True)
    
    # Run ensemble selection
    results = ensemble_selector.select_ensemble(X, y, "Synthetic Dataset")
    
    # Print results
    print("\n" + "="*60)
    print("ENSEMBLE SELECTION RESULTS")
    print("="*60)
    
    # Diversity analysis
    diversity_analysis = results['diversity_analysis']
    if 'overall_diversity_score' in diversity_analysis:
        print(f"\nAlgorithm Diversity Score: {diversity_analysis['overall_diversity_score']:.3f}")
        print(f"Average Disagreement: {diversity_analysis['disagreement']['average_disagreement']:.3f}")
    
    # Recommendations
    recommendations = results.get('recommendations', {})
    primary_rec = recommendations.get('primary_recommendation', {})
    
    if primary_rec:
        print(f"\nPrimary Recommendation:")
        print(f"- Ensemble: {primary_rec.get('ensemble_name', 'Unknown')}")
        print(f"- Algorithms: {', '.join(primary_rec.get('algorithms', []))}")
        print(f"- Strategy: {primary_rec.get('strategy', 'Unknown')}")
        print(f"- Combined Score: {primary_rec.get('combined_score', 0):.3f}")
        
        reasons = primary_rec.get('reasons', [])
        if reasons:
            print("- Reasons:")
            for reason in reasons:
                print(f"  * {reason}")
    
    # Alternative recommendations
    alternatives = recommendations.get('alternative_recommendations', [])
    if alternatives:
        print(f"\nAlternative Recommendations:")
        for alt in alternatives[:2]:  # Show top 2 alternatives
            print(f"- Rank {alt['rank']}: {', '.join(alt['algorithms'])} "
                  f"(Score: {alt['combined_score']:.3f}, Use case: {alt['use_case']})")
    
    # Use case specific recommendations
    use_case_specific = recommendations.get('use_case_specific', {})
    if use_case_specific:
        print(f"\nUse Case Specific Recommendations:")
        for use_case, recommendation in use_case_specific.items():
            print(f"- {use_case.replace('_', ' ').title()}: {recommendation}")
    
    # Validation results
    validation_results = results.get('validation_results', {})
    validation_stats = validation_results.get('validation_statistics', {})
    
    if validation_stats:
        print(f"\nValidation Results (Cross-Validation):")
        for metric, stats in validation_stats.items():
            print(f"- {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Implementation notes
    impl_notes = recommendations.get('implementation_notes', [])
    if impl_notes:
        print(f"\nImplementation Notes:")
        for i, note in enumerate(impl_notes[:3], 1):  # Show first 3 notes
            print(f"{i}. {note}")
    
    # Save results
    ensemble_selector.save_results('ensemble_selection_results.json')
    
    print(f"\nDetailed results saved to 'ensemble_selection_results.json'")
    print("Ensemble selection completed successfully!")


if __name__ == "__main__":
    main()