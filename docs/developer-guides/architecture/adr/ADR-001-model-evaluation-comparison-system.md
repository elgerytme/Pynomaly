# ADR-001: Model Evaluation and Comparison System

🍞 **Breadcrumb:** 🏠 [Home](../../../index.md) > 👨‍💻 [Developer Guides](../../README.md) > 🏗️ [Architecture](../README.md) > 📋 [ADR](./README.md)

---

## Status
**PROPOSED** - 2024-01-15

## Context

Pynomaly requires a comprehensive model evaluation and comparison system to:
- Support multiple task types (binary anomaly detection, multi-class classification, regression)
- Provide standardized metrics across different algorithms
- Enable statistical significance testing for model comparisons
- Offer configurable evaluation parameters for different use cases
- Support both automated and manual evaluation workflows

## Decision

We will implement a unified Model Evaluation and Comparison System with the following architectural components:

### 1. Supported Task Types
- **Binary Anomaly Detection**: Primary focus for anomaly detection tasks
- **Multi-class Classification**: Support for categorical prediction tasks
- **Regression**: Support for continuous value prediction tasks

### 2. Primary/Secondary Metrics
**Primary Metrics:**
- F1 Score (weighted and macro averages)
- Precision (per-class and overall)
- Recall (per-class and overall)
- AUC-ROC (Area Under ROC Curve)
- PR-AUC (Precision-Recall Area Under Curve)

**Secondary Metrics:**
- Latency (prediction time per sample)
- Memory Usage (peak memory consumption)
- Training Time
- Model Size
- Throughput (samples per second)

### 3. Statistical Tests
- **McNemar's Test**: For comparing two binary classifiers
- **Wilcoxon Signed-Rank Test**: For non-parametric comparison of two models
- **Bootstrap Confidence Intervals**: For robust confidence estimation
- **Friedman Test + Nemenyi Post-hoc**: For comparing >2 models across multiple datasets

### 4. Configuration Parameters
- **Significance Level (α)**: Default 0.05, configurable from 0.01 to 0.10
- **Minimum Practical Difference**: Configurable threshold for meaningful differences
- **Top-K Selection**: Number of top-performing models to retain
- **Pareto Optimization**: Multi-objective optimization for accuracy vs. performance trade-offs

## Implementation Strategy

### Phase 1: Core Evaluation Engine
```python
# Core evaluation interfaces
class ModelEvaluator:
    def evaluate_model(self, model, dataset, task_type: TaskType) -> EvaluationResult
    def compare_models(self, models: List[Model], dataset) -> ComparisonResult
    def statistical_test(self, results: List[EvaluationResult], test_type: StatTest) -> TestResult

class EvaluationResult:
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    execution_time: float
    memory_usage: float
```

### Phase 2: Statistical Testing Framework
```python
class StatisticalTestSuite:
    def mcnemar_test(self, model_a_results, model_b_results) -> McnemarResult
    def wilcoxon_test(self, model_a_scores, model_b_scores) -> WilcoxonResult
    def bootstrap_ci(self, scores, confidence_level=0.95) -> BootstrapResult
    def friedman_nemenyi(self, model_results_matrix) -> FriedmanResult
```

### Phase 3: Configuration Management
```python
class EvaluationConfig:
    significance_level: float = 0.05
    min_practical_difference: float = 0.01
    top_k: int = 5
    pareto_objectives: List[str] = ["f1_score", "latency"]
    bootstrap_samples: int = 1000
    cv_folds: int = 5
```

## Architecture Components

### Domain Layer
- `ModelEvaluationService`: Core evaluation logic
- `StatisticalTestService`: Statistical significance testing
- `MetricsCalculator`: Computation of evaluation metrics
- `ComparisonEngine`: Model comparison and ranking

### Application Layer
- `EvaluateModelUseCase`: Single model evaluation
- `CompareModelsUseCase`: Multi-model comparison
- `StatisticalTestUseCase`: Statistical significance testing
- `BenchmarkSuiteUseCase`: Comprehensive benchmarking

### Infrastructure Layer
- `MetricsRepository`: Storage and retrieval of evaluation results
- `StatisticalTestAdapter`: Integration with statistical libraries
- `PerformanceMonitor`: System resource monitoring during evaluation

### Presentation Layer
- `EvaluationEndpoints`: REST API for evaluation services
- `EvaluationCLI`: Command-line interface for batch evaluations
- `EvaluationDashboard`: Web interface for result visualization

## Configuration Examples

### Basic Configuration
```yaml
evaluation:
  task_type: "binary_anomaly"
  metrics:
    primary: ["f1_score", "precision", "recall", "auc_roc"]
    secondary: ["latency", "memory_usage"]
  statistical_tests:
    enabled: true
    significance_level: 0.05
    tests: ["mcnemar", "wilcoxon", "bootstrap_ci"]
```

### Advanced Configuration
```yaml
evaluation:
  task_type: "multi_class"
  metrics:
    primary: ["f1_weighted", "precision_macro", "recall_macro"]
    secondary: ["training_time", "model_size"]
  statistical_tests:
    significance_level: 0.01
    min_practical_difference: 0.02
    bootstrap_samples: 2000
  pareto_optimization:
    objectives: ["f1_score", "latency"]
    weights: [0.7, 0.3]
  top_k: 10
```

## Consequences

### Positive
- Standardized evaluation framework across all model types
- Robust statistical testing for model comparison
- Configurable evaluation parameters for different use cases
- Support for both automated and manual evaluation workflows
- Clear separation of concerns with domain-driven design

### Negative
- Increased complexity in the evaluation pipeline
- Additional dependencies on statistical libraries
- Potential performance overhead for comprehensive evaluations
- Learning curve for advanced statistical testing features

### Risks
- Statistical test selection may require domain expertise
- Large-scale evaluations may require significant computational resources
- Configuration complexity may overwhelm basic users

## Mitigation Strategies
- Provide sensible defaults for all configuration parameters
- Implement progressive disclosure in UI (basic → advanced options)
- Create evaluation templates for common use cases
- Comprehensive documentation with examples
- Performance optimization for large-scale evaluations

## Acceptance Criteria

### Task Type Support
- [ ] Binary anomaly detection evaluation with appropriate metrics
- [ ] Multi-class classification evaluation with class-wise metrics
- [ ] Regression evaluation with continuous value metrics
- [ ] Automatic task type detection based on data characteristics

### Metrics Implementation
- [ ] F1 Score calculation (weighted, macro, micro averages)
- [ ] Precision/Recall per class and overall
- [ ] AUC-ROC and PR-AUC computation
- [ ] Latency measurement (prediction time per sample)
- [ ] Memory usage monitoring during evaluation

### Statistical Testing
- [ ] McNemar's test for binary classifier comparison
- [ ] Wilcoxon signed-rank test implementation
- [ ] Bootstrap confidence interval calculation
- [ ] Friedman test with Nemenyi post-hoc analysis
- [ ] Multiple testing correction (Bonferroni, FDR)

### Configuration Management
- [ ] Configurable significance level (α) with validation
- [ ] Minimum practical difference threshold setting
- [ ] Top-K model selection configuration
- [ ] Pareto optimization objective configuration
- [ ] Configuration validation and error handling

### Integration Points
- [ ] Integration with existing model training pipeline
- [ ] REST API endpoints for evaluation services
- [ ] CLI commands for batch evaluation
- [ ] Dashboard visualization of evaluation results
- [ ] Export capabilities (JSON, CSV, PDF reports)

## Related ADRs
- ADR-002: Statistical Testing Framework Integration
- ADR-003: Performance Monitoring Architecture
- ADR-004: Evaluation Results Storage Strategy

## References
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Statistical Tests for Machine Learning](https://jmlr.org/papers/v7/demsar06a.html)
- [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/)
