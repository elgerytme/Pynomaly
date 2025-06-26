# Explainability and Model Interpretation Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 📄 Explainability

---


## Overview

Pynomaly provides comprehensive model explainability features through SHAP and LIME integration, enabling users to understand why specific anomalies were detected and how models make decisions. This guide covers local explanations, global feature importance, cohort analysis, and visualization techniques.

## 🔍 Introduction to Explainability

### Why Explainability Matters

- **Trust**: Understand model decisions for critical applications
- **Debugging**: Identify model bias and potential issues
- **Compliance**: Meet regulatory requirements for interpretable AI
- **Business Insights**: Extract actionable insights from anomaly patterns
- **Model Improvement**: Guide feature engineering and algorithm selection

### Types of Explanations

| Type | Scope | Use Case | Example |
|------|-------|----------|---------|
| **Local** | Single prediction | Why is this specific instance anomalous? | "High transaction amount + unusual location" |
| **Global** | Entire model | Which features are most important overall? | "Transaction amount is the top predictor" |
| **Cohort** | Group of instances | How do explanations differ by segment? | "Weekend vs weekday transaction patterns" |

---

## 🎯 Quick Start

### Basic Explanation

```python
from pynomaly import create_detector, explain_anomaly

# Train detector
detector = create_detector("IsolationForest")
detector.fit(X_train)

# Detect anomalies
anomaly_scores = detector.predict(X_test)
anomalies = anomaly_scores > 0.5

# Explain a specific anomaly
explanation = explain_anomaly(
    detector=detector,
    instance=X_test[0],  # First test instance
    method="shap"        # or "lime"
)

print(f"Anomaly score: {anomaly_scores[0]:.3f}")
print(f"Top contributing features:")
for feature, importance in explanation.feature_importance.items():
    print(f"  {feature}: {importance:.3f}")
```

### Batch Explanations

```python
# Explain multiple anomalies
explanations = explain_anomaly(
    detector=detector,
    instances=X_test[anomalies],  # All anomalous instances
    method="shap",
    return_format="dataframe"
)

# Analyze explanation patterns
top_features = explanations.mean().abs().sort_values(ascending=False)
print("Most important features across all anomalies:")
print(top_features.head(10))
```

---

## 🔬 SHAP Integration

### TreeExplainer for Tree-Based Models

```python
from pynomaly.infrastructure.explainers import SHAPExplainer

# Initialize SHAP explainer for tree-based models
explainer = SHAPExplainer(
    method="tree",
    model=detector.model,
    feature_names=feature_names
)

# Generate SHAP explanations
shap_values = explainer.explain_instance(X_test[0])

print(f"Base value (expected model output): {explainer.expected_value:.3f}")
print(f"Instance prediction: {shap_values.sum() + explainer.expected_value:.3f}")

# Feature contributions
for i, (feature, value) in enumerate(zip(feature_names, shap_values)):
    print(f"{feature}: {value:.3f}")
```

### KernelExplainer for Black-Box Models

```python
# For complex models (neural networks, ensembles)
explainer = SHAPExplainer(
    method="kernel",
    model=detector.predict,  # Prediction function
    data=X_train[:100],      # Background dataset
    feature_names=feature_names
)

# Explain instance (may be slower)
shap_values = explainer.explain_instance(
    X_test[0],
    nsamples=1000  # Number of samples for estimation
)

# Get confidence intervals
explanation_with_confidence = explainer.explain_with_confidence(
    X_test[0],
    confidence_level=0.95
)

print(f"Feature importance (95% CI):")
for feature, (value, ci_low, ci_high) in explanation_with_confidence.items():
    print(f"{feature}: {value:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
```

### Linear Models (LinearExplainer)

```python
# For linear models
explainer = SHAPExplainer(
    method="linear",
    model=detector.model,
    feature_names=feature_names
)

# Fast linear explanations
shap_values = explainer.explain_instance(X_test[0])
```

### Advanced SHAP Features

```python
# Interaction values (feature interactions)
interaction_values = explainer.explain_interactions(X_test[0])

print("Feature interactions:")
for i, feature_i in enumerate(feature_names):
    for j, feature_j in enumerate(feature_names):
        if i < j and abs(interaction_values[i][j]) > 0.01:
            print(f"{feature_i} × {feature_j}: {interaction_values[i][j]:.3f}")

# Cohort analysis
cohorts = {
    "high_value": X_test[X_test[:, feature_idx] > threshold],
    "low_value": X_test[X_test[:, feature_idx] <= threshold]
}

cohort_explanations = explainer.explain_cohorts(cohorts)
```

---

## 🍋 LIME Integration

### Tabular Data Explanation

```python
from pynomaly.infrastructure.explainers import LIMEExplainer

# Initialize LIME explainer
explainer = LIMEExplainer(
    mode="tabular",
    training_data=X_train,
    feature_names=feature_names,
    categorical_features=[2, 5],  # Indices of categorical features
    categorical_names={2: ["A", "B", "C"], 5: ["X", "Y"]}  # Category names
)

# Explain instance
explanation = explainer.explain_instance(
    instance=X_test[0],
    predict_fn=detector.predict,
    num_features=10,      # Show top 10 features
    num_samples=5000      # Samples for local approximation
)

# Extract explanation
lime_values = explanation.as_map()[1]  # Get explanation for anomaly class
sorted_features = sorted(lime_values, key=lambda x: abs(x[1]), reverse=True)

print("LIME explanation (top features):")
for feature_idx, importance in sorted_features[:5]:
    feature_name = feature_names[feature_idx]
    print(f"{feature_name}: {importance:.3f}")
```

### Advanced LIME Configuration

```python
# Custom distance metrics and kernel
explainer = LIMEExplainer(
    mode="tabular",
    training_data=X_train,
    feature_names=feature_names,
    discretize_continuous=True,      # Discretize continuous features
    kernel_width=0.75,               # Kernel width for weighting
    feature_selection="auto",        # Feature selection method
    distance_metric="euclidean"      # Distance metric
)

# Robust explanation with multiple runs
explanations = []
for _ in range(10):  # 10 independent runs
    exp = explainer.explain_instance(
        X_test[0], 
        detector.predict, 
        num_samples=1000
    )
    explanations.append(exp.as_map()[1])

# Aggregate results
aggregated = explainer.aggregate_explanations(explanations)
```

---

## 📊 Visualization Techniques

### SHAP Visualizations

```python
import shap
import matplotlib.pyplot as plt

# Summary plot (global importance)
shap.summary_plot(
    shap_values_matrix,
    X_test,
    feature_names=feature_names,
    plot_type="bar"
)

# Waterfall plot (single instance)
shap.waterfall_plot(
    explainer.expected_value,
    shap_values[0],
    X_test[0],
    feature_names=feature_names
)

# Force plot (single instance, interactive)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test[0],
    feature_names=feature_names
)

# Dependence plot (partial dependence)
shap.dependence_plot(
    feature_idx,
    shap_values_matrix,
    X_test,
    feature_names=feature_names,
    interaction_index="auto"
)
```

### Custom Visualizations

```python
from pynomaly.application.services import ExplainabilityVisualizer

visualizer = ExplainabilityVisualizer()

# Feature importance heatmap
visualizer.plot_feature_importance_heatmap(
    explanations=shap_values_matrix,
    feature_names=feature_names,
    instance_labels=["Anomaly 1", "Anomaly 2", "Normal 1"]
)

# Explanation stability plot
visualizer.plot_explanation_stability(
    multiple_explanations,
    confidence_intervals=True
)

# Comparison plot (SHAP vs LIME)
visualizer.plot_method_comparison(
    shap_explanations=shap_values,
    lime_explanations=lime_values,
    feature_names=feature_names
)
```

---

## 🔄 Method Comparison and Consistency

### Comparing SHAP and LIME

```python
from pynomaly.domain.services import ExplainabilityService

service = ExplainabilityService()

# Generate explanations with both methods
comparison = service.compare_explanation_methods(
    detector=detector,
    instance=X_test[0],
    methods=["shap", "lime"],
    comparison_metrics=["correlation", "rank_correlation", "agreement"]
)

print(f"Explanation correlation: {comparison.correlation:.3f}")
print(f"Rank correlation: {comparison.rank_correlation:.3f}")
print(f"Top-5 feature agreement: {comparison.top_k_agreement[5]:.3f}")

# Consistency analysis
consistency = service.analyze_explanation_consistency(
    detector=detector,
    instances=X_test[:10],
    methods=["shap", "lime"],
    stability_runs=5
)

print(f"Average consistency score: {consistency.average_score:.3f}")
```

### Explanation Robustness Testing

```python
# Test explanation robustness to small perturbations
robustness = service.test_explanation_robustness(
    detector=detector,
    instance=X_test[0],
    method="shap",
    perturbation_std=0.01,
    num_perturbations=100
)

print(f"Robustness score: {robustness.stability_score:.3f}")
print(f"Feature ranking stability: {robustness.ranking_stability:.3f}")
```

---

## 📈 Global Model Analysis

### Feature Importance Analysis

```python
# Global feature importance across all data
global_importance = service.analyze_global_importance(
    detector=detector,
    data=X_test,
    method="shap",
    sample_size=1000  # Subsample for efficiency
)

print("Global feature importance:")
for feature, importance in global_importance.items():
    print(f"{feature}: {importance:.3f}")

# Feature importance by prediction class
class_importance = service.analyze_importance_by_class(
    detector=detector,
    data=X_test,
    labels=y_test,  # True labels if available
    method="shap"
)

print("\nFeature importance by class:")
for class_name, importances in class_importance.items():
    print(f"\n{class_name}:")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.3f}")
```

### Model Behavior Analysis

```python
# Analyze model behavior patterns
behavior_analysis = service.analyze_model_behavior(
    detector=detector,
    data=X_test,
    feature_names=feature_names,
    include_interactions=True,
    include_thresholds=True
)

print("Model behavior insights:")
print(f"- Most influential feature: {behavior_analysis.top_feature}")
print(f"- Average number of important features: {behavior_analysis.avg_important_features:.1f}")
print(f"- Feature interaction strength: {behavior_analysis.interaction_strength:.3f}")

# Decision boundary analysis
boundary_analysis = service.analyze_decision_boundaries(
    detector=detector,
    data=X_test,
    resolution=100
)
```

---

## 🎯 Cohort and Segment Analysis

### Demographic Cohorts

```python
# Define cohorts based on feature values
cohorts = {
    "young": X_test[X_test[:, age_idx] < 30],
    "middle_aged": X_test[(X_test[:, age_idx] >= 30) & (X_test[:, age_idx] < 60)],
    "senior": X_test[X_test[:, age_idx] >= 60]
}

# Analyze explanations by cohort
cohort_analysis = service.analyze_cohort_explanations(
    detector=detector,
    cohorts=cohorts,
    method="shap"
)

print("Cohort analysis results:")
for cohort_name, analysis in cohort_analysis.items():
    print(f"\n{cohort_name} cohort:")
    print(f"  Top feature: {analysis.top_feature}")
    print(f"  Avg anomaly score: {analysis.avg_anomaly_score:.3f}")
    print(f"  Explanation complexity: {analysis.explanation_complexity:.3f}")
```

### Time-Based Analysis

```python
# Temporal explanation analysis
temporal_analysis = service.analyze_temporal_explanations(
    detector=detector,
    data=X_test,
    timestamps=timestamps,
    time_windows=["daily", "weekly", "monthly"]
)

print("Temporal explanation patterns:")
for window, patterns in temporal_analysis.items():
    print(f"\n{window} patterns:")
    print(f"  Dominant features: {patterns.dominant_features}")
    print(f"  Seasonal effects: {patterns.seasonal_strength:.3f}")
```

---

## ⚡ Real-time Explanation

### Streaming Explanations

```python
from pynomaly.application.services import StreamingExplainer

# Initialize streaming explainer
streaming_explainer = StreamingExplainer(
    detector=detector,
    method="shap",
    buffer_size=1000,
    explanation_frequency=10  # Explain every 10th anomaly
)

# Process streaming data
for batch in data_stream:
    results = streaming_explainer.process_batch(batch)
    
    for result in results:
        if result.is_anomaly and result.explanation:
            print(f"Anomaly detected: {result.anomaly_score:.3f}")
            print(f"Top contributing features: {result.explanation.top_features}")
```

### Cached Explanations

```python
# Use caching for repeated explanations
cached_explainer = service.create_cached_explainer(
    detector=detector,
    method="lime",
    cache_size=1000,
    similarity_threshold=0.95  # Cache hit threshold
)

# Explanations will be cached and reused for similar instances
explanation = cached_explainer.explain(X_test[0])
```

---

## 🔧 Advanced Configuration

### Custom Explanation Methods

```python
from pynomaly.infrastructure.explainers import CustomExplainer

class BusinessLogicExplainer(CustomExplainer):
    """Custom explainer incorporating business rules"""
    
    def explain_instance(self, instance, **kwargs):
        # Custom business logic explanation
        explanation = {}
        
        # Rule-based explanations
        if instance[transaction_amount_idx] > 10000:
            explanation["high_amount"] = 0.8
        
        if instance[location_idx] not in known_locations:
            explanation["unusual_location"] = 0.6
        
        # Combine with ML explanation
        ml_explanation = self.base_explainer.explain_instance(instance)
        explanation.update(ml_explanation)
        
        return explanation

# Use custom explainer
custom_explainer = BusinessLogicExplainer(
    base_explainer=shap_explainer,
    business_rules=business_rules
)
```

### Explanation Templates

```python
# Pre-configured explanation templates for common use cases
from pynomaly.infrastructure.explainers import ExplanationTemplates

# Fraud detection template
fraud_explainer = ExplanationTemplates.fraud_detection(
    detector=detector,
    feature_names=feature_names,
    business_rules=fraud_rules
)

# IoT monitoring template
iot_explainer = ExplanationTemplates.iot_monitoring(
    detector=detector,
    sensor_names=sensor_names,
    normal_ranges=sensor_ranges
)

# Quality control template
quality_explainer = ExplanationTemplates.quality_control(
    detector=detector,
    measurement_names=measurement_names,
    specification_limits=spec_limits
)
```

---

## 📊 Explanation Quality Assessment

### Explanation Metrics

```python
# Assess explanation quality
quality_metrics = service.assess_explanation_quality(
    detector=detector,
    explanations=explanations,
    ground_truth=ground_truth_explanations,  # If available
    data=X_test
)

print("Explanation quality metrics:")
print(f"Faithfulness: {quality_metrics.faithfulness:.3f}")
print(f"Stability: {quality_metrics.stability:.3f}")
print(f"Comprehensiveness: {quality_metrics.comprehensiveness:.3f}")
print(f"Compactness: {quality_metrics.compactness:.3f}")
```

### Human Evaluation Interface

```python
# Generate human evaluation interface
evaluation_interface = service.create_evaluation_interface(
    explanations=explanations,
    instances=X_test[:10],
    output_format="html"
)

# Save interactive evaluation tool
evaluation_interface.save("explanation_evaluation.html")
```

---

## 📚 Best Practices

### Choosing the Right Method

```python
# Method selection guide
def select_explanation_method(detector, data_characteristics):
    if detector.model_type == "tree_based":
        return "shap_tree"  # Fast and exact
    elif data_characteristics["size"] == "large":
        return "shap_sampling"  # Efficient sampling
    elif data_characteristics["complexity"] == "high":
        return "lime"  # Model-agnostic
    else:
        return "shap_kernel"  # General purpose

method = select_explanation_method(detector, data_profile)
```

### Performance Optimization

```python
# Optimize explanation performance
optimized_explainer = service.create_optimized_explainer(
    detector=detector,
    method="shap",
    optimization_strategy="speed",  # speed, accuracy, memory
    max_samples=1000,
    use_gpu=True,
    parallel_processing=True
)
```

### Explanation Validation

```python
# Validate explanations against domain knowledge
validation_rules = {
    "transaction_amount": "positive_correlation",
    "account_age": "negative_correlation",
    "location_risk": "positive_correlation"
}

validation_results = service.validate_explanations(
    explanations=explanations,
    validation_rules=validation_rules,
    tolerance=0.1
)

print(f"Explanation validity: {validation_results.overall_validity:.3f}")
```

---

## 🚀 Integration Examples

### Web Dashboard Integration

```python
# Create explanation dashboard
from pynomaly.presentation.web import ExplanationDashboard

dashboard = ExplanationDashboard(
    detector=detector,
    explainer=explainer,
    feature_names=feature_names
)

# Add to web application
app.mount("/explanations", dashboard.create_app())
```

### API Integration

```python
# RESTful explanation API
from fastapi import FastAPI
from pynomaly.presentation.api.endpoints import ExplanationEndpoints

app = FastAPI()
explanation_endpoints = ExplanationEndpoints(
    detector=detector,
    explainer=explainer
)

app.include_router(explanation_endpoints.router, prefix="/api/explanations")
```

---

## 📖 Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- [Algorithm-Specific Explanation Guides](../reference/algorithm-explanations.md)
- [Business Use Case Examples](../examples/explanation-use-cases.md)

---

## 🤝 Contributing

See our [Contributing Guide](../development/contributing.md) for information on extending explainability features.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
