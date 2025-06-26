# Autonomous Mode Preprocessing Integration

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Autonomous

---


This guide covers the intelligent preprocessing capabilities integrated into Pynomaly's autonomous anomaly detection mode, providing truly end-to-end automated data preparation and anomaly detection.

## Overview

The autonomous preprocessing integration transforms the autonomous detection workflow from a simple "load and detect" process into a comprehensive "assess, prepare, optimize, and detect" pipeline. The system now automatically:

1. **Assesses data quality** using 10+ quality metrics
2. **Identifies preprocessing needs** based on data characteristics
3. **Applies intelligent preprocessing** with strategy selection
4. **Monitors quality improvements** and performance impact
5. **Proceeds with optimized detection** on prepared data

## Key Components

### 1. Quality Analyzer (`AutonomousQualityAnalyzer`)

Performs comprehensive data quality assessment with intelligent issue detection:

**Detected Issues:**
- **Missing Values**: Multiple strategies based on data type and missing patterns
- **Outliers**: IQR-based detection with severity assessment
- **Duplicates**: Row-level duplicate detection and impact analysis
- **Constant Features**: Zero-variance feature identification
- **Infinite Values**: Numerical stability issue detection
- **Poor Scaling**: Scale difference analysis between features
- **High Cardinality**: Categorical feature cardinality assessment
- **Imbalanced Categories**: Category frequency distribution analysis

**Quality Scoring:**
- Overall quality score (0.0 to 1.0)
- Issue-specific severity ratings
- Improvement potential estimation
- Processing time and memory impact predictions

### 2. Preprocessing Orchestrator (`AutonomousPreprocessingOrchestrator`)

Manages the intelligent application of preprocessing steps:

**Decision Logic:**
- Quality threshold evaluation (default: 0.8)
- Processing time budget enforcement (default: 5 minutes)
- Strategy selection based on data characteristics
- Error handling and fallback mechanisms

**Applied Preprocessing:**
- **Data Cleaning**: Missing values, outliers, duplicates, infinite values
- **Feature Scaling**: Standard, robust, or min-max scaling based on distribution
- **Categorical Handling**: Intelligent encoding selection (frequency, target, one-hot)
- **Feature Selection**: Removal of constant and low-variance features

### 3. Enhanced Autonomous Service

Integrates preprocessing seamlessly into the existing autonomous workflow:

**New Workflow Steps:**
1. **Data Loading** (existing)
2. **Quality Assessment** (new) - Analyze data quality and determine preprocessing needs
3. **Intelligent Preprocessing** (new) - Apply optimal preprocessing strategy
4. **Data Profiling** (enhanced) - Profile processed data with quality metadata
5. **Algorithm Recommendation** (existing, enhanced with preprocessing context)
6. **Detection Execution** (existing)
7. **Results & Export** (enhanced with preprocessing metadata)

## Configuration Options

### CLI Options

The autonomous detection command now includes comprehensive preprocessing options:

```bash
pynomaly auto detect data.csv \
  --preprocess/--no-preprocess \
  --quality-threshold 0.8 \
  --max-preprocess-time 300 \
  --preprocessing-strategy auto
```

**Preprocessing Options:**
- `--preprocess/--no-preprocess`: Enable/disable intelligent preprocessing (default: enabled)
- `--quality-threshold`: Data quality threshold for preprocessing (default: 0.8)
- `--max-preprocess-time`: Maximum preprocessing time in seconds (default: 300)
- `--preprocessing-strategy`: Strategy selection (auto, aggressive, conservative, minimal)

### Configuration Object

```python
config = AutonomousConfig(
    enable_preprocessing=True,
    quality_threshold=0.8,
    max_preprocessing_time=300.0,
    preprocessing_strategy="auto"
)
```

**Strategy Options:**
- **`auto`**: Intelligent strategy selection based on data characteristics (default)
- **`aggressive`**: Apply comprehensive preprocessing for maximum quality improvement
- **`conservative`**: Apply only essential preprocessing to minimize data changes
- **`minimal`**: Apply only critical fixes (infinite values, severe outliers)

## Usage Examples

### Basic Autonomous Detection with Preprocessing

```bash
# Default autonomous detection with intelligent preprocessing
pynomaly auto detect transactions.csv --output results.csv

# Verbose output showing preprocessing steps
pynomaly auto detect transactions.csv --verbose --output results.csv
```

### Custom Preprocessing Configuration

```bash
# Conservative preprocessing with higher quality threshold
pynomaly auto detect data.csv \
  --preprocessing-strategy conservative \
  --quality-threshold 0.9 \
  --verbose

# Aggressive preprocessing with extended time budget
pynomaly auto detect large_dataset.csv \
  --preprocessing-strategy aggressive \
  --max-preprocess-time 600 \
  --output processed_results.csv

# Disable preprocessing for pre-cleaned data
pynomaly auto detect clean_data.csv \
  --no-preprocess \
  --output clean_results.csv
```

### Programmatic Usage

```python
import asyncio
from pynomaly.application.services.autonomous_service import (
    AutonomousDetectionService, AutonomousConfig
)

# Setup service
service = AutonomousDetectionService(
    detector_repository=detector_repo,
    result_repository=result_repo,
    data_loaders=data_loaders
)

# Configure with preprocessing
config = AutonomousConfig(
    enable_preprocessing=True,
    quality_threshold=0.8,
    max_preprocessing_time=300.0,
    preprocessing_strategy="auto",
    verbose=True
)

# Run autonomous detection
results = await service.detect_autonomous("data.csv", config)

# Access preprocessing information
profile = results["data_profile"]
if profile.preprocessing_applied:
    print(f"Quality improvement: {profile.quality_score:.2f}")
    print(f"Steps applied: {len(profile.preprocessing_metadata['applied_steps'])}")
```

## Output and Reporting

### Enhanced CLI Output

The autonomous detection now provides comprehensive preprocessing information:

```
🤖 Autonomous Anomaly Detection
Data Source: transactions.csv
Max Algorithms: 5
Auto-tune: Yes
Preprocessing: Enabled
Quality Threshold: 0.80

📊 Dataset Profile
┌─────────────────────────┬─────────────┐
│ Property                │ Value       │
├─────────────────────────┼─────────────┤
│ Samples                 │ 10,000      │
│ Features                │ 12          │
│ Numeric Features        │ 8           │
│ Categorical Features    │ 4           │
│ Missing Values          │ 2.3%        │
└─────────────────────────┴─────────────┘

🔧 Data Quality & Preprocessing
┌─────────────────────────┬─────────────┐
│ Metric                  │ Value       │
├─────────────────────────┼─────────────┤
│ Quality Score           │ 0.65        │
│ Preprocessing Recommended│ Yes         │
│ Preprocessing Applied   │ Yes         │
└─────────────────────────┴─────────────┘

✓ Preprocessing Applied
┌─────────────────────────┬─────────────┬─────────────┐
│ Step                    │ Action      │ Details     │
├─────────────────────────┼─────────────┼─────────────┤
│ Missing Values          │ Filled      │ 3 columns   │
│ Outliers                │ Clipped     │ 2 columns   │
│ Duplicates              │ Removed     │ 15 items    │
│ Scaling                 │ Standardized│ 8 columns   │
└─────────────────────────┴─────────────┴─────────────┘

Shape: 10,000×12 → 9,985×12
```

### Preprocessing Metadata

The results include comprehensive preprocessing metadata:

```python
{
    "preprocessing_applied": True,
    "applied_steps": [
        {
            "type": "missing_values",
            "action": "filled",
            "columns": ["amount", "balance", "fee"],
            "strategy": "median/mode"
        },
        {
            "type": "outliers", 
            "action": "clipped",
            "columns": ["amount", "balance"]
        },
        {
            "type": "duplicates",
            "action": "removed", 
            "count": 15
        },
        {
            "type": "scaling",
            "action": "standardized",
            "columns": ["amount", "balance", "fee", "duration", "frequency", "score", "rating", "risk"]
        }
    ],
    "original_shape": [10000, 12],
    "final_shape": [9985, 12],
    "quality_improvement": 0.23,
    "processing_successful": True
}
```

## Quality Assessment Details

### Quality Metrics

The system evaluates multiple quality dimensions:

1. **Missing Values Ratio**: Percentage of missing data across all features
2. **Outlier Density**: Proportion of outliers using IQR method
3. **Duplicate Rate**: Percentage of duplicate rows
4. **Feature Variance**: Identification of constant and low-variance features
5. **Scale Consistency**: Analysis of feature scale differences
6. **Categorical Balance**: Assessment of category distribution
7. **Numerical Stability**: Detection of infinite and extreme values

### Quality Score Calculation

```
Quality Score = 1.0 - Weighted Average of Issue Severities

Weights:
- Missing Values: 0.30
- Infinite Values: 0.25  
- Outliers: 0.20
- Poor Scaling: 0.20
- Constant Features: 0.15
- Duplicates: 0.10
- Other Issues: 0.05-0.10
```

### Issue Severity Assessment

Each detected issue receives a severity score (0.0 to 1.0):

- **Low Severity (0.0-0.3)**: Minor issues with minimal impact
- **Medium Severity (0.3-0.7)**: Moderate issues requiring attention
- **High Severity (0.7-1.0)**: Critical issues requiring immediate fixing

## Strategy Selection Logic

### Auto Strategy (Default)

Intelligent strategy selection based on data characteristics:

```python
# Missing values
if missing_ratio > 0.3:
    strategy = "drop_columns"
elif data_type == "numeric" and missing_ratio < 0.1:
    strategy = "fill_median"
else:
    strategy = "knn_impute"

# Outliers  
if outlier_ratio > 0.1:
    strategy = "winsorize"
elif outlier_ratio > 0.05:
    strategy = "clip"
else:
    strategy = "transform_log"

# Scaling
if scale_ratio > 1000:
    strategy = "robust"
elif scale_ratio > 100:
    strategy = "standard"
else:
    strategy = "minmax"
```

### Aggressive Strategy

Maximum preprocessing for optimal data quality:
- Apply all available cleaning operations
- Use sophisticated imputation methods (KNN, iterative)
- Apply advanced transformations (polynomial features, feature selection)
- Optimize for detection performance over processing speed

### Conservative Strategy

Minimal preprocessing to preserve data integrity:
- Only fix critical issues (infinite values, severe outliers)
- Use simple imputation methods (median, mode)
- Avoid feature transformations that change data distribution
- Prioritize data preservation over quality improvements

### Minimal Strategy

Emergency preprocessing for compatibility:
- Only fix issues that cause algorithm failures
- Remove infinite values and extreme outliers
- Fill missing values with simple strategies
- No feature engineering or complex transformations

## Performance Considerations

### Processing Time Management

The system enforces processing time budgets to maintain responsiveness:

- **Default Limit**: 5 minutes (300 seconds)
- **Time Estimation**: Based on data size and selected operations
- **Early Termination**: Skip preprocessing if estimated time exceeds budget
- **Progressive Application**: Apply most critical fixes first

### Memory Management

Preprocessing operations are designed for memory efficiency:

- **In-Place Operations**: Minimize memory overhead where possible
- **Data Type Optimization**: Convert to efficient data types
- **Feature Reduction**: Remove unnecessary features early
- **Memory Monitoring**: Track memory usage during operations

### Quality vs. Speed Trade-offs

Different strategies balance quality improvements with processing speed:

| Strategy | Quality Gain | Speed | Memory Usage | Recommended For |
|----------|--------------|-------|--------------|-----------------|
| Minimal | Low | Fast | Low | Quick analysis, clean data |
| Conservative | Medium | Medium | Medium | Production pipelines |
| Auto | High | Medium | Medium | General use (default) |
| Aggressive | Maximum | Slow | High | Research, maximum accuracy |

## Integration Points

### Algorithm Recommendation Enhancement

Preprocessing context improves algorithm recommendations:

```python
# Enhanced algorithm selection considers preprocessing
if profile.preprocessing_applied:
    # Prefer algorithms that work well with preprocessed data
    recommendations.append({
        "algorithm": "IsolationForest",
        "confidence": 0.9,  # Higher confidence due to preprocessing
        "reasoning": "Isolation Forest performs optimally on preprocessed data with normalized features"
    })
```

### Export Integration

Preprocessing metadata is included in all export formats:

```json
{
    "metadata": {
        "preprocessing": {
            "applied": true,
            "quality_improvement": 0.23,
            "steps": [...],
            "processing_time": 45.2
        }
    },
    "anomalies": [...],
    "summary": {...}
}
```

### Pipeline Integration

Autonomous preprocessing integrates with the existing pipeline CLI:

```bash
# Export autonomous preprocessing as reusable pipeline
pynomaly auto detect data.csv --save-pipeline auto_pipeline.json

# Apply autonomous-generated pipeline to new data
pynomaly data pipeline load --config auto_pipeline.json --name auto_preprocessing
pynomaly data pipeline apply --name auto_preprocessing --dataset new_data_id
```

## Best Practices

### When to Enable Preprocessing

**Enable preprocessing when:**
- Working with raw, uncleaned data
- Data quality is unknown or suspected to be poor
- Maximum detection accuracy is required
- Data exploration and analysis workflow

**Disable preprocessing when:**
- Data has been pre-cleaned and validated
- Processing time is strictly limited
- Data integrity must be preserved exactly
- Testing specific algorithm behavior on raw data

### Strategy Selection Guidelines

**Use Auto strategy for:**
- General-purpose anomaly detection
- Unknown data quality scenarios
- Balanced quality vs. speed requirements
- Default production workflows

**Use Aggressive strategy for:**
- Research and analysis scenarios
- Maximum accuracy requirements
- Large datasets with ample processing time
- Data exploration workflows

**Use Conservative strategy for:**
- Production environments with strict SLAs
- Sensitive data that must be minimally modified
- Real-time or near-real-time processing
- Regulatory compliance scenarios

**Use Minimal strategy for:**
- Quick data compatibility checks
- Emergency processing situations
- Very large datasets with time constraints
- Algorithm behavior testing

### Monitoring and Validation

**Quality Improvement Tracking:**
```python
# Monitor quality improvements
if profile.preprocessing_applied:
    improvement = profile.preprocessing_metadata.get("quality_improvement", 0)
    if improvement > 0.2:
        print(f"Significant quality improvement: {improvement:.1%}")
    elif improvement < 0.05:
        print("Minimal quality improvement - consider disabling preprocessing")
```

**Processing Time Analysis:**
```python
# Analyze processing efficiency
processing_time = profile.preprocessing_metadata.get("processing_time", 0)
if processing_time > config.max_preprocessing_time * 0.8:
    print("Preprocessing approaching time limit - consider faster strategy")
```

## Troubleshooting

### Common Issues

**"Preprocessing skipped: Processing time too long"**
- Increase `--max-preprocess-time` value
- Use `--preprocessing-strategy minimal` or `conservative`
- Consider data sampling for very large datasets

**"No quality improvement from preprocessing"**
- Data may already be high quality - use `--no-preprocess`
- Try `--preprocessing-strategy aggressive` for more comprehensive cleaning
- Check data characteristics with `pynomaly auto profile`

**"Preprocessing failed with error"**
- Data may have format issues or corrupted values
- Use `--verbose` flag to see detailed error messages
- Try `--preprocessing-strategy minimal` for basic compatibility

**"Algorithm performance worse after preprocessing"**
- Some algorithms work better with raw data
- Use `--no-preprocess` to compare results
- Consider `--preprocessing-strategy conservative`

### Performance Optimization

**For Large Datasets:**
```bash
# Sample data for faster processing
pynomaly auto detect large_data.csv --max-samples 5000 --preprocess

# Use minimal preprocessing
pynomaly auto detect large_data.csv --preprocessing-strategy minimal
```

**For Real-time Processing:**
```bash
# Disable preprocessing for speed
pynomaly auto detect stream_data.csv --no-preprocess

# Use very short time budget
pynomaly auto detect data.csv --max-preprocess-time 30
```

## Future Enhancements

### Planned Features

1. **Adaptive Learning**: Learn from preprocessing effectiveness across datasets
2. **Custom Strategies**: User-defined preprocessing strategies and rules
3. **Parallel Processing**: Multi-threaded preprocessing for large datasets
4. **Quality Prediction**: Predict preprocessing impact before application
5. **Strategy Optimization**: Automatic strategy tuning based on detection performance

### Research Directions

1. **ML-Powered Strategy Selection**: Use machine learning for optimal strategy selection
2. **Real-time Quality Monitoring**: Continuous quality assessment during detection
3. **Domain-Specific Preprocessing**: Specialized preprocessing for different data domains
4. **Preprocessing Impact Analysis**: Quantify preprocessing impact on detection accuracy

## Conclusion

The autonomous preprocessing integration transforms Pynomaly's autonomous mode into a comprehensive, intelligent data preparation and anomaly detection pipeline. By automatically assessing data quality, applying optimal preprocessing strategies, and monitoring improvements, the system provides truly autonomous anomaly detection that works effectively on real-world, imperfect data.

Key benefits:

- **Automated Quality Assessment**: 10+ quality metrics with intelligent issue detection
- **Intelligent Preprocessing**: Context-aware strategy selection and application
- **Seamless Integration**: Transparent operation within existing autonomous workflow
- **Comprehensive Reporting**: Detailed preprocessing metadata and quality improvements
- **Flexible Configuration**: Multiple strategies and customizable parameters
- **Production Ready**: Time limits, error handling, and performance optimization

The integration maintains backward compatibility while significantly enhancing the autonomous detection capabilities, making Pynomaly suitable for production deployment on diverse, real-world datasets without manual data preparation.