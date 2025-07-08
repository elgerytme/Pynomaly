# Anomaly Detection System Design Specification

## Anomaly Type Taxonomy

1. **Statistical Anomalies**
   - **Definition**: Deviations from statistical norms.
   - **Examples**: Z-score, Moving Average Deviation (MAD).

2. **Pattern-based Anomalies**
   - **Definition**: Deviations from expected patterns.
   - **Examples**: Rule-based detection.

3. **Temporal Anomalies**
   - **Definition**: Occur based on time-based evaluation.
   - **Examples**: Time series analysis.

4. **Seasonal Anomalies**
   - **Definition**: Deviations from the expected seasonal patterns.
   - **Examples**: Seasonality checks in time series.

5. **Contextual Anomalies**
   - **Definition**: Deviations based on contextual attributes.
   - **Examples**: Unusual activity in typical conditions.

6. **Point Anomalies**
   - **Definition**: Single data point deviations.
   - **Examples**: Outliers.

7. **Collective Anomalies**
   - **Definition**: Deviations occurring collectively.
   - **Examples**: Cluster analysis.

8. **Contextual-Collective Anomalies**
   - **Definition**: Collective deviations based on contextual information.
   - **Examples**: Group outliers with contextual information.

## Severity Level Scale

1. **Informational (Info)**
   - **Algorithmic Criteria**: Basic threshold alerts.
   - **Methods**: Domain thresholds.

2. **Low**
   - **Algorithmic Criteria**: Minor deviations.
   - **Methods**: Z-score between 1 and 2.

3. **Medium**
   - **Algorithmic Criteria**: Moderate concern.
   - **Methods**: MAD deviations, model residuals.

4. **High**
   - **Algorithmic Criteria**: Serious issue requiring attention.
   - **Methods**: Z-score over 3.

5. **Critical**
   - **Algorithmic Criteria**: Immediate reaction required.
   - **Methods**: High confidence anomalies with low interval width.

6. **Catastrophic**
   - **Algorithmic Criteria**: System-wide failure potential.
   - **Methods**: Infra level threshold breaches.

## Public API Changes

1. **Enums**
   - **AnomalyType**: Defines various anomaly types.
   - **SeverityLevel**: Defines expanded severity levels.

2. **Strategy Interfaces**
   - **IAnomalyDetectionStrategy**: Interface for implementing different detection strategies.
   - **ISeverityAssessment**: Interface for assessing severity levels.

## Backward-Compatibility Plan

- Maintain existing API endpoints with legacy options.
- Provide versioning for new methods to prevent disruption.

## Migration Path

- Support old API methods for 6 months post-release.
- Automated migration scripts for data conversion.

## Test Matrix

1. **Functional Tests**
   - Test anomaly detection across types.

2. **Performance Tests**
   - Ensure system performance under load.

3. **Integration Tests**
   - Validate integration with current systems.

4. **Backward Compatibility**
   - Ensure existing systems work with minimal changes.
