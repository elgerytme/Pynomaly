# ADR-003: Anomaly Type Taxonomy and Severity Framework

## Status
**Proposed** - 2025-01-08

## Context
The current Pynomaly system needs a formal taxonomy for anomaly types and an expanded severity assessment framework to improve classification accuracy, user understanding, and system response capabilities. The existing system has basic severity levels but lacks a comprehensive framework for anomaly classification and algorithmic criteria.

## Decision
We will implement a comprehensive anomaly type taxonomy with an expanded severity level scale, including algorithmic criteria, new enums, strategy interfaces, and a backward-compatible migration path.

## Formal Anomaly Type Taxonomy

### Core Anomaly Categories

#### 1. Statistical Anomalies
**Definition**: Deviations from statistical norms based on distributional properties.

**Subcategories**:
- **Z-Score Anomalies**: `|z| > threshold` (typically 2.5-3.0)
- **MAD-based Anomalies**: Modified Z-score using Median Absolute Deviation
- **Percentile-based Anomalies**: Values beyond p1/p99 percentiles
- **IQR Outliers**: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR

**Algorithmic Criteria**:
```python
# Z-score calculation
z_score = (value - mean) / std_dev
is_anomaly = abs(z_score) > 2.5

# MAD-based calculation  
mad_score = 0.6745 * (value - median) / mad
is_anomaly = abs(mad_score) > 3.5
```

#### 2. Pattern-Based Anomalies
**Definition**: Deviations from expected behavioral or structural patterns.

**Subcategories**:
- **Rule-based Violations**: Business logic rule breaches
- **Sequence Anomalies**: Unexpected event sequences
- **Frequency Anomalies**: Unusual occurrence rates
- **Dependency Anomalies**: Broken correlations between variables

**Algorithmic Criteria**:
```python
# Pattern matching with confidence threshold
pattern_score = pattern_matcher.score(sequence)
is_anomaly = pattern_score < confidence_threshold
```

#### 3. Temporal Anomalies
**Definition**: Time-dependent deviations in sequential data.

**Subcategories**:
- **Trend Anomalies**: Unexpected trend changes
- **Cyclical Anomalies**: Broken periodic patterns
- **Burst Anomalies**: Sudden spikes in activity
- **Gap Anomalies**: Unexpected silence periods

**Algorithmic Criteria**:
```python
# LSTM-based temporal scoring
temporal_score = temporal_model.predict_proba(sequence)
is_anomaly = temporal_score > anomaly_threshold
```

#### 4. Seasonal Anomalies
**Definition**: Deviations from expected seasonal or periodic patterns.

**Subcategories**:
- **Daily Seasonality**: Hour-of-day pattern deviations
- **Weekly Seasonality**: Day-of-week pattern deviations  
- **Monthly Seasonality**: Day-of-month pattern deviations
- **Annual Seasonality**: Month-of-year pattern deviations

**Algorithmic Criteria**:
```python
# Seasonal decomposition residual analysis
seasonal_residual = observed - seasonal_component
seasonal_score = abs(seasonal_residual) / seasonal_std
is_anomaly = seasonal_score > seasonal_threshold
```

#### 5. Contextual Anomalies
**Definition**: Conditional anomalies dependent on contextual attributes.

**Subcategories**:
- **Conditional Anomalies**: Anomalous given specific conditions
- **Multi-dimensional Context**: Anomalous in feature space combinations
- **Environmental Context**: Anomalous given external factors
- **Behavioral Context**: Anomalous given user/entity behavior

**Algorithmic Criteria**:
```python
# Context-aware scoring
context_score = contextual_model.score(features, context)
is_anomaly = context_score > context_threshold
```

#### 6. Point Anomalies
**Definition**: Individual data points that deviate from normal patterns.

**Subcategories**:
- **Global Outliers**: Anomalous across entire dataset
- **Local Outliers**: Anomalous within local neighborhoods
- **Distributional Outliers**: Tail distribution extremes
- **Feature-specific Outliers**: Anomalous in specific dimensions

**Algorithmic Criteria**:
```python
# Local Outlier Factor (LOF)
lof_score = lof_model.decision_function(point)
is_anomaly = lof_score < lof_threshold
```

#### 7. Collective Anomalies
**Definition**: Groups of data points that are collectively anomalous.

**Subcategories**:
- **Cluster Anomalies**: Unusual cluster formations
- **Subspace Anomalies**: Anomalous in projected dimensions
- **Network Anomalies**: Unusual graph structures
- **Ensemble Anomalies**: Collective model disagreement

**Algorithmic Criteria**:
```python
# Collective scoring using ensemble methods
collective_scores = [model.score(subset) for model in ensemble]
collective_anomaly = np.mean(collective_scores) > collective_threshold
```

#### 8. Contextual-Collective Anomalies
**Definition**: Groups of data points anomalous under specific contextual conditions.

**Subcategories**:
- **Conditional Clusters**: Groups anomalous given conditions
- **Temporal Collectives**: Time-based group anomalies
- **Multi-entity Collectives**: Cross-entity group patterns
- **Hierarchical Collectives**: Nested group anomalies

**Algorithmic Criteria**:
```python
# Context-aware collective scoring
context_collective_score = contextual_collective_model.score(
    group, context, temporal_window
)
is_anomaly = context_collective_score > cc_threshold
```

## Expanded Severity Level Scale

### 6-Level Severity Framework

#### 1. INFO (Informational)
**Score Range**: 0.0 - 0.2  
**Algorithmic Criteria**:
- Domain-specific thresholds
- Configuration violations
- Data quality notices

**Response**: Log only, no alerts
```python
def assess_info_severity(score: float, context: dict) -> bool:
    return 0.0 <= score <= 0.2 and context.get('info_enabled', True)
```

#### 2. LOW  
**Score Range**: 0.2 - 0.4  
**Algorithmic Criteria**:
- Z-score: 1.0 < |z| <= 2.0
- MAD score: 1.5 < |mad| <= 2.5
- Model residuals: 1 < residual <= 2 standard deviations

**Response**: Dashboard notification
```python
def assess_low_severity(score: float, z_score: float) -> bool:
    return 0.2 < score <= 0.4 or (1.0 < abs(z_score) <= 2.0)
```

#### 3. MEDIUM
**Score Range**: 0.4 - 0.6  
**Algorithmic Criteria**:
- Z-score: 2.0 < |z| <= 3.0
- MAD score: 2.5 < |mad| <= 3.5
- Model residuals: 2 < residual <= 3 standard deviations
- Confidence interval width < 0.3

**Response**: Email alert, dashboard highlight
```python
def assess_medium_severity(score: float, confidence_width: float) -> bool:
    return (0.4 < score <= 0.6) and (confidence_width < 0.3)
```

#### 4. HIGH
**Score Range**: 0.6 - 0.8  
**Algorithmic Criteria**:
- Z-score: |z| > 3.0
- MAD score: |mad| > 3.5
- Model confidence > 0.8
- Confidence interval width < 0.2

**Response**: Immediate notification, escalation
```python
def assess_high_severity(score: float, model_confidence: float) -> bool:
    return (0.6 < score <= 0.8) and (model_confidence > 0.8)
```

#### 5. CRITICAL
**Score Range**: 0.8 - 0.95  
**Algorithmic Criteria**:
- Extreme statistical deviations (|z| > 4.0)
- High confidence anomalies (confidence > 0.9)
- Narrow confidence intervals (width < 0.1)
- Multiple model agreement (>80% consensus)

**Response**: Immediate escalation, automated response
```python
def assess_critical_severity(
    score: float, 
    confidence: float, 
    interval_width: float,
    model_consensus: float
) -> bool:
    return (
        0.8 < score <= 0.95 and 
        confidence > 0.9 and 
        interval_width < 0.1 and 
        model_consensus > 0.8
    )
```

#### 6. CATASTROPHIC
**Score Range**: 0.95 - 1.0  
**Algorithmic Criteria**:
- Infrastructure-level threshold breaches
- System-wide failure indicators
- Security breach patterns
- Data integrity violations

**Response**: Emergency protocols, system protection
```python
def assess_catastrophic_severity(
    score: float, 
    system_health: dict,
    security_indicators: dict
) -> bool:
    return (
        score > 0.95 or 
        system_health.get('critical_failure', False) or
        security_indicators.get('breach_detected', False)
    )
```

## Public API Changes

### New Enumerations

#### AnomalyType Enum
```python
from enum import Enum

class AnomalyType(str, Enum):
    """Formal anomaly type classification."""
    
    # Statistical Anomalies
    STATISTICAL = "statistical"
    Z_SCORE = "z_score"
    MAD_BASED = "mad_based"
    PERCENTILE = "percentile"
    IQR_OUTLIER = "iqr_outlier"
    
    # Pattern-Based Anomalies
    PATTERN_BASED = "pattern_based"
    RULE_VIOLATION = "rule_violation"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    DEPENDENCY_ANOMALY = "dependency_anomaly"
    
    # Temporal Anomalies
    TEMPORAL = "temporal"
    TREND_ANOMALY = "trend_anomaly"
    CYCLICAL_ANOMALY = "cyclical_anomaly"
    BURST_ANOMALY = "burst_anomaly"
    GAP_ANOMALY = "gap_anomaly"
    
    # Seasonal Anomalies
    SEASONAL = "seasonal"
    DAILY_SEASONAL = "daily_seasonal"
    WEEKLY_SEASONAL = "weekly_seasonal"
    MONTHLY_SEASONAL = "monthly_seasonal"
    ANNUAL_SEASONAL = "annual_seasonal"
    
    # Contextual Anomalies
    CONTEXTUAL = "contextual"
    CONDITIONAL = "conditional"
    MULTI_DIMENSIONAL = "multi_dimensional"
    ENVIRONMENTAL = "environmental"
    BEHAVIORAL = "behavioral"
    
    # Point Anomalies
    POINT = "point"
    GLOBAL_OUTLIER = "global_outlier"
    LOCAL_OUTLIER = "local_outlier"
    DISTRIBUTIONAL_OUTLIER = "distributional_outlier"
    FEATURE_SPECIFIC = "feature_specific"
    
    # Collective Anomalies
    COLLECTIVE = "collective"
    CLUSTER_ANOMALY = "cluster_anomaly"
    SUBSPACE_ANOMALY = "subspace_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    ENSEMBLE_ANOMALY = "ensemble_anomaly"
    
    # Contextual-Collective Anomalies
    CONTEXTUAL_COLLECTIVE = "contextual_collective"
    CONDITIONAL_CLUSTER = "conditional_cluster"
    TEMPORAL_COLLECTIVE = "temporal_collective"
    MULTI_ENTITY_COLLECTIVE = "multi_entity_collective"
    HIERARCHICAL_COLLECTIVE = "hierarchical_collective"
```

#### Enhanced SeverityLevel Enum
```python
class SeverityLevel(str, Enum):
    """Enhanced severity level classification."""
    
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"
    
    @property
    def score_range(self) -> tuple[float, float]:
        """Get score range for severity level."""
        ranges = {
            self.INFO: (0.0, 0.2),
            self.LOW: (0.2, 0.4),
            self.MEDIUM: (0.4, 0.6),
            self.HIGH: (0.6, 0.8),
            self.CRITICAL: (0.8, 0.95),
            self.CATASTROPHIC: (0.95, 1.0)
        }
        return ranges[self]
    
    @property
    def priority(self) -> int:
        """Get numeric priority for severity level."""
        priorities = {
            self.INFO: 1,
            self.LOW: 2,
            self.MEDIUM: 3,
            self.HIGH: 4,
            self.CRITICAL: 5,
            self.CATASTROPHIC: 6
        }
        return priorities[self]
```

### Strategy Interfaces

#### Anomaly Detection Strategy Interface
```python
from abc import ABC, abstractmethod
from typing import Protocol, Any, Dict, List
from dataclasses import dataclass

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    anomaly_type: AnomalyType
    severity: SeverityLevel
    score: float
    confidence: float
    explanation: str
    metadata: Dict[str, Any]

class IAnomalyDetectionStrategy(Protocol):
    """Interface for anomaly detection strategies."""
    
    @abstractmethod
    def detect(self, data: Any, context: Dict[str, Any]) -> AnomalyDetectionResult:
        """Detect anomalies in data."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[AnomalyType]:
        """Get supported anomaly types."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the detection strategy."""
        pass
```

#### Severity Assessment Interface
```python
class ISeverityAssessment(Protocol):
    """Interface for severity assessment strategies."""
    
    @abstractmethod
    def assess_severity(
        self, 
        score: float, 
        anomaly_type: AnomalyType,
        context: Dict[str, Any]
    ) -> SeverityLevel:
        """Assess severity of an anomaly."""
        pass
    
    @abstractmethod
    def get_severity_criteria(self, anomaly_type: AnomalyType) -> Dict[str, Any]:
        """Get severity criteria for anomaly type."""
        pass
```

### Enhanced Domain Entities

#### Updated Anomaly Entity
```python
@dataclass
class Anomaly:
    """Enhanced anomaly entity with taxonomy support."""
    
    # Core identification
    id: UUID = field(default_factory=uuid4)
    score: AnomalyScore
    data_point: Dict[str, Any]
    detector_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Enhanced classification
    anomaly_type: AnomalyType
    severity: SeverityLevel
    
    # Detailed assessment
    confidence: float
    confidence_interval: ConfidenceInterval | None = None
    explanation: str | None = None
    
    # Contextual information
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Algorithmic details
    algorithm_details: Dict[str, Any] = field(default_factory=dict)
    
    def get_severity_details(self) -> Dict[str, Any]:
        """Get detailed severity assessment information."""
        return {
            "severity": self.severity,
            "score": self.score.value,
            "confidence": self.confidence,
            "criteria_met": self.algorithm_details.get("criteria_met", []),
            "assessment_method": self.algorithm_details.get("assessment_method")
        }
```

## Backward-Compatibility Plan

### Legacy Support Strategy

#### 1. API Versioning
- Maintain existing `/v1/` endpoints unchanged
- Introduce new `/v2/` endpoints with enhanced taxonomy
- Support both versions for 6 months minimum

#### 2. Enum Migration
```python
class LegacySeverityMapper:
    """Maps legacy severity to new severity levels."""
    
    LEGACY_TO_NEW = {
        "low": SeverityLevel.LOW,
        "medium": SeverityLevel.MEDIUM,
        "high": SeverityLevel.HIGH,
        "critical": SeverityLevel.CRITICAL
    }
    
    @classmethod
    def map_legacy_severity(cls, legacy_severity: str) -> SeverityLevel:
        return cls.LEGACY_TO_NEW.get(legacy_severity, SeverityLevel.MEDIUM)
```

#### 3. Response Format Compatibility
```python
class BackwardCompatibilitySerializer:
    """Maintains backward compatibility for API responses."""
    
    def serialize_legacy_format(self, anomaly: Anomaly) -> Dict[str, Any]:
        """Serialize anomaly in legacy format."""
        return {
            "id": str(anomaly.id),
            "score": anomaly.score.value,
            "severity": anomaly.severity.value if hasattr(anomaly, 'severity') else "medium",
            "detector_name": anomaly.detector_name,
            "timestamp": anomaly.timestamp.isoformat(),
            "data_point": anomaly.data_point,
            "metadata": anomaly.metadata
        }
```

## Migration Path

### Phase 1: Foundation (Weeks 1-2)
1. Implement new enums and interfaces
2. Create backward-compatibility layer
3. Add enhanced entity definitions
4. Update existing severity calculations

### Phase 2: Strategy Implementation (Weeks 3-4)
1. Implement core detection strategies
2. Add severity assessment algorithms
3. Create taxonomy classification logic
4. Develop configuration framework

### Phase 3: Integration (Weeks 5-6)
1. Update API endpoints with new taxonomy
2. Enhance CLI commands
3. Update web interface
4. Add documentation and examples

### Phase 4: Migration Tools (Weeks 7-8)
1. Create automated migration scripts
2. Develop data conversion utilities
3. Add migration verification tools
4. Update monitoring and alerting

### Migration Scripts
```python
class AnomalyDataMigrator:
    """Migrates existing anomaly data to new taxonomy."""
    
    def migrate_anomaly_records(self, records: List[Dict]) -> List[Dict]:
        """Migrate legacy anomaly records."""
        migrated_records = []
        
        for record in records:
            migrated_record = self._migrate_single_record(record)
            migrated_records.append(migrated_record)
        
        return migrated_records
    
    def _migrate_single_record(self, record: Dict) -> Dict:
        """Migrate a single anomaly record."""
        # Infer anomaly type from detector name and metadata
        anomaly_type = self._infer_anomaly_type(record)
        
        # Map legacy severity to new severity
        new_severity = self._map_severity(record.get('severity', 'medium'))
        
        # Enhance with new fields
        record.update({
            'anomaly_type': anomaly_type.value,
            'severity': new_severity.value,
            'confidence': record.get('confidence', 0.5),
            'algorithm_details': self._extract_algorithm_details(record)
        })
        
        return record
```

## Test Matrix

### Functional Testing

#### 1. Anomaly Type Classification Tests
```python
class TestAnomalyTypeClassification:
    """Test anomaly type classification accuracy."""
    
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly classification."""
        pass
    
    def test_pattern_based_anomaly_detection(self):
        """Test pattern-based anomaly classification."""
        pass
    
    def test_temporal_anomaly_detection(self):
        """Test temporal anomaly classification."""
        pass
    
    # ... tests for each anomaly type
```

#### 2. Severity Assessment Tests
```python
class TestSeverityAssessment:
    """Test severity level assessment."""
    
    def test_severity_score_ranges(self):
        """Test severity level score ranges."""
        pass
    
    def test_algorithmic_criteria(self):
        """Test algorithmic severity criteria."""
        pass
    
    def test_confidence_interval_impact(self):
        """Test confidence interval impact on severity."""
        pass
```

### Performance Testing

#### 1. Scalability Tests
- Test taxonomy classification with 1M+ records
- Benchmark severity assessment algorithms
- Measure memory usage with enhanced entities

#### 2. Latency Tests
- API response time with new taxonomy
- Real-time classification performance
- Batch processing throughput

### Integration Testing

#### 1. System Integration
- End-to-end workflow testing
- Database migration verification
- API backward compatibility

#### 2. External Integration
- Webhook notification with new severity levels
- Monitoring system integration
- Alert routing with enhanced taxonomy

### Backward Compatibility Testing

#### 1. Legacy API Tests
```python
class TestBackwardCompatibility:
    """Test backward compatibility with legacy systems."""
    
    def test_legacy_api_endpoints(self):
        """Test legacy API endpoint compatibility."""
        pass
    
    def test_legacy_response_format(self):
        """Test legacy response format preservation."""
        pass
    
    def test_migration_data_integrity(self):
        """Test data integrity during migration."""
        pass
```

#### 2. Configuration Compatibility
- Legacy configuration file support
- Environment variable mapping
- Default value preservation

## Implementation Timeline

### Sprint 1 (Weeks 1-2): Foundation
- [ ] Implement AnomalyType and SeverityLevel enums
- [ ] Create strategy interfaces
- [ ] Design backward compatibility layer
- [ ] Update domain entities

### Sprint 2 (Weeks 3-4): Core Algorithms
- [ ] Implement statistical anomaly detection
- [ ] Create severity assessment algorithms
- [ ] Add confidence calculation methods
- [ ] Develop configuration framework

### Sprint 3 (Weeks 5-6): Integration
- [ ] Update API endpoints
- [ ] Enhance CLI commands
- [ ] Update web interface
- [ ] Add monitoring integration

### Sprint 4 (Weeks 7-8): Migration & Documentation
- [ ] Create migration scripts
- [ ] Add comprehensive documentation
- [ ] Implement monitoring and alerting
- [ ] Conduct final testing

## Consequences

### Positive
- **Improved Classification**: More precise anomaly categorization
- **Enhanced Severity Assessment**: Better prioritization and response
- **Algorithmic Transparency**: Clear criteria for severity levels
- **Backward Compatibility**: Seamless migration for existing users
- **Extensibility**: Framework for future anomaly types

### Negative
- **Increased Complexity**: More sophisticated classification logic
- **Migration Effort**: Requires careful data migration
- **Performance Impact**: Additional computation for taxonomy
- **Learning Curve**: Users need to understand new taxonomy

### Mitigation Strategies
- Comprehensive documentation and examples
- Automated migration tools
- Performance optimization and caching
- Gradual rollout with feature flags
- Training materials and tutorials

## References
- [Anomaly Detection: A Survey](https://example.com/anomaly-survey)
- [Statistical Outlier Detection Methods](https://example.com/statistical-methods)
- [Temporal Anomaly Detection Techniques](https://example.com/temporal-methods)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Pynomaly Architecture Overview](../overview.md)
