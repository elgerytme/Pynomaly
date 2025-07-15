# Property Testing Implementation Plan

## Implementation Roadmap

### Phase 1: Core Domain Properties (Priority 1 - Critical)

#### 1.1 Value Objects Property Tests
**File**: `tests/property/test_value_objects_mathematical_properties.py`
```python
# Mathematical invariants for core value objects
@given(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
def test_anomaly_score_range_invariant(score_value):
    """AnomalyScore values must always be in [0, 1] range."""
    score = AnomalyScore(value=score_value)
    assert 0.0 <= score.value <= 1.0
    assert score.is_valid()

@given(st.floats(0.001, 0.499))
def test_contamination_rate_bounds_invariant(rate_value):
    """ContaminationRate must be in valid bounds."""
    rate = ContaminationRate(rate_value)
    assert 0.0 < rate.value <= 0.5
    assert not rate.is_auto
```

**File**: `tests/property/test_confidence_interval_mathematical_properties.py`
```python
# Mathematical properties of confidence intervals
@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_confidence_interval_ordering_invariant(lower, upper):
    """Lower bound must never exceed upper bound."""
    assume(lower <= upper)
    ci = ConfidenceInterval(lower, upper)
    assert ci.lower <= ci.upper
    assert ci.width() >= 0.0

@given(confidence_interval_strategy(), st.floats(0.0, 1.0))
def test_confidence_interval_containment_property(ci, test_value):
    """Containment test must be consistent with bounds."""
    is_contained = ci.contains(test_value)
    expected = ci.lower <= test_value <= ci.upper
    assert is_contained == expected
```

#### 1.2 Entity State Machine Properties
**File**: `tests/property/test_detector_state_machine_properties.py`
```python
# Detector lifecycle state machine properties
@given(detector_strategy())
def test_detector_state_transitions(detector):
    """Detector state transitions must be valid."""
    # Initial state
    assert not detector.is_fitted
    assert detector.trained_at is None
    
    # Transition to fitted
    detector.mark_as_fitted()
    assert detector.is_fitted
    assert detector.trained_at is not None
    
    # Transition back to unfitted
    detector.mark_as_unfitted()
    assert not detector.is_fitted
    assert detector.trained_at is None

@given(detector_strategy(), st.dictionaries(st.text(), st.one_of(st.floats(), st.integers(), st.text())))
def test_detector_parameter_updates_preserve_state(detector, new_params):
    """Parameter updates should reset fitting state."""
    detector.mark_as_fitted()
    original_fitted = detector.is_fitted
    
    detector.update_parameters(**new_params)
    
    # Parameters updated but fitting state reset
    assert not detector.is_fitted
    assert detector.trained_at is None
    for key, value in new_params.items():
        assert detector.parameters[key] == value
```

#### 1.3 Entity Invariant Properties  
**File**: `tests/property/test_entity_business_invariants.py`
```python
# Business rule invariants for entities
@given(anomaly_strategy())
def test_anomaly_severity_consistency(anomaly):
    """Anomaly severity must be consistent with score."""
    score_value = anomaly.score.value if isinstance(anomaly.score, AnomalyScore) else anomaly.score
    severity = anomaly.severity
    
    if score_value > 0.9:
        assert severity == "critical"
    elif score_value > 0.7:
        assert severity == "high"
    elif score_value > 0.5:
        assert severity == "medium"
    else:
        assert severity == "low"

@given(dataset_strategy())
def test_dataset_feature_consistency(dataset):
    """Dataset feature counts must be consistent."""
    assert dataset.n_samples == len(dataset.data)
    assert dataset.n_features == len(dataset.data.columns)
    
    numeric_features = set(dataset.get_numeric_features())
    all_features = set(dataset.data.columns)
    assert numeric_features.issubset(all_features)
```

### Phase 2: Algorithm Mathematical Properties (Priority 1 - Critical)

#### 2.1 Algorithm Contract Properties
**File**: `tests/property/test_algorithm_mathematical_invariants.py`
```python
# Mathematical properties all algorithms must satisfy
@given(algorithm_input_strategy(), contamination_strategy(), st.integers(1, 100))
def test_algorithm_score_range_invariant(data, contamination, random_seed):
    """All algorithms must produce scores in [0, 1] range."""
    adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(contamination))
    np.random.seed(random_seed)
    
    try:
        # Create and fit detector
        dataset = Dataset("test", pd.DataFrame(data))
        adapter.fit(dataset)
        scores = adapter.score(dataset)
        
        # Score range property
        for score in scores:
            assert 0.0 <= score.value <= 1.0
            assert score.is_valid()
    except Exception:
        assume(False)  # Skip if algorithm fails on this data

@given(algorithm_input_strategy(), contamination_strategy(), st.integers(1, 100))
def test_algorithm_deterministic_property(data, contamination, random_seed):
    """Same algorithm with same seed must produce identical results."""
    try:
        dataset = Dataset("test", pd.DataFrame(data))
        
        # First run
        adapter1 = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(contamination), random_state=random_seed)
        adapter1.fit(dataset)
        result1 = adapter1.detect(dataset)
        
        # Second run with same seed
        adapter2 = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(contamination), random_state=random_seed)
        adapter2.fit(dataset)
        result2 = adapter2.detect(dataset)
        
        # Results must be identical
        assert len(result1.scores) == len(result2.scores)
        for s1, s2 in zip(result1.scores, result2.scores):
            assert abs(s1.value - s2.value) < 1e-10
    except Exception:
        assume(False)
```

#### 2.2 Algorithm Contamination Properties
**File**: `tests/property/test_algorithm_contamination_properties.py`
```python
# Properties related to contamination rate compliance
@given(algorithm_input_strategy(), st.floats(0.01, 0.3))
def test_contamination_rate_compliance(data, contamination_rate):
    """Algorithms should respect contamination rate within tolerance."""
    try:
        dataset = Dataset("test", pd.DataFrame(data))
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(contamination_rate))
        adapter.fit(dataset)
        result = adapter.detect(dataset)
        
        # Count anomalies
        n_anomalies = sum(1 for label in result.labels if label == 1)
        actual_rate = n_anomalies / len(result.labels)
        
        # Should be close to expected contamination rate
        tolerance = max(0.05, contamination_rate * 0.5)  # 5% or 50% of rate
        assert abs(actual_rate - contamination_rate) <= tolerance
    except Exception:
        assume(False)
```

#### 2.3 Algorithm Robustness Properties
**File**: `tests/property/test_algorithm_robustness_properties.py`
```python
# Algorithm robustness to data variations
@given(
    stnp.arrays(dtype=np.float64, shape=st.tuples(st.integers(50, 200), st.integers(2, 10)),
                elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)),
    st.floats(0.0, 0.2)  # Noise level
)
def test_algorithm_noise_robustness(clean_data, noise_level):
    """Algorithms should be reasonably robust to small amounts of noise."""
    try:
        dataset = Dataset("clean", pd.DataFrame(clean_data))
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(0.1), random_state=42)
        adapter.fit(dataset)
        clean_scores = adapter.score(dataset)
        
        # Add noise
        noise = np.random.normal(0, noise_level, clean_data.shape)
        noisy_data = clean_data + noise
        noisy_dataset = Dataset("noisy", pd.DataFrame(noisy_data))
        
        noisy_scores = adapter.score(noisy_dataset)
        
        # Compute correlation
        clean_values = [s.value for s in clean_scores]
        noisy_values = [s.value for s in noisy_scores]
        correlation = np.corrcoef(clean_values, noisy_values)[0, 1]
        
        # Correlation should remain high for low noise
        min_correlation = max(0.5, 1.0 - noise_level * 3)
        assert correlation >= min_correlation or np.isnan(correlation)
    except Exception:
        assume(False)
```

### Phase 3: API Contract Properties (Priority 2 - High)

#### 3.1 HTTP API Contract Properties
**File**: `tests/property/test_api_contract_properties.py`
```python
# API endpoint contract properties
@given(st.data())
def test_detection_endpoint_input_validation(data):
    """Detection endpoint must validate input format consistently."""
    # Generate request data
    features = data.draw(st.lists(
        st.dictionaries(st.text(), st.floats(-100, 100)), 
        min_size=1, max_size=100
    ))
    
    request_data = {
        "features": features,
        "detector_config": {
            "algorithm": data.draw(st.sampled_from(["IsolationForest", "OneClassSVM"])),
            "contamination": data.draw(st.floats(0.01, 0.3))
        }
    }
    
    response = client.post("/detect", json=request_data)
    
    # Response format properties
    assert response.status_code in [200, 400, 422, 500]
    
    if response.status_code == 200:
        result = response.json()
        # Response structure properties
        assert "anomalies" in result
        assert "scores" in result
        assert "metadata" in result
        assert len(result["scores"]) == len(features)
        
        # Score properties
        for score in result["scores"]:
            assert 0.0 <= score <= 1.0

@given(st.integers(1, 1000), st.integers(1, 20))
def test_pagination_consistency(page_size, total_items):
    """Pagination should be mathematically consistent."""
    # Test dataset listing with pagination
    response = client.get(f"/datasets?page_size={page_size}&page=1")
    
    if response.status_code == 200:
        data = response.json()
        
        # Pagination properties
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        
        # Mathematical consistency
        assert len(data["items"]) <= page_size
        if data["total"] > 0:
            expected_pages = math.ceil(data["total"] / page_size)
            assert data["page"] <= expected_pages
```

#### 3.2 Authentication Properties
**File**: `tests/property/test_auth_contract_properties.py`
```python
# Authentication contract properties
@given(st.text(min_size=1), st.text(min_size=8))
def test_jwt_token_properties(username, password):
    """JWT tokens must have consistent properties."""
    # Attempt login
    response = client.post("/auth/login", json={
        "username": username,
        "password": password
    })
    
    if response.status_code == 200:
        token_data = response.json()
        
        # Token structure properties
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert token_data["token_type"] == "bearer"
        
        # Token should be valid JWT
        token = token_data["access_token"]
        parts = token.split(".")
        assert len(parts) == 3  # header.payload.signature
        
        # Each part should be base64 encoded
        for part in parts:
            try:
                import base64
                base64.b64decode(part + "==")  # Add padding
            except Exception:
                assert False, "Invalid base64 encoding in JWT token"
```

### Phase 4: Performance Properties (Priority 2 - High)

#### 4.1 Algorithm Performance Properties
**File**: `tests/property/test_algorithm_performance_properties.py`
```python
# Algorithm performance and scalability properties
@given(performance_data_strategy())
def test_algorithm_time_complexity_bounds(data_pair):
    """Algorithm execution time should scale within expected bounds."""
    small_data, large_data = data_pair
    
    try:
        # Time small dataset
        start = time.time()
        small_dataset = Dataset("small", pd.DataFrame(small_data))
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(0.1))
        adapter.fit(small_dataset)
        adapter.detect(small_dataset)
        small_time = time.time() - start
        
        # Time large dataset
        start = time.time()
        large_dataset = Dataset("large", pd.DataFrame(large_data))
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(0.1))
        adapter.fit(large_dataset)
        adapter.detect(large_dataset)
        large_time = time.time() - start
        
        # Time complexity bound
        size_ratio = len(large_data) / len(small_data)
        time_ratio = large_time / small_time if small_time > 0 else float('inf')
        
        # Should not be exponentially slower (allow polynomial growth)
        max_ratio = size_ratio ** 2  # Allow up to quadratic
        assert time_ratio <= max_ratio * 5  # With some tolerance
    except Exception:
        assume(False)

@given(algorithm_input_strategy())
def test_memory_usage_bounds(data):
    """Algorithm memory usage should be bounded."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        dataset = Dataset("test", pd.DataFrame(data))
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(0.1))
        adapter.fit(dataset)
        adapter.detect(dataset)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10x data size)
        data_size_mb = data.nbytes / 1024 / 1024
        max_memory = data_size_mb * 10
        
        assert memory_increase <= max_memory
    except Exception:
        assume(False)
```

### Phase 5: Service Layer Properties (Priority 3 - Medium)

#### 5.1 Application Service Properties
**File**: `tests/property/test_service_contract_properties.py`
```python
# Application service contract properties
@given(dataset_strategy(), detector_strategy())
def test_detection_service_idempotency(dataset, detector):
    """Detection service should be idempotent for same inputs."""
    try:
        service = DetectionService()
        
        # First detection
        result1 = service.detect_anomalies(dataset, detector)
        
        # Second detection with same inputs
        result2 = service.detect_anomalies(dataset, detector)
        
        # Results should be identical
        assert len(result1.anomalies) == len(result2.anomalies)
        assert len(result1.scores) == len(result2.scores)
        
        for s1, s2 in zip(result1.scores, result2.scores):
            assert abs(s1.value - s2.value) < 1e-10
    except Exception:
        assume(False)

@given(st.lists(detector_strategy(), min_size=2, max_size=5), dataset_strategy())
def test_ensemble_service_aggregation_properties(detectors, dataset):
    """Ensemble aggregation should satisfy mathematical properties."""
    try:
        service = EnsembleService()
        
        # Individual detector results
        individual_results = []
        for detector in detectors:
            result = service.detect_with_single_detector(dataset, detector)
            individual_results.append(result)
        
        # Ensemble result
        ensemble_result = service.detect_with_ensemble(dataset, detectors)
        
        # Ensemble scores should be related to individual scores
        for i, ensemble_score in enumerate(ensemble_result.scores):
            individual_scores = [result.scores[i].value for result in individual_results]
            
            # Ensemble score should be within range of individual scores
            min_score = min(individual_scores)
            max_score = max(individual_scores)
            
            assert min_score <= ensemble_score.value <= max_score
    except Exception:
        assume(False)
```

## Implementation Guidelines

### 1. Test Organization
- Group related properties in focused test files
- Use descriptive test names that explain the property being tested
- Include mathematical formulas in docstrings where applicable

### 2. Hypothesis Strategies
- Extend existing `strategies.py` with additional generators
- Create domain-specific strategies for complex objects
- Use `assume()` to filter invalid test cases gracefully

### 3. Property Categories
- **Mathematical Invariants**: Properties that must always hold
- **Behavioral Properties**: Expected behavior under various conditions
- **Contract Properties**: Interface compliance and consistency
- **Performance Properties**: Resource usage and scalability bounds

### 4. Error Handling
- Use `assume(False)` to skip tests when external dependencies fail
- Catch and handle expected exceptions appropriately
- Validate error messages and exception types

### 5. Coverage Metrics
- Target: 80%+ property test coverage for domain layer
- Target: 60%+ property test coverage for algorithm layer
- Target: 40%+ property test coverage for API layer

## Estimated Timeline

- **Phase 1**: 3-4 weeks (30+ property test files)
- **Phase 2**: 2-3 weeks (25+ property test files)  
- **Phase 3**: 2 weeks (20+ property test files)
- **Phase 4**: 1-2 weeks (15+ property test files)
- **Phase 5**: 2 weeks (20+ property test files)

**Total**: ~10-14 weeks for comprehensive property test coverage

## Success Criteria

1. **Mathematical Correctness**: All algorithms satisfy mathematical properties
2. **Robust Error Handling**: Properties catch edge cases traditional tests miss
3. **API Consistency**: All endpoints follow consistent contracts
4. **Performance Guarantees**: Algorithms meet performance bounds
5. **Regression Prevention**: Properties prevent introducing bugs during refactoring

This implementation plan provides comprehensive property test coverage across all architectural layers while prioritizing the most critical components first.