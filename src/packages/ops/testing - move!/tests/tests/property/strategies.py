"""Hypothesis strategies for generating test data for Pynomaly domain objects."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp

from monorepo.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from monorepo.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)


# Basic value strategies
@st.composite
def contamination_rate_strategy(draw):
    """Generate valid ContaminationRate instances."""
    rate = draw(
        st.floats(min_value=0.0, max_value=0.5, exclude_min=False, exclude_max=False)
    )
    return ContaminationRate(rate)


@st.composite
def anomaly_score_strategy(draw):
    """Generate valid AnomalyScore instances."""
    score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return AnomalyScore(score)


@st.composite
def confidence_interval_strategy(draw):
    """Generate valid ConfidenceInterval instances."""
    lower = draw(
        st.floats(min_value=0.0, max_value=0.95, allow_nan=False, allow_infinity=False)
    )
    upper = draw(
        st.floats(min_value=lower, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    confidence_level = draw(st.floats(min_value=0.01, max_value=0.99))
    return ConfidenceInterval(
        lower=lower, upper=upper, confidence_level=confidence_level
    )


# Entity strategies
@st.composite
def dataset_strategy(draw):
    """Generate valid Dataset instances."""
    # Generate feature data
    n_samples = draw(st.integers(min_value=10, max_value=1000))
    n_features = draw(st.integers(min_value=1, max_value=50))

    features = draw(
        stnp.arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Optional target data
    targets = draw(
        st.one_of(
            st.none(),
            stnp.arrays(
                dtype=np.int32,
                shape=(n_samples,),
                elements=st.integers(min_value=0, max_value=1),
            ),
        )
    )

    name = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        )
    )

    return Dataset(name=name, features=features, targets=targets)


@st.composite
def detector_strategy(draw):
    """Generate valid Detector instances."""
    name = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        )
    )
    algorithm = draw(
        st.sampled_from(
            [
                "isolation_forest",
                "local_outlier_factor",
                "one_class_svm",
                "pyod_abod",
                "pyod_knn",
                "pyod_lof",
                "pyod_ocsvm",
            ]
        )
    )
    contamination = draw(contamination_rate_strategy())

    # Optional hyperparameters
    hyperparameters = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(
                    st.floats(min_value=0.1, max_value=10.0),
                    st.integers(min_value=1, max_value=100),
                    st.booleans(),
                    st.text(min_size=1, max_size=20),
                ),
                min_size=0,
                max_size=5,
            ),
        )
    )

    return Detector(
        name=name,
        algorithm=algorithm,
        contamination=contamination,
        hyperparameters=hyperparameters,
    )


@st.composite
def anomaly_strategy(draw):
    """Generate valid Anomaly instances."""
    score = draw(anomaly_score_strategy())
    index = draw(st.integers(min_value=0, max_value=10000))
    confidence = draw(st.one_of(st.none(), confidence_interval_strategy()))

    # Optional feature values
    features = draw(
        st.one_of(
            st.none(),
            stnp.arrays(
                dtype=np.float64,
                shape=draw(st.integers(min_value=1, max_value=20)),
                elements=st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
        )
    )

    # Optional explanation
    explanation = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.floats(min_value=-1.0, max_value=1.0),
                min_size=1,
                max_size=10,
            ),
        )
    )

    return Anomaly(
        score=score,
        index=index,
        confidence=confidence,
        features=features,
        explanation=explanation,
    )


@st.composite
def detection_result_strategy(draw):
    """Generate valid DetectionResult instances."""
    detector = draw(detector_strategy())
    dataset = draw(dataset_strategy())

    # Generate anomalies for some indices
    n_anomalies = draw(
        st.integers(min_value=0, max_value=min(10, len(dataset.features)))
    )
    anomaly_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(dataset.features) - 1),
            min_size=n_anomalies,
            max_size=n_anomalies,
            unique=True,
        )
    )

    anomalies = []
    for idx in anomaly_indices:
        anomaly = draw(anomaly_strategy())
        # Override index to match dataset
        anomaly = Anomaly(
            score=anomaly.score,
            index=idx,
            confidence=anomaly.confidence,
            features=dataset.features[idx] if anomaly.features is not None else None,
            explanation=anomaly.explanation,
        )
        anomalies.append(anomaly)

    # Generate scores for all samples
    scores = draw(
        stnp.arrays(
            dtype=np.float64,
            shape=(len(dataset.features),),
            elements=st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return DetectionResult(
        detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
    )


# Algorithm input strategies
@st.composite
def algorithm_input_strategy(draw):
    """Generate valid algorithm input data."""
    n_samples = draw(st.integers(min_value=10, max_value=500))
    n_features = draw(st.integers(min_value=1, max_value=20))

    # Ensure data is well-formed for ML algorithms
    data = draw(
        stnp.arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return data


@st.composite
def contamination_strategy(draw):
    """Generate valid contamination rates for algorithms."""
    return draw(
        st.floats(min_value=0.001, max_value=0.5, exclude_min=False, exclude_max=False)
    )


# Performance testing strategies
@st.composite
def performance_data_strategy(draw):
    """Generate data for performance property testing."""
    small_size = draw(st.integers(min_value=50, max_value=200))
    large_size = draw(st.integers(min_value=small_size * 2, max_value=small_size * 10))
    n_features = draw(st.integers(min_value=2, max_value=10))

    small_data = draw(
        stnp.arrays(
            dtype=np.float64,
            shape=(small_size, n_features),
            elements=st.floats(
                min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    large_data = draw(
        stnp.arrays(
            dtype=np.float64,
            shape=(large_size, n_features),
            elements=st.floats(
                min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return small_data, large_data
