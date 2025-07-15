Feature: Anomaly Detection Workflow
  As a data scientist
  I want to detect anomalies in my datasets
  So that I can identify unusual patterns and outliers

  Background:
    Given I have a clean dataset with known anomalies
    And the Pynomaly system is properly configured

  Scenario: Basic anomaly detection with default settings
    Given I have a dataset with 100 samples and 2 features
    And the dataset contains 10% anomalies
    When I create an Isolation Forest detector with default contamination
    And I train the detector on the dataset
    And I run anomaly detection
    Then the detector should identify approximately 10 anomalies
    And all anomaly scores should be between 0 and 1
    And anomalies should have higher scores than normal points

  Scenario: Anomaly detection with custom contamination rate
    Given I have a dataset with 200 samples and 3 features
    And the dataset contains 5% anomalies
    When I create a Local Outlier Factor detector with 0.05 contamination
    And I train the detector on the dataset
    And I run anomaly detection
    Then the detector should identify approximately 10 anomalies
    And the contamination rate should be respected
    And the results should be deterministic with fixed random seed

  Scenario: Multi-algorithm ensemble detection
    Given I have a dataset with 150 samples and 4 features
    And the dataset contains mixed anomaly types
    When I create an ensemble with Isolation Forest, LOF, and One-Class SVM
    And I train all detectors on the dataset
    And I run ensemble anomaly detection with majority voting
    Then the ensemble should outperform individual detectors
    And the final scores should be aggregated properly
    And confidence intervals should be provided for anomalies

  Scenario: Real-time anomaly detection
    Given I have a streaming data source
    And I have a pre-trained anomaly detector
    When new data points arrive one by one
    And I score each point for anomalies
    Then each score should be computed quickly (< 100ms)
    And scores should be consistent with batch processing
    And the system should handle data drift gracefully

  Scenario: Anomaly detection with missing data
    Given I have a dataset with some missing values
    When I attempt to run anomaly detection
    Then the system should handle missing values appropriately
    And either impute the values or exclude incomplete samples
    And provide clear warnings about data quality issues

  Scenario: Large dataset anomaly detection
    Given I have a large dataset with 10,000 samples
    And the dataset has moderate dimensionality (20 features)
    When I run anomaly detection with memory-efficient settings
    Then the detection should complete within reasonable time
    And memory usage should remain bounded
    And results should be saved incrementally if needed

  Scenario Outline: Algorithm-specific detection patterns
    Given I have a dataset optimized for <algorithm>
    When I create a <algorithm> detector with appropriate parameters
    And I run anomaly detection
    Then the algorithm should detect the expected anomaly pattern
    And performance metrics should meet minimum thresholds

    Examples:
      | algorithm           |
      | isolation_forest    |
      | local_outlier_factor|
      | one_class_svm       |
      | pyod_abod          |
      | pyod_knn           |

  Scenario: Anomaly explanation and interpretability
    Given I have detected anomalies in my dataset
    When I request explanations for the top anomalies
    Then each anomaly should have feature importance scores
    And explanations should highlight the most contributing features
    And the explanations should be human-readable

  Scenario: Anomaly detection model persistence
    Given I have trained an anomaly detector
    When I save the model to disk
    And load the model in a new session
    Then the loaded model should produce identical results
    And all hyperparameters should be preserved
    And the model metadata should be intact

  Scenario: Handling edge cases in anomaly detection
    Given I have datasets with edge cases
    When I run anomaly detection on data with all identical values
    Then the system should handle the case gracefully
    And provide appropriate warnings or default behavior
    When I run detection on a single data point
    Then the system should either process it or give clear error messages
    When I provide data with extreme outliers (1000x normal range)
    Then the algorithm should remain stable and not crash
