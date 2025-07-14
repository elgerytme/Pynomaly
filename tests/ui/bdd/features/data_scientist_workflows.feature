Feature: Data Scientist Research and Analysis Workflows
  As a data scientist
  I want to conduct comprehensive anomaly detection research
  So that I can discover insights and build robust detection models

  Background:
    Given I am a data scientist with expertise in anomaly detection
    And the Pynomaly web application is running and accessible
    And I have appropriate permissions for data analysis

  @smoke @critical
  Scenario: Complete Research Workflow - Financial Fraud Detection
    Given I have a financial transactions dataset with known fraud cases
    And the dataset contains 50,000 transactions with 2% fraud rate
    When I navigate to the datasets page
    And I upload the dataset named "financial_fraud_detection.csv"
    Then I should see the upload progress indicator
    And I should see "Dataset uploaded successfully" message within 30 seconds
    And I should see the dataset "financial_fraud_detection.csv" in the datasets list
    And I should see dataset statistics showing "50,000 records, 15 features"

    When I click on "View Dataset Details"
    Then I should see the dataset preview with first 10 records
    And I should see feature distribution charts
    And I should see data quality metrics
    And I should see recommendations for preprocessing

    When I navigate to the detectors page
    And I click "Create New Detector"
    And I fill in the detector name as "Fraud Detection Model"
    And I select "Isolation Forest" algorithm
    And I select the dataset "financial_fraud_detection.csv"
    And I configure contamination rate to 0.02
    And I click "Create Detector"
    Then I should see "Detector created successfully" message
    And I should see the detector "Fraud Detection Model" in the detectors list

    When I click "Train Detector" for "Fraud Detection Model"
    Then I should see training progress indicator
    And I should see "Training in progress..." status
    And training should complete within 60 seconds
    And I should see "Training completed successfully" message
    And I should see training metrics including accuracy and F1-score

    When I navigate to the detection page
    And I select the trained detector "Fraud Detection Model"
    And I upload new transaction data for detection
    Then I should see detection results within 10 seconds
    And I should see anomaly scores for each transaction
    And I should see highlighted fraudulent transactions
    And I should see confidence intervals for predictions

    When I navigate to the visualizations page
    Then I should see scatter plot of anomaly scores
    And I should see feature importance chart
    And I should see ROC curve analysis
    And I should see precision-recall curve
    And I should be able to filter results by confidence threshold

    When I navigate to the export page
    And I select CSV format
    And I click "Export Results"
    Then I should receive a download of "fraud_detection_results.csv"
    And the file should contain all detected anomalies with scores
    And the file should include confidence metrics

  @analysis @advanced
  Scenario: Multi-Algorithm Comparison Study
    Given I have a benchmark dataset for anomaly detection
    When I create multiple detectors with different algorithms:
      | Algorithm | Contamination | Features |
      | Isolation Forest | 0.05 | all |
      | Local Outlier Factor | 0.05 | all |
      | One-Class SVM | 0.05 | all |
      | Autoencoder | 0.05 | all |
    Then I should see all detectors created successfully

    When I train all detectors on the same dataset
    Then all training should complete successfully
    And I should see training metrics for each detector

    When I run detection with all trained detectors
    Then I should see detection results for each algorithm
    And I should see performance comparison metrics
    And I should see ensemble voting results
    And I should be able to export comparison report

  @preprocessing @data-quality
  Scenario: Advanced Data Preprocessing Workflow
    Given I have a raw dataset with quality issues
    And the dataset contains missing values, outliers, and mixed data types
    When I upload the dataset
    Then I should see data quality warnings
    And I should see preprocessing recommendations

    When I click "Apply Preprocessing"
    And I select the following options:
      | Option | Setting |
      | Handle Missing Values | Forward Fill |
      | Outlier Treatment | IQR Method |
      | Feature Scaling | StandardScaler |
      | Feature Selection | Variance Threshold |
    And I click "Apply Preprocessing"
    Then I should see preprocessing progress indicator
    And I should see "Preprocessing completed" message
    And I should see before/after data quality metrics
    And I should see feature transformation summary

  @autonomous @intelligent
  Scenario: Autonomous Algorithm Selection
    Given I have an unlabeled dataset for anomaly detection
    When I navigate to the autonomous detection page
    And I upload my dataset
    And I click "Start Autonomous Analysis"
    Then I should see "Analyzing dataset characteristics..." message
    And I should see algorithm recommendations within 30 seconds
    And I should see reasoning for each recommendation
    And I should see expected performance estimates

    When I accept the top recommendation
    And I click "Run Autonomous Detection"
    Then I should see automatic model training progress
    And I should see hyperparameter optimization in progress
    And I should see final model performance metrics
    And I should see detection results with explanations

  @explainability @interpretation
  Scenario: Model Explainability and Interpretation
    Given I have a trained anomaly detection model
    And the model has detected several anomalies
    When I navigate to the explainability page
    And I select a detected anomaly
    Then I should see SHAP values for the anomaly
    And I should see feature contribution chart
    And I should see local explanation details
    And I should see similar anomalies for comparison

    When I click "Global Explanations"
    Then I should see overall feature importance
    And I should see model behavior analysis
    And I should see decision boundary visualization
    And I should see model confidence regions

  @time-series @temporal
  Scenario: Time Series Anomaly Detection
    Given I have time series data with temporal patterns
    And the data includes timestamps and multiple metrics
    When I upload the time series dataset
    Then I should see time series visualization
    And I should see seasonal pattern analysis
    And I should see trend detection results

    When I create a time series anomaly detector
    And I configure temporal parameters:
      | Parameter | Value |
      | Window Size | 24 hours |
      | Seasonality | Weekly |
      | Trend Sensitivity | Medium |
    And I train the detector
    Then I should see time series model metrics
    And I should see forecasting accuracy

    When I run detection on new time series data
    Then I should see temporal anomalies highlighted
    And I should see anomaly timeline visualization
    And I should see contextual explanations for each anomaly

  @streaming @real-time
  Scenario: Real-time Streaming Detection
    Given I have a trained model for real-time detection
    When I navigate to the streaming page
    And I configure streaming parameters:
      | Parameter | Value |
      | Buffer Size | 1000 records |
      | Update Frequency | 10 seconds |
      | Alert Threshold | 0.8 |
    And I start streaming detection
    Then I should see live data visualization
    And I should see real-time anomaly alerts
    And I should see streaming performance metrics

    When an anomaly is detected in the stream
    Then I should receive an immediate alert
    And I should see the anomaly highlighted in real-time
    And I should see confidence score for the detection
    And I should be able to mark it as false positive

  @collaboration @teamwork
  Scenario: Research Collaboration Features
    Given I am working on a research project with colleagues
    When I create a new project workspace
    And I invite team members to the project
    Then team members should receive collaboration invitations

    When I share a detector with the team
    And I add research notes and observations
    Then team members should see the shared detector
    And team members should see my research notes
    And team members should be able to add their own comments

    When a team member makes changes to the model
    Then I should see version history
    And I should see change notifications
    And I should be able to compare different versions
    And I should be able to merge improvements

  @performance @scalability
  Scenario: Large-scale Data Analysis
    Given I have a large dataset with 1 million records
    When I upload the dataset
    Then I should see chunked upload progress
    And upload should complete within 5 minutes
    And I should see memory usage indicators

    When I train a detector on the large dataset
    Then I should see distributed training options
    And I should see training progress with ETA
    And training should utilize available computing resources efficiently
    And I should see scalability recommendations

  @validation @statistical
  Scenario: Statistical Validation and Cross-validation
    Given I have a labeled dataset for validation
    When I configure cross-validation settings:
      | Setting | Value |
      | Folds | 5 |
      | Stratification | Yes |
      | Random Seed | 42 |
    And I run cross-validation
    Then I should see validation results for each fold
    And I should see average performance metrics
    And I should see confidence intervals
    And I should see statistical significance tests

    When I compare multiple models using validation
    Then I should see statistical comparison results
    And I should see model ranking with significance
    And I should see recommendations based on validation

  @error-handling @resilience
  Scenario: Robust Error Handling During Research
    Given I am conducting anomaly detection research
    When I upload a corrupted dataset
    Then I should see a clear error message about data corruption
    And I should see suggestions for data repair
    And the system should remain stable

    When training fails due to insufficient memory
    Then I should see memory usage warnings
    And I should see recommendations for reducing memory usage
    And I should be able to retry with different settings

    When detection is interrupted by network issues
    Then I should see connection status indicators
    And I should be able to resume detection when connection is restored
    And I should not lose any previous work or results

  @accessibility @inclusive
  Scenario: Accessible Research Interface
    Given I am using screen reader technology
    When I navigate through the research workflow
    Then all data tables should be properly announced
    And all charts should have alternative text descriptions
    And all form controls should have clear labels
    And keyboard navigation should work throughout

    When I interact with data visualizations
    Then I should be able to access chart data via keyboard
    And I should hear audio descriptions of trends
    And I should be able to navigate data points sequentially
    And I should receive meaningful descriptions of patterns
