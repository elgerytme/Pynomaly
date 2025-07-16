Feature: Complete Anomaly Detection User Workflows
  As a data scientist
  I want to use Pynomaly's web interface
  So that I can detect anomalies in my data through an intuitive workflow

  Background:
    Given I am a data scientist working with anomaly detection
    And the Pynomaly web application is running

  Scenario: Complete Data Scientist Research Workflow
    Given I have a dataset with known anomalies
    When I navigate to the datasets page
    And I upload my dataset
    Then I should see the uploaded dataset in the datasets list
    When I create a new anomaly detector
    And I train the detector with my dataset
    Then I should see the detector training successfully
    When I run anomaly detection on new data
    Then I should see anomaly detection results
    And I should see visualizations of the anomalies
    And I should be able to export the results

  Scenario: Dataset Upload and Management
    Given I have a dataset with known anomalies
    When I navigate to the datasets page
    And I upload my dataset
    Then I should see the uploaded dataset in the datasets list
    And I should be able to view dataset statistics
    And I should be able to preview the dataset content

  Scenario: Detector Creation and Configuration
    Given I have a dataset with known anomalies
    And the dataset is already uploaded
    When I navigate to the detectors page
    And I create a new detector with Isolation Forest algorithm
    And I configure the detector parameters
    Then I should see the detector in the detectors list
    And I should be able to view detector configuration
    And I should be able to edit detector settings

  Scenario: Model Training and Validation
    Given I have a trained detector
    When I start the training process
    Then I should see training progress indicators
    And I should see training completion notification
    And I should be able to view training metrics
    And I should be able to validate model performance

  Scenario: Real-time Anomaly Detection
    Given I have a trained detector
    When I navigate to the detection page
    And I select my trained detector
    And I upload new data for detection
    Then I should see real-time detection results
    And I should see anomaly scores for each data point
    And I should see highlighted anomalous records

  Scenario: Visualization and Analysis
    Given I have detection results
    When I navigate to the visualizations page
    Then I should see scatter plots of the data
    And I should see anomaly scores distribution
    And I should see time series plots if applicable
    And I should be able to interact with the visualizations
    And I should be able to filter and zoom the charts

  Scenario: Results Export and Reporting
    Given I have detection results
    When I navigate to the export page
    Then I should be able to export results as CSV
    And I should be able to export results as JSON
    And I should be able to generate a PDF report
    And I should be able to schedule automated reports

  Scenario: Error Handling - Invalid File Upload
    Given I am on the datasets page
    When I try to upload an invalid file
    Then I should see an appropriate error message
    And the system should remain stable
    And I should be able to upload a valid file afterward

  Scenario: Error Handling - Insufficient Data
    Given I have a very small dataset
    When I try to create a detector
    And I attempt to train with insufficient data
    Then I should see a warning about data size
    And I should get recommendations for minimum data requirements

  Scenario: Responsive Design - Mobile Usage
    Given I am using a mobile device
    When I navigate through the application
    Then all pages should be mobile-friendly
    And navigation should work with touch
    And forms should be easily fillable
    And visualizations should be responsive

  Scenario: Performance - Large Dataset Handling
    Given I have a large dataset (10k+ records)
    When I upload and process the dataset
    Then the upload should complete within reasonable time
    And the interface should remain responsive
    And progress indicators should show status
    And I should be able to cancel long-running operations

  Scenario: Security Analyst Monitoring Workflow
    Given I am a security analyst
    And I have network traffic data
    When I create an anomaly detector for security monitoring
    And I configure real-time monitoring
    Then I should see real-time anomaly alerts
    And I should be able to investigate flagged events
    And I should be able to mark false positives
    And I should be able to create custom alert rules

  Scenario: ML Engineer Deployment Workflow
    Given I am an ML engineer
    And I have a trained model
    When I navigate to the deployment page
    And I configure model deployment settings
    Then I should be able to deploy the model to production
    And I should see deployment status
    And I should be able to monitor model performance
    And I should be able to roll back if needed

  Scenario: Business User Reporting Workflow
    Given I am a business user
    When I navigate to the dashboard
    Then I should see high-level metrics
    And I should see trend analysis
    And I should be able to drill down into details
    And I should be able to create custom reports
    And I should be able to share reports with colleagues

  Scenario: Collaborative Features
    Given I am working with a team
    When I create a detector project
    Then I should be able to share it with team members
    And team members should be able to view results
    And we should be able to add comments and annotations
    And we should see activity history and changes

  Scenario: Cross-browser Compatibility
    Given I am using different browsers
    When I access the application
    Then all features should work in Chrome
    And all features should work in Firefox
    And all features should work in Safari
    And all features should work in Edge
    And visual consistency should be maintained

  Scenario: Accessibility Compliance
    Given I am using assistive technology
    When I navigate the application
    Then all content should be accessible via screen reader
    And all interactive elements should be keyboard accessible
    And color contrast should meet WCAG guidelines
    And form labels should be properly associated
    And error messages should be announced clearly

  Scenario: Progressive Web App Features
    Given I am using the web application
    When I install it as a PWA
    Then it should work offline for basic functionality
    And it should sync data when connection is restored
    And I should receive push notifications for alerts
    And it should have app-like navigation
    And it should be installable on mobile devices
