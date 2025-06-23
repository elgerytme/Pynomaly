Feature: API Integration and Workflows
  As a developer
  I want to use the Pynomaly REST API
  So that I can integrate anomaly detection into my applications

  Background:
    Given the Pynomaly API server is running
    And I have valid API credentials

  Scenario: API authentication and authorization
    Given I have API credentials
    When I request an access token with valid credentials
    Then I should receive a valid JWT token
    And the token should have appropriate expiration time
    When I use the token to access protected endpoints
    Then I should be able to access the resources
    When the token expires
    Then I should be denied access to protected resources

  Scenario: Dataset management via API
    Given I am authenticated with the API
    When I upload a dataset via POST /api/datasets
    Then the dataset should be stored successfully
    And I should receive a dataset ID
    When I retrieve the dataset via GET /api/datasets/{id}
    Then I should get the complete dataset information
    When I list all datasets via GET /api/datasets
    Then my uploaded dataset should appear in the list

  Scenario: Detector creation and configuration
    Given I have a dataset uploaded to the API
    When I create a detector via POST /api/detectors
    With algorithm "isolation_forest" and contamination 0.1
    Then the detector should be created successfully
    And I should receive a detector ID
    When I retrieve the detector via GET /api/detectors/{id}
    Then the detector configuration should match my specifications

  Scenario: Training detectors via API
    Given I have a dataset and detector created via API
    When I start training via POST /api/detectors/{id}/train
    Then the training should start successfully
    And I should receive a training job ID
    When I poll the training status via GET /api/detectors/{id}/status
    Then I should see the training progress
    When training completes
    Then the detector status should be "trained"

  Scenario: Running anomaly detection via API
    Given I have a trained detector
    And I have data to analyze
    When I submit detection request via POST /api/detectors/{id}/detect
    Then the detection should execute successfully
    And I should receive anomaly scores and predictions
    And the response should include confidence intervals
    And anomalies should be ranked by score

  Scenario: Batch processing via API
    Given I have multiple datasets to process
    When I submit a batch detection job via POST /api/batch/detect
    Then the batch job should be queued
    And I should receive a batch job ID
    When I monitor the batch job via GET /api/batch/{id}/status
    Then I should see progress updates
    When the batch job completes
    Then I should be able to download all results

  Scenario: API rate limiting and throttling
    Given I am making API requests
    When I exceed the rate limit for my tier
    Then I should receive HTTP 429 Too Many Requests
    And the response should include retry-after headers
    When I wait for the rate limit window to reset
    Then I should be able to make requests again

  Scenario: API error handling and validation
    Given I am using the API
    When I submit invalid data (missing required fields)
    Then I should receive HTTP 400 Bad Request
    And the error message should specify what's wrong
    When I try to access a non-existent resource
    Then I should receive HTTP 404 Not Found
    When I submit data that causes processing errors
    Then I should receive HTTP 422 Unprocessable Entity
    And the error should include helpful debugging information

  Scenario: API versioning and compatibility
    Given I am using API version 1
    When I make requests with version header "v1"
    Then I should receive responses in v1 format
    When a new API version is released
    Then v1 should continue to work for backward compatibility
    And I should be able to gradually migrate to the new version

  Scenario: Asynchronous processing via webhooks
    Given I have configured a webhook URL
    When I submit a long-running detection job
    Then the API should return immediately with a job ID
    And when the job completes
    Then my webhook should receive a notification
    And the webhook payload should include job results

  Scenario: API monitoring and health checks
    When I check the API health via GET /api/health
    Then I should receive status information
    And the response should include service dependencies status
    When I check detailed metrics via GET /api/metrics
    Then I should receive performance and usage metrics
    And the metrics should be in Prometheus format

  Scenario: Data privacy and security
    Given I upload sensitive data via the API
    When the data is processed
    Then it should be encrypted at rest
    And access should be logged for audit trails
    When I delete a dataset via DELETE /api/datasets/{id}
    Then the data should be permanently removed
    And I should receive confirmation of deletion

  Scenario: API documentation and discoverability
    When I access the API documentation via GET /api/docs
    Then I should see interactive Swagger/OpenAPI documentation
    And all endpoints should be documented with examples
    And authentication requirements should be clearly specified
    When I use the interactive docs to test endpoints
    Then the requests should work correctly