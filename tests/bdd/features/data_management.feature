Feature: Data Management and Processing
  As a data scientist
  I want to efficiently load and process various data formats
  So that I can prepare data for anomaly detection

  Background:
    Given the Pynomaly system is initialized
    And I have access to various data formats

  Scenario: Loading CSV data with different encodings
    Given I have a CSV file with UTF-8 encoding
    When I load the data using the CSV loader
    Then the data should be loaded correctly
    And the feature matrix should have the expected shape
    And all numeric values should be preserved
    
    Given I have a CSV file with Windows-1252 encoding
    When I load the data specifying the encoding
    Then the data should load without character encoding errors
    And special characters should be preserved

  Scenario: Loading large datasets efficiently
    Given I have a large CSV file (>100MB)
    When I load the data using chunked processing
    Then the data should load without memory errors
    And processing should show progress indicators
    And the final dataset should be complete

  Scenario: Loading Parquet files with metadata
    Given I have a Parquet file with embedded metadata
    When I load the data using the Parquet loader
    Then the data types should be preserved from the file
    And categorical columns should maintain their categories
    And the schema information should be accessible

  Scenario: High-performance loading with Polars
    Given I have a large dataset suitable for Polars
    When I load the data using the Polars loader with lazy evaluation
    Then the initial load should be very fast
    And data processing should be parallelized
    And memory usage should be optimized
    And I can apply filters before materialization

  Scenario: Distributed processing with Spark
    Given I have a very large dataset requiring distributed processing
    When I initialize a Spark session for data loading
    And I load the data using the Spark loader
    Then the data should be distributed across available cores
    And processing should scale with data size
    And I can perform distributed anomaly detection

  Scenario: Data validation and quality checks
    Given I have loaded a dataset
    When I run data validation checks
    Then missing values should be identified and reported
    And data types should be validated
    And outliers beyond reasonable ranges should be flagged
    And duplicate rows should be detected

  Scenario: Feature preprocessing and transformation
    Given I have raw data with mixed types
    When I apply feature preprocessing
    Then categorical variables should be encoded appropriately
    And numerical features should be scaled if requested
    And missing values should be handled according to strategy
    And the feature matrix should be suitable for ML algorithms

  Scenario: Data sampling for large datasets
    Given I have a dataset with 1 million rows
    When I request a representative sample of 10,000 rows
    Then the sample should maintain the original data distribution
    And anomalies should be proportionally represented
    And the sampling should be reproducible with a fixed seed

  Scenario: Multi-format data integration
    Given I have related data in CSV and Parquet formats
    When I load and merge the datasets
    Then the join should work correctly on common keys
    And data types should be aligned appropriately
    And the merged dataset should be validated

  Scenario: Streaming data processing
    Given I have a streaming data source
    When I set up real-time data ingestion
    Then data should be processed as it arrives
    And buffers should be managed to prevent memory overflow
    And processing should maintain consistent throughput

  Scenario: Data export and persistence
    Given I have processed a dataset
    When I export the data to different formats
    Then CSV export should preserve all data and formatting
    And Parquet export should maintain optimal compression
    And the exported data should be identical to the original

  Scenario: Handling corrupted or malformed data
    Given I have a file with some corrupted rows
    When I attempt to load the data with error handling
    Then the system should skip corrupted rows
    And provide detailed error reports
    And continue processing valid data
    And log all data quality issues

  Scenario: Memory-efficient data processing
    Given I have limited available memory
    And I need to process a large dataset
    When I use memory-efficient processing options
    Then the system should use streaming or chunked processing
    And memory usage should stay within specified limits
    And processing should complete successfully

  Scenario: Data lineage and provenance tracking
    Given I have loaded and processed data through multiple steps
    When I query the data lineage
    Then I should see the complete processing history
    And transformation steps should be documented
    And original data sources should be traceable