# CI/CD Integration Guide

This guide explains how to integrate the Pynomaly Performance Regression Testing Framework into various CI/CD pipelines and automation workflows.

## GitHub Actions Integration

### Workflow Overview

The framework includes a comprehensive GitHub Actions workflow that automatically runs performance tests and provides detailed reporting.

**Workflow File:** `.github/workflows/performance-regression-testing.yml`

### Triggers

The workflow automatically runs on:

1. **Push events** to `main` and `develop` branches
2. **Pull requests** to `main` and `develop` branches  
3. **Scheduled runs** daily at 2 AM UTC
4. **Manual workflow dispatch** with configurable parameters

### Manual Workflow Dispatch

You can manually trigger performance tests with custom parameters:

```yaml
workflow_dispatch:
  inputs:
    test_duration:
      description: 'Test duration in seconds'
      required: false
      default: '30'
      type: string
    concurrent_users:
      description: 'Number of concurrent users'
      required: false
      default: '5'
      type: string
    fail_on_regression:
      description: 'Fail build on performance regression'
      required: false
      default: true
      type: boolean
    establish_baselines:
      description: 'Establish new baselines from results'
      required: false
      default: false
      type: boolean
```

### Workflow Configuration

#### Matrix Strategy

The workflow uses a matrix strategy to run different test scenarios in parallel:

```yaml
strategy:
  fail-fast: false
  matrix:
    test-scenario:
      - name: 'API Performance'
        config: 'api_performance'
        timeout: 25
      - name: 'Database Performance' 
        config: 'database_performance'
        timeout: 20
      - name: 'System Resources'
        config: 'system_resources'
        timeout: 15
```

#### Environment Variables

The workflow sets up necessary environment variables:

```yaml
env:
  PYTHON_VERSION: '3.12'
  NODE_VERSION: '18'
  PYNOMALY_TEST_URL: "http://localhost:8000"
  PERF_TEST_DURATION: "${{ github.event.inputs.test_duration || '30' }}"
  PERF_CONCURRENT_USERS: "${{ github.event.inputs.concurrent_users || '5' }}"
  GITHUB_TOKEN: "${{ secrets.GITHUB_TOKEN }}"
```

### Key Workflow Steps

1. **Environment Setup**
   - Install Python and Node.js
   - Install dependencies including k6 for load testing
   - Setup test databases and directories

2. **Service Startup**
   - Start the Pynomaly application server
   - Wait for service health checks to pass
   - Verify API endpoints are accessible

3. **Performance Testing**
   - Load historical baselines
   - Execute performance tests with configured scenarios
   - Record metrics and analyze regressions

4. **Reporting and Artifacts**
   - Generate HTML and JSON reports
   - Upload artifacts for 30-day retention
   - Comment on PRs with test results

5. **Baseline Management**
   - Update baselines on main branch commits
   - Commit baseline changes back to repository

### PR Comments

The workflow automatically comments on pull requests with performance results:

```markdown
## ✅ Performance Regression Test Results

**Status:** PASSED  
**Test Scenario:** API Performance  
**Duration:** 45.2s

### 📊 Summary

| Metric | Value |
|--------|-------|
| Total Tests | 3 |
| Successful Tests | 3 |
| Regressions | 0 |
| Critical Regressions | 0 |
| Improvements | 1 |
| Baseline Health | 0.85 |

### 💡 Recommendations

- ✅ All performance metrics are within expected ranges.
- 🎉 1 performance improvement detected!
```

## GitLab CI Integration

### GitLab CI Configuration

Create `.gitlab-ci.yml` with performance testing stages:

```yaml
stages:
  - test
  - performance
  - deploy

variables:
  PYTHON_VERSION: "3.12"
  PYNOMALY_TEST_URL: "http://localhost:8000"

performance-regression:
  stage: performance
  image: python:${PYTHON_VERSION}
  services:
    - postgres:13
    - redis:6.2
  variables:
    POSTGRES_DB: pynomaly_test
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: "postgresql://test:test@postgres:5432/pynomaly_test"
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - # Start application
    - python -m uvicorn src.pynomaly.presentation.api.main:app --host 0.0.0.0 --port 8000 &
    - sleep 10
    - # Run performance tests
    - python scripts/ci/performance_regression_check.py 
        --config performance_test_config.json
        --output performance_report.json
        --fail-on-regression
  artifacts:
    reports:
      performance: performance_report.json
    paths:
      - performance_reports/
      - performance_baselines/
    expire_in: 30 days
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_PIPELINE_SOURCE == "schedule"

performance-baseline-update:
  stage: performance
  extends: performance-regression
  script:
    - python scripts/ci/performance_regression_check.py
        --config performance_test_config.json
        --establish-baselines
  rules:
    - if: $CI_COMMIT_BRANCH == "main" && $CI_COMMIT_MESSAGE =~ /\[update-baselines\]/
  artifacts:
    paths:
      - performance_baselines/
```

### GitLab Performance Reports

GitLab can display performance metrics in merge requests:

```yaml
artifacts:
  reports:
    performance: performance_report.json
```

The `performance_report.json` should follow GitLab's performance report format:

```json
[
  {
    "name": "response_time_mean",
    "value": 125.5,
    "unit": "ms",
    "desiredSize": "smaller"
  },
  {
    "name": "throughput",
    "value": 45.2,
    "unit": "req/s", 
    "desiredSize": "larger"
  }
]
```

## Jenkins Integration

### Jenkinsfile Pipeline

```groovy
pipeline {
    agent any
    
    parameters {
        string(name: 'TEST_DURATION', defaultValue: '30', description: 'Test duration in seconds')
        string(name: 'CONCURRENT_USERS', defaultValue: '5', description: 'Number of concurrent users')
        booleanParam(name: 'FAIL_ON_REGRESSION', defaultValue: true, description: 'Fail build on regression')
        booleanParam(name: 'ESTABLISH_BASELINES', defaultValue: false, description: 'Establish new baselines')
    }
    
    environment {
        PYTHON_VERSION = '3.12'
        PYNOMALY_TEST_URL = 'http://localhost:8000'
        PERF_TEST_DURATION = "${params.TEST_DURATION}"
        PERF_CONCURRENT_USERS = "${params.CONCURRENT_USERS}"
    }
    
    stages {
        stage('Setup') {
            steps {
                // Install Python dependencies
                sh 'pip install -r requirements.txt'
                sh 'pip install -r requirements-dev.txt'
                
                // Setup test environment
                sh 'mkdir -p performance_baselines performance_reports'
            }
        }
        
        stage('Start Services') {
            steps {
                // Start application server
                sh '''
                python -m uvicorn src.pynomaly.presentation.api.main:app --host 0.0.0.0 --port 8000 &
                APP_PID=$!
                echo $APP_PID > app.pid
                
                # Wait for service to be ready
                timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 3; done'
                '''
            }
        }
        
        stage('Performance Tests') {
            steps {
                script {
                    def baselineFlag = params.ESTABLISH_BASELINES ? '--establish-baselines' : ''
                    def failFlag = params.FAIL_ON_REGRESSION ? '--fail-on-regression' : ''
                    
                    sh """
                    python scripts/ci/performance_regression_check.py \\
                        --config performance_test_config.json \\
                        --output performance_results/report_${BUILD_NUMBER}.json \\
                        --format json \\
                        ${baselineFlag} \\
                        ${failFlag} \\
                        --verbose
                    """
                }
            }
            post {
                always {
                    // Archive performance artifacts
                    archiveArtifacts artifacts: 'performance_results/*, performance_reports/*, performance_baselines/*', 
                                   allowEmptyArchive: true
                    
                    // Publish performance test results
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'performance_reports',
                        reportFiles: '*.html',
                        reportName: 'Performance Report'
                    ])
                }
            }
        }
        
        stage('Update Baselines') {
            when {
                allOf {
                    branch 'main'
                    params.ESTABLISH_BASELINES
                }
            }
            steps {
                // Commit baseline updates
                sh '''
                git config user.name "Jenkins Performance Bot"
                git config user.email "jenkins@company.com"
                git add performance_baselines/
                git commit -m "Update performance baselines [skip ci]" || true
                git push origin main || true
                '''
            }
        }
    }
    
    post {
        always {
            // Stop application server
            sh 'kill $(cat app.pid) || true'
            sh 'rm -f app.pid'
        }
        
        failure {
            // Send alerts on failure
            emailext (
                subject: "Performance Regression Detected - Build ${BUILD_NUMBER}",
                body: "Performance regression tests failed. Check ${BUILD_URL} for details.",
                to: "${env.ALERT_EMAIL_RECIPIENTS}"
            )
        }
    }
}
```

### Jenkins Performance Plugin

Use the Jenkins Performance Plugin to track metrics over time:

```groovy
stage('Publish Performance') {
    steps {
        // Convert results to JMeter format for Performance Plugin
        sh 'python scripts/ci/convert_to_jmeter.py performance_results/report.json'
        
        // Publish performance results
        publishPerformanceTestResults ([
            sourceDataFiles: 'performance_results/*.jtl',
            modeOfThreshold: true,
            configType: 'PRT',
            failBuildIfNoResultFile: false,
            compareBuildPrevious: true,
            percentiles: '50,90,95,99',
            modePerformancePerTestCase: true
        ])
    }
}
```

## Azure DevOps Integration

### Azure Pipeline YAML

```yaml
trigger:
  branches:
    include:
      - main
      - develop
  paths:
    include:
      - src/*
      - tests/performance/*

pr:
  branches:
    include:
      - main
      - develop

schedules:
- cron: "0 2 * * *"
  displayName: Daily performance tests
  branches:
    include:
    - main

variables:
  pythonVersion: '3.12'
  testUrl: 'http://localhost:8000'

stages:
- stage: PerformanceTests
  displayName: 'Performance Regression Tests'
  jobs:
  - job: RunPerformanceTests
    displayName: 'Run Performance Tests'
    pool:
      vmImage: 'ubuntu-latest'
    
    strategy:
      matrix:
        API_Performance:
          testScenario: 'api_performance'
          timeout: 25
        Database_Performance:
          testScenario: 'database_performance'
          timeout: 20
        System_Resources:
          testScenario: 'system_resources'
          timeout: 15
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
      displayName: 'Install dependencies'
    
    - script: |
        # Start application
        python -m uvicorn src.pynomaly.presentation.api.main:app --host 0.0.0.0 --port 8000 &
        sleep 10
        
        # Wait for health check
        timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 3; done'
      displayName: 'Start application services'
    
    - script: |
        python scripts/ci/performance_regression_check.py \
          --config performance_test_config.json \
          --output "$(Agent.TempDirectory)/performance_report_$(testScenario).json" \
          --fail-on-regression \
          --verbose
      displayName: 'Run performance tests'
      env:
        PYNOMALY_TEST_URL: $(testUrl)
        PERF_TEST_SCENARIO: $(testScenario)
    
    - task: PublishTestResults@2
      condition: always()
      inputs:
        testResultsFormat: 'JUnit'
        testResultsFiles: 'performance_results/*.xml'
        mergeTestResults: true
        testRunTitle: 'Performance Tests - $(testScenario)'
    
    - task: PublishBuildArtifacts@1
      condition: always()
      inputs:
        pathToPublish: '$(Agent.TempDirectory)'
        artifactName: 'performance-results-$(testScenario)'
        publishLocation: 'Container'

- stage: UpdateBaselines
  displayName: 'Update Performance Baselines'
  dependsOn: PerformanceTests
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - job: UpdateBaselines
    displayName: 'Update Baselines'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - script: |
        git config user.name "Azure DevOps Performance Bot"
        git config user.email "devops@company.com"
        git add performance_baselines/
        git commit -m "Update performance baselines [skip ci]" || true
        git push origin HEAD:main
      displayName: 'Commit baseline updates'
      env:
        SYSTEM_ACCESSTOKEN: $(System.AccessToken)
```

## Docker Integration

### Dockerfile for Performance Testing

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Create performance directories
RUN mkdir -p performance_baselines performance_reports performance_results

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYNOMALY_TEST_URL=http://app:8000

# Copy performance test configuration
COPY performance_test_config.json ./

# Default command
CMD ["python", "scripts/ci/performance_regression_check.py", "--config", "performance_test_config.json"]
```

### Docker Compose for Testing

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/pynomaly
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    command: >
      bash -c "python -m uvicorn src.pynomaly.presentation.api.main:app --host 0.0.0.0 --port 8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  performance-tests:
    build: .
    environment:
      - PYNOMALY_TEST_URL=http://app:8000
      - PERF_TEST_DURATION=30
      - PERF_CONCURRENT_USERS=5
    depends_on:
      app:
        condition: service_healthy
    volumes:
      - ./performance_results:/app/performance_results
      - ./performance_reports:/app/performance_reports
      - ./performance_baselines:/app/performance_baselines
    command: >
      bash -c "
      sleep 5 &&
      python scripts/ci/performance_regression_check.py 
        --config performance_test_config.json 
        --output performance_results/report.json 
        --verbose
      "

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:6.2-alpine
```

Run performance tests with Docker Compose:

```bash
# Run performance tests
docker-compose up performance-tests

# Run with baseline establishment
docker-compose run -e ESTABLISH_BASELINES=true performance-tests

# Clean up
docker-compose down -v
```

## Kubernetes Integration

### Performance Test Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: performance-regression-test
  labels:
    app: pynomaly-performance
spec:
  template:
    metadata:
      labels:
        app: pynomaly-performance
    spec:
      restartPolicy: Never
      containers:
      - name: performance-test
        image: pynomaly:latest
        command: ["python", "scripts/ci/performance_regression_check.py"]
        args: 
        - "--config"
        - "/config/performance.json"
        - "--output"
        - "/results/performance_report.json"
        - "--fail-on-regression"
        env:
        - name: PYNOMALY_TEST_URL
          value: "http://pynomaly-service:8000"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        volumeMounts:
        - name: config
          mountPath: /config
        - name: results
          mountPath: /results
        - name: baselines
          mountPath: /app/performance_baselines
      volumes:
      - name: config
        configMap:
          name: performance-config
      - name: results
        emptyDir: {}
      - name: baselines
        persistentVolumeClaim:
          claimName: performance-baselines-pvc
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: performance-config
data:
  performance.json: |
    {
      "base_url": "http://pynomaly-service:8000",
      "test_duration": 30,
      "concurrent_users": 5,
      "test_scenarios": [
        {
          "name": "health_check",
          "type": "api",
          "endpoint": "/health",
          "duration": 10
        }
      ]
    }
```

### CronJob for Scheduled Tests

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scheduled-performance-tests
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: performance-test
            image: pynomaly:latest
            command: ["python", "scripts/ci/performance_regression_check.py"]
            args: ["--config", "/config/performance.json", "--establish-baselines"]
            env:
            - name: PYNOMALY_TEST_URL
              value: "http://pynomaly-service:8000"
            volumeMounts:
            - name: config
              mountPath: /config
            - name: baselines
              mountPath: /app/performance_baselines
          volumes:
          - name: config
            configMap:
              name: performance-config
          - name: baselines
            persistentVolumeClaim:
              claimName: performance-baselines-pvc
```

## Monitoring and Alerting Integration

### Prometheus Metrics

Export performance metrics to Prometheus:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
performance_test_counter = Counter('performance_tests_total', 'Total performance tests run')
performance_regression_gauge = Gauge('performance_regressions_detected', 'Number of regressions detected')
test_duration_histogram = Histogram('performance_test_duration_seconds', 'Test execution duration')

# Update metrics in your tests
performance_test_counter.inc()
performance_regression_gauge.set(regression_count)
test_duration_histogram.observe(test_duration)
```

### Grafana Dashboard

Create a Grafana dashboard for performance monitoring:

```json
{
  "dashboard": {
    "title": "Pynomaly Performance Monitoring",
    "panels": [
      {
        "title": "Performance Test Frequency",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(performance_tests_total[1h])"
          }
        ]
      },
      {
        "title": "Regression Detection Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "performance_regressions_detected"
          }
        ]
      },
      {
        "title": "Test Duration Trends",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, performance_test_duration_seconds)"
          }
        ]
      }
    ]
  }
}
```

## Best Practices for CI/CD Integration

### 1. Environment Isolation

- Use dedicated test environments
- Isolate performance tests from functional tests
- Ensure consistent resource allocation

### 2. Test Scheduling

- Run performance tests during off-peak hours
- Use scheduled runs for baseline establishment
- Avoid running multiple performance test suites simultaneously

### 3. Artifact Management

- Archive performance reports and baselines
- Set appropriate retention policies
- Version control baseline data

### 4. Alert Configuration

- Configure different alert channels for different environments
- Use severity-based routing
- Implement proper alert throttling

### 5. Security Considerations

- Store credentials in CI/CD secrets
- Use least-privilege access for test accounts
- Audit and rotate API tokens regularly

### 6. Performance Test Hygiene

- Clean up test data after runs
- Monitor resource usage during tests
- Implement proper error handling and cleanup