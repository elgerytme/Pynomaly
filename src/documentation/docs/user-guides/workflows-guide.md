# Workflow Guides for Different User Types

## Overview

This guide provides step-by-step workflows tailored to different user types and use cases. Choose the workflow that best matches your role and objectives.

## Table of Contents

1. [Business Analyst Workflows](#business-analyst)
2. [Data Scientist Workflows](#data-scientist)
3. [DevOps/MLOps Engineer Workflows](#devops-mlops)
4. [Application Developer Workflows](#application-developer)
5. [Industry-Specific Workflows](#industry-specific)

---

## Business Analyst Workflows {#business-analyst}

### Quick Business Insights Workflow

**Time Required**: 15-30 minutes  
**Skills Needed**: Basic Excel/CSV knowledge  
**Objective**: Get actionable insights from business data

#### Step 1: Prepare Your Data
```csv
# Example: customer_transactions.csv
customer_id,transaction_amount,transaction_date,category,is_weekend
CUST001,45.67,2024-01-15,grocery,false
CUST002,1250.00,2024-01-15,electronics,false
CUST003,23.45,2024-01-15,grocery,false
# ... more data
```

#### Step 2: Quick Analysis with Web UI
1. **Upload Data**: Go to Pynomaly web interface → Upload CSV
2. **Auto-Detect**: Click "Auto-Detect Anomalies" 
3. **Review Results**: See highlighted unusual transactions
4. **Export Report**: Download Excel report with findings

#### Step 3: Business Interpretation
```python
# If using Python API for deeper analysis
from pynomaly import PynomalyClient

client = PynomalyClient()
results = client.detect_anomalies('customer_transactions.csv')

# Get business-friendly summary
summary = client.generate_business_summary(results)
print(summary['executive_summary'])
print(f"Revenue at Risk: ${summary['potential_impact']['revenue_risk']:,.2f}")
print(f"Transactions to Review: {summary['action_items']['manual_review_count']}")
```

#### Step 4: Create Business Report
```python
# Generate presentation-ready charts
charts = client.create_business_charts(results)
charts.save_dashboard('anomaly_dashboard.html')

# Export action items
action_items = client.export_action_items(results, format='excel')
```

**Expected Outcomes**:
- Executive summary of unusual patterns
- List of specific transactions/customers to investigate
- Risk assessment and potential impact
- Visual dashboard for presentations

---

### Fraud Monitoring Workflow

**Time Required**: 5 minutes setup + ongoing monitoring  
**Objective**: Continuous fraud detection and alerting

#### Step 1: Set Up Monitoring
```bash
# Using Pynomaly CLI for automated monitoring
pynomaly monitor setup \
  --data-source "transactions_table" \
  --algorithm "isolation_forest" \
  --contamination 0.02 \
  --alert-threshold 0.8 \
  --email-alerts "risk@company.com"
```

#### Step 2: Configure Business Rules
```yaml
# fraud_rules.yaml
business_rules:
  high_risk_amount: 5000
  velocity_threshold: 5  # transactions per hour
  geographical_risk:
    - "high_risk_country_codes"
  
alert_levels:
  yellow: 0.6  # Review within 4 hours
  orange: 0.75 # Review within 1 hour  
  red: 0.9     # Immediate review
```

#### Step 3: Dashboard Monitoring
- **Real-time alerts** appear on fraud dashboard
- **Weekly reports** automatically emailed
- **Mobile notifications** for high-risk cases

#### Step 4: Investigation Workflow
```python
# When an alert triggers, investigate quickly
case_id = "ALERT_20240715_001"
investigation = client.investigate_anomaly(case_id)

print(f"Customer Risk Profile: {investigation['customer_profile']}")
print(f"Similar Historical Cases: {investigation['similar_cases']}")
print(f"Recommended Action: {investigation['recommendation']}")
```

---

## Data Scientist Workflows {#data-scientist}

### Model Development and Evaluation Workflow

**Time Required**: 2-4 hours  
**Skills Needed**: Python, ML experience  
**Objective**: Build robust anomaly detection models

#### Step 1: Exploratory Data Analysis
```python
import pandas as pd
import numpy as np
from pynomaly import PynomalyClient
from pynomaly.utils import data_profiling

# Load and profile data
data = pd.read_csv('sensor_data.csv')
profile = data_profiling.profile_dataset(data)

print("Data Quality Report:")
print(f"Missing values: {profile['missing_values']}")
print(f"Outlier candidates: {profile['outlier_analysis']}")
print(f"Feature correlations: {profile['correlation_analysis']}")
```

#### Step 2: Algorithm Comparison
```python
from pynomaly.evaluation import compare_algorithms

# Compare multiple algorithms
algorithms = [
    'isolation_forest',
    'local_outlier_factor', 
    'one_class_svm',
    'autoencoder',
    'lstm_autoencoder'
]

comparison = compare_algorithms(
    data=data,
    algorithms=algorithms,
    validation_method='cross_validation',
    metrics=['precision', 'recall', 'f1', 'auc_roc']
)

print("Algorithm Performance Comparison:")
print(comparison.to_string())
```

#### Step 3: Hyperparameter Optimization
```python
from pynomaly.optimization import optimize_hyperparameters

# Optimize best-performing algorithm
best_algorithm = comparison.iloc[0]['algorithm']
optimized_params = optimize_hyperparameters(
    algorithm=best_algorithm,
    data=data,
    search_space='bayesian',
    n_trials=100,
    cv_folds=5
)

print(f"Optimized parameters: {optimized_params}")
```

#### Step 4: Model Validation and Testing
```python
from pynomaly.validation import comprehensive_validation
from sklearn.model_selection import train_test_split

# Split data for proper validation
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train final model
client = PynomalyClient(
    algorithm=best_algorithm,
    **optimized_params
)
model = client.fit(train_data)

# Comprehensive validation
validation_results = comprehensive_validation(
    model=model,
    test_data=test_data,
    validation_tests=[
        'distribution_stability',
        'adversarial_robustness', 
        'feature_importance',
        'edge_case_handling'
    ]
)

print("Model Validation Results:")
for test, result in validation_results.items():
    print(f"{test}: {'PASS' if result['passed'] else 'FAIL'}")
```

#### Step 5: Model Deployment Preparation
```python
# Model serialization and deployment prep
model_package = client.package_model(
    model=model,
    metadata={
        'training_data_hash': hash(str(train_data.values)),
        'validation_metrics': validation_results,
        'feature_schema': data.dtypes.to_dict(),
        'preprocessing_steps': ['standardization', 'outlier_clipping']
    }
)

# Save deployment artifacts
model_package.save('models/fraud_detector_v1.0.pkl')
model_package.export_config('models/deployment_config.yaml')
```

---

### Research and Experimentation Workflow

**Time Required**: 1-2 weeks  
**Objective**: Explore new approaches and publish findings

#### Step 1: Literature Review Integration
```python
from pynomaly.research import literature_survey, benchmark_datasets

# Get state-of-the-art comparison
sota_methods = literature_survey.get_recent_methods(
    domain='time_series_anomaly_detection',
    years=2
)

# Access benchmark datasets
benchmarks = benchmark_datasets.load_benchmark_suite([
    'credit_card_fraud',
    'network_intrusion', 
    'sensor_anomalies'
])
```

#### Step 2: Custom Algorithm Development
```python
from pynomaly.base import BaseAnomalyDetector
from pynomaly.research import experimental_framework

class MyNovelDetector(BaseAnomalyDetector):
    def __init__(self, novel_param=0.5):
        self.novel_param = novel_param
        
    def fit(self, X):
        # Your novel algorithm implementation
        pass
        
    def decision_function(self, X):
        # Return anomaly scores
        pass

# Register for experimental framework
experimental_framework.register_algorithm('my_novel_detector', MyNovelDetector)
```

#### Step 3: Comprehensive Benchmarking
```python
# Run comprehensive benchmarks
benchmark_results = experimental_framework.run_benchmark_suite(
    algorithms=['my_novel_detector'] + sota_methods,
    datasets=benchmarks,
    metrics=['precision', 'recall', 'f1', 'auc_roc', 'runtime'],
    statistical_tests=True,
    visualizations=True
)

# Generate research report
report = experimental_framework.generate_research_report(
    results=benchmark_results,
    template='ieee_conference'
)
```

#### Step 4: Statistical Analysis
```python
from pynomaly.research import statistical_analysis

# Statistical significance testing
significance_tests = statistical_analysis.compare_algorithms(
    results=benchmark_results,
    baseline_algorithm='isolation_forest',
    alpha=0.05,
    corrections=['bonferroni', 'holm']
)

# Effect size analysis
effect_sizes = statistical_analysis.calculate_effect_sizes(
    results=benchmark_results,
    metric='f1_score'
)
```

---

## DevOps/MLOps Engineer Workflows {#devops-mlops}

### Production Deployment Workflow

**Time Required**: 4-8 hours initial setup  
**Skills Needed**: Docker, Kubernetes, CI/CD  
**Objective**: Deploy scalable, monitored anomaly detection service

#### Step 1: Environment Setup
```bash
# Clone production configuration
git clone https://github.com/pynomaly/production-templates
cd production-templates

# Set up environment variables
cp .env.template .env
# Edit .env with your configuration
```

#### Step 2: Docker Containerization
```dockerfile
# Dockerfile.production
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "pynomaly.api:app"]
```

```bash
# Build and test container
docker build -t pynomaly:v1.0 -f Dockerfile.production .
docker run -p 8000:8000 pynomaly:v1.0

# Test health endpoint
curl http://localhost:8000/health
```

#### Step 3: Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
    spec:
      containers:
      - name: pynomaly
        image: pynomaly:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

#### Step 4: Monitoring and Observability
```yaml
# monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/
kubectl apply -f grafana/dashboards/
```

#### Step 5: CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: |
        python -m pytest tests/
        python -m pytest tests/integration/
        
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Security scan
      run: |
        bandit -r src/
        safety check
        
  deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/staging/
        ./scripts/validate_deployment.sh staging
        
    - name: Run integration tests
      run: |
        pytest tests/e2e/ --env=staging
        
    - name: Deploy to production
      run: |
        kubectl apply -f k8s/production/
        ./scripts/validate_deployment.sh production
```

---

### Monitoring and Maintenance Workflow

**Time Required**: 1-2 hours setup + ongoing monitoring  
**Objective**: Ensure reliable production operation

#### Step 1: Set Up Alerting
```yaml
# alerting/rules.yaml
groups:
  - name: pynomaly.rules
    rules:
    - alert: HighErrorRate
      expr: rate(pynomaly_http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        
    - alert: ModelDrift
      expr: pynomaly_model_drift_score > 0.7
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Model drift detected"
        
    - alert: HighLatency
      expr: histogram_quantile(0.95, pynomaly_request_duration_seconds) > 1.0
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected"
```

#### Step 2: Automated Model Retraining
```python
# scripts/model_retraining.py
from pynomaly.mlops import ModelRetrainer, DataDriftDetector

def automated_retraining_pipeline():
    # Check for data drift
    drift_detector = DataDriftDetector()
    drift_report = drift_detector.check_drift(
        reference_data='training_data.csv',
        current_data='recent_data.csv'
    )
    
    if drift_report['drift_detected']:
        logger.info("Drift detected, triggering retraining")
        
        # Retrain model
        retrainer = ModelRetrainer()
        new_model = retrainer.retrain(
            current_model='models/current_model.pkl',
            new_data='recent_data.csv',
            validation_strategy='time_series_split'
        )
        
        # Validate new model
        if new_model.performance_score > 0.85:
            # Deploy new model
            new_model.deploy('production')
            logger.info("New model deployed successfully")
        else:
            logger.warning("New model performance insufficient")
```

#### Step 3: Performance Monitoring Dashboard
```python
# monitoring/create_dashboard.py
from pynomaly.monitoring import create_monitoring_dashboard

dashboard = create_monitoring_dashboard([
    'request_rate',
    'error_rate', 
    'response_time',
    'model_performance',
    'data_quality',
    'resource_utilization'
])

dashboard.deploy('grafana')
```

---

## Application Developer Workflows {#application-developer}

### API Integration Workflow

**Time Required**: 2-4 hours  
**Skills Needed**: REST APIs, JSON, basic programming  
**Objective**: Integrate Pynomaly into existing applications

#### Step 1: API Authentication Setup
```python
# Python example
import requests
from pynomaly import PynomalyClient

# Initialize client with API key
client = PynomalyClient(
    api_key='your_api_key_here',
    base_url='https://api.pynomaly.com/v1'
)

# Or use environment variable
# export PYNOMALY_API_KEY=your_api_key_here
client = PynomalyClient()  # Auto-detects from environment
```

```javascript
// JavaScript example
const PynomalyClient = require('pynomaly-js');

const client = new PynomalyClient({
    apiKey: process.env.PYNOMALY_API_KEY,
    baseUrl: 'https://api.pynomaly.com/v1'
});
```

#### Step 2: Basic Integration Pattern
```python
# Real-time anomaly detection in application
def process_transaction(transaction_data):
    # Your existing business logic
    result = process_payment(transaction_data)
    
    # Add anomaly detection
    anomaly_result = client.detect_anomalies(
        data=[transaction_data],
        algorithm='isolation_forest',
        return_confidence=True
    )
    
    if anomaly_result['predictions'][0]:
        # Handle anomalous transaction
        confidence = anomaly_result['scores'][0]
        
        if confidence > 0.9:
            # High confidence anomaly - block transaction
            return {'status': 'blocked', 'reason': 'fraud_risk'}
        elif confidence > 0.7:
            # Medium confidence - flag for review
            flag_for_manual_review(transaction_data, confidence)
            return {'status': 'flagged', 'confidence': confidence}
    
    return result
```

#### Step 3: Batch Processing Integration
```python
# Batch processing for large datasets
def daily_fraud_analysis():
    # Get yesterday's transactions
    transactions = get_transactions(date='yesterday')
    
    # Process in batches to handle large volumes
    batch_size = 1000
    anomaly_results = []
    
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        
        batch_results = client.detect_anomalies_batch(
            data=batch,
            algorithm='ensemble',
            include_explanations=True
        )
        
        anomaly_results.extend(batch_results)
    
    # Process results
    high_risk_transactions = [
        result for result in anomaly_results 
        if result['score'] > 0.8
    ]
    
    # Send alerts
    send_fraud_alerts(high_risk_transactions)
    
    # Update database
    update_fraud_scores(anomaly_results)
```

#### Step 4: Error Handling and Resilience
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        logger.error(f"API call failed after {max_retries} attempts: {e}")
                        raise
                    
                    # Wait before retry
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
                    logger.warning(f"API call failed, retrying in {wait_time}s: {e}")
            
        return wrapper
    return decorator

@retry_on_failure()
def robust_anomaly_detection(data):
    return client.detect_anomalies(data)

# Fallback mechanism
def anomaly_detection_with_fallback(data):
    try:
        return robust_anomaly_detection(data)
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        # Fallback to simple rule-based detection
        return simple_rule_based_detection(data)
```

---

### Real-time Streaming Integration

**Time Required**: 4-6 hours  
**Objective**: Process streaming data for real-time anomaly detection

#### Step 1: Kafka Integration
```python
from kafka import KafkaConsumer, KafkaProducer
import json

# Set up Kafka consumer
consumer = KafkaConsumer(
    'transaction-stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Set up producer for results
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Process streaming data
for message in consumer:
    transaction = message.value
    
    # Real-time anomaly detection
    result = client.detect_anomalies(
        data=[transaction],
        algorithm='online_isolation_forest'  # Optimized for streaming
    )
    
    if result['predictions'][0]:
        # Send alert
        alert = {
            'transaction_id': transaction['id'],
            'anomaly_score': result['scores'][0],
            'timestamp': transaction['timestamp']
        }
        producer.send('fraud-alerts', alert)
```

#### Step 2: Redis Caching for Performance
```python
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_anomaly_detection(data, cache_key=None):
    if cache_key:
        # Check cache first
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return pickle.loads(cached_result)
    
    # Compute result
    result = client.detect_anomalies(data)
    
    # Cache result for 1 hour
    if cache_key:
        redis_client.setex(
            cache_key, 
            3600,  # 1 hour
            pickle.dumps(result)
        )
    
    return result
```

---

## Industry-Specific Workflows {#industry-specific}

### Financial Services: Fraud Detection

**Objective**: Detect fraudulent transactions in real-time

#### Implementation
```python
def financial_fraud_detection(transaction):
    # Feature engineering for financial fraud
    features = {
        'amount': transaction['amount'],
        'time_since_last': calculate_time_since_last_transaction(transaction),
        'amount_vs_avg': transaction['amount'] / get_customer_avg_amount(transaction['customer_id']),
        'merchant_risk_score': get_merchant_risk_score(transaction['merchant_id']),
        'location_risk': calculate_location_risk(transaction['location']),
        'hour_of_day': get_hour_of_day(transaction['timestamp']),
        'day_of_week': get_day_of_week(transaction['timestamp'])
    }
    
    # Use ensemble approach for high accuracy
    result = client.detect_anomalies(
        data=[features],
        algorithm='ensemble',
        ensemble_methods=['isolation_forest', 'one_class_svm', 'local_outlier_factor'],
        contamination=0.001  # Very low fraud rate expected
    )
    
    return {
        'is_fraud': result['predictions'][0],
        'fraud_score': result['scores'][0],
        'risk_level': categorize_risk_level(result['scores'][0])
    }
```

### Manufacturing: Quality Control

**Objective**: Detect defective products on production line

#### Implementation
```python
def quality_control_monitoring(sensor_readings):
    # Real-time quality monitoring
    quality_features = {
        'temperature': sensor_readings['temperature'],
        'pressure': sensor_readings['pressure'], 
        'vibration': sensor_readings['vibration'],
        'speed': sensor_readings['production_speed'],
        'power_consumption': sensor_readings['power']
    }
    
    # Use time-series aware detection
    result = client.detect_anomalies(
        data=[quality_features],
        algorithm='lstm_autoencoder',  # Good for sensor data
        contamination=0.05  # 5% defect rate expected
    )
    
    if result['predictions'][0]:
        # Trigger quality alert
        trigger_quality_alert(
            anomaly_score=result['scores'][0],
            sensor_readings=sensor_readings
        )
    
    return result
```

### Healthcare: Patient Monitoring

**Objective**: Detect anomalous patient vital signs

#### Implementation
```python
def patient_monitoring(vital_signs):
    # Patient vital signs monitoring
    patient_features = {
        'heart_rate': vital_signs['heart_rate'],
        'blood_pressure_systolic': vital_signs['bp_systolic'],
        'blood_pressure_diastolic': vital_signs['bp_diastolic'],
        'temperature': vital_signs['temperature'],
        'oxygen_saturation': vital_signs['spo2']
    }
    
    # Use patient-specific model if available
    patient_id = vital_signs['patient_id']
    if has_patient_specific_model(patient_id):
        model_id = get_patient_model(patient_id)
        result = client.detect_anomalies(
            data=[patient_features],
            model_id=model_id
        )
    else:
        # Use general population model
        result = client.detect_anomalies(
            data=[patient_features],
            algorithm='local_outlier_factor',
            contamination=0.1
        )
    
    if result['predictions'][0] and result['scores'][0] > 0.8:
        # High-priority medical alert
        send_medical_alert(patient_id, result['scores'][0])
    
    return result
```

---

## Best Practices for All Workflows

### 1. Data Quality
- Always validate input data before detection
- Handle missing values appropriately
- Monitor for data drift over time

### 2. Performance Optimization
- Use appropriate contamination rates
- Choose algorithms suited to your data size
- Implement caching for repeated queries

### 3. Monitoring and Alerting
- Set up appropriate alert thresholds
- Monitor model performance metrics
- Plan for model retraining

### 4. Security
- Use API keys securely (environment variables)
- Implement proper authentication
- Audit anomaly detection logs

### 5. Documentation
- Document your feature engineering decisions
- Keep track of model versions and performance
- Maintain runbooks for incident response

---

## Getting Help

- **Workflow-specific questions**: [GitHub Discussions](https://github.com/pynomaly/pynomaly/discussions)
- **Technical issues**: [GitHub Issues](https://github.com/pynomaly/pynomaly/issues)
- **Enterprise support**: [Contact Sales](mailto:sales@pynomaly.com)

## Next Steps

1. **Choose your workflow** based on your role and use case
2. **Follow the step-by-step guide** for your specific scenario
3. **Adapt the examples** to your specific data and requirements
4. **Monitor and iterate** on your implementation
5. **Share your experience** with the community

---

Ready to get started? Pick the workflow that matches your needs and dive in!