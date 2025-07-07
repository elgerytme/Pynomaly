# Banking Data Anomaly Detection Examples

This directory contains comprehensive examples of anomaly detection applied to various banking and financial transaction types. The examples demonstrate how to identify fraud, money laundering, data quality issues, and regulatory compliance violations using Pynomaly.

## Overview

The banking examples cover nine critical transaction types:

1. **Deposits** - Money laundering and structuring detection
2. **Loans** - Application fraud and suspicious lending patterns  
3. **Investments** - Market manipulation and insider trading
4. **Foreign Exchange (FX)** - Currency fraud and trade-based money laundering
5. **ATM Transactions** - Card skimming and location-based fraud
6. **Debit Card Transactions** - Real-time fraud detection
7. **Credit Card Transactions** - Advanced fraud pattern recognition
8. **Corporate Expenses** - Expense fraud and policy violations
9. **General Ledger (GL)** - Accounting anomalies and data quality issues

## Directory Structure

```
examples/banking/
├── datasets/           # Generated sample data with embedded anomalies
├── scripts/           # Python analysis scripts for each transaction type
├── notebooks/         # Jupyter notebooks with comprehensive workflows
├── outputs/           # Generated reports, visualizations, and results
└── README.md          # This documentation
```

## Quick Start

### 1. Generate Sample Data

```bash
cd examples/banking/scripts
python generate_sample_data.py
```

This creates realistic banking datasets with embedded anomalies:
- ~111,000 total transactions across all types
- 4,670 embedded anomalies (various types)
- Realistic transaction patterns and amounts
- Multiple anomaly types per transaction category

### 2. Run Individual Analysis Scripts

```bash
# Analyze deposit transactions for money laundering
python analyze_deposits.py

# Detect credit card fraud patterns
python analyze_credit_cards.py

# Identify suspicious foreign exchange activity
python analyze_fx_transactions.py
```

### 3. Comprehensive Analysis Notebook

```bash
jupyter notebook notebooks/banking_anomaly_detection_comprehensive.ipynb
```

## Transaction Types and Anomaly Patterns

### Deposits (`analyze_deposits.py`)

**Anomaly Types Detected:**
- **Structuring**: Transactions just under $10,000 reporting threshold
- **Large Cash Deposits**: Unusual amounts for customer profile
- **Timing Anomalies**: Deposits outside business hours
- **Velocity Patterns**: Multiple deposits in short timeframes
- **Source Anomalies**: Unusual funding sources

**Key Features:**
- Customer behavior profiling
- Velocity analysis (hourly/daily patterns)
- Amount deviation detection
- Source type analysis (cash, check, wire, ACH)

**Sample Output:**
```
=== DEPOSIT ANOMALY ANALYSIS ===
Total transactions: 10,000
Detected anomalies: 487 (4.9%)
Actual anomalies: 500 (5.0%)

Anomaly Characteristics:
Average amount: $89,234.56 vs $12,345.67 (normal)
Deposits near $10K threshold: 23
Weekend deposits: 45 (9.2%)
After-hours deposits: 67
```

### Credit Card Transactions (`analyze_credit_cards.py`)

**Anomaly Types Detected:**
- **Velocity Fraud**: Multiple transactions in short periods
- **Geographic Anomalies**: Impossible travel patterns
- **Amount Deviations**: Transactions inconsistent with customer patterns
- **Merchant Category Fraud**: Unusual spending categories
- **Card-Not-Present Fraud**: Online/phone transaction fraud

**Key Features:**
- Real-time fraud scoring
- Customer spending pattern analysis
- Merchant category analysis
- Card-present vs card-not-present analysis
- Geographic and temporal pattern detection

**Sample Output:**
```
=== CREDIT CARD FRAUD ANALYSIS ===
Total fraud detected: 1,247 (5.0%)
Fraud amount: $2,456,789.12
High-value fraud (>$1000): 234
Card-not-present fraud: 456 (36.6%)
Night-time fraud: 123 (9.9%)
```

### Foreign Exchange Transactions (`analyze_fx_transactions.py`)

**Anomaly Types Detected:**
- **Trade-Based Money Laundering**: Unusual rate manipulation
- **Structuring**: Breaking large amounts into smaller transactions
- **Rate Anomalies**: Suspicious exchange rates
- **High-Velocity Trading**: Rapid transaction sequences
- **Purpose Misrepresentation**: Inconsistent transaction purposes

**Key Features:**
- Exchange rate deviation analysis
- Cross-border transaction monitoring
- Purpose and method validation
- Customer velocity profiling
- Fee structure analysis

### ATM Transactions

**Anomaly Types Detected:**
- **Card Skimming**: Multiple failed attempts patterns
- **Geographic Impossibilities**: ATM usage in impossible locations
- **Unusual Timing**: Transactions at odd hours
- **Amount Patterns**: Consistent withdrawal amounts (testing limits)

### Investment Transactions

**Anomaly Types Detected:**
- **Insider Trading**: Suspicious timing patterns
- **Market Manipulation**: Pump and dump schemes
- **Wash Trading**: Artificial volume creation
- **Large Block Trading**: Unusual volume patterns

### Corporate Expenses

**Anomaly Types Detected:**
- **Expense Fraud**: Inflated or fictitious expenses
- **Policy Violations**: Expenses outside guidelines
- **Duplicate Submissions**: Same expense submitted multiple times
- **Personal Expenses**: Non-business expenses submitted

### General Ledger

**Anomaly Types Detected:**
- **Data Quality Issues**: Imbalanced journal entries
- **Timing Anomalies**: Off-hours posting
- **Manual Overrides**: Unusual manual adjustments
- **Amount Anomalies**: Transactions outside normal ranges

## Technical Implementation

### Algorithms Used

1. **Isolation Forest**: Effective for high-dimensional fraud detection
2. **Local Outlier Factor (LOF)**: Density-based anomaly detection
3. **One-Class SVM**: Support vector-based outlier detection
4. **CBLOF**: Cluster-based outlier factor
5. **Ensemble Methods**: Combining multiple algorithms for better accuracy

### Feature Engineering

**Common Features Across Transaction Types:**
- **Temporal Features**: Hour, day of week, business hours indicators
- **Amount Features**: Log transformation, z-scores, deviation metrics
- **Customer Behavior**: Historical patterns, velocity metrics
- **Categorical Encoding**: One-hot encoding for transaction types
- **Velocity Metrics**: Transaction frequency in time windows

**Transaction-Specific Features:**
- **Deposits**: Source type, branch location, teller information
- **Credit Cards**: Merchant category, card present/not present, location
- **FX**: Currency pairs, exchange rates, purpose codes
- **ATM**: Location, transaction type, success/failure rates

### Performance Metrics

**Model Evaluation:**
- **Precision**: Percentage of flagged transactions that are actual anomalies
- **Recall**: Percentage of actual anomalies that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **False Positive Rate**: Normal transactions incorrectly flagged

**Business Metrics:**
- **Detection Rate**: Percentage of fraud/anomalies caught
- **Investigation Efficiency**: Quality of flagged cases
- **Financial Impact**: Amount of fraud prevented vs investigation costs
- **Regulatory Compliance**: Meeting AML/KYC requirements

## Data Quality and Characteristics

### Dataset Statistics

| Transaction Type | Records | Anomalies | Anomaly Rate | Key Risk Areas |
|-----------------|---------|-----------|--------------|----------------|
| Deposits | 10,000 | 500 | 5.0% | Money laundering, structuring |
| Loans | 5,000 | 400 | 8.0% | Application fraud, suspicious patterns |
| Investments | 8,000 | 480 | 6.0% | Market manipulation, insider trading |
| FX Transactions | 3,000 | 300 | 10.0% | Trade-based money laundering |
| ATM Transactions | 15,000 | 450 | 3.0% | Card skimming, location fraud |
| Debit Cards | 25,000 | 1,250 | 5.0% | Real-time fraud patterns |
| Credit Cards | 25,000 | 1,250 | 5.0% | Sophisticated fraud schemes |
| Expenses | 8,000 | 560 | 7.0% | Expense fraud, policy violations |
| General Ledger | 12,000 | 480 | 4.0% | Data quality, manual errors |

### Anomaly Types Distribution

**Financial Crime (60%)**
- Money laundering: 25%
- Fraud (cards, loans): 20%
- Market manipulation: 10%
- Regulatory violations: 5%

**Operational Issues (25%)**
- Data quality problems: 15%
- System errors: 10%

**Policy Violations (15%)**
- Expense violations: 10%
- Trading policy breaches: 5%

## Business Use Cases

### 1. Real-Time Fraud Detection
- **Credit/Debit Cards**: Immediate transaction scoring
- **ATM Monitoring**: Real-time skimming detection
- **Wire Transfers**: Instant risk assessment

### 2. Anti-Money Laundering (AML)
- **Deposit Monitoring**: Structuring detection
- **FX Surveillance**: Trade-based money laundering
- **Customer Due Diligence**: Risk profiling

### 3. Regulatory Compliance
- **Suspicious Activity Reporting (SAR)**: Automated flagging
- **Know Your Customer (KYC)**: Enhanced due diligence triggers
- **Currency Transaction Reporting (CTR)**: Threshold monitoring

### 4. Operational Risk Management
- **Data Quality**: GL transaction validation
- **Process Monitoring**: Unusual operational patterns
- **Internal Fraud**: Employee expense monitoring

### 5. Market Surveillance
- **Trading Anomalies**: Unusual investment patterns
- **Market Manipulation**: Pump and dump detection
- **Insider Trading**: Suspicious timing analysis

## Advanced Features

### 1. Cross-Transaction Analysis
```python
# Example: Linking suspicious deposits with FX transactions
def cross_analyze_deposits_fx(deposits_df, fx_df):
    # Find customers with both suspicious deposits and FX activity
    suspicious_deposits = deposits_df[deposits_df['predicted_anomaly'] == 1]
    suspicious_fx = fx_df[fx_df['predicted_anomaly'] == 1]
    
    common_customers = set(suspicious_deposits['customer_id']) & set(suspicious_fx['customer_id'])
    return common_customers
```

### 2. Temporal Pattern Analysis
```python
# Example: Time-series anomaly detection
def detect_temporal_anomalies(df, time_window='1D'):
    # Aggregate transactions by time window
    time_series = df.groupby(pd.Grouper(key='timestamp', freq=time_window))['amount'].sum()
    
    # Detect anomalies in the time series
    # Implementation would use statistical methods or ML
    return anomalous_periods
```

### 3. Network Analysis
```python
# Example: Customer relationship network analysis
def build_customer_network(transactions_df):
    # Build graph of customers connected by transactions
    # Detect suspicious clusters or patterns
    # Implementation would use graph algorithms
    return suspicious_networks
```

## Configuration and Customization

### Contamination Rates
Adjust detection sensitivity by modifying contamination rates:

```python
# Conservative detection (higher precision, lower recall)
contamination = 0.01  # 1% of transactions flagged

# Balanced detection
contamination = 0.05  # 5% of transactions flagged

# Aggressive detection (higher recall, lower precision)  
contamination = 0.10  # 10% of transactions flagged
```

### Algorithm Selection
Choose algorithms based on data characteristics:

```python
# High-dimensional data (many features)
algorithm = 'isolation_forest'

# Density-based anomalies
algorithm = 'lof'

# Linear separable anomalies
algorithm = 'one_class_svm'

# Cluster-based detection
algorithm = 'cblof'
```

### Feature Selection
Customize features based on business requirements:

```python
# Basic feature set (fast processing)
features = ['amount', 'hour', 'day_of_week']

# Comprehensive feature set (better accuracy)
features = ['amount', 'amount_log', 'hour', 'day_of_week', 
           'customer_velocity', 'amount_deviation', 'category_encoded']

# Domain-specific features
# For FX: exchange_rate, currency_pair, purpose
# For Credit Cards: merchant_category, card_present, location
```

## Integration Examples

### 1. Real-Time Processing
```python
# Example integration with streaming data
from kafka import KafkaConsumer
import json

def process_transaction_stream():
    consumer = KafkaConsumer('transactions', 
                           value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
    for message in consumer:
        transaction = message.value
        
        # Real-time anomaly detection
        anomaly_score = detect_anomaly(transaction)
        
        if anomaly_score > threshold:
            # Trigger alert or block transaction
            handle_suspicious_transaction(transaction, anomaly_score)
```

### 2. Batch Processing
```python
# Example daily batch processing
def daily_anomaly_analysis():
    # Load yesterday's transactions
    transactions = load_daily_transactions(date.today() - timedelta(days=1))
    
    # Run anomaly detection
    results = detect_anomalies(transactions)
    
    # Generate reports
    generate_daily_report(results)
    
    # Update ML models
    retrain_models_if_needed(results)
```

### 3. API Integration
```python
# Example REST API for real-time scoring
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/score_transaction', methods=['POST'])
def score_transaction():
    transaction = request.json
    
    # Extract features
    features = extract_features(transaction)
    
    # Score transaction
    anomaly_score = model.predict(features)
    
    return jsonify({
        'transaction_id': transaction['id'],
        'anomaly_score': float(anomaly_score),
        'risk_level': classify_risk(anomaly_score)
    })
```

## Regulatory Considerations

### Suspicious Activity Reporting (SAR)
- Automatically flag transactions with high anomaly scores
- Include supporting documentation and analysis
- Track investigation outcomes for model improvement

### Model Validation and Governance
- Regular backtesting with known fraud cases
- Model performance monitoring and drift detection
- Documentation of model changes and approvals
- Audit trails for regulatory examinations

### Privacy and Data Protection
- Ensure compliance with GDPR, CCPA, and other regulations
- Implement data anonymization where required
- Maintain audit logs of data access and usage

## Troubleshooting

### Common Issues

1. **High False Positive Rates**
   - Reduce contamination rate
   - Improve feature engineering
   - Use ensemble methods
   - Incorporate business rules

2. **Missing Known Fraud**
   - Increase contamination rate
   - Add domain-specific features
   - Use multiple algorithms
   - Investigate feature importance

3. **Performance Issues**
   - Reduce feature dimensionality
   - Use sampling for large datasets
   - Optimize algorithm parameters
   - Consider approximate algorithms

4. **Data Quality Problems**
   - Implement data validation checks
   - Handle missing values appropriately
   - Normalize features consistently
   - Monitor data drift

### Performance Optimization

```python
# Example performance optimization
def optimize_detection_pipeline():
    # Feature selection
    important_features = select_features_by_importance(X, y)
    
    # Sampling for large datasets
    if len(data) > 100000:
        sample_data = data.sample(n=50000, random_state=42)
    
    # Parallel processing
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(
        delayed(detect_anomalies)(chunk) for chunk in data_chunks
    )
    
    return results
```

## Further Resources

### Academic Papers
- "Isolation Forest" by Liu et al. (2008)
- "Local Outlier Factor" by Breunig et al. (2000)
- "Anomaly Detection in Financial Data" by Various Authors

### Industry Standards
- FATF Guidelines on Anti-Money Laundering
- Basel Committee on Banking Supervision Guidelines
- COSO Enterprise Risk Management Framework

### Related Tools and Libraries
- **Scikit-learn**: Machine learning algorithms
- **PyOD**: Outlier detection library
- **SMOTE**: Synthetic data generation for imbalanced datasets
- **SHAP**: Model explainability and feature importance

### Support and Contributing
- Report issues or request features via GitHub Issues
- Contribute improvements via Pull Requests
- Join discussions in the Pynomaly community forums

---

**Note**: This is a demonstration dataset with synthetic data. In production environments, ensure compliance with all applicable regulations and privacy requirements when analyzing financial transaction data.
