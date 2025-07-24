# Interactive Jupyter Notebook Examples

This directory contains comprehensive Jupyter notebook tutorials for hands-on learning with the anomaly detection package. Each notebook is designed to be educational, interactive, and practically useful.

## 📚 Available Notebooks

### 🚀 Getting Started Notebooks
1. **[01_introduction_to_anomaly_detection.ipynb](01_introduction_to_anomaly_detection.ipynb)**
   - Introduction to anomaly detection concepts
   - Basic terminology and use cases
   - Interactive visualizations
   - **Time**: 30 minutes | **Level**: Beginner

2. **[02_algorithm_comparison_tutorial.ipynb](02_algorithm_comparison_tutorial.ipynb)**
   - Compare different algorithms side-by-side
   - Interactive parameter tuning
   - Performance visualization
   - **Time**: 45 minutes | **Level**: Intermediate

### 🏦 Use Case Notebooks
3. **[03_fraud_detection_end_to_end.ipynb](03_fraud_detection_end_to_end.ipynb)**
   - Complete fraud detection pipeline
   - Feature engineering for financial data
   - Model evaluation and deployment
   - **Time**: 60 minutes | **Level**: Intermediate

4. **[04_network_security_analysis.ipynb](04_network_security_analysis.ipynb)**
   - Network intrusion detection
   - Time series anomaly patterns
   - Real-time alerting simulation
   - **Time**: 45 minutes | **Level**: Intermediate

5. **[05_iot_sensor_monitoring.ipynb](05_iot_sensor_monitoring.ipynb)**
   - IoT device anomaly detection
   - Streaming data processing
   - Predictive maintenance insights
   - **Time**: 50 minutes | **Level**: Intermediate

### 🔬 Advanced Techniques
6. **[06_ensemble_methods_deep_dive.ipynb](06_ensemble_methods_deep_dive.ipynb)**
   - Advanced ensemble techniques
   - Voting, stacking, and hierarchical methods
   - Custom ensemble creation
   - **Time**: 55 minutes | **Level**: Advanced

7. **[07_real_time_streaming_detection.ipynb](07_real_time_streaming_detection.ipynb)**
   - Real-time anomaly detection
   - Kafka integration examples
   - Performance optimization
   - **Time**: 50 minutes | **Level**: Advanced

8. **[08_model_explainability_tutorial.ipynb](08_model_explainability_tutorial.ipynb)**
   - SHAP explainability integration
   - Feature importance analysis
   - Interactive explanation visualizations
   - **Time**: 40 minutes | **Level**: Intermediate

### 🏭 Production Notebooks
9. **[09_production_deployment_guide.ipynb](09_production_deployment_guide.ipynb)**
   - Production deployment strategies
   - Docker containerization
   - Monitoring and logging setup
   - **Time**: 75 minutes | **Level**: Advanced

10. **[10_performance_optimization_lab.ipynb](10_performance_optimization_lab.ipynb)**
    - Benchmarking different approaches
    - Memory and CPU optimization
    - Scaling strategies
    - **Time**: 60 minutes | **Level**: Advanced

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for beginners)
Click the "Open in Colab" button at the top of any notebook to run it directly in your browser.

### Option 2: Local Jupyter Setup
```bash
# Install dependencies
pip install jupyter anomaly-detection

# Clone repository and navigate to notebooks
cd docs/notebooks

# Start Jupyter
jupyter notebook
```

### Option 3: JupyterLab
```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

## 📊 Learning Paths

### 🟢 Beginner Path (2-3 hours)
1. [Introduction to Anomaly Detection](01_introduction_to_anomaly_detection.ipynb)
2. [Algorithm Comparison](02_algorithm_comparison_tutorial.ipynb)
3. [Fraud Detection End-to-End](03_fraud_detection_end_to_end.ipynb)

### 🟡 Intermediate Path (4-5 hours)
1. [Algorithm Comparison](02_algorithm_comparison_tutorial.ipynb)
2. [Network Security Analysis](04_network_security_analysis.ipynb)
3. [IoT Sensor Monitoring](05_iot_sensor_monitoring.ipynb)
4. [Model Explainability](08_model_explainability_tutorial.ipynb)

### 🔴 Advanced Path (6-8 hours)
1. [Ensemble Methods Deep Dive](06_ensemble_methods_deep_dive.ipynb)
2. [Real-time Streaming](07_real_time_streaming_detection.ipynb)
3. [Production Deployment](09_production_deployment_guide.ipynb)
4. [Performance Optimization](10_performance_optimization_lab.ipynb)

## 🎯 By Use Case

### 🏦 Financial Services
- [Fraud Detection End-to-End](03_fraud_detection_end_to_end.ipynb)
- [Ensemble Methods](06_ensemble_methods_deep_dive.ipynb)
- [Model Explainability](08_model_explainability_tutorial.ipynb)

### 🔒 Cybersecurity
- [Network Security Analysis](04_network_security_analysis.ipynb)
- [Real-time Streaming](07_real_time_streaming_detection.ipynb)
- [Performance Optimization](10_performance_optimization_lab.ipynb)

### 🏭 Manufacturing & IoT
- [IoT Sensor Monitoring](05_iot_sensor_monitoring.ipynb)
- [Real-time Streaming](07_real_time_streaming_detection.ipynb)
- [Production Deployment](09_production_deployment_guide.ipynb)

## 🛠️ Interactive Features

Each notebook includes:
- **📊 Interactive visualizations** using Plotly and Matplotlib
- **🎛️ Parameter widgets** for real-time experimentation
- **📈 Live performance metrics** and comparisons
- **🔧 Code exercises** with solutions
- **💡 Best practice tips** and production insights
- **🔗 Links to documentation** for deeper learning

## 📦 Dependencies

All notebooks use the following core dependencies:
```python
# Core packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Anomaly detection
from anomaly_detection import DetectionService, EnsembleService, StreamingService

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, HTML

# Machine learning utilities
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
```

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing packages
pip install plotly ipywidgets scikit-learn

# Enable widgets in Jupyter
jupyter nbextension enable --py widgetsnbextension
```

**Dataset Loading Issues**
```python
# Datasets are in the ../datasets/ directory
import pandas as pd
df = pd.read_csv('../datasets/credit_card_transactions.csv')
```

**Visualization Problems**
```python
# For Plotly in JupyterLab
import plotly.io as pio
pio.renderers.default = "jupyterlab"
```

## 🤝 Contributing

We welcome contributions! To add a new notebook:

1. **Follow naming convention**: `NN_descriptive_name.ipynb`
2. **Include metadata**: Title, description, difficulty, time estimate
3. **Add to this README**: Update the appropriate section
4. **Test thoroughly**: Ensure all cells run without errors
5. **Add Google Colab link**: For easy access

## 📱 Mobile Support

While Jupyter notebooks work best on desktop, you can view them on mobile:
- **GitHub**: All notebooks render with basic formatting
- **NBViewer**: Better mobile rendering at nbviewer.jupyter.org
- **Google Colab**: Mobile app available for basic editing

## 🆘 Getting Help

- **Documentation Issues**: Check the main [troubleshooting guide](../troubleshooting.md)
- **Notebook-Specific**: Each notebook has a "Getting Help" section
- **Community**: Join discussions in the project repository
- **Examples**: See the [code templates](../templates/) for reference implementations

---

**Ready to start learning?** Begin with the [Introduction to Anomaly Detection](01_introduction_to_anomaly_detection.ipynb) notebook! 🚀