# Web Interface Quickstart Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > 📄 Web Interface Quickstart

---


Get up and running with Pynomaly's web interface in 5 minutes. This guide walks you through your first anomaly detection using the progressive web app.

## 🎯 Prerequisites

- Pynomaly server running (see [Installation Guide](installation.md))
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- Sample dataset (we'll provide one below)

## 🚀 Step 1: Access the Web Interface

### Launch the Application

1. **Open your browser** and navigate to your Pynomaly instance:
   ```
   http://localhost:8000  # Local development
   https://your-domain.com  # Production
   ```

2. **Install as PWA** (optional but recommended):
   - Look for the install icon (⊕) in your browser's address bar
   - Click "Install" to add Pynomaly to your apps
   - Launch from your desktop/app menu for native app experience

### First-Time Setup

If authentication is enabled, you'll see a login screen:
```
Username: admin
Password: admin  # Change this in production!
```

## 📊 Step 2: Upload Your First Dataset

### Using Sample Data

Create a sample CSV file with anomalies:

```python
# Generate sample data (save as sample_data.csv)
import pandas as pd
import numpy as np

# Normal data
normal_data = np.random.normal(0, 1, (1000, 3))

# Add some anomalies
anomalies = np.random.normal(5, 0.5, (50, 3))  # Outliers
data = np.vstack([normal_data, anomalies])

# Create DataFrame
df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'feature_3'])
df.to_csv('sample_data.csv', index=False)
```

### Upload Process

1. **Navigate to Datasets** - Click "Datasets" in the sidebar
2. **Click "Upload Dataset"**
3. **Select your CSV file** (sample_data.csv)
4. **Configure settings**:
   - Name: "Sample Anomaly Data"
   - Description: "Test dataset with known anomalies"
   - Type: "Tabular"
5. **Click "Upload"**

The system will validate and process your data automatically.

## 🔍 Step 3: Run Anomaly Detection

### Quick Detection

1. **Go to Detection** - Click "Detection" in the sidebar
2. **Select your dataset** - Choose "Sample Anomaly Data"
3. **Choose algorithm** - Select "Isolation Forest" (great for beginners)
4. **Set parameters**:
   ```
   Contamination: 0.05  # Expected 5% anomalies
   Estimators: 100      # Number of trees
   Max Features: 1.0    # Use all features
   ```
5. **Click "Run Detection"**

### View Results

The detection will complete in seconds. You'll see:
- **Summary**: Total samples, anomalies found, confidence score
- **Visualization**: Scatter plot showing normal (green) vs anomalous (red) points
- **Statistics**: Detailed metrics and performance data

## 📈 Step 4: Explore Visualizations

### Interactive Dashboard

1. **Navigate to Dashboard** - Click "Dashboard" in the sidebar
2. **Overview Cards** show:
   - Total datasets uploaded
   - Detections run today
   - Anomalies found
   - System status

### Data Exploration

1. **Select your dataset** from the dropdown
2. **View distribution charts**:
   - Feature histograms
   - Correlation matrix
   - Statistical summaries
3. **Explore anomaly results**:
   - Anomaly distribution pie chart
   - Score histograms
   - Timeline of detections

### Advanced Visualizations

1. **Click "Advanced Visualization"**
2. **Choose visualization type**:
   - Scatter plots with anomaly overlay
   - 3D projections for multi-dimensional data
   - Time series plots for temporal data
3. **Interactive features**:
   - Zoom and pan
   - Hover for details
   - Export as PNG/PDF

## ⚡ Step 5: Try Offline Features

### Install PWA (if not already done)

1. **Click the install prompt** or look for the install icon
2. **Install the app** to your device
3. **Launch from your apps** for native experience

### Test Offline Capability

1. **Disconnect from internet** (airplane mode or disable WiFi)
2. **Open Pynomaly PWA**
3. **Access cached data**:
   - View your uploaded datasets
   - Review previous detection results
   - Explore saved visualizations

### Run Offline Analysis

1. **Select a cached dataset**
2. **Choose offline algorithm**:
   - Z-Score Detection (fast, simple)
   - IQR Method (robust to outliers)
   - MAD Detection (median-based)
3. **Configure parameters** and run
4. **View results immediately** - no server required!

## 🎛️ Step 6: Advanced Features

### AutoML Optimization

1. **Go to AutoML** section
2. **Select your dataset**
3. **Choose optimization goal**:
   - Best accuracy
   - Fastest execution
   - Balanced performance
4. **Start optimization** - the system will test multiple algorithms
5. **Review recommendations** and apply the best configuration

### Ensemble Methods

1. **Navigate to Ensemble**
2. **Select multiple algorithms**:
   - Isolation Forest
   - Local Outlier Factor
   - One-Class SVM
3. **Configure voting strategy**:
   - Majority voting
   - Average scoring
   - Weighted combination
4. **Run ensemble detection** for improved accuracy

### Model Explainability

1. **Select a detection result**
2. **Click "Explain Results"**
3. **View explanations**:
   - Feature importance scores
   - SHAP value plots
   - Individual anomaly explanations
4. **Export explanation reports**

## 📱 Step 7: Mobile Usage

### Mobile Browser

1. **Open mobile browser** (Chrome, Safari, Firefox)
2. **Navigate to Pynomaly**
3. **Add to home screen**:
   - iOS: Share → Add to Home Screen
   - Android: Menu → Add to Home Screen

### Mobile Features

✅ **Available on Mobile**:
- Dataset browsing and basic upload
- View detection results and visualizations
- Dashboard with responsive design
- Offline analysis with cached data
- Export results and reports

📱 **Optimized for Touch**:
- Large touch targets
- Swipe gestures for navigation
- Pinch-to-zoom for charts
- Responsive design for all screen sizes

## 🔧 Step 8: Customization

### User Preferences

1. **Click your profile** (top right)
2. **Go to Settings**
3. **Configure preferences**:
   - Default algorithm settings
   - Visualization themes
   - Notification preferences
   - Export formats

### Dashboard Customization

1. **Go to Dashboard**
2. **Click "Customize"**
3. **Drag and drop widgets**:
   - Rearrange chart positions
   - Show/hide specific metrics
   - Resize visualization panels
4. **Save your layout**

### Alert Configuration

1. **Navigate to Monitoring**
2. **Set up alerts**:
   - Anomaly rate thresholds
   - Detection failure notifications
   - Data quality warnings
3. **Configure delivery**:
   - In-app notifications
   - Email alerts (if configured)
   - Browser push notifications

## 📊 Step 9: Export and Sharing

### Export Results

1. **Select detection results**
2. **Click "Export"**
3. **Choose format**:
   - CSV for data analysis
   - Excel for business reports
   - JSON for API integration
   - PDF for presentations

### Generate Reports

1. **Go to Reports section**
2. **Select report type**:
   - Executive summary
   - Technical analysis
   - Comparison report
3. **Configure parameters**:
   - Date range
   - Datasets included
   - Metrics to display
4. **Generate and download**

### Share Visualizations

1. **Create visualization**
2. **Click "Share"**
3. **Options available**:
   - Direct link (if permissions allow)
   - Export as image
   - Embed code for websites
   - Print-friendly version

## 🚀 Next Steps

### Continue Learning

- **[Complete User Guide](../user-guides/)** - Comprehensive feature overview
- **[Algorithm Selection](../reference/algorithms/)** - Choose optimal algorithms
- **[Advanced Features](../user-guides/advanced-features/)** - Expert-level capabilities

### Production Setup

- **[Deployment Guide](../deployment/)** - Production deployment
- **[Security Configuration](../deployment/security.md)** - Secure your installation
- **[Monitoring Setup](../user-guides/basic-usage/monitoring.md)** - Production monitoring

### Integration

- **[API Integration](../developer-guides/api-integration/)** - Connect external systems
- **[Python SDK](../developer-guides/api-integration/python-sdk.md)** - Programmatic access
- **[Authentication](../developer-guides/api-integration/authentication.md)** - Security setup

## 🎯 Quick Reference

### Essential Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Upload Dataset | `Ctrl/Cmd + U` |
| Run Detection | `Ctrl/Cmd + R` |
| Go to Dashboard | `Ctrl/Cmd + D` |
| Search/Filter | `Ctrl/Cmd + F` |
| Export Results | `Ctrl/Cmd + E` |
| Help | `F1` or `?` |

### Common Parameters

#### **Isolation Forest** (Recommended for beginners)
```
contamination: 0.05-0.1    # Expected anomaly rate
n_estimators: 100-200      # Number of trees
max_features: 1.0          # Feature subset size
```

#### **Local Outlier Factor**
```
n_neighbors: 20            # Neighborhood size
contamination: 0.05        # Expected anomaly rate
```

#### **One-Class SVM**
```
kernel: 'rbf'              # Kernel function
gamma: 'scale'             # Kernel coefficient
nu: 0.05                   # Anomaly fraction
```

### Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow upload | Check file size (<50MB recommended) |
| Detection fails | Verify data format and parameters |
| Charts not loading | Enable JavaScript and refresh |
| Offline features missing | Install as PWA and ensure cache |
| Mobile layout issues | Update browser to latest version |

## 💡 Pro Tips

### Performance Optimization
- **Sample large datasets** for faster visualization (>10K rows)
- **Use appropriate algorithms** for your data type and size
- **Cache frequently used datasets** for offline access
- **Close unused browser tabs** for better performance

### Best Practices
- **Start with simple algorithms** (Isolation Forest, Z-Score)
- **Validate results** with domain knowledge
- **Document your parameters** for reproducibility
- **Regular data quality checks** before detection
- **Backup important results** before clearing cache

---

**🎉 Congratulations!** You've completed the web interface quickstart. You're now ready to explore Pynomaly's full capabilities.

**Need Help?**
- 📖 [Full Documentation](../user-guides/)
- 🐛 [Troubleshooting Guide](../user-guides/troubleshooting/)
- 💬 [Community Support](https://github.com/your-org/pynomaly/discussions)
- 📧 [Contact Support](mailto:support@your-domain.com)

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
