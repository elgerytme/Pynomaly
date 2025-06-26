# Progressive Web App (PWA) User Guide

Welcome to Pynomaly's Progressive Web App - a modern, offline-capable web application that brings enterprise-grade anomaly detection to your fingertips with native app-like experiences.

## 🌟 Overview

Pynomaly's PWA provides:
- **Offline Functionality** - Continue working without internet connectivity
- **Native App Experience** - Install on desktop and mobile devices
- **Real-time Synchronization** - Seamless data sync when online
- **Advanced Visualizations** - Interactive charts and dashboards
- **Local Analysis** - Run anomaly detection algorithms in your browser

---

## 🚀 Getting Started

### Accessing the PWA

1. **Open your browser** and navigate to your Pynomaly instance
2. **Look for the install prompt** - Most browsers will show an install button in the address bar
3. **Click "Install"** or use the install button in the app interface
4. **Launch from your device** - Find Pynomaly in your apps menu or desktop

### Browser Support

✅ **Recommended Browsers:**
- Chrome 90+ (Desktop & Mobile)
- Edge 90+ (Desktop & Mobile)
- Safari 14+ (Desktop & Mobile)
- Firefox 88+ (Desktop & Mobile)

---

## 📱 Installation Guide

### Desktop Installation

#### **Windows**
1. Open Chrome/Edge and navigate to Pynomaly
2. Click the **install icon** (⊞) in the address bar
3. Click "Install" in the confirmation dialog
4. Find Pynomaly in your Start Menu or Desktop

#### **macOS**
1. Open Safari/Chrome and navigate to Pynomaly
2. Click **Share** → **Add to Dock** (Safari) or the install icon (Chrome)
3. Confirm installation
4. Launch from Launchpad or Applications folder

#### **Linux**
1. Open your supported browser and navigate to Pynomaly
2. Click the **install icon** in the address bar
3. Confirm installation
4. Find Pynomaly in your applications menu

### Mobile Installation

#### **iOS (iPhone/iPad)**
1. Open **Safari** and navigate to Pynomaly
2. Tap the **Share button** (□↗)
3. Scroll down and tap **"Add to Home Screen"**
4. Customize the name and tap **"Add"**
5. Find Pynomaly on your home screen

#### **Android**
1. Open **Chrome** and navigate to Pynomaly
2. Tap the **three dots menu** (⋮)
3. Select **"Add to Home screen"**
4. Customize the name and tap **"Add"**
5. Find Pynomaly in your app drawer or home screen

---

## 🔄 Offline Capabilities

### What Works Offline

✅ **Available Offline:**
- View cached datasets and analysis results
- Run local anomaly detection algorithms
- Interactive data visualizations
- Dashboard with statistical overviews
- Export analysis results
- User preferences and settings

❌ **Requires Internet:**
- Upload new datasets
- Advanced ML algorithms (server-side)
- User authentication
- Real-time collaboration
- External data sources

### Offline Algorithms

Pynomaly includes browser-based algorithms for offline analysis:

#### **Z-Score Detection**
```javascript
// Statistical outlier detection using Z-scores
Parameters:
- threshold: 3.0 (standard deviations)
- Best for: Normally distributed data
```

#### **Interquartile Range (IQR)**
```javascript
// Outlier detection using IQR method
Parameters:
- factor: 1.5 (IQR multiplier)
- Best for: Skewed distributions
```

#### **Simple Isolation Detection**
```javascript
// Basic isolation-based anomaly detection
Parameters:
- contamination: 0.1 (expected anomaly rate)
- Best for: High-dimensional data
```

#### **Median Absolute Deviation (MAD)**
```javascript
// Robust outlier detection using MAD
Parameters:
- threshold: 3.5 (modified Z-score threshold)
- Best for: Data with outliers in distribution
```

---

## 📊 Dashboard Features

### Overview Cards

The dashboard displays real-time statistics:

- **Total Datasets** - Number of cached datasets available offline
- **Detections Run** - Count of completed anomaly detections
- **Anomalies Found** - Total anomalies detected across all analyses
- **Cached Data Size** - Amount of data available offline

### Interactive Charts

#### **Dataset Distribution**
- Pie chart showing dataset types and categories
- Click segments for detailed breakdowns
- Export as PNG or PDF

#### **Algorithm Performance**
- Bar chart comparing algorithm processing times
- Line overlay showing anomaly detection rates
- Performance trends over time

#### **Detection Timeline**
- Activity timeline showing detection frequency
- Anomaly discovery patterns
- Time-based analysis trends

### Real-time Analysis

Run anomaly detection directly in your browser:

1. **Select Dataset** - Choose from cached datasets
2. **Choose Algorithm** - Select appropriate detection method
3. **Configure Parameters** - Adjust algorithm settings
4. **Run Analysis** - Execute detection locally
5. **View Results** - Interactive visualization of findings

---

## 🔄 Data Synchronization

### Sync Strategies

Pynomaly offers three synchronization modes:

#### **Immediate Sync**
- Changes sync automatically when online
- Real-time data consistency
- Best for: Always-connected environments

#### **Smart Sync** (Default)
- Intelligent batching of changes
- Optimized for mobile networks
- Background sync when idle

#### **Manual Sync**
- User-controlled synchronization
- Perfect for bandwidth-limited scenarios
- Sync only when needed

### Conflict Resolution

When data conflicts occur, choose your resolution strategy:

#### **Server Wins**
- Accept server version
- Overwrite local changes
- Use when: Server data is authoritative

#### **Client Wins**
- Keep local version
- Overwrite server data
- Use when: Local changes are critical

#### **Manual Merge**
- Review both versions
- Create custom resolution
- Use when: Both versions have value

### Sync Status Indicators

Monitor synchronization status:

- 🟢 **Online** - Connected and synced
- 🟡 **Pending** - Changes waiting to sync
- 🔴 **Offline** - No internet connection
- ⚠️ **Conflict** - Requires resolution

---

## 📈 Data Visualization

### Chart Types

#### **Distribution Analysis**
- **Histograms** - Data distribution patterns
- **Box Plots** - Quartile and outlier visualization
- **Violin Plots** - Distribution density curves

#### **Correlation Analysis**
- **Heatmaps** - Feature correlation matrices
- **Scatter Plots** - Relationship exploration
- **Pair Plots** - Multi-dimensional analysis

#### **Anomaly Visualization**
- **Anomaly Scatter** - Normal vs. anomalous points
- **Score Distribution** - Anomaly score histograms
- **Time Series** - Temporal anomaly patterns

### Interactive Features

- **Zoom and Pan** - Explore data in detail
- **Hover Tooltips** - Detailed point information
- **Selection Tools** - Highlight specific regions
- **Export Options** - Save charts as images

### Customization

#### **Chart Appearance**
- Color schemes and themes
- Axis labels and titles
- Legend positioning
- Grid and background options

#### **Data Filtering**
- Date range selection
- Value range filtering
- Category exclusion
- Sample size limits

---

## 💾 Offline Storage

### Storage Management

Pynomaly automatically manages offline storage:

#### **Data Types Cached**
- **Datasets** - Raw data and metadata
- **Analysis Results** - Detection outcomes and scores
- **User Preferences** - Settings and configurations
- **Visualization State** - Chart configurations and views

#### **Storage Limits**
- **Mobile Browsers** - ~5-50 MB (varies by device)
- **Desktop Browsers** - ~100-500 MB (varies by browser)
- **Automatic Cleanup** - Removes old data when space is low

### Manual Storage Control

#### **Clear Cache**
```javascript
// Clear specific data types
PWA.clearCache('datasets');  // Clear dataset cache
PWA.clearCache('results');   // Clear analysis results
PWA.clearCache();           // Clear all cached data
```

#### **Storage Status**
```javascript
// Check storage usage
const status = await PWA.getStatus();
console.log(`Cache size: ${status.cacheInfo.totalSize} bytes`);
```

---

## ⚡ Performance Optimization

### Best Practices

#### **Data Management**
- **Limit Dataset Size** - Keep datasets under 10MB for optimal performance
- **Regular Cleanup** - Remove old analysis results periodically
- **Selective Caching** - Only cache frequently used datasets

#### **Visualization Performance**
- **Sample Large Datasets** - Use data sampling for >10,000 points
- **Optimize Chart Types** - Choose appropriate visualizations
- **Debounce Interactions** - Smooth user experience

#### **Browser Optimization**
- **Close Unused Tabs** - Free up memory for analysis
- **Regular Updates** - Keep browser updated for performance
- **Hardware Acceleration** - Enable GPU acceleration if available

### Memory Management

#### **Monitoring Usage**
```javascript
// Check memory usage
if (performance.memory) {
  const used = performance.memory.usedJSHeapSize;
  const limit = performance.memory.jsHeapSizeLimit;
  console.log(`Memory: ${used}/${limit} bytes`);
}
```

#### **Optimization Tips**
- Close inactive visualizations
- Clear completed analysis results
- Restart app if memory usage is high
- Use simpler algorithms for large datasets

---

## 🔧 Troubleshooting

### Common Issues

#### **Installation Problems**

**Problem**: Install button not appearing
```
Solution:
1. Ensure HTTPS connection
2. Check browser PWA support
3. Clear browser cache
4. Try incognito/private mode
```

**Problem**: App won't install on iOS
```
Solution:
1. Use Safari browser (not Chrome)
2. Ensure iOS 11.3+ version
3. Add to Home Screen manually
4. Check storage space availability
```

#### **Sync Issues**

**Problem**: Data not syncing
```
Solution:
1. Check internet connection
2. Verify authentication status
3. Clear sync queue manually
4. Restart application
```

**Problem**: Sync conflicts not resolving
```
Solution:
1. Choose manual resolution
2. Review conflict details
3. Select appropriate strategy
4. Force sync if needed
```

#### **Performance Issues**

**Problem**: Slow analysis performance
```
Solution:
1. Reduce dataset size
2. Use simpler algorithms
3. Close other applications
4. Clear browser cache
```

**Problem**: Charts not rendering
```
Solution:
1. Check browser compatibility
2. Enable hardware acceleration
3. Reduce chart complexity
4. Try different chart type
```

### Error Messages

#### **Storage Errors**
```
"Insufficient storage space"
→ Clear cached data or free device storage

"Storage quota exceeded"
→ Reduce dataset sizes or clear old results

"Unable to save data"
→ Check browser permissions and storage
```

#### **Network Errors**
```
"Sync failed"
→ Check internet connection and retry

"Authentication expired"
→ Re-login and restart sync

"Server unavailable"
→ Try again later or contact admin
```

### Recovery Procedures

#### **Reset Application State**
```javascript
// Complete reset (use with caution)
PWA.clearCache();           // Clear all data
localStorage.clear();       // Clear preferences
location.reload();          // Restart app
```

#### **Backup Important Data**
```javascript
// Export critical analysis results
const results = await OfflineDetector.getDetectionHistory();
const backup = JSON.stringify(results);
// Save backup externally
```

---

## 🔒 Security & Privacy

### Data Protection

#### **Local Storage Security**
- **Encrypted Storage** - Sensitive data encrypted at rest
- **Session Management** - Automatic logout on inactivity
- **Secure Transmission** - HTTPS for all network communication

#### **Privacy Controls**
- **Local Processing** - Analysis runs in your browser
- **Data Retention** - Configurable data expiration
- **User Control** - Full control over cached data

### Best Practices

#### **Secure Usage**
- Always log out on shared devices
- Regular cache cleanup on public computers
- Use device screen locks for mobile installations
- Keep browsers updated for security patches

#### **Corporate Environments**
- Verify PWA policies with IT department
- Use managed browser deployments
- Configure appropriate data retention policies
- Monitor storage usage and compliance

---

## 🚀 Advanced Features

### Custom Algorithms

Extend offline capabilities with custom algorithms:

```javascript
// Register custom algorithm
OfflineDetector.registerAlgorithm('custom_zscore', {
  name: 'Custom Z-Score',
  description: 'Modified Z-score with custom parameters',
  parameters: { threshold: 2.5, window: 100 },
  detect: (data, config) => {
    // Custom algorithm implementation
    return { anomalies, scores, statistics };
  }
});
```

### Integration APIs

#### **External Tool Integration**
```javascript
// Export data for external tools
const results = await OfflineDetector.getDetectionHistory();
const csv = convertToCSV(results);
downloadFile(csv, 'anomaly_results.csv');
```

#### **Custom Visualizations**
```javascript
// Create custom chart types
const customChart = echarts.init(container);
customChart.setOption({
  // Custom ECharts configuration
  series: [{
    type: 'custom',
    data: anomalyData,
    renderItem: customRenderFunction
  }]
});
```

### Automation

#### **Scheduled Analysis**
```javascript
// Set up recurring analysis
const scheduler = new AnalysisScheduler();
scheduler.schedule('daily', {
  dataset: 'production_data',
  algorithm: 'isolation',
  notify: true
});
```

#### **Alert Integration**
```javascript
// Configure alert thresholds
PWAManager.setAlertThreshold({
  anomalyRate: 0.15,
  scoreThreshold: 0.8,
  notification: true
});
```

---

## 📚 API Reference

### PWA Manager API

#### **Installation Management**
```javascript
// Check installation status
const isInstalled = PWAManager.isAppInstalled();

// Trigger install prompt
await PWAManager.installPWA();

// Get app status
const status = await PWAManager.getAppStatus();
```

#### **Sync Management**
```javascript
// Manual sync trigger
await SyncManager.forceSyncAll();

// Configure sync strategy
SyncManager.setSyncStrategy('smart');

// Get sync status
const syncStatus = SyncManager.getSyncStatus();
```

### Offline Detector API

#### **Algorithm Management**
```javascript
// Get available algorithms
const algorithms = OfflineDetector.getAlgorithms();

// Run detection
const result = await OfflineDetector.detectAnomalies(
  datasetId,
  algorithmId,
  parameters
);

// Get detection history
const history = await OfflineDetector.getDetectionHistory();
```

### Visualization API

#### **Chart Management**
```javascript
// Create visualization
const visualizer = new OfflineVisualizer();
await visualizer.selectDataset(datasetId);
await visualizer.renderDatasetVisualization();

// Export visualization
visualizer.exportVisualization('png');
```

---

## 🎯 Best Practices

### Performance
- Keep datasets under 10MB for optimal performance
- Use data sampling for large visualizations
- Clear old analysis results regularly
- Monitor memory usage during long sessions

### User Experience
- Provide clear offline/online status indicators
- Use progressive enhancement for optional features
- Implement graceful degradation for unsupported browsers
- Optimize for both desktop and mobile usage

### Data Management
- Implement data retention policies
- Regular cache cleanup and optimization
- Backup critical analysis results
- Monitor storage quota usage

### Security
- Encrypt sensitive cached data
- Implement session timeout policies
- Regular security updates and patches
- Audit data access and usage patterns

---

## 🔗 Related Documentation

- **[Installation Guide](../getting-started/installation.md)** - Basic setup and installation
- **[User Interface Guide](./basic-usage/dashboard.md)** - Dashboard and interface usage
- **[API Integration](../developer-guides/api-integration/)** - Development and integration
- **[Deployment Guide](../deployment/)** - Production deployment and configuration

---

*Need help? Check our [troubleshooting guide](./troubleshooting/) or [contact support](https://github.com/your-org/pynomaly/discussions).*