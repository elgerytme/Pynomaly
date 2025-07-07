# Business User UI - Analytics Dashboard Wireframe

## Layout
```
+----------------------------------------------------------+
|  Header: Anomaly Detection Analytics                    |
+----------------------------------------------------------+
|  KPI Overview                        |  Quick Actions   |
|  +--------------------------------+  |  +------------+  |
|  | Anomalies Detected: 47         |  |  | Run Scan   |  |
|  | Detection Accuracy: 92.5%      |  |  | Export     |  |
|  | False Positives: 3.2%          |  |  | Schedule   |  |
|  | Last Scan: 2 hours ago         |  |  +------------+  |
|  +--------------------------------+  |                  |
|                                      |                  |
|  Anomaly Trends                      |  Risk Score      |
|  +--------------------------------+  |  +------------+  |
|  | [Time series chart showing     |  |  | HIGH: 12   |  |
|  |  anomaly detection over time]  |  |  | MED:  23   |  |
|  |                                |  |  | LOW:  12   |  |
|  +--------------------------------+  |  +------------+  |
|                                      |                  |
|  Recent Detections                   |                  |
|  +------------------------------------------------+    |
|  | Time     | Type    | Severity | Actions        |    |
|  | 10:30    | Fraud   | High     | [Investigate]  |    |
|  | 09:15    | Outlier | Medium   | [Review]       |    |
|  +------------------------------------------------+    |
+----------------------------------------------------------+
|  Footer: Data refresh status                            |
+----------------------------------------------------------+
```

## Key Features
- Real-time anomaly monitoring
- Interactive charts and graphs
- Customizable KPI widgets
- Export capabilities
- Alert configuration

## User Persona: Business User
- **Needs**: Monitor business metrics, identify trends
- **Goals**: Quick insights, actionable intelligence
- **Pain Points**: Complex technical interfaces, delayed data
