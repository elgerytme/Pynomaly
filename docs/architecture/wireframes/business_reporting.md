# Business User UI - Reporting Interface Wireframe

## Layout
```
+----------------------------------------------------------+
|  Header: Anomaly Detection Reports                      |
+----------------------------------------------------------+
|  Report Templates           |  Report Configuration      |
|  +------------------------+ |  +----------------------+  |
|  | □ Executive Summary     | |  | Date Range:          |  |
|  | □ Technical Analysis    | |  | [From] [To]          |  |
|  | □ Trend Report         | |  |                      |  |
|  | □ Custom Template      | |  | Filters:             |  |
|  +------------------------+ |  | ☑ High Priority      |  |
|                             |  | ☐ False Positives    |  |
|  Recent Reports             |  | ☑ Include Charts     |  |
|  +------------------------+ |  +----------------------+  |
|  | Weekly Summary - 7/1    | |                          |
|  | Monthly Trend - 6/30    | |  [Generate Report]       |
|  | Custom Analysis - 6/28  | |  [Schedule Report]       |
|  +------------------------+ |  [Save Template]         |
|                             |                          |
|  Report Preview             |                          |
|  +-----------------------------------------------+     |
|  | [PDF/HTML preview of selected report]        |     |
|  |                                               |     |
|  | Executive Summary                             |     |
|  | - 47 anomalies detected this week            |     |
|  | - 12% increase from last week                |     |
|  | - Top risk areas: Payment processing         |     |
|  +-----------------------------------------------+     |
+----------------------------------------------------------+
|  Footer: Export options (PDF, Excel, Email)            |
+----------------------------------------------------------+
```

## Key Features
- Pre-built report templates
- Custom report builder
- Scheduled report generation
- Multiple export formats
- Email distribution

## User Persona: Business User
- **Needs**: Regular reports for stakeholders, trend analysis
- **Goals**: Automated reporting, clear visualizations
- **Pain Points**: Manual report creation, inconsistent formatting
