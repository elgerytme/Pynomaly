# Data Studio Package

An interactive data exploration and analysis workspace for data scientists and analysts.

## Overview

Data Studio provides a comprehensive interactive environment for data exploration, analysis, and collaborative research. It combines notebook-style interfaces with advanced data visualization and analysis tools.

## Features

- **Interactive Notebooks**: Jupyter-based notebook environment with enhanced features
- **Advanced Visualizations**: Rich, interactive charts and graphs for data exploration
- **Collaborative Workspaces**: Shared environments for team collaboration
- **Data Connectors**: Direct connections to various data sources and formats
- **Real-time Analysis**: Live data analysis and streaming capabilities
- **Export Capabilities**: Export analyses, visualizations, and reports in multiple formats
- **Version Control**: Built-in version control for notebooks and analyses
- **Template Library**: Pre-built templates for common analysis patterns

## Architecture

```
data_studio/
├── domain/                 # Core studio business logic
│   ├── entities/          # Notebook, Analysis, Workspace entities
│   ├── services/          # Analysis and collaboration services
│   └── value_objects/     # Studio-specific value objects
├── application/           # Use cases and orchestration  
│   ├── services/          # Application services
│   ├── use_cases/         # Studio workflows
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations
│   ├── adapters/          # Jupyter, visualization adapters
│   ├── storage/           # Notebook and workspace storage
│   └── connectors/        # Data source connectors
└── presentation/          # Interfaces
    ├── web/               # Web-based studio interface
    ├── api/               # REST API endpoints
    └── cli/               # Command-line tools
```

## Quick Start

```python
from src.packages.data.data_studio.application.services import StudioService
from src.packages.data.data_studio.domain.entities import Workspace, Notebook

# Create studio service
studio = StudioService()

# Create a new workspace
workspace = studio.create_workspace(
    name="Customer Analysis Project",
    description="Analysis of customer behavior patterns",
    team_members=["analyst1@company.com", "analyst2@company.com"]
)

# Create a new notebook
notebook = studio.create_notebook(
    workspace_id=workspace.id,
    name="Customer Segmentation Analysis",
    template="customer_analysis"
)

# Load and analyze data
data_connector = studio.get_data_connector("postgresql")
customer_data = data_connector.load_data("SELECT * FROM customers")

# Create visualization
viz = studio.create_visualization(
    data=customer_data,
    chart_type="scatter",
    x_axis="age",
    y_axis="lifetime_value",
    color_by="segment"
)

# Save and share analysis
studio.save_notebook(notebook.id)
studio.share_workspace(workspace.id, permissions={"read": ["team@company.com"]})
```

## Use Cases

- **Exploratory Data Analysis**: Interactive data exploration and hypothesis testing
- **Data Science Research**: Advanced analytics and machine learning experimentation
- **Business Intelligence**: Self-service analytics for business users
- **Collaborative Analysis**: Team-based data analysis projects
- **Report Generation**: Automated and interactive reporting
- **Data Visualization**: Advanced charting and dashboard creation

## Integration

Integrates with other data domain packages:

```python
# With data quality for validated analysis
from src.packages.data.data_quality.application.services import DataQualityService

quality_service = DataQualityService()
quality_report = quality_service.assess_data_quality(dataset)
studio.display_quality_report(quality_report)

# With data observability for lineage tracking
from src.packages.data.observability.application.services import LineageService

lineage_service = LineageService()
studio.show_data_lineage(dataset_id)
```

## Installation

```bash
# Install from package directory
cd src/packages/data/data_studio
pip install -e .

# Install with visualization dependencies
pip install -e ".[viz,jupyter]"
```

## Configuration

```yaml
# studio_config.yaml
data_studio:
  jupyter:
    kernel: "python3"
    extensions: ["plotly", "altair", "bokeh"]
    memory_limit: "4GB"
  
  visualization:
    default_theme: "corporate"
    interactive: true
    export_formats: ["png", "svg", "html", "pdf"]
  
  collaboration:
    real_time_editing: true
    version_control: "git"
    backup_interval: "5min"
  
  data_connectors:
    - name: "postgresql"
      enabled: true
    - name: "s3"
      enabled: true
    - name: "bigquery"
      enabled: false
```

## License

MIT License
