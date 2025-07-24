# Data Quality Package

This package provides functionalities for data profiling, data quality checks, and data quality rules.

## Architecture Overview

The `data_quality` package follows a Clean Architecture approach, separating concerns into distinct layers:

- **Domain Layer (`src/data_quality/domain`):** Contains the core business logic and entities, such as `DataProfile`, `DataQualityCheck`, and `DataQualityRule`.
- **Application Layer (`src/data_quality/application`):** Orchestrates the domain entities and defines use cases (e.g., `CreateDataProfileUseCase`, `RunDataQualityCheckUseCase`). It also defines abstract repository interfaces (ports).
- **Infrastructure Layer (`src/data_quality/infrastructure`):** Provides concrete implementations for external concerns, including database repositories (using SQLAlchemy) and data source adapters (e.g., `PandasCSVAdapter`).
- **Presentation Layer (`src/data_quality/presentation`):** Handles user interaction through a Command Line Interface (CLI) and a RESTful API.

## Usage Examples

### Command Line Interface (CLI)

To use the CLI, navigate to the monorepo root and activate your virtual environment. Then, you can run commands like:

**Create a Data Profile:**

```bash
python src/packages/data/data_quality/src/data_quality/presentation/cli/main.py profile create my_dataset --file-path /path/to/your/data.csv
```

**Create a Data Quality Rule:**

```bash
python src/packages/data/data_quality/src/data_quality/presentation/cli/main.py rule create "NotNullCheck" "my_dataset" --description "Ensures a column is not null" --rule-type NOT_NULL --severity ERROR --column-name my_column --operator IS_NOT_NULL
```

**Run a Data Quality Check:**

```bash
# First, create a check using the API or by directly interacting with the service
# (assuming you have a check_id, e.g., from the API response)
python src/packages/data/data_quality/src/data_quality/presentation/cli/main.py check run <your_check_id> --file-path /path/to/your/data.csv
```

**Get a Data Quality Rule:**

```bash
python src/packages/data/data_quality/src/data_quality/presentation/cli/main.py rule get <your_rule_id>
```

### RESTful API

To start the API server, navigate to the `data_quality` package directory and run:

```bash
cd src/packages/data/data_quality/src/data_quality/presentation/api
uvicorn main:app --reload --port 8000
```

Once the server is running, you can access the API documentation at `http://localhost:8000/docs`.

**Example API Usage (using `curl` or a similar tool):**

**Create a Data Profile:**

```bash
curl -X POST "http://localhost:8000/profiles" \
-H "Content-Type: application/json" \
-d '{ "dataset_name": "api_test_dataset", "file_path": "/path/to/your/api_data.csv" }'
```

**Create a Data Quality Rule:**

```bash
curl -X POST "http://localhost:8000/rules" \
-H "Content-Type: application/json" \
-d '{ "name": "APINotNullRule", "description": "API rule for not null", "rule_type": "NOT_NULL", "severity": "WARNING", "dataset_name": "api_test_dataset", "conditions": [{"column_name": "api_column", "operator": "IS_NOT_NULL"}] }'
```

**Create a Data Quality Check (requires a rule_id from the previous step):**

```bash
curl -X POST "http://localhost:8000/checks" \
-H "Content-Type: application/json" \
-d '{ "name": "APICheck", "description": "API check for not null", "check_type": "COMPLETENESS", "rule_id": "<your_rule_id>", "dataset_name": "api_test_dataset", "column_name": "api_column" }'
```

**Run a Data Quality Check:**

```bash
curl -X POST "http://localhost:8000/checks/<your_check_id>/run" \
-H "Content-Type: application/json" \
-d '{ "file_path": "/path/to/your/api_data.csv" }'
```