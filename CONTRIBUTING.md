# Contributing to Pynomaly
## Environment Recreation

To set up a development environment, you'll need to have Poetry installed. Then, run the following commands to install the dependencies:

```bash
poetry install
```

If you are using Hatch, you can set up environments as specified in the `pyproject.toml` under `[tool.hatch.envs]`.

## Build Hook Safety

Build hooks are configured with `pre-commit` to ensure code quality and secure development practices. Run the following command to set up the pre-commit hooks:

```bash
pre-commit install
```

You can check all hooks by running:

```bash
pre-commit run --all-files
```

## Running Tests

Tests are managed with `pytest` using the Hatch environment specified in `pyproject.toml`. To run the tests, use:

```bash
hatch run test
```

For a coverage report, execute:

```bash
hatch run test-cov
```

## Recommended PowerShell Commands

The following PowerShell commands are provided to help manage development tasks:

- **Build the project**:
  ```powershell
  Build
  ```

- **Clean build artifacts**:
  ```powershell
  Clean
  ```

- **Start development server**:
  ```powershell
  Start-Dev
  ```

Ensure you have imported `tasks.ps1` in your PowerShell session like so:

```powershell
. .\tasks.ps1
```
## Development Setup

This project uses Poetry for dependency management. Make sure you have Poetry installed before proceeding.

## PowerShell Build Tasks

For Windows users, we provide a `tasks.ps1` PowerShell script that proxies common make targets. To use these tasks:

1. Import the script in your PowerShell session:
   ```powershell
   . .\tasks.ps1
   ```

2. Use the available functions:

### Available Tasks

- **Build**: Build the project using Poetry
  ```powershell
  Build
  ```
  This runs `poetry run python -m build` to create distribution packages.

- **Clean**: Remove build artifacts
  ```powershell
  Clean
  ```
  This removes the `dist`, `build`, and `.pytest_cache` directories.

- **Start-Dev**: Start the development server
  ```powershell
  Start-Dev
  ```
  This runs `poetry run uvicorn pynomaly.api:app --reload` to start the API server with auto-reload enabled.

## Example Workflow

```powershell
# Import the tasks
. .\tasks.ps1

# Clean previous builds
Clean

# Build the project
Build

# Start development server
Start-Dev
```
