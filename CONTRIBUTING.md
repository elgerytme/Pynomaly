# Contributing to Pynomaly

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
