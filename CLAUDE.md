# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Pynomaly is a Python project currently in its initial setup phase. The project uses Python 3.11.4 with a virtual environment.

## Development Environment

### Python Version
- Python 3.11.4 (managed via pyenv-win)
- Virtual environment located at `.venv/`

### Activating the Virtual Environment
```bash
# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# On Linux/Mac (bash/zsh)
source .venv/bin/activate
```

### Installing Dependencies
Once the project has a `requirements.txt` or `pyproject.toml`, use:
```bash
pip install -r requirements.txt
# or
pip install -e .  # if using pyproject.toml
```

## Project Structure
The project is newly initialized and doesn't have established conventions yet. When adding code:
- Follow PEP 8 style guidelines for Python code
- Create appropriate directory structure as needed (e.g., `src/`, `tests/`, `docs/`)
- Add a `requirements.txt` or `pyproject.toml` when introducing dependencies

## Common Commands
Since this is a new project without established tooling, here are typical Python development commands:

### Running Python Files
```bash
python <filename>.py
```

### Installing Packages
```bash
pip install <package-name>
pip freeze > requirements.txt  # Save dependencies
```

### Testing
When tests are added, typical commands would be:
```bash
pytest  # if using pytest
python -m unittest  # if using unittest
```

### Linting and Formatting
When linting tools are added:
```bash
black .  # if using black formatter
flake8  # if using flake8 linter
mypy .  # if using mypy for type checking
```

## Important Notes
- The project name "Pynomaly" suggests potential anomaly detection functionality
- Always ensure the virtual environment is activated before installing packages or running code
- The `.gitignore` currently excludes the VS Code workspace file (`Pynomaly.code-workspace`)