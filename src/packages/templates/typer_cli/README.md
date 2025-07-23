# Typer CLI Template

A feature-rich CLI application template built with Typer and Rich.

## Features

- Modern CLI with Typer framework
- Beautiful output with Rich formatting
- Configuration management
- Data processing commands
- Process management
- Structured logging
- Environment configuration
- Interactive prompts

## Usage

This template uses placeholders that should be replaced when creating a new project:

- `{{package_name}}` - Your package name (e.g., "my_cli")
- `{{PACKAGE_NAME_UPPER}}` - Uppercase package name for env vars (e.g., "MY_CLI")
- `{{description}}` - Your project description
- `{{author}}` - Author name

## Project Structure

```
typer_cli/
├── src/
│   └── {{package_name}}/
│       ├── commands/     # CLI command modules
│       ├── core/         # Core functionality
│       └── utils/        # Utility modules
├── .env.template
├── pyproject.toml.template
└── README.md
```

## Commands

### Main Commands
- `{{package_name}} --help` - Show help
- `{{package_name}} --version` - Show version
- `{{package_name}} init` - Initialize configuration
- `{{package_name}} info` - Show application info

### Configuration Commands
- `{{package_name}} config show` - Show configuration
- `{{package_name}} config set KEY VALUE` - Set config value
- `{{package_name}} config get KEY` - Get config value
- `{{package_name}} config reset` - Reset to defaults
- `{{package_name}} config interactive` - Interactive setup

### Data Commands
- `{{package_name}} data process INPUT` - Process data file
- `{{package_name}} data stats FILE` - Show file statistics
- `{{package_name}} data merge FILE1 FILE2` - Merge files
- `{{package_name}} data validate FILE` - Validate data

### Process Commands
- `{{package_name}} process run COMMAND` - Run a process
- `{{package_name}} process list` - List processes
- `{{package_name}} process stop NAME` - Stop process
- `{{package_name}} process logs NAME` - Show logs
- `{{package_name}} process monitor` - Monitor processes

## Getting Started

1. Replace all template variables
2. Install dependencies: `poetry install`
3. Set up environment variables in `.env`
4. Initialize configuration: `{{package_name}} init`
5. Run commands: `{{package_name}} --help`