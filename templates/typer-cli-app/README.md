# Typer CLI App Template

A comprehensive command-line interface template built with Typer, featuring rich output, configuration management, and extensible command structure.

## Features

- **Typer Framework**: Modern CLI framework with type hints
- **Rich Output**: Beautiful terminal output with colors and formatting
- **Configuration Management**: YAML/JSON/TOML configuration support
- **Plugin System**: Extensible command structure
- **Auto-completion**: Shell completion for commands and options
- **Progress Bars**: Visual progress indicators
- **Logging**: Structured logging with multiple levels
- **Testing**: Comprehensive CLI testing with Typer's test client
- **Packaging**: Distribution-ready with entry points
- **Cross-platform**: Works on Windows, macOS, and Linux

## Directory Structure

```
typer-cli-app/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── cli_app/
│       ├── commands/     # Command modules
│       ├── core/         # Core logic
│       ├── config/       # Configuration handling
│       ├── utils/        # Utility functions
│       ├── plugins/      # Plugin system
│       └── cli.py        # Main CLI entry point
├── pkg/                  # Package metadata
├── examples/             # Usage examples
├── tests/                # Test suites
├── .github/              # GitHub workflows
├── scripts/              # Automation scripts
├── pyproject.toml        # Project configuration
├── Dockerfile           # Container configuration
├── README.md            # Project documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-cli-app
   cd my-cli-app
   ```

2. **Install the CLI**:
   ```bash
   pip install -e .
   ```

3. **Run the CLI**:
   ```bash
   cli-app --help
   cli-app hello World
   ```

4. **Enable shell completion**:
   ```bash
   # Bash
   cli-app --install-completion bash
   
   # Zsh  
   cli-app --install-completion zsh
   
   # Fish
   cli-app --install-completion fish
   ```

## Development

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"
```

### Running the CLI in Development

```bash
# Run directly with Python
python -m cli_app --help

# Or use the installed entry point
cli-app --help
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cli_app

# Run specific test file
pytest tests/test_commands.py

# Test CLI commands
pytest tests/test_cli.py -v
```

## CLI Commands

### Core Commands

- `cli-app hello <name>` - Greet someone
- `cli-app config` - Configuration management
- `cli-app init` - Initialize new project
- `cli-app serve` - Start development server
- `cli-app build` - Build project

### File Operations

- `cli-app file list <path>` - List files in directory
- `cli-app file copy <src> <dst>` - Copy files with progress
- `cli-app file watch <path>` - Watch directory for changes

### Data Operations

- `cli-app data process <input>` - Process data files
- `cli-app data export <format>` - Export data in various formats
- `cli-app data validate <schema>` - Validate data against schema

### Utility Commands

- `cli-app utils hash <file>` - Calculate file hashes
- `cli-app utils compress <path>` - Compress files/directories
- `cli-app utils convert <input> <output>` - Convert file formats

## Configuration

The CLI supports multiple configuration formats:

### YAML Configuration (config.yaml)
```yaml
app:
  name: "My CLI App"
  version: "1.0.0"
  debug: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "app.log"

database:
  url: "sqlite:///app.db"
  pool_size: 5

api:
  base_url: "https://api.example.com"
  timeout: 30
  retry_count: 3
```

### JSON Configuration (config.json)
```json
{
  "app": {
    "name": "My CLI App",
    "version": "1.0.0",
    "debug": false
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "app.log"
  }
}
```

### TOML Configuration (config.toml)
```toml
[app]
name = "My CLI App"
version = "1.0.0"
debug = false

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file = "app.log"
```

## Usage Examples

### Basic Usage

```bash
# Get help
cli-app --help

# Greet someone
cli-app hello "World"

# List files with rich output
cli-app file list /path/to/directory --format table

# Process data with progress bar
cli-app data process input.csv --output output.json --verbose
```

### Advanced Usage

```bash
# Use custom configuration
cli-app --config custom-config.yaml data process input.csv

# Enable debug mode
cli-app --debug file copy large-file.zip backup/

# Export data in different formats
cli-app data export --format json --output results.json
cli-app data export --format csv --output results.csv
cli-app data export --format xml --output results.xml
```

### Pipeline Usage

```bash
# Chain commands using pipes
cli-app file list /data --format json | jq '.files[0].name'

# Use with other CLI tools
find /path -name "*.py" | xargs -I {} cli-app utils hash {}
```

## Command Structure

### Adding New Commands

1. **Create command module**:
   ```python
   # src/cli_app/commands/new_command.py
   import typer
   from rich.console import Console
   
   console = Console()
   app = typer.Typer()
   
   @app.command()
   def action(
       input_file: str = typer.Argument(..., help="Input file path"),
       output: str = typer.Option("output.txt", help="Output file path"),
       verbose: bool = typer.Option(False, help="Enable verbose output")
   ):
       """Perform some action on the input file."""
       if verbose:
           console.print(f"Processing {input_file}...")
       
       # Your command logic here
       console.print(f"✅ Successfully processed to {output}")
   ```

2. **Register in main CLI**:
   ```python
   # src/cli_app/cli.py
   from cli_app.commands import new_command
   
   app.add_typer(new_command.app, name="new")
   ```

### Command Categories

Commands are organized into logical groups:

- **Core**: Essential application commands
- **File**: File and directory operations  
- **Data**: Data processing and transformation
- **Utils**: Utility and helper commands
- **Config**: Configuration management
- **Plugin**: Plugin system commands

## Rich Output Features

### Progress Bars

```python
from rich.progress import track
import time

for i in track(range(100), description="Processing..."):
    time.sleep(0.01)  # Simulate work
```

### Tables

```python
from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="File List")
table.add_column("Name", style="cyan")
table.add_column("Size", style="magenta")
table.add_column("Modified", style="green")

table.add_row("file1.txt", "1.2 KB", "2024-01-15")
table.add_row("file2.txt", "3.4 KB", "2024-01-14")

console.print(table)
```

### Panels and Syntax Highlighting

```python
from rich.panel import Panel
from rich.syntax import Syntax

# Panel with content
console.print(Panel("Hello, World!", title="Greeting"))

# Syntax highlighting
code = '''
def hello(name: str) -> str:
    return f"Hello, {name}!"
'''
syntax = Syntax(code, "python", theme="github-dark")
console.print(syntax)
```

## Plugin System

### Creating Plugins

1. **Create plugin module**:
   ```python
   # plugins/my_plugin.py
   import typer
   from cli_app.core.plugin import Plugin
   
   class MyPlugin(Plugin):
       name = "my-plugin"
       version = "1.0.0"
       description = "My custom plugin"
       
       def create_commands(self) -> typer.Typer:
           app = typer.Typer()
           
           @app.command()
           def custom_command():
               """Custom plugin command."""
               print("Hello from plugin!")
           
           return app
   ```

2. **Register plugin**:
   ```python
   # Register in plugin system
   from cli_app.core.plugin_manager import plugin_manager
   plugin_manager.register_plugin(MyPlugin())
   ```

## Error Handling

### Custom Exceptions

```python
from cli_app.core.exceptions import CLIError, FileNotFoundError

try:
    # Some operation
    pass
except FileNotFoundError as e:
    console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(1)
except CLIError as e:
    console.print(f"[red]CLI Error:[/red] {e}")
    raise typer.Exit(1)
```

### Graceful Error Handling

```python
@app.command()
def safe_command():
    """Command with proper error handling."""
    try:
        # Command logic
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if typer.get_app_dir("cli-app"):
            # Log error details
            pass
        raise typer.Exit(1)
```

## Configuration Management

### Loading Configuration

```python
from cli_app.config import load_config

# Load default configuration
config = load_config()

# Load custom configuration file
config = load_config("custom-config.yaml")

# Access configuration values
database_url = config.database.url
log_level = config.logging.level
```

### Environment Variables

```bash
# Override configuration with environment variables
export CLI_APP_DEBUG=true
export CLI_APP_LOG_LEVEL=DEBUG
export CLI_APP_DATABASE_URL=postgresql://localhost/mydb

cli-app --help  # Will use environment values
```

## Testing

### Testing Commands

```python
from typer.testing import CliRunner
from cli_app.cli import app

runner = CliRunner()

def test_hello_command():
    result = runner.invoke(app, ["hello", "World"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.stdout

def test_config_command():
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "Configuration" in result.stdout
```

### Testing with Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
app:
  name: "Test App"
  debug: true
""")
    return config_file

def test_with_config(temp_config_file):
    result = runner.invoke(app, ["--config", str(temp_config_file), "hello", "Test"])
    assert result.exit_code == 0
```

## Deployment

### Building Standalone Executables

```bash
# Install PyInstaller
pip install pyinstaller

# Build single executable
pyinstaller --onefile --name cli-app src/cli_app/cli.py

# Build with custom icon
pyinstaller --onefile --icon=icon.ico --name cli-app src/cli_app/cli.py
```

### Docker Deployment

```bash
# Build Docker image
docker build -t cli-app:latest .

# Run CLI in container
docker run --rm cli-app:latest --help

# Run with volume mount
docker run --rm -v $(pwd):/data cli-app:latest file list /data
```

### Package Distribution

```bash
# Build distribution packages
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
pip install my-cli-app
```

## Performance

### Optimization Tips

1. **Lazy Imports**: Import heavy modules only when needed
2. **Caching**: Cache expensive operations
3. **Async Operations**: Use async for I/O operations
4. **Progress Feedback**: Show progress for long operations
5. **Memory Management**: Handle large files efficiently

### Example Optimizations

```python
# Lazy imports
def heavy_command():
    import heavy_module  # Import only when needed
    return heavy_module.process()

# Caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(data):
    # Expensive computation
    return result

# Progress feedback
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Processing...", total=total_items)
    for item in items:
        # Process item
        progress.update(task, advance=1)
```

## Best Practices

1. **Type Hints**: Use type hints for all parameters
2. **Documentation**: Document all commands and options
3. **Error Handling**: Provide clear error messages
4. **User Experience**: Use rich output for better UX
5. **Testing**: Write comprehensive tests
6. **Configuration**: Make everything configurable
7. **Logging**: Implement proper logging
8. **Security**: Validate all user inputs

## License

MIT License - see LICENSE file for details