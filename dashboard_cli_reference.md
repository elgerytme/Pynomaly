# Dashboard CLI Reference - Click Implementation

**File Location:** `src/pynomaly/presentation/cli/dashboard.py`

## Click Imports

```python
import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
```

## Click Group Definition

```python
@click.group(name="dashboard")
def dashboard_commands():
    """Visualization dashboard management commands."""
    pass
```

## Commands and Their Exact Signatures

### 1. `generate` Command

```python
@dashboard_commands.command()
@click.option(
    "--type",
    "dashboard_type",
    type=click.Choice([
        "executive",
        "operational", 
        "analytical",
        "performance",
        "real_time",
        "compliance",
    ]),
    default="analytical",
    help="Type of dashboard to generate",
)
@click.option("--output-path", help="Path to save dashboard files")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "png", "pdf", "svg", "json"]),
    default="html",
    help="Export format for dashboard",
)
@click.option(
    "--theme",
    type=click.Choice(["default", "dark", "light", "corporate"]),
    default="default",
    help="Dashboard theme",
)
@click.option("--real-time", is_flag=True, help="Enable real-time updates")
@click.option("--websocket-endpoint", help="WebSocket endpoint for real-time data")
def generate(
    dashboard_type: str,
    output_path: str | None,
    export_format: str,
    theme: str,
    real_time: bool,
    websocket_endpoint: str | None,
):
    """Generate comprehensive visualization dashboard."""
```

**Key Click Features:**
- Uses `click.Choice` for dashboard type validation
- Uses `click.Choice` for export format validation  
- Uses `click.Choice` for theme validation
- Uses `is_flag=True` for boolean flag
- Parameter renaming: `"--type"` → `"dashboard_type"`, `"--format"` → `"export_format"`
- Optional parameters with `str | None` type hints

### 2. `status` Command

```python
@dashboard_commands.command()
@click.option("--dashboard-id", help="Specific dashboard ID to show status for")
@click.option("--detailed", is_flag=True, help="Show detailed status information")
def status(dashboard_id: str | None, detailed: bool):
    """Show dashboard service status and active dashboards."""
```

**Key Click Features:**
- Optional string parameter
- Boolean flag with `is_flag=True`

### 3. `monitor` Command

```python
@dashboard_commands.command()
@click.option("--interval", type=int, default=5, help="Update interval in seconds")
@click.option(
    "--websocket-endpoint", default="ws://localhost:8000/ws", help="WebSocket endpoint"
)
@click.option("--duration", type=int, help="Duration to monitor in seconds")
def monitor(interval: int, websocket_endpoint: str, duration: int | None):
    """Start real-time dashboard monitoring."""
```

**Key Click Features:**
- `type=int` for integer parameters
- Default values for options
- Optional integer parameter

### 4. `compare` Command

```python
@dashboard_commands.command()
@click.option(
    "--dashboard-type",
    type=click.Choice(["executive", "operational", "analytical", "performance"]),
    default="analytical",
    help="Dashboard type to compare",
)
@click.option("--metrics", multiple=True, help="Specific metrics to compare")
@click.option("--time-period", type=int, default=30, help="Time period in days")
def compare(dashboard_type: str, metrics: list[str], time_period: int):
    """Compare dashboard metrics across different time periods."""
```

**Key Click Features:**
- `click.Choice` for type validation
- `multiple=True` for list parameters
- `type=int` with default value

### 5. `export` Command

```python
@dashboard_commands.command()
@click.option("--dashboard-id", required=True, help="Dashboard ID to export")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "png", "pdf", "svg", "json"]),
    default="html",
    help="Export format",
)
@click.option("--output", required=True, help="Output file path")
@click.option("--config-file", help="Export configuration file")
def export(dashboard_id: str, export_format: str, output: str, config_file: str | None):
    """Export dashboard to various formats."""
```

**Key Click Features:**
- `required=True` for mandatory parameters
- `click.Choice` for format validation
- Parameter renaming: `"--format"` → `"export_format"`

### 6. `cleanup` Command

```python
@dashboard_commands.command()
@click.option("--clear-cache", is_flag=True, help="Clear dashboard cache")
@click.option("--reset-metrics", is_flag=True, help="Reset metrics history")
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
def cleanup(clear_cache: bool, reset_metrics: bool, force: bool):
    """Clean up dashboard service resources."""
```

**Key Click Features:**
- Multiple boolean flags with `is_flag=True`
- Uses `click.confirm()` for user confirmation when not forced

## Click Helper Functions Used

### User Confirmation
```python
# Used in cleanup command
confirm = click.confirm("Clear all cached dashboards?")
confirm = click.confirm("Reset all metrics history?")
```

## Command Structure Summary

| Command | Required Args | Optional Args | Flags | Choice Parameters |
|---------|---------------|---------------|-------|-------------------|
| `generate` | None | `--output-path`, `--websocket-endpoint` | `--real-time` | `--type`, `--format`, `--theme` |
| `status` | None | `--dashboard-id` | `--detailed` | None |
| `monitor` | None | `--duration` | None | None |
| `compare` | None | `--metrics` (multiple) | None | `--dashboard-type` |
| `export` | `--dashboard-id`, `--output` | `--config-file` | None | `--format` |
| `cleanup` | None | None | `--clear-cache`, `--reset-metrics`, `--force` | None |

## Type Patterns Used

- **String parameters:** `str | None` for optional, `str` for required
- **Integer parameters:** `type=int` with optional default values
- **Boolean flags:** `is_flag=True`
- **Choice parameters:** `click.Choice([...])` with validation lists
- **Multiple values:** `multiple=True` → `list[str]`
- **Parameter renaming:** Using second parameter for different variable names

## Default Values

- `--type`: `"analytical"`
- `--format`: `"html"`
- `--theme`: `"default"`
- `--interval`: `5`
- `--websocket-endpoint`: `"ws://localhost:8000/ws"`
- `--time-period`: `30`
- `--dashboard-type`: `"analytical"`

## Validation Constraints

### Dashboard Types
- `executive`, `operational`, `analytical`, `performance`, `real_time`, `compliance`

### Export Formats  
- `html`, `png`, `pdf`, `svg`, `json`

### Themes
- `default`, `dark`, `light`, `corporate`

### Compare Dashboard Types (Subset)
- `executive`, `operational`, `analytical`, `performance`

## Special Behaviors

1. **Interactive Confirmation:** `cleanup` command uses `click.confirm()` unless `--force` is provided
2. **Async Execution:** All commands use `asyncio.run()` to handle async operations
3. **Rich Console Integration:** Uses Rich library for enhanced terminal output
4. **Parameter Validation:** Choice parameters provide automatic validation
5. **Error Handling:** Try-catch blocks with console error output

## Entry Point

```python
if __name__ == "__main__":
    dashboard_commands()
```

This reference captures all Click-specific behavior that must be preserved during Typer conversion to ensure complete behavior-parity.
