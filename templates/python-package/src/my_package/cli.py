"""Command-line interface for the package."""

import typer
from rich.console import Console
from rich.table import Table

from .core import Calculator, DataProcessor, DataPoint
from .utils import format_output, validate_input

app = typer.Typer(help="My Package CLI - Command-line interface for the package")
console = Console()


@app.command()
def calculate(
    operation: str = typer.Argument(..., help="Operation (add, subtract, multiply, divide, power)"),
    x: float = typer.Argument(..., help="First number"),
    y: float = typer.Argument(..., help="Second number"),
) -> None:
    """Perform mathematical calculations."""
    
    calculator = Calculator()
    
    try:
        if operation == "add":
            result = calculator.add(x, y)
        elif operation == "subtract":
            result = calculator.subtract(x, y)
        elif operation == "multiply":
            result = calculator.multiply(x, y)
        elif operation == "divide":
            result = calculator.divide(x, y)
        elif operation == "power":
            result = calculator.power(x, y)
        else:
            console.print(f"[red]Error:[/red] Unknown operation '{operation}'")
            console.print("Available operations: add, subtract, multiply, divide, power")
            raise typer.Exit(1)
        
        console.print(f"[green]Result:[/green] {result}")
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def process_data(
    values: str = typer.Option(..., "--values", "-v", help="Comma-separated values"),
    operation: str = typer.Option("summary", "--operation", "-o", help="Operation (summary, average, sum, min, max)"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (json, csv, table)"),
) -> None:
    """Process data points."""
    
    processor = DataProcessor()
    
    # Parse values
    try:
        value_list = [float(v.strip()) for v in values.split(",")]
    except ValueError:
        console.print("[red]Error:[/red] Invalid values format. Use comma-separated numbers.")
        raise typer.Exit(1)
    
    # Add data points
    for i, value in enumerate(value_list):
        data_point = DataPoint(
            value=value,
            metadata={"index": i},
            timestamp=None
        )
        processor.add_data_point(data_point)
    
    # Perform operation
    if operation == "summary":
        result = {
            "count": processor.count(),
            "sum": processor.calculate_sum(),
            "average": processor.calculate_average(),
            "min": processor.find_min(),
            "max": processor.find_max(),
        }
    elif operation == "average":
        result = processor.calculate_average()
    elif operation == "sum":
        result = processor.calculate_sum()
    elif operation == "min":
        result = processor.find_min()
    elif operation == "max":
        result = processor.find_max()
    else:
        console.print(f"[red]Error:[/red] Unknown operation '{operation}'")
        console.print("Available operations: summary, average, sum, min, max")
        raise typer.Exit(1)
    
    # Format and display result
    if format_type == "table" and operation == "summary":
        table = Table(title="Data Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in result.items():
            table.add_row(key.title(), str(value))
        
        console.print(table)
    else:
        formatted_result = format_output(result, format_type)
        console.print(formatted_result)


@app.command()
def validate(
    value: str = typer.Argument(..., help="Value to validate"),
    value_type: str = typer.Option("str", "--type", "-t", help="Expected type (str, int, float)"),
    min_value: float = typer.Option(None, "--min", help="Minimum value for numeric types"),
) -> None:
    """Validate input values."""
    
    # Convert value to expected type
    try:
        if value_type == "int":
            typed_value = int(value)
            expected_type = int
        elif value_type == "float":
            typed_value = float(value)
            expected_type = float
        else:
            typed_value = value
            expected_type = str
    except ValueError:
        console.print(f"[red]Error:[/red] Cannot convert '{value}' to {value_type}")
        raise typer.Exit(1)
    
    # Validate
    is_valid = validate_input(typed_value, expected_type, min_value)
    
    if is_valid:
        console.print(f"[green]✓[/green] Value '{value}' is valid")
    else:
        console.print(f"[red]✗[/red] Value '{value}' is invalid")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"[bold green]My Package[/bold green] v{__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()