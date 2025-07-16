"""
CLI commands for interactive tutorials.
"""

import typer
from rich.console import Console

from interfaces.presentation.cli.tutorials import tutorial_manager

console = Console()

# Create tutorial CLI app
app = typer.Typer(
    name="tutorial",
    help="📚 Interactive tutorials for learning Pynomaly",
    add_completion=True
)


@app.command()
def list():
    """List all available tutorials."""
    tutorial_manager.list_tutorials()


@app.command()
def run(
    tutorial_id: str = typer.Argument(..., help="Tutorial ID to run"),
    skip_intro: bool = typer.Option(False, "--skip-intro", help="Skip tutorial introduction")
):
    """Run an interactive tutorial."""
    if not skip_intro:
        console.print("🎓 [bold blue]Welcome to Pynomaly Tutorials![/bold blue]")
        console.print("Interactive learning experience to master anomaly detection.\n")

    success = tutorial_manager.run_tutorial(tutorial_id)

    if success:
        console.print("\n🎉 [bold green]Tutorial completed successfully![/bold green]")
        console.print("Keep learning with more tutorials: [cyan]pynomaly tutorial list[/cyan]")
    else:
        console.print("\n❌ [bold red]Tutorial was not completed.[/bold red]")
        console.print("You can restart it anytime with: [cyan]pynomaly tutorial run {tutorial_id}[/cyan]")


@app.command()
def info(
    tutorial_id: str = typer.Argument(..., help="Tutorial ID to get information about")
):
    """Show detailed information about a tutorial."""
    from interfaces.presentation.cli.tutorials import tutorial_info_command
    tutorial_info_command(tutorial_id)


@app.command()
def progress(
    tutorial_id: str = typer.Argument(..., help="Tutorial ID to check progress for")
):
    """Check progress on a tutorial (future feature)."""
    console.print(f"[yellow]Progress tracking for '{tutorial_id}' - Coming soon![/yellow]")
    console.print("This feature will track your progress through tutorials and allow resuming.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for tutorials")
):
    """Search tutorials by topic or keyword."""
    console.print(f"[cyan]Searching tutorials for: '{query}'[/cyan]")

    # Simple search implementation
    results = []
    for tutorial_id, tutorial in tutorial_manager.tutorials.items():
        if (query.lower() in tutorial.name.lower() or
            query.lower() in tutorial.description.lower() or
            query.lower() in tutorial.difficulty.lower()):
            results.append((tutorial_id, tutorial))

    if results:
        console.print(f"\n[green]Found {len(results)} matching tutorial(s):[/green]")
        for tutorial_id, tutorial in results:
            console.print(f"  • [bold]{tutorial_id}[/bold]: {tutorial.name}")
            console.print(f"    {tutorial.description}")
            console.print(f"    Difficulty: {tutorial.difficulty}, Time: {tutorial.estimated_time} min\n")
    else:
        console.print(f"[red]No tutorials found matching '{query}'[/red]")
        console.print("Try: [cyan]pynomaly tutorial list[/cyan] to see all available tutorials")


@app.command()
def quickstart():
    """Quick start guide - automatically run the basic tutorial."""
    console.print("🚀 [bold blue]Quick Start Guide[/bold blue]")
    console.print("Starting the basic tutorial to get you up and running quickly!\n")

    success = tutorial_manager.run_tutorial("basic")

    if success:
        console.print("\n✅ [bold green]Quick start completed![/bold green]")
        console.print("You're now ready to use Pynomaly effectively!")
        console.print("\n[bold cyan]Next steps:[/bold cyan]")
        console.print("• Try the advanced tutorial: [white]pynomaly tutorial run advanced[/white]")
        console.print("• Explore the API tutorial: [white]pynomaly tutorial run api[/white]")
        console.print("• Check out the documentation: [white]pynomaly --help[/white]")
    else:
        console.print("\n❌ [bold red]Quick start was not completed.[/bold red]")
        console.print("You can restart it anytime with: [cyan]pynomaly tutorial quickstart[/cyan]")


@app.command()
def create(
    name: str = typer.Argument(..., help="Name of the custom tutorial"),
    description: str = typer.Option("", "--description", help="Tutorial description"),
    difficulty: str = typer.Option("Beginner", "--difficulty", help="Tutorial difficulty"),
    time: int = typer.Option(15, "--time", help="Estimated time in minutes")
):
    """Create a custom tutorial template (future feature)."""
    console.print(f"[yellow]Creating custom tutorial '{name}' - Coming soon![/yellow]")
    console.print("This feature will allow you to create custom tutorials for your team.")
    console.print(f"Tutorial: {name}")
    console.print(f"Description: {description or 'Custom tutorial'}")
    console.print(f"Difficulty: {difficulty}")
    console.print(f"Estimated time: {time} minutes")


if __name__ == "__main__":
    app()
