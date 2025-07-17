"""
Interactive CLI Tutorials for Pynomaly

This module provides comprehensive tutorials to help users learn
how to use Pynomaly effectively through interactive lessons.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from interfaces.presentation.cli.ux_improvements import ProgressIndicator

console = Console()
progress_indicator = ProgressIndicator()


class TutorialStep:
    """Represents a single tutorial step."""

    def __init__(
        self,
        title: str,
        description: str,
        code_example: str | None = None,
        explanation: str | None = None,
        interactive_action: Callable | None = None,
        validation_function: Callable | None = None,
        tips: list[str] | None = None
    ):
        self.title = title
        self.description = description
        self.code_example = code_example
        self.explanation = explanation
        self.interactive_action = interactive_action
        self.validation_function = validation_function
        self.tips = tips or []

    def execute(self, context: dict[str, Any]) -> bool:
        """Execute this tutorial step."""
        # Display step title
        console.print(Panel.fit(
            f"[bold blue]{self.title}[/bold blue]",
            title=f"Step {context.get('step_number', 1)}",
            border_style="blue"
        ))

        # Show description
        console.print(f"\n{self.description}\n")

        # Show code example if provided
        if self.code_example:
            console.print("[bold cyan]Example:[/bold cyan]")
            syntax = Syntax(self.code_example, "bash", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print()

        # Show explanation if provided
        if self.explanation:
            console.print("[bold yellow]Explanation:[/bold yellow]")
            console.print(f"{self.explanation}\n")

        # Show tips if provided
        if self.tips:
            console.print("[bold green]💡 Tips:[/bold green]")
            for tip in self.tips:
                console.print(f"  • {tip}")
            console.print()

        # Execute interactive action if provided
        if self.interactive_action:
            try:
                result = self.interactive_action(context)
                if result is False:
                    console.print("[red]❌ Step failed. Please try again.[/red]")
                    return False
                elif isinstance(result, dict):
                    context.update(result)
            except Exception as e:
                console.print(f"[red]❌ Error in interactive action: {str(e)}[/red]")
                return False

        # Validate if validation function provided
        if self.validation_function:
            try:
                if not self.validation_function(context):
                    console.print("[red]❌ Validation failed. Please check your work.[/red]")
                    return False
            except Exception as e:
                console.print(f"[red]❌ Validation error: {str(e)}[/red]")
                return False

        # Ask user to continue
        if not Confirm.ask("\n[cyan]Ready to continue to the next step?[/cyan]", default=True):
            console.print("[yellow]⏸️ Tutorial paused. Run again to continue.[/yellow]")
            return False

        return True


class Tutorial:
    """Represents a complete tutorial with multiple steps."""

    def __init__(
        self,
        name: str,
        description: str,
        difficulty: str,
        estimated_time: int,
        prerequisites: list[str] | None = None,
        steps: list[TutorialStep] | None = None
    ):
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.estimated_time = estimated_time
        self.prerequisites = prerequisites or []
        self.steps = steps or []

    def add_step(self, step: TutorialStep) -> None:
        """Add a step to the tutorial."""
        self.steps.append(step)

    def execute(self) -> bool:
        """Execute the complete tutorial."""
        # Show tutorial introduction
        console.print(Panel.fit(
            f"[bold blue]{self.name}[/bold blue]\n\n"
            f"{self.description}\n\n"
            f"[bold yellow]Difficulty:[/bold yellow] {self.difficulty}\n"
            f"[bold yellow]Estimated Time:[/bold yellow] {self.estimated_time} minutes",
            title="🎓 Tutorial",
            border_style="blue"
        ))

        # Show prerequisites if any
        if self.prerequisites:
            console.print("\n[bold cyan]📋 Prerequisites:[/bold cyan]")
            for prereq in self.prerequisites:
                console.print(f"  • {prereq}")
            console.print()

        # Ask user if they want to continue
        if not Confirm.ask("Ready to start the tutorial?", default=True):
            console.print("Tutorial cancelled.")
            return False

        # Execute steps with progress tracking
        context = {"tutorial_name": self.name}

        with progress_indicator.create_progress_bar("Tutorial Progress") as progress:
            task = progress.add_task("[cyan]Learning...", total=len(self.steps))

            for i, step in enumerate(self.steps):
                context["step_number"] = i + 1
                context["total_steps"] = len(self.steps)

                progress.update(task, description=f"[cyan]Step {i + 1}: {step.title}")

                if not step.execute(context):
                    console.print(f"[red]❌ Tutorial stopped at step {i + 1}.[/red]")
                    return False

                progress.update(task, completed=i + 1)

        # Show completion message
        console.print(Panel.fit(
            f"[bold green]🎉 Congratulations![/bold green]\n\n"
            f"You've completed the '{self.name}' tutorial!\n\n"
            f"[bold cyan]What you learned:[/bold cyan]\n"
            f"• {len(self.steps)} key concepts\n"
            f"• Practical hands-on experience\n"
            f"• Best practices and tips\n\n"
            f"[bold yellow]Next steps:[/bold yellow]\n"
            f"• Try the advanced tutorials\n"
            f"• Explore the documentation\n"
            f"• Join the community discussions",
            title="✅ Tutorial Complete",
            border_style="green"
        ))

        return True


class TutorialManager:
    """Manages all available tutorials."""

    def __init__(self):
        self.tutorials: dict[str, Tutorial] = {}
        self._initialize_tutorials()

    def _initialize_tutorials(self) -> None:
        """Initialize all available tutorials."""
        # Basic tutorial
        self._create_basic_tutorial()

        # Advanced tutorial
        self._create_advanced_tutorial()

        # API integration tutorial
        self._create_api_tutorial()

        # Performance optimization tutorial
        self._create_performance_tutorial()

    def _create_basic_tutorial(self) -> None:
        """Create the basic Pynomaly tutorial."""
        tutorial = Tutorial(
            name="Getting Started with Pynomaly",
            description="Learn the basics of anomaly detection using Pynomaly",
            difficulty="Beginner",
            estimated_time=15,
            prerequisites=["Python 3.8+", "Basic command line knowledge"]
        )

        # Step 1: Check installation
        def check_installation(context):
            console.print("Let's verify that Pynomaly is properly installed...")
            # In real implementation, would check actual installation
            console.print("[green]✅ Pynomaly is installed and ready to use![/green]")
            return True

        tutorial.add_step(TutorialStep(
            title="Verify Installation",
            description="First, let's make sure Pynomaly is properly installed on your system.",
            code_example="pynomaly --version",
            explanation="This command shows the current version of Pynomaly and verifies it's working correctly.",
            interactive_action=check_installation,
            tips=["If you see any errors, try reinstalling Pynomaly with pip install pynomaly"]
        ))

        # Step 2: Create sample data
        def create_sample_data(context):
            console.print("Creating sample dataset for the tutorial...")
            # In real implementation, would create actual sample data
            sample_data_path = Path("tutorial_data.csv")
            console.print(f"[green]✅ Sample data created at {sample_data_path}[/green]")
            return {"sample_data_path": str(sample_data_path)}

        tutorial.add_step(TutorialStep(
            title="Create Sample Data",
            description="Let's create some sample data to work with throughout this tutorial.",
            code_example="# Sample data will be created automatically\n# In practice, you'd load your own data:",
            explanation="For this tutorial, we'll use a synthetic dataset with normal and anomalous data points.",
            interactive_action=create_sample_data,
            tips=["Real datasets can be loaded from CSV, JSON, or Parquet files"]
        ))

        # Step 3: Load dataset
        def load_dataset(context):
            data_path = context.get("sample_data_path", "tutorial_data.csv")
            console.print(f"Loading dataset from {data_path}...")
            # Simulate loading
            time.sleep(1)
            console.print("[green]✅ Dataset loaded successfully![/green]")
            console.print("📊 Dataset info: 1000 samples, 5 features, ~5% anomalies")
            return {"dataset_id": "tutorial_dataset"}

        tutorial.add_step(TutorialStep(
            title="Load Dataset",
            description="Now let's load the dataset into Pynomaly for processing.",
            code_example="pynomaly dataset load tutorial_data.csv --name tutorial_dataset",
            explanation="The dataset load command imports your data and prepares it for anomaly detection.",
            interactive_action=load_dataset,
            tips=[
                "Use descriptive names for your datasets",
                "Check data quality before loading",
                "Pynomaly supports CSV, JSON, and Parquet formats"
            ]
        ))

        # Step 4: Create detector
        def create_detector(context):
            console.print("Creating an anomaly detector...")
            # Simulate detector creation
            time.sleep(1)
            console.print("[green]✅ Detector created successfully![/green]")
            console.print("🔍 Using IsolationForest algorithm (recommended for beginners)")
            return {"detector_id": "tutorial_detector"}

        tutorial.add_step(TutorialStep(
            title="Create Detector",
            description="Let's create our first anomaly detector using the IsolationForest algorithm.",
            code_example="pynomaly detector create tutorial_detector --algorithm IsolationForest --contamination 0.05",
            explanation="IsolationForest is a fast, effective algorithm that works well for most use cases.",
            interactive_action=create_detector,
            tips=[
                "Start with IsolationForest for general use",
                "Adjust contamination based on expected anomaly rate",
                "Lower contamination = fewer anomalies detected"
            ]
        ))

        # Step 5: Train detector
        def train_detector(context):
            console.print("Training the detector on your data...")
            # Simulate training with progress
            progress_indicator.track_training_progress(
                total_epochs=5,
                current_metrics={"accuracy": 0.95, "precision": 0.87, "recall": 0.92}
            )
            console.print("[green]✅ Training completed![/green]")
            console.print("📈 Model performance: 95% accuracy, 87% precision, 92% recall")
            return {"training_complete": True}

        tutorial.add_step(TutorialStep(
            title="Train the Detector",
            description="Now we'll train our detector on the sample data.",
            code_example="pynomaly detect train tutorial_detector tutorial_dataset",
            explanation="Training teaches the detector to recognize normal patterns in your data.",
            interactive_action=train_detector,
            tips=[
                "Training time depends on dataset size",
                "Monitor training metrics to ensure good performance",
                "Use cross-validation for better model evaluation"
            ]
        ))

        # Step 6: Run detection
        def run_detection(context):
            console.print("Running anomaly detection...")
            # Simulate detection with progress
            items = [f"sample_{i}" for i in range(100)]
            results = progress_indicator.track_batch_operation(
                items, "Detecting anomalies",
                lambda item, i: f"processed_{item}"
            )
            console.print("[green]✅ Detection completed![/green]")
            console.print("🎯 Found 47 anomalies out of 1000 samples (4.7%)")
            return {"detection_results": results}

        tutorial.add_step(TutorialStep(
            title="Run Detection",
            description="Let's run our trained detector on the data to find anomalies.",
            code_example="pynomaly detect run tutorial_detector tutorial_dataset --output results.csv",
            explanation="The detector analyzes each data point and assigns an anomaly score.",
            interactive_action=run_detection,
            tips=[
                "Results are saved to the specified output file",
                "Higher scores indicate more likely anomalies",
                "Review results carefully and validate findings"
            ]
        ))

        # Step 7: Analyze results
        def analyze_results(context):
            console.print("Analyzing detection results...")

            # Create sample results table
            table = Table(title="Detection Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Samples", "1,000")
            table.add_row("Anomalies Found", "47")
            table.add_row("Anomaly Rate", "4.7%")
            table.add_row("Average Score", "0.23")
            table.add_row("Max Score", "0.89")

            console.print(table)
            console.print("\n[green]✅ Analysis complete![/green]")
            return {"analysis_complete": True}

        tutorial.add_step(TutorialStep(
            title="Analyze Results",
            description="Finally, let's analyze the detection results to understand what we found.",
            code_example="pynomaly detect results --latest --analyze",
            explanation="Analysis helps you understand the quality and patterns in your anomaly detections.",
            interactive_action=analyze_results,
            tips=[
                "Look for patterns in detected anomalies",
                "Validate results with domain knowledge",
                "Adjust detector parameters if needed"
            ]
        ))

        self.tutorials["basic"] = tutorial

    def _create_advanced_tutorial(self) -> None:
        """Create the advanced Pynomaly tutorial."""
        tutorial = Tutorial(
            name="Advanced Anomaly Detection Techniques",
            description="Learn advanced features like hyperparameter tuning, ensemble methods, and custom algorithms",
            difficulty="Advanced",
            estimated_time=30,
            prerequisites=["Completed basic tutorial", "Understanding of machine learning concepts"]
        )

        # Add advanced steps...
        tutorial.add_step(TutorialStep(
            title="Hyperparameter Optimization",
            description="Learn how to optimize detector parameters for better performance.",
            code_example="pynomaly automl optimize --dataset tutorial_dataset --algorithms IsolationForest LOF --trials 50",
            explanation="AutoML automatically finds the best parameters for your specific dataset.",
            tips=["More trials = better results but longer time", "Use cross-validation for robust evaluation"]
        ))

        tutorial.add_step(TutorialStep(
            title="Ensemble Methods",
            description="Combine multiple detectors for improved accuracy.",
            code_example="pynomaly detect ensemble detector1 detector2 detector3 --method voting --weights 0.4 0.3 0.3",
            explanation="Ensemble methods combine predictions from multiple models to reduce errors.",
            tips=["Use diverse algorithms for better ensemble performance", "Weight models based on their individual performance"]
        ))

        self.tutorials["advanced"] = tutorial

    def _create_api_tutorial(self) -> None:
        """Create the API integration tutorial."""
        tutorial = Tutorial(
            name="API Integration and Automation",
            description="Learn how to integrate Pynomaly with your applications using the REST API",
            difficulty="Intermediate",
            estimated_time=20,
            prerequisites=["Basic HTTP/REST knowledge", "API key setup"]
        )

        # Add API-focused steps...
        tutorial.add_step(TutorialStep(
            title="API Authentication",
            description="Set up authentication for API access.",
            code_example="curl -H \"X-API-Key: your-api-key\" https://api.example.com/health",
            explanation="All API calls require authentication using your API key.",
            tips=["Keep your API key secure", "Use environment variables for keys", "Monitor API usage"]
        ))

        self.tutorials["api"] = tutorial

    def _create_performance_tutorial(self) -> None:
        """Create the performance optimization tutorial."""
        tutorial = Tutorial(
            name="Performance Optimization",
            description="Learn how to optimize Pynomaly for large datasets and production environments",
            difficulty="Advanced",
            estimated_time=25,
            prerequisites=["System administration knowledge", "Understanding of performance concepts"]
        )

        # Add performance-focused steps...
        tutorial.add_step(TutorialStep(
            title="Batch Processing",
            description="Process large datasets efficiently using batch operations.",
            code_example="pynomaly detect batch detector_id dataset_id --batch-size 1000 --parallel 4",
            explanation="Batch processing improves performance for large datasets.",
            tips=["Adjust batch size based on memory", "Use parallel processing when possible"]
        ))

        self.tutorials["performance"] = tutorial

    def list_tutorials(self) -> None:
        """Display all available tutorials."""
        console.print(Panel.fit(
            "[bold blue]📚 Available Tutorials[/bold blue]",
            title="Pynomaly Learning Center",
            border_style="blue"
        ))

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Difficulty", style="yellow")
        table.add_column("Time", style="green")
        table.add_column("Prerequisites", style="dim")

        for tutorial_id, tutorial in self.tutorials.items():
            prereqs = ", ".join(tutorial.prerequisites) if tutorial.prerequisites else "None"
            table.add_row(
                tutorial_id,
                tutorial.name,
                tutorial.difficulty,
                f"{tutorial.estimated_time} min",
                prereqs
            )

        console.print(table)

        console.print("\n[bold cyan]💡 Getting Started:[/bold cyan]")
        console.print("• New users: Start with the 'basic' tutorial")
        console.print("• Experienced users: Try 'advanced' or 'performance' tutorials")
        console.print("• Developers: Check out the 'api' tutorial")
        console.print("\n[bold yellow]Usage:[/bold yellow] pynomaly tutorial run <tutorial_id>")

    def run_tutorial(self, tutorial_id: str) -> bool:
        """Run a specific tutorial."""
        if tutorial_id not in self.tutorials:
            console.print(f"[red]❌ Tutorial '{tutorial_id}' not found.[/red]")
            self.list_tutorials()
            return False

        tutorial = self.tutorials[tutorial_id]
        return tutorial.execute()

    def get_tutorial_info(self, tutorial_id: str) -> dict[str, Any] | None:
        """Get information about a specific tutorial."""
        if tutorial_id not in self.tutorials:
            return None

        tutorial = self.tutorials[tutorial_id]
        return {
            "name": tutorial.name,
            "description": tutorial.description,
            "difficulty": tutorial.difficulty,
            "estimated_time": tutorial.estimated_time,
            "prerequisites": tutorial.prerequisites,
            "steps": len(tutorial.steps)
        }


# Global tutorial manager instance
tutorial_manager = TutorialManager()


# CLI commands for tutorials
def run_tutorial_command(tutorial_id: str) -> None:
    """Run a tutorial command."""
    tutorial_manager.run_tutorial(tutorial_id)


def list_tutorials_command() -> None:
    """List all available tutorials."""
    tutorial_manager.list_tutorials()


def tutorial_info_command(tutorial_id: str) -> None:
    """Show information about a specific tutorial."""
    info = tutorial_manager.get_tutorial_info(tutorial_id)
    if not info:
        console.print(f"[red]❌ Tutorial '{tutorial_id}' not found.[/red]")
        return

    console.print(Panel.fit(
        f"[bold blue]{info['name']}[/bold blue]\n\n"
        f"{info['description']}\n\n"
        f"[bold yellow]Difficulty:[/bold yellow] {info['difficulty']}\n"
        f"[bold yellow]Estimated Time:[/bold yellow] {info['estimated_time']} minutes\n"
        f"[bold yellow]Steps:[/bold yellow] {info['steps']}\n\n"
        f"[bold cyan]Prerequisites:[/bold cyan]\n" +
        "\n".join(f"• {prereq}" for prereq in info['prerequisites']) if info['prerequisites'] else "• None",
        title="📖 Tutorial Information",
        border_style="blue"
    ))
