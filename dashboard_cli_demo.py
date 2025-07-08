#!/usr/bin/env python3
"""Demo script showing dashboard CLI usage."""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pynomaly.presentation.cli.dashboard import app
import typer

def demo_dashboard_cli():
    """Demonstrate dashboard CLI capabilities."""
    print("=== Dashboard CLI Demo ===\n")
    
    print("1. Available Dashboard Commands:")
    print("   - generate: Generate comprehensive visualization dashboard")
    print("   - status: Show dashboard service status and active dashboards")
    print("   - monitor: Start real-time dashboard monitoring")
    print("   - compare: Compare dashboard metrics across different time periods")
    print("   - export: Export dashboard to various formats")
    print("   - cleanup: Clean up dashboard service resources")
    
    print("\n2. Example Usage:")
    print("   # Generate an executive dashboard")
    print("   python -m pynomaly.presentation.cli.dashboard generate --type executive")
    
    print("\n   # Generate with specific output and format")
    print("   python -m pynomaly.presentation.cli.dashboard generate --type analytical --output-path ./dash.html --format html")
    
    print("\n   # Monitor real-time dashboard")
    print("   python -m pynomaly.presentation.cli.dashboard monitor --interval 10 --duration 300")
    
    print("\n   # Export dashboard to PDF")
    print("   python -m pynomaly.presentation.cli.dashboard export --dashboard-id my-dash --format pdf --output report.pdf")
    
    print("\n   # Check service status")
    print("   python -m pynomaly.presentation.cli.dashboard status --detailed")
    
    print("\n3. Integration with main CLI:")
    print("   # Access via main CLI")
    print("   python -m pynomaly.presentation.cli.app dashboard generate --type executive")
    
    print("\n4. Validation Features:")
    print("   - Dashboard types: executive, operational, analytical, performance, real_time, compliance")
    print("   - Export formats: html, png, pdf, svg, json")
    print("   - Themes: default, dark, light, corporate")
    
    print("\n5. Stand-alone Execution:")
    print("   The dashboard CLI can be run independently or integrated into the main CLI.")
    print("   Both approaches provide the same functionality with proper sub-command registration.")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_dashboard_cli()
