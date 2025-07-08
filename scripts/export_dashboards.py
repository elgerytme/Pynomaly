#!/usr/bin/env python3
"""Export Grafana dashboard JSON files for deployment.

This script generates the complete Pynomaly dashboard JSON configuration
and exports it to the deploy/grafana/provisioning/dashboards directory.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynomaly.infrastructure.monitoring.dashboards import (
    DashboardGenerator,
    get_complete_dashboard_template,
)


def export_complete_dashboard(output_dir: str = "deploy/grafana/provisioning/dashboards") -> None:
    """Export the complete Pynomaly dashboard JSON.
    
    Args:
        output_dir: Directory to save the dashboard JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get complete dashboard template
    dashboard_config = get_complete_dashboard_template()
    
    # Export main dashboard
    main_dashboard_path = os.path.join(output_dir, "pynomaly.json")
    with open(main_dashboard_path, "w") as f:
        json.dump(dashboard_config, f, indent=2)
    
    print(f"✓ Exported complete dashboard to {main_dashboard_path}")
    
    # Export individual dashboards
    dashboards = DashboardGenerator.generate_all_dashboards()
    
    for name, config_json in dashboards.items():
        dashboard_path = os.path.join(output_dir, f"pynomaly-{name}.json")
        with open(dashboard_path, "w") as f:
            f.write(config_json)
        print(f"✓ Exported {name} dashboard to {dashboard_path}")
    
    print(f"\nTotal dashboards exported: {len(dashboards) + 1}")
    print(f"Dashboard files location: {output_dir}")


def export_provisioning_config(output_dir: str = "deploy/grafana/provisioning/dashboards") -> None:
    """Export Grafana provisioning configuration for dashboards.
    
    Args:
        output_dir: Directory to save the provisioning config
    """
    provisioning_config = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "pynomaly-dashboards",
                "orgId": 1,
                "folder": "Pynomaly",
                "type": "file",
                "disableDeletion": False,
                "editable": True,
                "options": {
                    "path": "/var/lib/grafana/dashboards"
                }
            }
        ]
    }
    
    config_path = os.path.join(output_dir, "provisioning.yml")
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(provisioning_config, f, default_flow_style=False)
    
    print(f"✓ Exported provisioning config to {config_path}")


def main():
    """Main function to export all dashboard configurations."""
    print("🚀 Exporting Pynomaly Grafana dashboards...")
    
    # Export dashboards
    export_complete_dashboard()
    
    # Export provisioning config
    try:
        export_provisioning_config()
    except ImportError:
        print("⚠️  Warning: PyYAML not installed, skipping provisioning config export")
    
    print("\n✅ Dashboard export completed successfully!")


if __name__ == "__main__":
    main()
