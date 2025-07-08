#!/usr/bin/env python3
"""Simple validation script for dashboard JSON without dependencies."""

import json
import sys
from pathlib import Path


def validate_dashboard_json():
    """Validate the generated dashboard JSON file."""
    dashboard_path = Path("deploy/grafana/provisioning/dashboards/pynomaly.json")
    
    print(f"Checking for dashboard file: {dashboard_path}")
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    print("✓ Dashboard file exists")
    
    try:
        with open(dashboard_path, "r") as f:
            dashboard_data = json.load(f)
        print("✓ Dashboard JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Dashboard JSON is invalid: {e}")
        return False
    
    # Check structure
    if "dashboard" not in dashboard_data:
        print("❌ Dashboard missing 'dashboard' key")
        return False
    
    dashboard = dashboard_data["dashboard"]
    
    # Check required fields
    required_fields = ["title", "panels", "tags", "refresh", "time"]
    for field in required_fields:
        if field not in dashboard:
            print(f"❌ Dashboard missing required field: {field}")
            return False
    
    print("✓ Dashboard has required fields")
    
    # Check panels
    panels = dashboard["panels"]
    if not isinstance(panels, list):
        print("❌ Dashboard panels is not a list")
        return False
    
    if len(panels) == 0:
        print("❌ Dashboard has no panels")
        return False
    
    print(f"✓ Dashboard has {len(panels)} panels")
    
    # Check each panel
    panel_ids = set()
    for i, panel in enumerate(panels):
        if "id" not in panel:
            print(f"❌ Panel {i} missing 'id' field")
            return False
        
        panel_id = panel["id"]
        if panel_id in panel_ids:
            print(f"❌ Duplicate panel ID: {panel_id}")
            return False
        panel_ids.add(panel_id)
        
        # Check required panel fields
        required_panel_fields = ["title", "type", "gridPos", "targets"]
        for field in required_panel_fields:
            if field not in panel:
                print(f"❌ Panel {panel_id} missing required field: {field}")
                return False
        
        # Check targets
        targets = panel["targets"]
        if not isinstance(targets, list) or len(targets) == 0:
            print(f"❌ Panel {panel_id} has no targets")
            return False
        
        for target in targets:
            if "expr" not in target:
                print(f"❌ Panel {panel_id} target missing 'expr' field")
                return False
            
            expr = target["expr"]
            if not isinstance(expr, str) or len(expr) == 0:
                print(f"❌ Panel {panel_id} target has invalid expr")
                return False
            
            # Basic Prometheus syntax check
            if "pynomaly_" not in expr:
                print(f"⚠️  Panel {panel_id} target doesn't use pynomaly metrics: {expr}")
    
    print("✓ All panels have valid structure")
    
    # Check for expected panel categories
    expected_panel_ranges = {
        "Overview": (1, 3),
        "Detection": (10, 13),
        "Streaming": (20, 23),
        "Ensemble": (30, 31),
        "System": (40, 43),
    }
    
    for category, (start, end) in expected_panel_ranges.items():
        found_in_range = [pid for pid in panel_ids if start <= pid <= end]
        if found_in_range:
            print(f"✓ Found {category} panels: {found_in_range}")
        else:
            print(f"⚠️  No {category} panels found in range {start}-{end}")
    
    print(f"\n✅ Dashboard validation completed successfully!")
    print(f"   - Total panels: {len(panels)}")
    print(f"   - Unique panel IDs: {len(panel_ids)}")
    print(f"   - Title: {dashboard['title']}")
    print(f"   - Tags: {dashboard['tags']}")
    
    return True


def validate_all_dashboard_files():
    """Validate all exported dashboard files."""
    dashboard_dir = Path("deploy/grafana/provisioning/dashboards")
    
    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found: {dashboard_dir}")
        return False
    
    json_files = list(dashboard_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {dashboard_dir}")
        return False
    
    print(f"Found {len(json_files)} dashboard files:")
    
    all_valid = True
    for json_file in json_files:
        print(f"\nValidating {json_file.name}...")
        
        try:
            with open(json_file, "r") as f:
                dashboard_data = json.load(f)
            
            if "dashboard" not in dashboard_data:
                print(f"❌ {json_file.name} missing 'dashboard' key")
                all_valid = False
                continue
            
            dashboard = dashboard_data["dashboard"]
            
            if "title" not in dashboard:
                print(f"❌ {json_file.name} missing title")
                all_valid = False
                continue
            
            if "panels" not in dashboard:
                print(f"❌ {json_file.name} missing panels")
                all_valid = False
                continue
            
            panel_count = len(dashboard["panels"])
            title = dashboard["title"]
            
            print(f"✓ {json_file.name}: '{title}' ({panel_count} panels)")
            
        except json.JSONDecodeError as e:
            print(f"❌ {json_file.name} has invalid JSON: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {json_file.name} error: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Main validation function."""
    print("🔍 Validating Pynomaly Grafana dashboard JSON files...")
    
    # Validate main dashboard
    main_valid = validate_dashboard_json()
    
    print("\n" + "="*60)
    
    # Validate all dashboard files
    all_valid = validate_all_dashboard_files()
    
    if main_valid and all_valid:
        print("\n✅ All dashboard validations passed!")
        return 0
    else:
        print("\n❌ Some dashboard validations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
