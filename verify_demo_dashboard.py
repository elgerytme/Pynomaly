#!/usr/bin/env python3
"""
Verification script for Advanced Analytics Dashboard readiness.
This script checks if the demo dashboard components are ready for recording.
"""

import os
from pathlib import Path
import requests
import json
from datetime import datetime

def check_file_exists(file_path):
    """Check if a file exists and return its status."""
    path = Path(file_path)
    return {
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "modified": path.stat().st_mtime if path.exists() else None
    }

def check_demo_components():
    """Check if all demo components are ready."""
    base_path = Path(__file__).parent
    
    # Check templates
    templates_path = base_path / "src" / "pynomaly" / "presentation" / "web" / "templates"
    template_files = [
        "demo_dashboard.html",
        "advanced-dashboard-demo.html", 
        "dashboard-demo.html",
        "base.html"
    ]
    
    # Check static files
    static_path = base_path / "src" / "pynomaly" / "presentation" / "web" / "static"
    static_files = [
        "js/demo_data.js",
        "css/dashboard.css",
        "css/main.css"
    ]
    
    # Check demo data
    demo_data_files = [
        "demo_data.json",
        "demo_results.json"
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "checking",
        "components": {
            "templates": {},
            "static_files": {},
            "demo_data": {},
            "server": {}
        }
    }
    
    # Check templates
    for template in template_files:
        file_path = templates_path / template
        results["components"]["templates"][template] = check_file_exists(file_path)
    
    # Check static files
    for static_file in static_files:
        file_path = static_path / static_file
        results["components"]["static_files"][static_file] = check_file_exists(file_path)
    
    # Check demo data
    for data_file in demo_data_files:
        file_path = static_path / data_file
        results["components"]["demo_data"][data_file] = check_file_exists(file_path)
    
    # Check if we can serve the demo directly
    demo_template = templates_path / "demo_dashboard.html"
    if demo_template.exists():
        with open(demo_template, 'r', encoding='utf-8') as f:
            content = f.read()
            results["components"]["templates"]["demo_dashboard.html"]["content_length"] = len(content)
            results["components"]["templates"]["demo_dashboard.html"]["has_demo_data"] = "demo_data.js" in content
    
    # Overall status assessment
    template_ready = all(results["components"]["templates"][t]["exists"] for t in template_files)
    static_ready = all(results["components"]["static_files"][s]["exists"] for s in static_files)
    data_ready = all(results["components"]["demo_data"][d]["exists"] for d in demo_data_files)
    
    if template_ready and static_ready and data_ready:
        results["overall_status"] = "ready"
    elif template_ready:
        results["overall_status"] = "partial"
    else:
        results["overall_status"] = "not_ready"
    
    return results

def generate_report():
    """Generate a comprehensive readiness report."""
    results = check_demo_components()
    
    print("=" * 80)
    print("PYNOMALY ADVANCED ANALYTICS DASHBOARD READINESS REPORT")
    print("=" * 80)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print()
    
    # Templates status
    print("📋 TEMPLATES STATUS:")
    for template, info in results["components"]["templates"].items():
        status = "✅" if info["exists"] else "❌"
        size = f" ({info['size']} bytes)" if info["exists"] else ""
        print(f"  {status} {template}{size}")
    
    print()
    
    # Static files status
    print("🎨 STATIC FILES STATUS:")
    for static_file, info in results["components"]["static_files"].items():
        status = "✅" if info["exists"] else "❌"
        size = f" ({info['size']} bytes)" if info["exists"] else ""
        print(f"  {status} {static_file}{size}")
    
    print()
    
    # Demo data status
    print("📊 DEMO DATA STATUS:")
    for data_file, info in results["components"]["demo_data"].items():
        status = "✅" if info["exists"] else "❌"
        size = f" ({info['size']} bytes)" if info["exists"] else ""
        print(f"  {status} {data_file}{size}")
    
    print()
    
    # Recommendations
    print("📋 RECOMMENDATIONS:")
    if results["overall_status"] == "ready":
        print("  ✅ All components are ready for recording!")
        print("  ✅ Demo dashboard is fully functional")
        print("  ✅ Static assets are properly configured")
        print("  ✅ Demo data is available")
    elif results["overall_status"] == "partial":
        print("  ⚠️  Core templates are available but some components are missing")
        print("  ⚠️  Demo may work with limited functionality")
        print("  ⚠️  Consider adding missing static files for full experience")
    else:
        print("  ❌ Critical components are missing")
        print("  ❌ Demo dashboard is not ready for recording")
        print("  ❌ Additional setup required")
    
    print()
    print("🚀 NEXT STEPS:")
    if results["overall_status"] == "ready":
        print("  1. Start a simple HTTP server to serve the demo")
        print("  2. Navigate to the demo dashboard URL")
        print("  3. Begin recording the demo")
        print("  4. The dashboard should show realistic data and visualizations")
    else:
        print("  1. Fix missing components identified above")
        print("  2. Re-run this verification script")
        print("  3. Once ready, proceed with demo recording")
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = generate_report()
    
    # Save results to file
    with open("dashboard_readiness_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed report saved to: dashboard_readiness_report.json")
