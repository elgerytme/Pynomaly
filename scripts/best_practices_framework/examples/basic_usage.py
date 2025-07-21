#!/usr/bin/env python3
"""
Basic Usage Example for Best Practices Framework
===============================================
Demonstrates common usage patterns and API calls
"""

import asyncio
import json
from pathlib import Path

# Import the main components
from best_practices_framework import BestPracticesValidator, quick_validate


async def basic_validation_example():
    """Example 1: Basic comprehensive validation"""
    print("🔍 Example 1: Basic Comprehensive Validation")
    print("=" * 50)
    
    # Initialize validator with project root
    validator = BestPracticesValidator(project_root=".")
    
    # Run comprehensive validation
    report = await validator.validate_all()
    
    # Display results
    print(f"Overall Score: {report.compliance_score.overall_score:.1f}%")
    print(f"Grade: {report.compliance_score.grade}")
    print(f"Total Violations: {report.compliance_score.total_violations}")
    print(f"Critical Violations: {report.compliance_score.critical_violations}")
    
    # Check if quality gate passes
    passes_quality_gate = validator.quality_gate(
        report, 
        enforce_critical=True, 
        enforce_high=False
    )
    
    print(f"Quality Gate: {'✅ PASSED' if passes_quality_gate else '❌ FAILED'}")
    print()


async def category_specific_validation_example():
    """Example 2: Category-specific validation"""
    print("🔒 Example 2: Security-Only Validation")
    print("=" * 50)
    
    validator = BestPracticesValidator(project_root=".")
    
    # Run security validation only
    security_results = await validator.validate_category("security")
    
    print(f"Security validators run: {len(security_results)}")
    
    for result in security_results:
        print(f"  - {result.validator_name}: {result.score:.1f}% ({len(result.violations)} violations)")
    
    print()


async def multiple_categories_example():
    """Example 3: Multiple specific categories"""
    print("🧪 Example 3: Testing and DevOps Validation")  
    print("=" * 50)
    
    validator = BestPracticesValidator(project_root=".")
    
    # Run multiple categories
    categories = ["testing", "devops"]
    results = {}
    
    for category in categories:
        results[category] = await validator.validate_category(category)
        
        total_violations = sum(len(r.violations) for r in results[category])
        avg_score = sum(r.score for r in results[category]) / len(results[category]) if results[category] else 0
        
        print(f"{category.title()}: {avg_score:.1f}% ({total_violations} violations)")
    
    print()


async def quick_validation_example():
    """Example 4: Quick validation helper"""
    print("⚡ Example 4: Quick Validation")
    print("=" * 50)
    
    # Use the quick validation helper function
    results = await quick_validate(
        project_root=".",
        categories=["security", "architecture"]
    )
    
    print(f"Quick validation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print()


async def custom_configuration_example():
    """Example 5: Custom configuration"""
    print("⚙️ Example 5: Custom Configuration")
    print("=" * 50)
    
    # Create a custom config file
    custom_config = """
framework_version: "1.0.0"
enabled_categories:
  - security
  - testing

global:
  enforcement_level: "moderate"
  fail_on_critical: true
  fail_on_high: true

security:
  enabled: true
  secrets_detection:
    enabled: true
    scan_all_files: true

testing:
  enabled: true
  coverage:
    unit_test_minimum: 70
    integration_test_minimum: 50
"""
    
    # Write config to temporary file
    config_path = Path("temp_config.yml")
    config_path.write_text(custom_config)
    
    try:
        # Initialize with custom configuration
        validator = BestPracticesValidator(
            project_root=".",
            config_path=config_path
        )
        
        # Run validation with custom rules
        report = await validator.validate_all()
        
        print(f"Custom validation completed:")
        print(f"  Categories: {list(report.category_results.keys())}")
        print(f"  Score: {report.compliance_score.overall_score:.1f}%")
        
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()
    
    print()


def generate_reports_example():
    """Example 6: Generate different report formats"""
    print("📊 Example 6: Report Generation")
    print("=" * 50)
    
    # Simulate validation results (normally you'd get this from actual validation)
    sample_results = {
        "project_name": "Example Project",
        "compliance_score": {
            "overall_score": 87.5,
            "grade": "B+",
            "total_violations": 12,
            "critical_violations": 0,
            "high_violations": 2,
            "category_scores": {
                "security": 95.0,
                "testing": 80.0,
                "architecture": 87.5
            },
            "recommendations": [
                "Address 2 high-priority security issues",
                "Improve test coverage in integration tests",
                "Review architecture documentation"
            ],
            "execution_time": 45.2,
            "timestamp": "2024-01-15T10:30:00Z"
        },
        "category_results": {
            "security": [{"validator_name": "secrets_detector", "score": 95.0, "violations": []}],
            "testing": [{"validator_name": "coverage_validator", "score": 80.0, "violations": []}]
        },
        "all_violations": []
    }
    
    # Generate different formats
    from best_practices_framework.reporting.report_generator import ReportGenerator
    
    generator = ReportGenerator()
    
    # JSON report
    json_report = generator.generate_json_report(sample_results)
    with open("example_report.json", "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    print("✅ Generated JSON report: example_report.json")
    
    # HTML report
    html_report = generator.generate_html_report(sample_results)
    with open("example_report.html", "w") as f:
        f.write(html_report)
    print("✅ Generated HTML report: example_report.html")
    
    # Markdown report
    markdown_report = generator.generate_markdown_report(sample_results)
    with open("example_report.md", "w") as f:
        f.write(markdown_report)
    print("✅ Generated Markdown report: example_report.md")
    
    # SARIF report (for security tools)
    sarif_report = generator.generate_sarif_report(sample_results)
    with open("example_report.sarif", "w") as f:
        json.dump(sarif_report, f, indent=2)
    print("✅ Generated SARIF report: example_report.sarif")
    
    print()


async def error_handling_example():
    """Example 7: Error handling and edge cases"""
    print("🛡️ Example 7: Error Handling")
    print("=" * 50)
    
    try:
        # Try to validate a non-existent directory
        validator = BestPracticesValidator(project_root="/non/existent/path")
        report = await validator.validate_all()
        
    except FileNotFoundError as e:
        print(f"⚠️ Handled file not found error: {e}")
    
    try:
        # Try to validate with invalid category
        validator = BestPracticesValidator(project_root=".")
        results = await validator.validate_category("invalid_category")
        
    except ValueError as e:
        print(f"⚠️ Handled invalid category error: {e}")
    
    print("✅ Error handling examples completed")
    print()


async def main():
    """Run all examples"""
    print("🏗️ Best Practices Framework - Usage Examples")
    print("=" * 60)
    print()
    
    try:
        # Run all examples
        await basic_validation_example()
        await category_specific_validation_example()  
        await multiple_categories_example()
        await quick_validation_example()
        await custom_configuration_example()
        generate_reports_example()
        await error_handling_example()
        
        print("🎉 All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Try running: best-practices validate")
        print("2. Customize .best-practices.yml for your project")
        print("3. Integrate with your CI/CD pipeline")
        print("4. Set up quality gates and automation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed the framework:")
        print("pip install best-practices-framework[full]")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your setup and try again")


if __name__ == "__main__":
    asyncio.run(main())