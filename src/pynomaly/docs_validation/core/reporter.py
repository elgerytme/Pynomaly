"""Report generation for documentation validation results."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ValidationReporter:
    """Generates validation reports in multiple formats."""

    def __init__(self, config):
        """Initialize reporter with configuration."""
        self.config = config

    def generate_report(self, result, output_format: str = "console") -> Optional[str]:
        """Generate validation report in specified format.

        Args:
            result: ValidationResult object containing validation results
            output_format: Format for report (console, json, html, markdown)

        Returns:
            Report content as string, or None for console output
        """
        if output_format == "console":
            return self._generate_console_report(result)
        elif output_format == "json":
            return self._generate_json_report(result)
        elif output_format == "html":
            return self._generate_html_report(result)
        elif output_format == "markdown":
            return self._generate_markdown_report(result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_console_report(self, result) -> None:
        """Generate colorized console report."""
        print("\n" + "=" * 60)
        print("📚 DOCUMENTATION VALIDATION REPORT")
        print("=" * 60)

        # Summary
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Files checked: {result.file_count}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")

        # Errors
        if result.errors:
            print("\n🚨 ERRORS:")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")

        # Warnings
        if result.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")

        # Metrics
        if result.metrics:
            print("\n📊 METRICS:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")

        print("=" * 60)
        return None

    def _generate_json_report(self, result) -> str:
        """Generate JSON report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "validation_result": {
                "passed": result.passed,
                "file_count": result.file_count,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
                "warnings": result.warnings,
                "metrics": result.metrics,
            },
            "configuration": (
                self.config.to_dict() if hasattr(self.config, "to_dict") else {}
            ),
        }

        return json.dumps(report_data, indent=2, ensure_ascii=False)

    def _generate_html_report(self, result) -> str:
        """Generate HTML report."""
        status_color = "green" if result.passed else "red"
        status_text = "PASSED" if result.passed else "FAILED"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Documentation Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .status {{ color: {status_color}; font-weight: bold; }}
        .errors {{ background: #ffebee; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .warnings {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .metrics {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Documentation Validation Report</h1>
        <p><strong>Status:</strong> <span class="status">{status_text}</span></p>
        <p><strong>Files checked:</strong> {result.file_count}</p>
        <p><strong>Duration:</strong> {result.duration_seconds:.2f}s</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        if result.errors:
            html += """
    <div class="errors">
        <h2>🚨 Errors</h2>
        <ul>
"""
            for error in result.errors:
                html += f"            <li>• {error}</li>\n"
            html += "        </ul>\n    </div>\n"

        if result.warnings:
            html += """
    <div class="warnings">
        <h2>⚠️ Warnings</h2>
        <ul>
"""
            for warning in result.warnings:
                html += f"            <li>• {warning}</li>\n"
            html += "        </ul>\n    </div>\n"

        if result.metrics:
            html += """
    <div class="metrics">
        <h2>📊 Metrics</h2>
        <ul>
"""
            for key, value in result.metrics.items():
                html += f"            <li><strong>{key}:</strong> {value}</li>\n"
            html += "        </ul>\n    </div>\n"

        html += """
</body>
</html>
"""
        return html

    def _generate_markdown_report(self, result) -> str:
        """Generate Markdown report."""
        status_emoji = "✅" if result.passed else "❌"
        status_text = "PASSED" if result.passed else "FAILED"

        md = f"""# 📚 Documentation Validation Report

## Summary

- **Status:** {status_emoji} {status_text}
- **Files checked:** {result.file_count}
- **Duration:** {result.duration_seconds:.2f}s
- **Errors:** {len(result.errors)}
- **Warnings:** {len(result.warnings)}
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        if result.errors:
            md += "## 🚨 Errors\n\n"
            for i, error in enumerate(result.errors, 1):
                md += f"{i}. {error}\n"
            md += "\n"

        if result.warnings:
            md += "## ⚠️ Warnings\n\n"
            for i, warning in enumerate(result.warnings, 1):
                md += f"{i}. {warning}\n"
            md += "\n"

        if result.metrics:
            md += "## 📊 Metrics\n\n"
            for key, value in result.metrics.items():
                md += f"- **{key}:** {value}\n"
            md += "\n"

        return md

    def save_report(
        self, result, output_format: str = "json", filename: Optional[str] = None
    ) -> str:
        """Save report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "html" if output_format == "html" else output_format
            filename = f"{self.config.output_path}_{timestamp}.{extension}"

        report_content = self.generate_report(result, output_format)

        if report_content is not None:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(f"Report saved to: {output_path}")
            return str(output_path)

        return ""
