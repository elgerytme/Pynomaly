#!/usr/bin/env python3
"""
Documentation Breadcrumb Navigation Implementation

This script adds breadcrumb navigation to all documentation files to provide
clear hierarchy navigation and improve user orientation within the docs.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class BreadcrumbImplementer:
    """Implements breadcrumb navigation across documentation."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.implemented_files: List[str] = []
        self.errors: List[str] = []

        # Define breadcrumb patterns for different documentation sections
        self.breadcrumb_patterns = {
            "getting-started": {
                "base": "🏠 [Home](../index.md) > 🚀 [Getting Started](README.md)",
                "sections": {
                    "installation.md": "🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > 📦 Installation",
                    "quickstart.md": "🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > ⚡ Quick Start",
                    "platform-specific": {
                        "base": "🏠 [Home](../../index.md) > 🚀 [Getting Started](../README.md) > 🖥️ [Platform Setup](README.md)",
                        "windows.md": "🏠 [Home](../../index.md) > 🚀 [Getting Started](../README.md) > 🖥️ [Platform Setup](README.md) > 🪟 Windows",
                        "macos.md": "🏠 [Home](../../index.md) > 🚀 [Getting Started](../README.md) > 🖥️ [Platform Setup](README.md) > 🍎 macOS",
                        "linux.md": "🏠 [Home](../../index.md) > 🚀 [Getting Started](../README.md) > 🖥️ [Platform Setup](README.md) > 🐧 Linux",
                    },
                },
            },
            "user-guides": {
                "base": "🏠 [Home](../index.md) > 👤 [User Guides](README.md)",
                "sections": {
                    "basic-usage": {
                        "base": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md)",
                        "autonomous-mode.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 🤖 Autonomous Mode",
                        "datasets.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 📊 Datasets",
                        "monitoring.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 📈 Monitoring",
                    },
                    "advanced-features": {
                        "base": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md)",
                        "performance-tuning.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > ⚡ Performance Tuning",
                        "automl-and-intelligence.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 🧠 AutoML",
                        "ensemble-methods.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 🎯 Ensemble Methods",
                    },
                    "troubleshooting": {
                        "base": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔧 [Troubleshooting](README.md)",
                        "troubleshooting.md": "🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔧 [Troubleshooting](README.md) > 🆘 Common Issues",
                    },
                },
            },
            "developer-guides": {
                "base": "🏠 [Home](../index.md) > 👨‍💻 [Developer Guides](README.md)",
                "sections": {
                    "architecture": {
                        "base": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🏗️ [Architecture](README.md)",
                        "overview.md": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🏗️ [Architecture](README.md) > 📋 Overview",
                    },
                    "api-integration": {
                        "base": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md)",
                        "rest-api.md": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 🌐 REST API",
                        "python-sdk.md": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 🐍 Python SDK",
                        "cli.md": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > ⌨️ CLI",
                    },
                    "contributing": {
                        "base": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md)",
                        "HATCH_GUIDE.md": "🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📦 Hatch Guide",
                    },
                },
            },
            "reference": {
                "base": "🏠 [Home](../index.md) > 📖 [Reference](README.md)",
                "sections": {
                    "algorithms": {
                        "base": "🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md)",
                        "core-algorithms.md": "🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md) > 🔵 Core Algorithms",
                        "algorithm-comparison.md": "🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md) > ⚖️ Algorithm Comparison",
                        "specialized-algorithms.md": "🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > 🧮 [Algorithms](README.md) > 🎯 Specialized Algorithms",
                    },
                    "configuration": {
                        "base": "🏠 [Home](../../index.md) > 📖 [Reference](../README.md) > ⚙️ [Configuration](README.md)"
                    },
                },
            },
            "examples": {
                "base": "🏠 [Home](../index.md) > 💡 [Examples](README.md)",
                "sections": {
                    "banking": {
                        "base": "🏠 [Home](../../index.md) > 💡 [Examples](../README.md) > 🏦 [Banking](README.md)"
                    },
                    "notebooks": {
                        "base": "🏠 [Home](../../index.md) > 💡 [Examples](../README.md) > 📓 [Notebooks](README.md)"
                    },
                },
            },
            "deployment": {
                "base": "🏠 [Home](../index.md) > 🚀 [Deployment](README.md)",
                "sections": {
                    "DOCKER_DEPLOYMENT_GUIDE.md": "🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 🐳 Docker Guide",
                    "PRODUCTION_DEPLOYMENT_GUIDE.md": "🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 🏭 Production Guide",
                    "SECURITY.md": "🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 🔒 Security",
                },
            },
        }

    def implement_breadcrumbs(self) -> Tuple[int, int]:
        """
        Implement breadcrumb navigation across all documentation.

        Returns:
            Tuple of (files_implemented, errors_encountered)
        """
        print("🍞 Starting breadcrumb navigation implementation...")

        # Process all markdown files
        for md_file in self.docs_root.rglob("*.md"):
            # Skip files that shouldn't have breadcrumbs
            if self._should_skip_file(md_file):
                continue

            try:
                self._add_breadcrumb_to_file(md_file)
            except Exception as e:
                self.errors.append(f"Error processing {md_file}: {e}")

        print(f"✅ Implemented breadcrumbs in {len(self.implemented_files)} files")
        if self.errors:
            print(f"❌ {len(self.errors)} errors encountered")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  • {error}")

        return len(self.implemented_files), len(self.errors)

    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped for breadcrumb implementation."""
        # Skip index.md files (they are top-level)
        if file_path.name == "index.md":
            return True

        # Skip files that already have breadcrumbs
        try:
            content = file_path.read_text(encoding="utf-8")
            if "🏠 [Home]" in content or "🍞 **Breadcrumb:" in content:
                return True
        except:
            return True

        return False

    def _add_breadcrumb_to_file(self, file_path: Path):
        """Add breadcrumb navigation to a specific file."""
        breadcrumb = self._generate_breadcrumb(file_path)

        if not breadcrumb:
            return  # No breadcrumb pattern found

        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Find the title (first # heading)
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)

        if title_match:
            title_line = title_match.group(0)
            # Insert breadcrumb after the title
            breadcrumb_section = f"\n\n🍞 **Breadcrumb:** {breadcrumb}\n\n---\n"
            content = content.replace(title_line, title_line + breadcrumb_section, 1)
        else:
            # If no title found, add at the beginning
            content = f"🍞 **Breadcrumb:** {breadcrumb}\n\n---\n\n" + content

        # Only write if content changed
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            self.implemented_files.append(str(file_path.relative_to(self.docs_root)))
            print(f"  ✅ Added breadcrumb to {file_path.name}")

    def _generate_breadcrumb(self, file_path: Path) -> str:
        """Generate appropriate breadcrumb for a file based on its location."""
        relative_path = file_path.relative_to(self.docs_root)
        path_parts = relative_path.parts

        # Handle root level files
        if len(path_parts) == 1:
            return "🏠 [Home](index.md)"

        # Handle different sections
        section = path_parts[0]

        if section not in self.breadcrumb_patterns:
            # Generic breadcrumb for unknown sections
            return f"🏠 [Home](../index.md) > 📁 {section.title()}"

        section_patterns = self.breadcrumb_patterns[section]

        # Handle README files in sections
        if file_path.name == "README.md":
            if len(path_parts) == 2:
                return section_patterns["base"]
            else:
                # Nested README
                subsection = path_parts[1]
                if subsection in section_patterns["sections"]:
                    subsection_data = section_patterns["sections"][subsection]
                    if isinstance(subsection_data, dict) and "base" in subsection_data:
                        return subsection_data["base"]
                # Fallback for unknown subsections
                return section_patterns["base"] + f" > 📁 {subsection.title()}"

        # Handle specific files
        if len(path_parts) == 2:
            # File directly in section
            filename = path_parts[1]
            if filename in section_patterns["sections"]:
                return section_patterns["sections"][filename]
            else:
                # Generic pattern for unknown files
                file_title = filename.replace(".md", "").replace("-", " ").title()
                return section_patterns["base"] + f" > 📄 {file_title}"

        elif len(path_parts) == 3:
            # File in subsection
            subsection = path_parts[1]
            filename = path_parts[2]

            if subsection in section_patterns["sections"]:
                subsection_data = section_patterns["sections"][subsection]
                if isinstance(subsection_data, dict):
                    if filename in subsection_data:
                        return subsection_data[filename]
                    elif "base" in subsection_data:
                        file_title = (
                            filename.replace(".md", "").replace("-", " ").title()
                        )
                        return subsection_data["base"] + f" > 📄 {file_title}"

            # Fallback
            file_title = filename.replace(".md", "").replace("-", " ").title()
            return (
                section_patterns["base"]
                + f" > 📁 {subsection.title()} > 📄 {file_title}"
            )

        # Fallback for deeply nested files
        breadcrumb_parts = ["🏠 [Home](../index.md)"]
        for i, part in enumerate(path_parts[:-1]):
            if i == 0:
                breadcrumb_parts.append(f"📁 [{part.title()}](README.md)")
            else:
                breadcrumb_parts.append(f"📁 {part.title()}")

        # Add current file
        file_title = path_parts[-1].replace(".md", "").replace("-", " ").title()
        breadcrumb_parts.append(f"📄 {file_title}")

        return " > ".join(breadcrumb_parts)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Implement breadcrumb navigation in documentation"
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs"),
        help="Path to documentation root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    if not args.docs_root.exists():
        print(f"❌ Documentation root not found: {args.docs_root}")
        return 1

    implementer = BreadcrumbImplementer(args.docs_root)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        return 0

    implemented_count, error_count = implementer.implement_breadcrumbs()

    print(f"\n📊 Breadcrumb Implementation Summary:")
    print(f"  • Files with breadcrumbs: {implemented_count}")
    print(f"  • Errors: {error_count}")

    if error_count == 0:
        print("✅ Breadcrumb implementation completed successfully!")
        return 0
    else:
        print("⚠️  Breadcrumb implementation completed with errors")
        return 1


if __name__ == "__main__":
    exit(main())
