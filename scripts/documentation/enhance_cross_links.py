#!/usr/bin/env python3
"""
Documentation Cross-Link Enhancement Script

This script systematically improves navigation by adding consistent cross-links
across all documentation following established patterns and user journeys.
"""

import argparse
import re
from pathlib import Path


class CrossLinkEnhancer:
    """Enhances documentation with consistent cross-linking."""

    def __init__(self, docs_root: Path):
        self.docs_root = docs_root
        self.enhanced_files: list[str] = []
        self.errors: list[str] = []

        # Define cross-linking patterns based on document analysis
        self.link_patterns = {
            # Getting started documents should link to user guides
            "getting-started": {
                "outgoing": [
                    ("../user-guides/README.md", "User Guides"),
                    (
                        "../user-guides/basic-usage/autonomous-mode.md",
                        "Autonomous Mode",
                    ),
                    ("../user-guides/basic-usage/datasets.md", "Dataset Management"),
                    ("../examples/README.md", "Examples and Tutorials"),
                    ("../developer-guides/README.md", "Developer Documentation"),
                ]
            },
            # User guides should cross-reference each other and link to references
            "user-guides/basic-usage": {
                "outgoing": [
                    ("../advanced-features/README.md", "Advanced Features"),
                    ("../../reference/algorithms/README.md", "Algorithm Reference"),
                    (
                        "../../developer-guides/api-integration/README.md",
                        "API Documentation",
                    ),
                    ("../troubleshooting/troubleshooting.md", "Troubleshooting"),
                ]
            },
            "user-guides/advanced-features": {
                "outgoing": [
                    ("../basic-usage/README.md", "Basic Usage"),
                    ("../../reference/algorithms/README.md", "Algorithm Reference"),
                    ("../../deployment/README.md", "Deployment Guides"),
                    ("../../examples/README.md", "Examples"),
                ]
            },
            # Algorithm reference should link to usage guides
            "reference/algorithms": {
                "outgoing": [
                    (
                        "../../user-guides/basic-usage/autonomous-mode.md",
                        "Autonomous Mode",
                    ),
                    (
                        "../../user-guides/advanced-features/performance-tuning.md",
                        "Performance Tuning",
                    ),
                    ("../../examples/README.md", "Examples"),
                    (
                        "../../developer-guides/api-integration/README.md",
                        "API Integration",
                    ),
                ]
            },
            # Developer guides should link to references and examples
            "developer-guides": {
                "outgoing": [
                    ("../reference/algorithms/README.md", "Algorithm Reference"),
                    ("../user-guides/README.md", "User Guides"),
                    ("../examples/README.md", "Examples"),
                    ("../deployment/README.md", "Deployment"),
                ]
            },
            # Examples should link to guides and references
            "examples": {
                "outgoing": [
                    ("../user-guides/README.md", "User Guides"),
                    ("../reference/algorithms/README.md", "Algorithm Reference"),
                    ("../developer-guides/README.md", "Developer Documentation"),
                    ("../getting-started/README.md", "Getting Started"),
                ]
            },
            # Deployment docs should link to security and configuration
            "deployment": {
                "outgoing": [
                    ("../user-guides/basic-usage/monitoring.md", "Monitoring"),
                    ("../developer-guides/contributing/README.md", "Development Setup"),
                    (
                        "../security/security-best-practices.md",
                        "Security Best Practices",
                    ),
                    ("SECURITY.md", "Security Configuration"),
                ]
            },
        }

        # Define common footer sections for different document types
        self.footer_sections = {
            "user-guide": """
---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
""",
            "developer-guide": """
---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
""",
            "reference": """
---

## 🔗 **Related Documentation**

### **User Guides**
- **[Basic Usage](../../user-guides/basic-usage/README.md)** - Getting started with algorithms
- **[Advanced Features](../../user-guides/advanced-features/README.md)** - Advanced algorithm usage
- **[Autonomous Mode](../../user-guides/basic-usage/autonomous-mode.md)** - Automatic algorithm selection

### **Examples**
- **[Algorithm Examples](../../examples/README.md)** - Practical usage examples
- **[Performance Benchmarks](../../examples/performance/)** - Algorithm performance data
- **[Use Case Examples](../../examples/tutorials/)** - Real-world applications

### **Development**
- **[API Integration](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Custom Algorithms](../../developer-guides/contributing/README.md)** - Adding new algorithms
- **[Testing](../../developer-guides/contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Algorithm testing

---

## 🆘 **Getting Help**

- **[Algorithm Selection Guide](README.md)** - Choosing the right algorithm
- **[Performance Tuning](../../user-guides/advanced-features/performance-tuning.md)** - Optimization tips
- **[Troubleshooting](../../user-guides/troubleshooting/README.md)** - Common issues
""",
        }

    def enhance_all_documents(self) -> tuple[int, int]:
        """
        Enhance all documentation with improved cross-linking.

        Returns:
            Tuple of (files_enhanced, errors_encountered)
        """
        print("🔗 Starting cross-link enhancement...")

        # Process each category of documents
        categories = [
            "getting-started",
            "user-guides",
            "developer-guides",
            "reference",
            "examples",
            "deployment",
        ]

        for category in categories:
            category_path = self.docs_root / category
            if category_path.exists():
                self._enhance_category(category, category_path)

        # Enhance orphaned documents
        self._enhance_orphaned_documents()

        # Add navigation to hub documents
        self._enhance_hub_documents()

        print(f"✅ Enhanced {len(self.enhanced_files)} files")
        if self.errors:
            print(f"❌ {len(self.errors)} errors encountered")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  • {error}")

        return len(self.enhanced_files), len(self.errors)

    def _enhance_category(self, category: str, category_path: Path):
        """Enhance all documents in a specific category."""
        print(f"📁 Processing {category}...")

        for md_file in category_path.rglob("*.md"):
            if md_file.name == "README.md":
                continue  # Skip README files for now

            try:
                self._enhance_document(md_file, category)
            except Exception as e:
                self.errors.append(f"Error processing {md_file}: {e}")

    def _enhance_document(self, file_path: Path, category: str):
        """Enhance a single document with cross-links."""
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Check if document already has a "Related Documentation" section
        if "## 🔗" in content or "Related Documentation" in content:
            return  # Skip if already has cross-links

        # Determine document type and add appropriate footer
        doc_type = self._determine_document_type(file_path, category)
        footer = self.footer_sections.get(doc_type, self.footer_sections["user-guide"])

        # Add footer before any existing "Getting Help" or at the end
        if "## 🆘" in content:
            # Insert before existing help section
            content = content.replace("## 🆘", f"{footer}## 🆘")
        else:
            # Add at the end
            content = content.rstrip() + "\n" + footer

        # Only write if content changed
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            self.enhanced_files.append(str(file_path.relative_to(self.docs_root)))
            print(f"  ✅ Enhanced {file_path.name}")

    def _determine_document_type(self, file_path: Path, category: str) -> str:
        """Determine the type of document for appropriate footer selection."""
        path_str = str(file_path)

        if "developer-guides" in path_str:
            return "developer-guide"
        elif "reference" in path_str:
            return "reference"
        else:
            return "user-guide"

    def _enhance_orphaned_documents(self):
        """Add navigation to documents that have no incoming links."""
        print("🔗 Enhancing orphaned documents...")

        # Key orphaned documents to enhance
        orphaned_files = [
            "accessibility/accessibility-guidelines.md",
            "design-system/README.md",
            "examples/README.md",
            "security/security-best-practices.md",
            "storybook/README.md",
            "testing/cross-browser-testing.md",
        ]

        for file_rel_path in orphaned_files:
            file_path = self.docs_root / file_rel_path
            if file_path.exists():
                try:
                    self._add_navigation_to_orphaned(file_path)
                except Exception as e:
                    self.errors.append(f"Error enhancing orphaned {file_path}: {e}")

    def _add_navigation_to_orphaned(self, file_path: Path):
        """Add basic navigation to an orphaned document."""
        content = file_path.read_text(encoding="utf-8")

        # Skip if already has navigation
        if "🔗" in content or "Related Documentation" in content:
            return

        # Add basic navigation based on file location
        if "accessibility" in str(file_path):
            nav = """
---

## 🔗 **Related Documentation**

- **[Design System](../design-system/README.md)** - UI design guidelines
- **[Testing](../testing/cross-browser-testing.md)** - Cross-browser testing
- **[User Guides](../user-guides/README.md)** - User documentation
"""
        elif "design-system" in str(file_path):
            nav = """
---

## 🔗 **Related Documentation**

- **[Accessibility](../accessibility/accessibility-guidelines.md)** - Accessibility guidelines
- **[Storybook](../storybook/README.md)** - Component documentation
- **[UI Testing](../testing/cross-browser-testing.md)** - UI testing guides
"""
        elif "examples" in str(file_path) and file_path.name == "README.md":
            nav = """
---

## 🔗 **Related Documentation**

- **[Getting Started](../getting-started/README.md)** - Installation and setup
- **[User Guides](../user-guides/README.md)** - Feature documentation
- **[Algorithm Reference](../reference/algorithms/README.md)** - Algorithm details
- **[Developer Guides](../developer-guides/README.md)** - Development documentation
"""
        else:
            nav = """
---

## 🔗 **Related Documentation**

- **[User Guides](../user-guides/README.md)** - Feature documentation
- **[Getting Started](../getting-started/README.md)** - Installation and setup
- **[Examples](../examples/README.md)** - Real-world examples
"""

        content = content.rstrip() + "\n" + nav
        file_path.write_text(content, encoding="utf-8")
        self.enhanced_files.append(str(file_path.relative_to(self.docs_root)))
        print(f"  ✅ Added navigation to {file_path.name}")

    def _enhance_hub_documents(self):
        """Enhance high-traffic hub documents with better outgoing links."""
        print("🔗 Enhancing hub documents...")

        # Hub documents identified in analysis
        hub_files = [
            "user-guides/basic-usage/monitoring.md",
            "user-guides/advanced-features/performance-tuning.md",
            "developer-guides/architecture/overview.md",
            "developer-guides/contributing/HATCH_GUIDE.md",
        ]

        for file_rel_path in hub_files:
            file_path = self.docs_root / file_rel_path
            if file_path.exists():
                try:
                    self._enhance_hub_document(file_path)
                except Exception as e:
                    self.errors.append(f"Error enhancing hub {file_path}: {e}")

    def _enhance_hub_document(self, file_path: Path):
        """Add comprehensive navigation to a hub document."""
        content = file_path.read_text(encoding="utf-8")

        # Skip if already enhanced
        if "🔗" in content and len(re.findall(r"\[.*?\]\(.*?\.md\)", content)) > 5:
            return

        # Add comprehensive navigation section based on document type
        if "monitoring.md" in str(file_path):
            enhanced_nav = """
---

## 🔗 **Related Documentation**

### **Monitoring & Observability**
- **[Performance Analysis](../advanced-features/performance.md)** - Performance monitoring and optimization
- **[Security Monitoring](../../deployment/SECURITY.md)** - Security monitoring and alerting
- **[Production Deployment](../../deployment/PRODUCTION_DEPLOYMENT_GUIDE.md)** - Production monitoring setup

### **Configuration**
- **[API Configuration](../../developer-guides/api-integration/rest-api.md)** - API monitoring setup
- **[Docker Monitoring](../../deployment/DOCKER_DEPLOYMENT_GUIDE.md)** - Container monitoring
- **[Kubernetes Monitoring](../../deployment/kubernetes.md)** - Orchestration monitoring

### **Development**
- **[Testing Infrastructure](../../developer-guides/contributing/COMPREHENSIVE_TEST_ANALYSIS.md)** - Test monitoring
- **[Development Setup](../../developer-guides/contributing/README.md)** - Development monitoring
"""
        elif "performance-tuning.md" in str(file_path):
            enhanced_nav = """
---

## 🔗 **Related Documentation**

### **Performance & Optimization**
- **[Performance Analysis](performance.md)** - Performance monitoring and analysis
- **[Algorithm Comparison](../../reference/algorithms/algorithm-comparison.md)** - Algorithm performance comparison
- **[AutoML](automl-and-intelligence.md)** - Automated performance optimization

### **Algorithms**
- **[Core Algorithms](../../reference/algorithms/core-algorithms.md)** - High-performance algorithms
- **[Specialized Algorithms](../../reference/algorithms/specialized-algorithms.md)** - Domain-optimized algorithms
- **[Autonomous Mode](../basic-usage/autonomous-mode.md)** - Automatic optimization

### **Deployment**
- **[Production Deployment](../../deployment/PRODUCTION_DEPLOYMENT_GUIDE.md)** - Production performance
- **[Docker Optimization](../../deployment/DOCKER_DEPLOYMENT_GUIDE.md)** - Container performance
- **[Monitoring](../basic-usage/monitoring.md)** - Performance monitoring
"""
        else:
            # Generic hub enhancement
            enhanced_nav = """
---

## 🔗 **Related Documentation**

- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[Examples](../../examples/README.md)** - Real-world examples
- **[API Documentation](../api-integration/README.md)** - Programming interfaces
"""

        # Insert navigation before any existing help section or at the end
        if "🆘" in content:
            content = content.replace("## 🆘", f"{enhanced_nav}\n## 🆘")
        else:
            content = content.rstrip() + "\n" + enhanced_nav

        file_path.write_text(content, encoding="utf-8")
        self.enhanced_files.append(str(file_path.relative_to(self.docs_root)))
        print(f"  ✅ Enhanced hub document {file_path.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhance documentation cross-linking")
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

    enhancer = CrossLinkEnhancer(args.docs_root)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        return 0

    enhanced_count, error_count = enhancer.enhance_all_documents()

    print("\n📊 Cross-link Enhancement Summary:")
    print(f"  • Files enhanced: {enhanced_count}")
    print(f"  • Errors: {error_count}")

    if error_count == 0:
        print("✅ Cross-link enhancement completed successfully!")
        return 0
    else:
        print("⚠️  Cross-link enhancement completed with errors")
        return 1


if __name__ == "__main__":
    exit(main())
