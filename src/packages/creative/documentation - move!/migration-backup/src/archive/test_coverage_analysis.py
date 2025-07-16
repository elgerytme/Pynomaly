#!/usr/bin/env python3
"""
Comprehensive test coverage analysis for Pynomaly project.
Identifies files that don't have corresponding test files.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TestGap:
    source_file: str
    expected_test_path: str
    test_type: str
    priority: str  # high, medium, low
    reason: str

class TestCoverageAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src" / "monorepo"
        self.test_dir = self.project_root / "tests"
        self.gaps: List[TestGap] = []
        
    def get_all_python_files(self, directory: Path) -> List[Path]:
        """Get all Python files in directory, excluding __pycache__"""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        return python_files
        
    def get_existing_tests(self) -> Set[str]:
        """Get all existing test files"""
        test_files = self.get_all_python_files(self.test_dir)
        existing_tests = set()
        
        for test_file in test_files:
            # Convert test path to relative path from tests directory
            rel_path = test_file.relative_to(self.test_dir)
            existing_tests.add(str(rel_path))
            
        return existing_tests
        
    def get_expected_test_path(self, source_file: Path) -> Tuple[str, str]:
        """Get expected test path for a source file"""
        # Get relative path from src/pynomaly
        rel_path = source_file.relative_to(self.src_dir)
        
        # Determine test type based on directory structure
        parts = rel_path.parts
        
        if parts[0] == "application":
            if "dto" in parts:
                test_type = "unit"
            elif "use_cases" in parts:
                test_type = "unit"
            elif "services" in parts:
                test_type = "unit"
            else:
                test_type = "unit"
        elif parts[0] == "domain":
            test_type = "unit"
        elif parts[0] == "infrastructure":
            test_type = "unit"
        elif parts[0] == "shared":
            test_type = "unit"
        elif parts[0] == "presentation":
            test_type = "integration"
        else:
            test_type = "unit"
            
        # Build expected test path
        test_file_name = f"test_{rel_path.stem}.py"
        expected_path = Path(test_type) / rel_path.parent / test_file_name
        
        return str(expected_path), test_type
        
    def categorize_priority(self, source_file: Path) -> Tuple[str, str]:
        """Categorize the priority and reason for testing"""
        rel_path = source_file.relative_to(self.src_dir)
        parts = rel_path.parts
        filename = source_file.name
        
        # High priority files
        if "protocol" in str(rel_path).lower():
            return "high", "Protocol definitions are critical interfaces"
        elif filename.endswith("_dto.py"):
            return "high", "DTOs define data contracts"
        elif "shared" in parts:
            return "high", "Shared modules are used across the application"
        elif "exceptions" in str(rel_path).lower():
            return "high", "Exception handling is critical for reliability"
        elif "value_objects" in parts:
            return "high", "Value objects contain business logic"
        elif "entities" in parts:
            return "high", "Domain entities are core business objects"
        elif "services" in parts:
            return "high", "Services contain business logic"
        elif "use_cases" in parts:
            return "high", "Use cases define application behavior"
        elif "repositories" in parts:
            return "medium", "Data access layer needs testing"
        elif "adapters" in parts:
            return "medium", "External integrations need testing"
        elif "config" in parts:
            return "medium", "Configuration modules affect behavior"
        elif filename == "__init__.py":
            return "low", "Init files typically just import modules"
        elif "monitoring" in parts:
            return "medium", "Monitoring affects operations"
        elif "security" in parts:
            return "high", "Security modules are critical"
        else:
            return "medium", "Standard module requiring test coverage"
            
    def analyze_coverage(self) -> Dict[str, List[TestGap]]:
        """Analyze test coverage and identify gaps"""
        source_files = self.get_all_python_files(self.src_dir)
        existing_tests = self.get_existing_tests()
        
        # Filter out files that shouldn't be tested
        testable_files = []
        for file in source_files:
            rel_path = file.relative_to(self.src_dir)
            
            # Skip certain files
            if (file.name in ['__main__.py', '_version.py', 'py.typed'] or
                'demo_functions.py' in file.name or
                '__pycache__' in str(file)):
                continue
                
            testable_files.append(file)
            
        # Check for missing tests
        gaps_by_category = defaultdict(list)
        
        for source_file in testable_files:
            expected_test_path, test_type = self.get_expected_test_path(source_file)
            priority, reason = self.categorize_priority(source_file)
            
            # Check if test exists
            if expected_test_path not in existing_tests:
                # Also check alternative paths
                alt_paths = self.get_alternative_test_paths(source_file)
                found_alternative = any(alt_path in existing_tests for alt_path in alt_paths)
                
                if not found_alternative:
                    gap = TestGap(
                        source_file=str(source_file.relative_to(self.project_root)),
                        expected_test_path=expected_test_path,
                        test_type=test_type,
                        priority=priority,
                        reason=reason
                    )
                    gaps_by_category[priority].append(gap)
                    
        return gaps_by_category
        
    def get_alternative_test_paths(self, source_file: Path) -> List[str]:
        """Get alternative test paths where the test might exist"""
        rel_path = source_file.relative_to(self.src_dir)
        alternatives = []
        
        # Try different test directories
        test_dirs = ["unit", "integration", "domain", "infrastructure", "application"]
        
        for test_dir in test_dirs:
            test_file_name = f"test_{rel_path.stem}.py"
            alt_path = Path(test_dir) / rel_path.parent / test_file_name
            alternatives.append(str(alt_path))
            
            # Also try without subdirectory structure
            alt_path = Path(test_dir) / test_file_name
            alternatives.append(str(alt_path))
            
        return alternatives
        
    def generate_report(self) -> str:
        """Generate a comprehensive test coverage report"""
        gaps_by_category = self.analyze_coverage()
        
        report = []
        report.append("# Test Coverage Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        total_gaps = sum(len(gaps) for gaps in gaps_by_category.values())
        report.append(f"Total files missing tests: {total_gaps}")
        report.append(f"High priority: {len(gaps_by_category['high'])}")
        report.append(f"Medium priority: {len(gaps_by_category['medium'])}")
        report.append(f"Low priority: {len(gaps_by_category['low'])}")
        report.append("")
        
        # Detailed breakdown by priority
        for priority in ["high", "medium", "low"]:
            gaps = gaps_by_category[priority]
            if not gaps:
                continue
                
            report.append(f"## {priority.upper()} Priority Files Missing Tests ({len(gaps)} files)")
            report.append("-" * 50)
            
            # Group by category
            by_category = defaultdict(list)
            for gap in gaps:
                category = self.get_file_category(gap.source_file)
                by_category[category].append(gap)
                
            for category, category_gaps in sorted(by_category.items()):
                report.append(f"\n### {category}")
                for gap in sorted(category_gaps, key=lambda x: x.source_file):
                    report.append(f"- **{gap.source_file}**")
                    report.append(f"  - Expected test: `{gap.expected_test_path}`")
                    report.append(f"  - Test type: {gap.test_type}")
                    report.append(f"  - Reason: {gap.reason}")
                    report.append("")
                    
        return "\n".join(report)
        
    def get_file_category(self, file_path: str) -> str:
        """Categorize files for reporting"""
        if "shared/protocols" in file_path:
            return "Protocol Definitions"
        elif "shared/" in file_path:
            return "Shared Modules"
        elif "application/dto" in file_path:
            return "Data Transfer Objects"
        elif "application/services" in file_path:
            return "Application Services"
        elif "application/use_cases" in file_path:
            return "Use Cases"
        elif "domain/entities" in file_path:
            return "Domain Entities"
        elif "domain/value_objects" in file_path:
            return "Value Objects"
        elif "domain/services" in file_path:
            return "Domain Services"
        elif "domain/exceptions" in file_path:
            return "Domain Exceptions"
        elif "infrastructure/adapters" in file_path:
            return "Infrastructure Adapters"
        elif "infrastructure/repositories" in file_path:
            return "Repositories"
        elif "infrastructure/config" in file_path:
            return "Configuration"
        elif "infrastructure/auth" in file_path:
            return "Authentication"
        elif "infrastructure/security" in file_path:
            return "Security"
        elif "infrastructure/monitoring" in file_path:
            return "Monitoring"
        elif "infrastructure/" in file_path:
            return "Infrastructure"
        elif "presentation/" in file_path:
            return "Presentation Layer"
        else:
            return "Other"
            
    def generate_creation_commands(self) -> List[str]:
        """Generate commands to create missing test files"""
        gaps_by_category = self.analyze_coverage()
        commands = []
        
        # Focus on high priority files first
        for priority in ["high"]:
            gaps = gaps_by_category[priority]
            for gap in sorted(gaps, key=lambda x: x.source_file):
                test_path = self.test_dir / gap.expected_test_path
                commands.append(f"mkdir -p {test_path.parent}")
                commands.append(f"touch {test_path}")
                
        return commands

def main():
    analyzer = TestCoverageAnalyzer("/mnt/c/Users/andre/Pynomaly")
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Also save to file
    with open("/mnt/c/Users/andre/Pynomaly/test_coverage_report.md", "w") as f:
        f.write(report)
        
    print("\nReport saved to test_coverage_report.md")

if __name__ == "__main__":
    main()