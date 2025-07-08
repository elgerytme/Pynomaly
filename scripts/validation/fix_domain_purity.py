#!/usr/bin/env python3
"""
Domain Layer Purity Remediation Script

This script identifies and provides recommendations for fixing domain layer
violations in Clean Architecture, particularly external dependency usage.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DomainPurityFixer:
    """Analyzes and fixes domain layer purity violations."""

    def __init__(self, src_root: Path):
        self.src_root = src_root
        self.domain_root = src_root / "pynomaly" / "domain"
        self.violations: List[Dict] = []
        self.fixes_applied: List[str] = []

        # External dependencies commonly found in domain that should be abstracted
        self.problematic_imports = {
            "pydantic": "Use dataclasses or custom validation in infrastructure",
            "numpy": "Abstract mathematical operations into domain services",
            "pandas": "Use domain entities with simple Python types",
            "scipy": "Abstract statistical operations into domain services",
            "sklearn": "Move ML-specific logic to infrastructure adapters",
            "torch": "Move deep learning logic to infrastructure",
            "tensorflow": "Move deep learning logic to infrastructure",
        }

        # Allowed standard library modules for domain layer
        self.allowed_stdlib = {
            "typing",
            "abc",
            "dataclasses",
            "enum",
            "functools",
            "itertools",
            "collections",
            "datetime",
            "uuid",
            "math",
            "re",
            "json",
            "pathlib",
            "logging",
            "time",
            "os",
            "sys",
            "warnings",
            "weakref",
            "copy",
            "decimal",
            "fractions",
            "operator",
            "random",
            "string",
            "textwrap",
            "unicodedata",
            "__future__",
        }

    def analyze_violations(self) -> List[Dict]:
        """
        Analyze domain layer for architectural violations.

        Returns:
            List of violation details with fix recommendations
        """
        print("🔍 Analyzing domain layer purity violations...")

        if not self.domain_root.exists():
            print(f"❌ Domain directory not found: {self.domain_root}")
            return []

        for py_file in self.domain_root.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            violations = self._analyze_file(py_file)
            self.violations.extend(violations)

        print(
            f"📊 Found {len(self.violations)} violations across {len(set(v['file'] for v in self.violations))} files"
        )
        return self.violations

    def _analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze a single file for violations."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in self.problematic_imports:
                            violations.append(
                                {
                                    "file": str(file_path.relative_to(self.src_root)),
                                    "line": node.lineno,
                                    "type": "problematic_import",
                                    "module": module,
                                    "import_stmt": f"import {alias.name}",
                                    "recommendation": self.problematic_imports[module],
                                }
                            )
                        elif (
                            module not in self.allowed_stdlib
                            and not module.startswith("pynomaly.domain")
                            and not module.startswith("pynomaly.shared")
                        ):
                            violations.append(
                                {
                                    "file": str(file_path.relative_to(self.src_root)),
                                    "line": node.lineno,
                                    "type": "external_dependency",
                                    "module": module,
                                    "import_stmt": f"import {alias.name}",
                                    "recommendation": "Move to infrastructure layer or abstract into domain service",
                                }
                            )

                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module in self.problematic_imports:
                        violations.append(
                            {
                                "file": str(file_path.relative_to(self.src_root)),
                                "line": node.lineno,
                                "type": "problematic_import",
                                "module": module,
                                "import_stmt": f"from {node.module} import {', '.join(alias.name for alias in node.names)}",
                                "recommendation": self.problematic_imports[module],
                            }
                        )
                    elif (
                        module not in self.allowed_stdlib
                        and not module.startswith("pynomaly.domain")
                        and not module.startswith("pynomaly.shared")
                    ):
                        violations.append(
                            {
                                "file": str(file_path.relative_to(self.src_root)),
                                "line": node.lineno,
                                "type": "external_dependency",
                                "module": module,
                                "import_stmt": f"from {node.module} import {', '.join(alias.name for alias in node.names)}",
                                "recommendation": "Move to infrastructure layer or abstract into domain service",
                            }
                        )

        except Exception as e:
            print(f"⚠️  Could not analyze {file_path}: {e}")

        return violations

    def generate_remediation_plan(self) -> Dict:
        """Generate a comprehensive remediation plan."""
        print("📋 Generating remediation plan...")

        # Group violations by file and type
        by_file = {}
        by_module = {}

        for violation in self.violations:
            file_path = violation["file"]
            module = violation["module"]

            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(violation)

            if module not in by_module:
                by_module[module] = []
            by_module[module].append(violation)

        # Generate specific remediation strategies
        remediation_plan = {
            "summary": {
                "total_violations": len(self.violations),
                "affected_files": len(by_file),
                "problematic_modules": list(by_module.keys()),
            },
            "by_file": by_file,
            "by_module": by_module,
            "strategies": self._generate_strategies(by_module),
            "implementation_order": self._determine_implementation_order(by_file),
        }

        return remediation_plan

    def _generate_strategies(self, by_module: Dict) -> Dict:
        """Generate specific remediation strategies for each module."""
        strategies = {}

        for module, violations in by_module.items():
            if module == "pydantic":
                strategies[module] = {
                    "strategy": "Replace with dataclasses and move validation to infrastructure",
                    "steps": [
                        "1. Convert Pydantic models to @dataclass entities",
                        "2. Move validation logic to infrastructure/validation/",
                        "3. Create domain protocols for validation interfaces",
                        "4. Update application layer to use validation services",
                    ],
                    "example": """
# Before (Domain - ❌)
from pydantic import BaseModel

class Anomaly(BaseModel):
    score: float
    threshold: float

# After (Domain - ✅)
from dataclasses import dataclass

@dataclass
class Anomaly:
    score: float
    threshold: float

# Infrastructure Validation (✅)
from pydantic import BaseModel
from pynomaly.domain.entities import Anomaly

class AnomalyValidator(BaseModel):
    score: float
    threshold: float

    def to_domain(self) -> Anomaly:
        return Anomaly(score=self.score, threshold=self.threshold)
""",
                }

            elif module == "numpy":
                strategies[module] = {
                    "strategy": "Abstract numerical operations into domain services",
                    "steps": [
                        "1. Create domain services for mathematical operations",
                        "2. Define protocols for numerical operations",
                        "3. Move numpy implementations to infrastructure",
                        "4. Use simple Python types in domain entities",
                    ],
                    "example": """
# Before (Domain - ❌)
import numpy as np

class AnomalyScore:
    def calculate_percentile(self, scores: np.ndarray) -> float:
        return np.percentile(scores, 95)

# After (Domain - ✅)
from typing import List, Protocol

class StatisticalService(Protocol):
    def calculate_percentile(self, scores: List[float], percentile: float) -> float: ...

class AnomalyScore:
    def __init__(self, stats_service: StatisticalService):
        self._stats = stats_service

    def calculate_percentile(self, scores: List[float]) -> float:
        return self._stats.calculate_percentile(scores, 95)

# Infrastructure Implementation (✅)
import numpy as np

class NumpyStatisticalService:
    def calculate_percentile(self, scores: List[float], percentile: float) -> float:
        return np.percentile(scores, percentile)
""",
                }

            elif module == "pandas":
                strategies[module] = {
                    "strategy": "Use domain entities with simple Python types",
                    "steps": [
                        "1. Replace DataFrame usage with domain entities",
                        "2. Use List, Dict, and simple types in domain",
                        "3. Move DataFrame operations to infrastructure",
                        "4. Create adapters for data transformation",
                    ],
                    "example": """
# Before (Domain - ❌)
import pandas as pd

class Dataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

# After (Domain - ✅)
from typing import List, Dict, Any

@dataclass
class DataPoint:
    features: Dict[str, Any]
    timestamp: datetime

@dataclass
class Dataset:
    name: str
    data_points: List[DataPoint]
    metadata: Dict[str, Any]

# Infrastructure Adapter (✅)
import pandas as pd

class PandasDatasetAdapter:
    def from_dataframe(self, df: pd.DataFrame) -> Dataset:
        data_points = [
            DataPoint(features=row.to_dict(), timestamp=row.get('timestamp'))
            for _, row in df.iterrows()
        ]
        return Dataset(name="", data_points=data_points, metadata={})
""",
                }

            else:
                strategies[module] = {
                    "strategy": f"Abstract {module} functionality into infrastructure layer",
                    "steps": [
                        f"1. Create domain protocols for {module} operations",
                        f"2. Move {module} implementations to infrastructure/adapters/",
                        "3. Use dependency injection in application layer",
                        "4. Keep domain layer pure with protocols only",
                    ],
                }

        return strategies

    def _determine_implementation_order(self, by_file: Dict) -> List[Dict]:
        """Determine the optimal order for implementing fixes."""
        # Prioritize by impact and dependency complexity
        priority_order = []

        for file_path, violations in by_file.items():
            violation_types = set(v["type"] for v in violations)
            modules = set(v["module"] for v in violations)

            # Calculate priority score
            priority_score = 0
            if "pydantic" in modules:
                priority_score += 10  # High priority - affects many entities
            if "numpy" in modules or "pandas" in modules:
                priority_score += 8  # Medium-high priority
            if len(violations) > 5:
                priority_score += 5  # Many violations

            priority_order.append(
                {
                    "file": file_path,
                    "violations_count": len(violations),
                    "modules": list(modules),
                    "priority_score": priority_score,
                    "estimated_effort": self._estimate_effort(violations),
                }
            )

        # Sort by priority score (highest first)
        priority_order.sort(key=lambda x: x["priority_score"], reverse=True)

        return priority_order

    def _estimate_effort(self, violations: List[Dict]) -> str:
        """Estimate the effort required to fix violations."""
        violation_count = len(violations)
        modules = set(v["module"] for v in violations)

        if violation_count <= 2:
            return "Low (1-2 hours)"
        elif violation_count <= 5:
            return "Medium (2-4 hours)"
        elif "pydantic" in modules or violation_count > 10:
            return "High (4-8 hours)"
        else:
            return "Medium-High (3-6 hours)"

    def print_remediation_report(self, plan: Dict):
        """Print a comprehensive remediation report."""
        print("\n" + "=" * 80)
        print("🏗️  DOMAIN LAYER PURITY REMEDIATION PLAN")
        print("=" * 80)

        # Summary
        summary = plan["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   • Total violations: {summary['total_violations']}")
        print(f"   • Affected files: {summary['affected_files']}")
        print(f"   • Problematic modules: {', '.join(summary['problematic_modules'])}")

        # Implementation order
        print(f"\n🎯 IMPLEMENTATION ORDER (by priority):")
        for i, item in enumerate(plan["implementation_order"][:10], 1):  # Top 10
            print(f"   {i:2d}. {item['file']}")
            print(f"       • {item['violations_count']} violations")
            print(f"       • Modules: {', '.join(item['modules'])}")
            print(f"       • Effort: {item['estimated_effort']}")
            print()

        # Strategies
        print(f"\n🔧 REMEDIATION STRATEGIES:")
        for module, strategy in plan["strategies"].items():
            print(f"\n   📦 {module.upper()}:")
            print(f"      Strategy: {strategy['strategy']}")
            if "steps" in strategy:
                for step in strategy["steps"]:
                    print(f"      {step}")

        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Start with highest priority files")
        print(f"   2. Create infrastructure adapters first")
        print(f"   3. Define domain protocols for interfaces")
        print(f"   4. Update application layer to use dependency injection")
        print(f"   5. Run validation after each file to track progress")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and fix domain layer purity violations"
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("src"),
        help="Path to source code root directory",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for remediation plan (JSON format)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed remediation examples"
    )

    args = parser.parse_args()

    if not args.src_root.exists():
        print(f"❌ Source root not found: {args.src_root}")
        return 1

    fixer = DomainPurityFixer(args.src_root)
    violations = fixer.analyze_violations()

    if not violations:
        print("✅ No domain purity violations found!")
        return 0

    plan = fixer.generate_remediation_plan()
    fixer.print_remediation_report(plan)

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\n📄 Detailed plan saved to: {args.output}")

    if args.detailed:
        print(f"\n" + "=" * 80)
        print("🔧 DETAILED REMEDIATION EXAMPLES")
        print("=" * 80)
        for module, strategy in plan["strategies"].items():
            if "example" in strategy:
                print(f"\n📦 {module.upper()} Example:")
                print(strategy["example"])

    return 0


if __name__ == "__main__":
    exit(main())
