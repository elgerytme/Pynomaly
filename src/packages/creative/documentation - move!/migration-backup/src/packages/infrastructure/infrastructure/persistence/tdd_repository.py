"""Test-Driven Development repository for tracking test requirements and compliance."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from monorepo.infrastructure.config.tdd_config import (
    TDDComplianceReport,
    TDDViolation,
    TestRequirement,
)
from monorepo.shared.protocols.repository_protocol import RepositoryProtocol


class TestDrivenDevelopmentRepositoryProtocol(RepositoryProtocol[TestRequirement]):
    """Protocol for TDD repository implementations."""

    def create_test_requirement(
        self,
        module_path: str,
        function_name: str,
        description: str,
        test_specification: str,
        coverage_target: float = 0.8,
        tags: set[str] | None = None,
    ) -> TestRequirement:
        """Create a new test requirement.

        Args:
            module_path: Path to the module requiring tests
            function_name: Name of the function/method to test
            description: Description of what should be tested
            test_specification: Detailed test specification
            coverage_target: Target coverage percentage
            tags: Optional tags for categorization

        Returns:
            Created test requirement
        """
        ...

    def find_requirements_by_module(self, module_path: str) -> list[TestRequirement]:
        """Find all test requirements for a specific module.

        Args:
            module_path: Path to the module

        Returns:
            List of test requirements for the module
        """
        ...

    def find_requirements_by_status(self, status: str) -> list[TestRequirement]:
        """Find test requirements by status.

        Args:
            status: Status to filter by (pending, implemented, validated)

        Returns:
            List of test requirements with the specified status
        """
        ...

    def find_requirements_by_tags(self, tags: set[str]) -> list[TestRequirement]:
        """Find test requirements by tags.

        Args:
            tags: Tags to filter by

        Returns:
            List of test requirements containing any of the specified tags
        """
        ...

    def mark_implementation_complete(
        self,
        requirement_id: str,
        implementation_path: str,
        test_file_path: str | None = None,
    ) -> None:
        """Mark a test requirement as implemented.

        Args:
            requirement_id: ID of the test requirement
            implementation_path: Path to the implementation file
            test_file_path: Path to the test file (optional)
        """
        ...

    def mark_validation_complete(self, requirement_id: str) -> None:
        """Mark a test requirement as validated.

        Args:
            requirement_id: ID of the test requirement
        """
        ...

    def update_coverage_target(
        self, requirement_id: str, coverage_target: float
    ) -> None:
        """Update the coverage target for a test requirement.

        Args:
            requirement_id: ID of the test requirement
            coverage_target: New coverage target
        """
        ...

    def get_compliance_report(self) -> TDDComplianceReport:
        """Generate a TDD compliance report.

        Returns:
            Comprehensive compliance report
        """
        ...

    def save_violation(self, violation: TDDViolation) -> None:
        """Save a TDD violation.

        Args:
            violation: TDD violation to save
        """
        ...

    def find_violations_by_file(self, file_path: str) -> list[TDDViolation]:
        """Find violations for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of violations for the file
        """
        ...

    def clear_violations_for_file(self, file_path: str) -> None:
        """Clear all violations for a specific file.

        Args:
            file_path: Path to the file
        """
        ...


class InMemoryTDDRepository(TestDrivenDevelopmentRepositoryProtocol):
    """In-memory TDD repository for testing and development."""

    def __init__(self):
        self._requirements: dict[str, TestRequirement] = {}
        self._violations: list[TDDViolation] = []
        self._coverage_data: dict[str, float] = {}

    def save(self, entity: TestRequirement) -> None:
        """Save a test requirement."""
        self._requirements[entity.id] = entity

    def find_by_id(self, entity_id: UUID) -> TestRequirement | None:
        """Find test requirement by ID."""
        return self._requirements.get(str(entity_id))

    def find_all(self) -> list[TestRequirement]:
        """Find all test requirements."""
        return list(self._requirements.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete test requirement by ID."""
        requirement_id = str(entity_id)
        if requirement_id in self._requirements:
            del self._requirements[requirement_id]
            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if test requirement exists."""
        return str(entity_id) in self._requirements

    def count(self) -> int:
        """Count total test requirements."""
        return len(self._requirements)

    def create_test_requirement(
        self,
        module_path: str,
        function_name: str,
        description: str,
        test_specification: str,
        coverage_target: float = 0.8,
        tags: set[str] | None = None,
    ) -> TestRequirement:
        """Create a new test requirement."""
        requirement = TestRequirement(
            id=str(uuid4()),
            module_path=module_path,
            function_name=function_name,
            description=description,
            test_specification=test_specification,
            created_at=datetime.now().isoformat(),
            status="pending",
            coverage_target=coverage_target,
            tags=tags or set(),
        )
        self.save(requirement)
        return requirement

    def find_requirements_by_module(self, module_path: str) -> list[TestRequirement]:
        """Find all test requirements for a specific module."""
        return [
            req for req in self._requirements.values() if req.module_path == module_path
        ]

    def find_requirements_by_status(self, status: str) -> list[TestRequirement]:
        """Find test requirements by status."""
        return [req for req in self._requirements.values() if req.status == status]

    def find_requirements_by_tags(self, tags: set[str]) -> list[TestRequirement]:
        """Find test requirements by tags."""
        return [
            req for req in self._requirements.values() if tags.intersection(req.tags)
        ]

    def mark_implementation_complete(
        self,
        requirement_id: str,
        implementation_path: str,
        test_file_path: str | None = None,
    ) -> None:
        """Mark a test requirement as implemented."""
        if requirement_id in self._requirements:
            requirement = self._requirements[requirement_id]
            requirement.status = "implemented"
            requirement.implementation_path = implementation_path
            if test_file_path:
                requirement.test_file_path = test_file_path

    def mark_validation_complete(self, requirement_id: str) -> None:
        """Mark a test requirement as validated."""
        if requirement_id in self._requirements:
            self._requirements[requirement_id].status = "validated"

    def update_coverage_target(
        self, requirement_id: str, coverage_target: float
    ) -> None:
        """Update the coverage target for a test requirement."""
        if requirement_id in self._requirements:
            self._requirements[requirement_id].coverage_target = coverage_target

    def get_compliance_report(self) -> TDDComplianceReport:
        """Generate a TDD compliance report."""
        requirements = list(self._requirements.values())
        total_requirements = len(requirements)

        if total_requirements == 0:
            return TDDComplianceReport(
                total_requirements=0,
                pending_requirements=0,
                implemented_requirements=0,
                validated_requirements=0,
                overall_compliance=1.0,
                module_compliance={},
                violations=[violation.description for violation in self._violations],
                coverage_report=self._coverage_data.copy(),
                last_updated=datetime.now().isoformat(),
            )

        pending_requirements = len([r for r in requirements if r.status == "pending"])
        implemented_requirements = len(
            [r for r in requirements if r.status == "implemented"]
        )
        validated_requirements = len(
            [r for r in requirements if r.status == "validated"]
        )

        # Calculate module-specific compliance
        module_compliance = {}
        modules = {req.module_path for req in requirements}

        for module in modules:
            module_reqs = [r for r in requirements if r.module_path == module]
            module_validated = len([r for r in module_reqs if r.status == "validated"])
            module_compliance[module] = (
                module_validated / len(module_reqs) if module_reqs else 0.0
            )

        # Overall compliance
        overall_compliance = validated_requirements / total_requirements

        return TDDComplianceReport(
            total_requirements=total_requirements,
            pending_requirements=pending_requirements,
            implemented_requirements=implemented_requirements,
            validated_requirements=validated_requirements,
            overall_compliance=overall_compliance,
            module_compliance=module_compliance,
            violations=[violation.description for violation in self._violations],
            coverage_report=self._coverage_data.copy(),
            last_updated=datetime.now().isoformat(),
        )

    def save_violation(self, violation: TDDViolation) -> None:
        """Save a TDD violation."""
        self._violations.append(violation)

    def find_violations_by_file(self, file_path: str) -> list[TDDViolation]:
        """Find violations for a specific file."""
        return [v for v in self._violations if v.file_path == file_path]

    def clear_violations_for_file(self, file_path: str) -> None:
        """Clear all violations for a specific file."""
        self._violations = [v for v in self._violations if v.file_path != file_path]

    def update_coverage_data(self, module_path: str, coverage: float) -> None:
        """Update coverage data for a module."""
        self._coverage_data[module_path] = coverage


class FileTDDRepository(TestDrivenDevelopmentRepositoryProtocol):
    """File-based TDD repository for persistent storage."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.requirements_file = storage_path / "test_requirements.json"
        self.violations_file = storage_path / "tdd_violations.json"
        self.coverage_file = storage_path / "coverage_data.json"

        # Ensure storage directory exists
        storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize files if they don't exist
        for file_path in [
            self.requirements_file,
            self.violations_file,
            self.coverage_file,
        ]:
            if not file_path.exists():
                with open(file_path, "w") as f:
                    json.dump({} if file_path != self.violations_file else [], f)

    def _load_requirements(self) -> dict[str, TestRequirement]:
        """Load test requirements from file."""
        try:
            with open(self.requirements_file) as f:
                data = json.load(f)
                return {
                    req_id: TestRequirement(**req_data)
                    for req_id, req_data in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_requirements(self, requirements: dict[str, TestRequirement]) -> None:
        """Save test requirements to file."""
        data = {
            req_id: {
                "id": req.id,
                "module_path": req.module_path,
                "function_name": req.function_name,
                "description": req.description,
                "test_specification": req.test_specification,
                "created_at": req.created_at,
                "status": req.status,
                "implementation_path": req.implementation_path,
                "test_file_path": req.test_file_path,
                "coverage_target": req.coverage_target,
                "tags": list(req.tags),
            }
            for req_id, req in requirements.items()
        }

        with open(self.requirements_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_violations(self) -> list[TDDViolation]:
        """Load violations from file."""
        try:
            with open(self.violations_file) as f:
                data = json.load(f)
                return [TDDViolation(**violation) for violation in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_violations(self, violations: list[TDDViolation]) -> None:
        """Save violations to file."""
        data = [
            {
                "violation_type": v.violation_type,
                "file_path": v.file_path,
                "line_number": v.line_number,
                "description": v.description,
                "severity": v.severity,
                "rule_name": v.rule_name,
                "suggestion": v.suggestion,
                "auto_fixable": v.auto_fixable,
            }
            for v in violations
        ]

        with open(self.violations_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_coverage_data(self) -> dict[str, float]:
        """Load coverage data from file."""
        try:
            with open(self.coverage_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_coverage_data(self, coverage_data: dict[str, float]) -> None:
        """Save coverage data to file."""
        with open(self.coverage_file, "w") as f:
            json.dump(coverage_data, f, indent=2)

    def save(self, entity: TestRequirement) -> None:
        """Save a test requirement."""
        requirements = self._load_requirements()
        requirements[entity.id] = entity
        self._save_requirements(requirements)

    def find_by_id(self, entity_id: UUID) -> TestRequirement | None:
        """Find test requirement by ID."""
        requirements = self._load_requirements()
        return requirements.get(str(entity_id))

    def find_all(self) -> list[TestRequirement]:
        """Find all test requirements."""
        requirements = self._load_requirements()
        return list(requirements.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete test requirement by ID."""
        requirements = self._load_requirements()
        requirement_id = str(entity_id)

        if requirement_id in requirements:
            del requirements[requirement_id]
            self._save_requirements(requirements)
            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if test requirement exists."""
        requirements = self._load_requirements()
        return str(entity_id) in requirements

    def count(self) -> int:
        """Count total test requirements."""
        requirements = self._load_requirements()
        return len(requirements)

    def create_test_requirement(
        self,
        module_path: str,
        function_name: str,
        description: str,
        test_specification: str,
        coverage_target: float = 0.8,
        tags: set[str] | None = None,
    ) -> TestRequirement:
        """Create a new test requirement."""
        requirement = TestRequirement(
            id=str(uuid4()),
            module_path=module_path,
            function_name=function_name,
            description=description,
            test_specification=test_specification,
            created_at=datetime.now().isoformat(),
            status="pending",
            coverage_target=coverage_target,
            tags=tags or set(),
        )
        self.save(requirement)
        return requirement

    def find_requirements_by_module(self, module_path: str) -> list[TestRequirement]:
        """Find all test requirements for a specific module."""
        requirements = self._load_requirements()
        return [req for req in requirements.values() if req.module_path == module_path]

    def find_requirements_by_status(self, status: str) -> list[TestRequirement]:
        """Find test requirements by status."""
        requirements = self._load_requirements()
        return [req for req in requirements.values() if req.status == status]

    def find_requirements_by_tags(self, tags: set[str]) -> list[TestRequirement]:
        """Find test requirements by tags."""
        requirements = self._load_requirements()
        return [req for req in requirements.values() if tags.intersection(req.tags)]

    def mark_implementation_complete(
        self,
        requirement_id: str,
        implementation_path: str,
        test_file_path: str | None = None,
    ) -> None:
        """Mark a test requirement as implemented."""
        requirements = self._load_requirements()

        if requirement_id in requirements:
            requirement = requirements[requirement_id]
            requirement.status = "implemented"
            requirement.implementation_path = implementation_path
            if test_file_path:
                requirement.test_file_path = test_file_path
            self._save_requirements(requirements)

    def mark_validation_complete(self, requirement_id: str) -> None:
        """Mark a test requirement as validated."""
        requirements = self._load_requirements()

        if requirement_id in requirements:
            requirements[requirement_id].status = "validated"
            self._save_requirements(requirements)

    def update_coverage_target(
        self, requirement_id: str, coverage_target: float
    ) -> None:
        """Update the coverage target for a test requirement."""
        requirements = self._load_requirements()

        if requirement_id in requirements:
            requirements[requirement_id].coverage_target = coverage_target
            self._save_requirements(requirements)

    def get_compliance_report(self) -> TDDComplianceReport:
        """Generate a TDD compliance report."""
        requirements = list(self._load_requirements().values())
        violations = self._load_violations()
        coverage_data = self._load_coverage_data()

        total_requirements = len(requirements)

        if total_requirements == 0:
            return TDDComplianceReport(
                total_requirements=0,
                pending_requirements=0,
                implemented_requirements=0,
                validated_requirements=0,
                overall_compliance=1.0,
                module_compliance={},
                violations=[violation.description for violation in violations],
                coverage_report=coverage_data,
                last_updated=datetime.now().isoformat(),
            )

        pending_requirements = len([r for r in requirements if r.status == "pending"])
        implemented_requirements = len(
            [r for r in requirements if r.status == "implemented"]
        )
        validated_requirements = len(
            [r for r in requirements if r.status == "validated"]
        )

        # Calculate module-specific compliance
        module_compliance = {}
        modules = {req.module_path for req in requirements}

        for module in modules:
            module_reqs = [r for r in requirements if r.module_path == module]
            module_validated = len([r for r in module_reqs if r.status == "validated"])
            module_compliance[module] = (
                module_validated / len(module_reqs) if module_reqs else 0.0
            )

        # Overall compliance
        overall_compliance = validated_requirements / total_requirements

        return TDDComplianceReport(
            total_requirements=total_requirements,
            pending_requirements=pending_requirements,
            implemented_requirements=implemented_requirements,
            validated_requirements=validated_requirements,
            overall_compliance=overall_compliance,
            module_compliance=module_compliance,
            violations=[violation.description for violation in violations],
            coverage_report=coverage_data,
            last_updated=datetime.now().isoformat(),
        )

    def save_violation(self, violation: TDDViolation) -> None:
        """Save a TDD violation."""
        violations = self._load_violations()
        violations.append(violation)
        self._save_violations(violations)

    def find_violations_by_file(self, file_path: str) -> list[TDDViolation]:
        """Find violations for a specific file."""
        violations = self._load_violations()
        return [v for v in violations if v.file_path == file_path]

    def clear_violations_for_file(self, file_path: str) -> None:
        """Clear all violations for a specific file."""
        violations = self._load_violations()
        violations = [v for v in violations if v.file_path != file_path]
        self._save_violations(violations)

    def update_coverage_data(self, module_path: str, coverage: float) -> None:
        """Update coverage data for a module."""
        coverage_data = self._load_coverage_data()
        coverage_data[module_path] = coverage
        self._save_coverage_data(coverage_data)
