#!/usr/bin/env python3
"""
Test script for GitHub Issues sync automation system.
This verifies that all components work correctly without requiring GitHub API access.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock data for testing
MOCK_ISSUES = [
    {
        "number": 119,
        "title": "Web UI Performance Optimization",
        "state": "open",
        "created_at": "2025-07-10T10:00:00Z",
        "updated_at": "2025-07-11T14:30:00Z",
        "body": "Optimize Core Web Vitals, implement lazy loading, and reduce bundle size.",
        "labels": [
            {"name": "P1-High"},
            {"name": "presentation"},
            {"name": "enhancement"},
        ],
    },
    {
        "number": 122,
        "title": "Advanced Web UI Testing Infrastructure",
        "state": "closed",
        "created_at": "2025-07-09T09:00:00Z",
        "updated_at": "2025-07-10T16:45:00Z",
        "body": "Enhanced Playwright testing, visual regression, accessibility automation.",
        "labels": [
            {"name": "P1-High"},
            {"name": "presentation"},
            {"name": "enhancement"},
            {"name": "completed"},
        ],
    },
    {
        "number": 118,
        "title": "Fix CLI Help Formatting Issues",
        "state": "open",
        "created_at": "2025-07-08T11:00:00Z",
        "updated_at": "2025-07-08T11:00:00Z",
        "body": "Improve CLI help system formatting and ensure consistent display.",
        "labels": [
            {"name": "P2-Medium"},
            {"name": "enhancement"},
            {"name": "in-progress"},
        ],
    },
]


def test_component_imports():
    """Test that all automation components can be imported."""
    print("🧪 Testing component imports...")

    try:
        from scripts.automation.sync_github_issues_to_todo import GitHubIssuesSync

        print("  ✅ GitHubIssuesSync class imported successfully")

        # Test class instantiation (will fail without env vars, but import works)
        try:
            os.environ["GITHUB_TOKEN"] = "test"
            os.environ["GITHUB_REPOSITORY"] = "test/test"
            GitHubIssuesSync()  # Test instantiation
            print("  ✅ GitHubIssuesSync instantiated successfully")
        except Exception as e:
            print(
                f"  ⚠️  GitHubIssuesSync instantiation failed (expected): {e}"
            )
        finally:
            # Clean up test env vars
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GITHUB_REPOSITORY", None)

    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

    return True


def test_priority_mapping():
    """Test priority mapping functionality."""
    print("🧪 Testing priority mapping...")

    try:
        from scripts.automation.sync_github_issues_to_todo import GitHubIssuesSync

        # Create a dummy instance for testing
        os.environ["GITHUB_TOKEN"] = "test"
        os.environ["GITHUB_REPOSITORY"] = "test/test"
        syncer = GitHubIssuesSync()

        # Test priority mapping
        test_cases = [
            ([{"name": "P1-High"}], "high"),
            ([{"name": "P2-Medium"}], "medium"),
            ([{"name": "P3-Low"}], "low"),
            ([{"name": "enhancement"}], "medium"),  # Default
            ([], "medium"),  # Default
        ]

        for labels, expected in test_cases:
            result = syncer.get_priority_from_labels(labels)
            if result == expected:
                print(
                    f"  ✅ Priority mapping correct: {[label['name'] for label in labels]} → {result}"
                )
            else:
                print(
                    f"  ❌ Priority mapping failed: {[l['name'] for l in labels]} → {result}, expected {expected}"
                )
                return False

        # Clean up
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GITHUB_REPOSITORY", None)

    except Exception as e:
        print(f"  ❌ Priority mapping test failed: {e}")
        return False

    return True


def test_status_detection():
    """Test status detection functionality."""
    print("🧪 Testing status detection...")

    try:
        from scripts.automation.sync_github_issues_to_todo import GitHubIssuesSync

        # Create a dummy instance for testing
        os.environ["GITHUB_TOKEN"] = "test"
        os.environ["GITHUB_REPOSITORY"] = "test/test"
        syncer = GitHubIssuesSync()

        # Test status detection
        test_cases = [
            ({"state": "closed", "labels": []}, "completed"),
            ({"state": "open", "labels": [{"name": "in-progress"}]}, "in_progress"),
            ({"state": "open", "labels": [{"name": "blocked"}]}, "blocked"),
            ({"state": "open", "labels": [{"name": "enhancement"}]}, "pending"),
        ]

        for issue_data, expected in test_cases:
            result = syncer.get_status_from_issue(issue_data)
            if result == expected:
                print(
                    f"  ✅ Status detection correct: {issue_data['state']} + labels → {result}"
                )
            else:
                print(
                    f"  ❌ Status detection failed: {issue_data} → {result}, expected {expected}"
                )
                return False

        # Clean up
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GITHUB_REPOSITORY", None)

    except Exception as e:
        print(f"  ❌ Status detection test failed: {e}")
        return False

    return True


def test_issue_formatting():
    """Test issue formatting functionality."""
    print("🧪 Testing issue formatting...")

    try:
        from scripts.automation.sync_github_issues_to_todo import GitHubIssuesSync

        # Create a dummy instance for testing
        os.environ["GITHUB_TOKEN"] = "test"
        os.environ["GITHUB_REPOSITORY"] = "test/test"
        syncer = GitHubIssuesSync()

        # Test with mock issue
        formatted = syncer.format_issue_for_todo(MOCK_ISSUES[0])

        # Check that formatted output contains expected elements
        expected_elements = [
            "#### **Issue #119: Web UI Performance Optimization**",
            "P1-High",
            "⏳ PENDING",
            "presentation, enhancement",
            "Jul 10, 2025",
            "https://github.com/test/test/issues/119",
        ]

        for element in expected_elements:
            if element in formatted:
                print(f"  ✅ Format contains: {element}")
            else:
                print(f"  ❌ Format missing: {element}")
                print(f"  🔍 Actual output:\n{formatted}")
                return False

        # Clean up
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GITHUB_REPOSITORY", None)

    except Exception as e:
        print(f"  ❌ Issue formatting test failed: {e}")
        return False

    return True


def test_section_generation():
    """Test complete section generation."""
    print("🧪 Testing section generation...")

    try:
        from scripts.automation.sync_github_issues_to_todo import GitHubIssuesSync

        # Create a dummy instance for testing
        os.environ["GITHUB_TOKEN"] = "test"
        os.environ["GITHUB_REPOSITORY"] = "test/test"
        syncer = GitHubIssuesSync()

        # Generate section with mock data
        section = syncer.generate_issues_section(MOCK_ISSUES)

        # Check that section contains expected structure
        expected_elements = [
            "## 📋 **GitHub Issues** (Auto-Synchronized)",
            "**Total Open Issues**: 3",
            "**Last Sync**:",
            "### 🔥 **P1-High Priority Issues**",
            "### 🔶 **P2-Medium Priority Issues**",
            "Issue #119: Web UI Performance Optimization",
            "Issue #122: Advanced Web UI Testing Infrastructure",
            "Issue #118: Fix CLI Help Formatting Issues",
        ]

        for element in expected_elements:
            if element in section:
                print(f"  ✅ Section contains: {element}")
            else:
                print(f"  ❌ Section missing: {element}")
                return False

        # Clean up
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GITHUB_REPOSITORY", None)

    except Exception as e:
        print(f"  ❌ Section generation test failed: {e}")
        return False

    return True


def test_file_operations():
    """Test file reading and writing operations."""
    print("🧪 Testing file operations...")

    # Test that TODO.md exists and is readable
    todo_path = project_root / "TODO.md"
    if not todo_path.exists():
        print("  ❌ TODO.md not found")
        return False

    try:
        with open(todo_path, encoding="utf-8") as f:
            content = f.read()
        print("  ✅ TODO.md readable")

        # Check for automation section
        if "## 📋 **GitHub Issues** (Auto-Synchronized)" in content:
            print("  ✅ Automation section found in TODO.md")
        else:
            print("  ⚠️  Automation section not found in TODO.md")

    except Exception as e:
        print(f"  ❌ Failed to read TODO.md: {e}")
        return False

    return True


def test_workflow_file():
    """Test that the GitHub Actions workflow file exists."""
    print("🧪 Testing workflow file...")

    workflow_path = project_root / ".github" / "workflows" / "issue-sync.yml"
    if not workflow_path.exists():
        print("  ❌ Workflow file not found")
        return False

    try:
        with open(workflow_path, encoding="utf-8") as f:
            content = f.read()

        # Check for key workflow elements
        required_elements = [
            "name: GitHub Issues Sync to TODO.md",
            "on:",
            "issues:",
            "scripts/automation/sync_github_issues_to_todo.py",
        ]

        for element in required_elements:
            if element in content:
                print(f"  ✅ Workflow contains: {element}")
            else:
                print(f"  ❌ Workflow missing: {element}")
                return False

    except Exception as e:
        print(f"  ❌ Failed to read workflow file: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("🚀 Starting GitHub Issues sync automation tests...\n")

    tests = [
        ("Component Imports", test_component_imports),
        ("Priority Mapping", test_priority_mapping),
        ("Status Detection", test_status_detection),
        ("Issue Formatting", test_issue_formatting),
        ("Section Generation", test_section_generation),
        ("File Operations", test_file_operations),
        ("Workflow File", test_workflow_file),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
            failed += 1

    print("\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed! The automation system is ready to use.")
        return True
    else:
        print(
            f"\n⚠️  {failed} test(s) failed. Please review and fix issues before deploying."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
