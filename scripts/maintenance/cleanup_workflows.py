#!/usr/bin/env python3
"""
Script to clean up redundant workflow files after CI/CD simplification.
This script safely removes old workflow files and creates a backup.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


def get_workflow_files() -> list[Path]:
    """Get all workflow files in the .github/workflows directory."""
    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        return []

    return list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))


def analyze_workflows() -> dict[str, list[Path]]:
    """Analyze workflows and categorize them."""
    workflows = get_workflow_files()

    # Define the new unified workflows (keep these)
    unified_workflows = {"ci-unified.yml", "cd-unified.yml", "maintenance-unified.yml"}

    # Categorize workflows
    analysis = {"keep": [], "remove": [], "unified": []}

    for workflow in workflows:
        if workflow.name in unified_workflows:
            analysis["unified"].append(workflow)
        elif workflow.name in ["ci.yml", "cd.yml", "ci-cd.yml"]:
            # Keep main CI/CD workflows temporarily for comparison
            analysis["keep"].append(workflow)
        else:
            # Remove all other workflows
            analysis["remove"].append(workflow)

    return analysis


def create_backup(workflows_to_remove: list[Path]) -> str:
    """Create a backup of workflows that will be removed."""
    backup_dir = Path("backups/workflows") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for workflow in workflows_to_remove:
        shutil.copy2(workflow, backup_dir / workflow.name)

    return str(backup_dir)


def generate_removal_report(analysis: dict[str, list[Path]], backup_dir: str) -> str:
    """Generate a report of the workflow cleanup."""
    report = f"""# Workflow Cleanup Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Backup Location:** {backup_dir}

## Summary

- **Total Workflows Before:** {len(analysis['keep']) + len(analysis['remove']) + len(analysis['unified'])}
- **Unified Workflows Created:** {len(analysis['unified'])}
- **Workflows Removed:** {len(analysis['remove'])}
- **Workflows Kept:** {len(analysis['keep'])}

## Unified Workflows (Created)

"""

    for workflow in analysis["unified"]:
        report += f"- ✅ `{workflow.name}` - New unified workflow\n"

    report += "\n## Workflows Removed\n\n"

    for workflow in analysis["remove"]:
        report += f"- 🗑️ `{workflow.name}` - Redundant, functionality moved to unified workflows\n"

    report += "\n## Workflows Kept (Temporary)\n\n"

    for workflow in analysis["keep"]:
        report += f"- 📋 `{workflow.name}` - Kept for comparison, can be removed after validation\n"

    report += f"""

## Benefits of Simplification

1. **Reduced Complexity:** From {len(analysis['keep']) + len(analysis['remove']) + len(analysis['unified'])} workflows to 3 unified workflows
2. **Better Maintainability:** Centralized CI/CD logic in fewer files
3. **Improved Performance:** Optimized caching and parallel execution
4. **Clearer Responsibilities:** Separate workflows for CI, CD, and maintenance
5. **Reduced Duplication:** Eliminated redundant configurations

## Next Steps

1. Test the unified workflows thoroughly
2. Remove the temporary workflows after validation
3. Update documentation to reflect the new structure
4. Train team members on the new simplified workflows

## Rollback Instructions

If needed, workflows can be restored from: `{backup_dir}`

```bash
# To restore a specific workflow
cp {backup_dir}/[workflow-name].yml .github/workflows/

# To restore all workflows
cp {backup_dir}/*.yml .github/workflows/
```

## Workflow Migration Mapping

| Old Workflow | New Workflow | Notes |
|-------------|-------------|--------|
| ci.yml | ci-unified.yml | Consolidated all CI tasks |
| cd.yml | cd-unified.yml | Unified deployment pipeline |
| ci-cd.yml | ci-unified.yml + cd-unified.yml | Split into separate CI and CD |
| test.yml | ci-unified.yml | Integrated into unified testing |
| build.yml | ci-unified.yml | Integrated into unified build |
| security.yml | ci-unified.yml + maintenance-unified.yml | Split between CI and maintenance |
| automated-test-coverage-analysis.yml | ci-unified.yml | Integrated into unified testing |
| buck2-enhanced-ci.yml | ci-unified.yml | Functionality consolidated |
| buck2-incremental-testing.yml | ci-unified.yml | Functionality consolidated |
| build-matrix.yml | ci-unified.yml | Matrix strategy simplified |
| comprehensive-testing.yml | ci-unified.yml | Integrated into unified testing |
| multi-python-testing.yml | ci-unified.yml | Matrix strategy in unified workflow |
| deploy.yml | cd-unified.yml | Unified deployment pipeline |
| production-deployment.yml | cd-unified.yml | Integrated into unified CD |
| container-security-c004.yml | ci-unified.yml | Security scan in unified CI |
| dependency-update-bot.yml | maintenance-unified.yml | Moved to maintenance |
| mutation-testing.yml | ci-unified.yml | Integrated into unified testing |
| quality-gates.yml | ci-unified.yml | Quality checks in unified CI |
| security-scan.yml | ci-unified.yml + maintenance-unified.yml | Split appropriately |
| ui-testing-ci.yml | ci-unified.yml | UI tests in unified CI |
| validation-suite.yml | ci-unified.yml | Validation in unified CI |
| branch-stash-cleanup.yml | maintenance-unified.yml | Moved to maintenance |
| changelog-check.yml | ci-unified.yml | Integrated into unified CI |
| complexity-monitoring.yml | maintenance-unified.yml | Moved to maintenance |
| file-organization.yml | maintenance-unified.yml | Moved to maintenance |
| maintenance.yml | maintenance-unified.yml | Consolidated maintenance |
| project-organization.yml | maintenance-unified.yml | Moved to maintenance |
| adr-toc.yml | maintenance-unified.yml | Moved to maintenance |
| pr-validation.yml | ci-unified.yml | PR validation in unified CI |
| release.yml | cd-unified.yml | Release process in unified CD |
| smart-test-selection.yml | ci-unified.yml | Test optimization in unified CI |
| ui-tests.yml | ci-unified.yml | UI tests in unified CI |
| ui_testing.yml | ci-unified.yml | UI tests in unified CI |
"""

    return report


def main():
    """Main function to clean up workflows."""
    print("🔧 Starting workflow cleanup process...")

    # Analyze workflows
    analysis = analyze_workflows()

    print("📊 Analysis complete:")
    print(f"  - Unified workflows: {len(analysis['unified'])}")
    print(f"  - Workflows to remove: {len(analysis['remove'])}")
    print(f"  - Workflows to keep: {len(analysis['keep'])}")

    if not analysis["remove"]:
        print("✅ No workflows to remove.")
        return

    # Create backup
    print("💾 Creating backup of workflows to be removed...")
    backup_dir = create_backup(analysis["remove"])
    print(f"✅ Backup created at: {backup_dir}")

    # Generate report
    print("📋 Generating cleanup report...")
    report = generate_removal_report(analysis, backup_dir)

    report_path = Path("reports/workflow-cleanup-report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    print(f"✅ Cleanup report generated: {report_path}")

    # Show what would be removed (dry run)
    print("\n🗑️ Workflows that would be removed:")
    for workflow in analysis["remove"]:
        print(f"  - {workflow.name}")

    # Ask for confirmation
    response = input(
        "\n❓ Do you want to proceed with removing these workflows? (y/N): "
    )

    if response.lower() in ["y", "yes"]:
        print("\n🗑️ Removing redundant workflows...")

        for workflow in analysis["remove"]:
            try:
                workflow.unlink()
                print(f"  ✅ Removed: {workflow.name}")
            except Exception as e:
                print(f"  ❌ Failed to remove {workflow.name}: {e}")

        print("\n🎉 Workflow cleanup completed!")
        print(
            f"📊 Reduced from {len(analysis['keep']) + len(analysis['remove']) + len(analysis['unified'])} to {len(analysis['unified']) + len(analysis['keep'])} workflows"
        )
        print(f"📋 Full report available at: {report_path}")
        print(f"💾 Backup available at: {backup_dir}")

        # Create summary JSON for automation
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_before": len(analysis["keep"])
            + len(analysis["remove"])
            + len(analysis["unified"]),
            "total_after": len(analysis["unified"]) + len(analysis["keep"]),
            "removed_count": len(analysis["remove"]),
            "unified_count": len(analysis["unified"]),
            "backup_location": backup_dir,
            "report_location": str(report_path),
        }

        summary_path = Path("reports/workflow-cleanup-summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"📄 Summary JSON: {summary_path}")

    else:
        print("\n❌ Workflow cleanup cancelled.")
        print("💾 Backup preserved for future use.")


if __name__ == "__main__":
    main()
