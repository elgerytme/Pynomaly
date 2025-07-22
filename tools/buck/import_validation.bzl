"""
Buck2 Import Validation Rules
============================

Custom Buck2 rules for validating import consolidation during build process.
These rules integrate with the monorepo's existing Buck2 build system to enforce
the single import per package rule at build time.
"""

def _import_validation_impl(ctx):
    """Implementation for import validation rule"""
    
    # Get the Python source files to validate
    python_sources = []
    for dep in ctx.attr.deps:
        if hasattr(dep, "py_info") and dep.py_info.sources:
            python_sources.extend(dep.py_info.sources.to_list())
    
    # Create validation script
    validation_script = ctx.actions.declare_file("{}_import_validation.sh".format(ctx.label.name))
    
    # Get validator script path
    validator_path = "scripts/import_consolidation_validator.py"
    
    # Create the validation command
    validation_cmd = """#!/bin/bash
set -e

echo "🔍 Running import consolidation validation for {target}..."

# Change to repository root
cd "{repo_root}"

# Create list of files to validate
FILES_TO_VALIDATE=()
{file_list}

# Run import consolidation validation
if python3 {validator_path} --changed-files "${{FILES_TO_VALIDATE[@]}}" --fail-on-violations; then
    echo "✅ Import consolidation validation passed for {target}"
    touch {success_marker}
else
    echo "❌ Import consolidation validation failed for {target}"
    echo "To fix violations, run:"
    echo "  python3 scripts/import_consolidation_refactor.py --files ${{FILES_TO_VALIDATE[@]}}"
    exit 1
fi
""".format(
        target = ctx.label,
        repo_root = ctx.label.package,
        validator_path = validator_path,
        success_marker = validation_script.path + ".success",
        file_list = "\n".join([
            'FILES_TO_VALIDATE+=("{}")'.format(src.path) 
            for src in python_sources
        ])
    )
    
    ctx.actions.write(
        output = validation_script,
        content = validation_cmd,
        is_executable = True
    )
    
    # Create success marker file
    success_marker = ctx.actions.declare_file("{}_validation_success".format(ctx.label.name))
    
    # Run the validation
    ctx.actions.run(
        inputs = python_sources + [validation_script],
        outputs = [success_marker],
        executable = validation_script,
        arguments = [],
        mnemonic = "ImportValidation",
        progress_message = "Validating import consolidation for %{label}"
    )
    
    return [DefaultInfo(files = depset([success_marker]))]

# Define the import validation rule
import_validation_test = rule(
    implementation = _import_validation_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [PyInfo],
            doc = "Python targets to validate for import consolidation"
        ),
    },
    doc = "Validates import consolidation for Python targets"
)

def validate_package_imports(name, package_targets, **kwargs):
    """
    Convenience macro to create import validation tests for a package.
    
    Args:
        name: Name for the validation test target
        package_targets: List of Python targets to validate
        **kwargs: Additional arguments passed to the rule
    """
    
    import_validation_test(
        name = name,
        deps = package_targets,
        **kwargs
    )

def _import_consolidation_fix_impl(ctx):
    """Implementation for import consolidation fix rule"""
    
    # Get the Python source files to fix
    python_sources = []
    for dep in ctx.attr.deps:
        if hasattr(dep, "py_info") and dep.py_info.sources:
            python_sources.extend(dep.py_info.sources.to_list())
    
    # Create fix script
    fix_script = ctx.actions.declare_file("{}_import_fix.sh".format(ctx.label.name))
    
    # Get refactor script path
    refactor_path = "scripts/import_consolidation_refactor.py"
    
    # Create the fix command
    fix_cmd = """#!/bin/bash
set -e

echo "🔧 Running import consolidation fixes for {target}..."

# Change to repository root
cd "{repo_root}"

# Create list of files to fix
FILES_TO_FIX=()
{file_list}

# Run import consolidation refactoring
if python3 {refactor_path} --files "${{FILES_TO_FIX[@]}}"; then
    echo "✅ Import consolidation fixes applied for {target}"
    echo "📝 Modified files have been updated"
    touch {success_marker}
else
    echo "❌ Import consolidation fixes failed for {target}"
    exit 1
fi
""".format(
        target = ctx.label,
        repo_root = ctx.label.package,
        refactor_path = refactor_path,
        success_marker = fix_script.path + ".success",
        file_list = "\n".join([
            'FILES_TO_FIX+=("{}")'.format(src.path) 
            for src in python_sources
        ])
    )
    
    ctx.actions.write(
        output = fix_script,
        content = fix_cmd,
        is_executable = True
    )
    
    # Create success marker file
    success_marker = ctx.actions.declare_file("{}_fix_success".format(ctx.label.name))
    
    return [DefaultInfo(files = depset([fix_script, success_marker]))]

# Define the import consolidation fix rule
import_consolidation_fix = rule(
    implementation = _import_consolidation_fix_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [PyInfo],
            doc = "Python targets to fix import consolidation issues"
        ),
    },
    executable = True,
    doc = "Fixes import consolidation issues in Python targets"
)

def create_import_validation_suite(name, packages):
    """
    Create a comprehensive import validation suite for multiple packages.
    
    Args:
        name: Base name for the validation suite
        packages: Dictionary mapping package names to their targets
    """
    
    validation_tests = []
    
    for package_name, package_targets in packages.items():
        test_name = "{}_{}".format(name, package_name.replace(".", "_"))
        
        # Create validation test for this package
        validate_package_imports(
            name = test_name,
            package_targets = package_targets,
            tags = ["import-validation", package_name]
        )
        
        validation_tests.append(":" + test_name)
    
    # Create test suite that runs all validation tests
    native.test_suite(
        name = name,
        tests = validation_tests,
        tags = ["import-validation-suite"]
    )

def create_import_fix_suite(name, packages):
    """
    Create a comprehensive import fix suite for multiple packages.
    
    Args:
        name: Base name for the fix suite
        packages: Dictionary mapping package names to their targets
    """
    
    fix_targets = []
    
    for package_name, package_targets in packages.items():
        fix_name = "{}_fix_{}".format(name, package_name.replace(".", "_"))
        
        # Create fix target for this package
        import_consolidation_fix(
            name = fix_name,
            deps = package_targets,
            tags = ["import-fix", package_name]
        )
        
        fix_targets.append(":" + fix_name)
    
    # Create filegroup that includes all fix targets
    native.filegroup(
        name = name,
        srcs = fix_targets,
        tags = ["import-fix-suite"]
    )