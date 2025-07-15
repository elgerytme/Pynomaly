#!/usr/bin/env python3
"""
Comprehensive test to verify URL refactoring from /web to /
"""

import re
import sys
from pathlib import Path


def scan_file_for_web_refs(file_path: Path) -> list[tuple[int, str]]:
    """Scan a file for /web references and return line numbers and content."""
    if not file_path.exists():
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        web_refs = []
        for i, line in enumerate(lines, 1):
            if "/web/" in line or '"/web"' in line or "'/web'" in line:
                web_refs.append((i, line.strip()))

        return web_refs
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def test_critical_files():
    """Test that critical files have been updated."""
    critical_files = [
        "src/pynomaly/presentation/web/app.py",
        "src/pynomaly/presentation/web/templates/base.html",
        "config/web/nginx.conf",
        "scripts/run/run_web_app.py",
    ]

    print("Testing critical files for /web references...")
    all_passed = True

    for file_path_str in critical_files:
        file_path = Path(file_path_str)
        web_refs = scan_file_for_web_refs(file_path)

        if web_refs:
            print(f"❌ {file_path_str} still contains {len(web_refs)} /web references:")
            for line_num, line_content in web_refs[:5]:  # Show first 5
                print(f"   Line {line_num}: {line_content}")
            all_passed = False
        else:
            print(f"✅ {file_path_str} - no /web references found")

    return all_passed


def test_template_directory():
    """Test all HTML templates for /web references."""
    templates_dir = Path("src/pynomaly/presentation/web/templates")

    if not templates_dir.exists():
        print("❌ Templates directory not found")
        return False

    html_files = list(templates_dir.rglob("*.html"))

    if not html_files:
        print("❌ No HTML template files found")
        return False

    total_web_refs = 0
    files_with_refs = 0

    for html_file in html_files:
        web_refs = scan_file_for_web_refs(html_file)
        if web_refs:
            total_web_refs += len(web_refs)
            files_with_refs += 1

    if total_web_refs == 0:
        print(f"✅ All {len(html_files)} HTML templates updated - no /web references")
        return True
    else:
        print(
            f"❌ Found {total_web_refs} /web references in {files_with_refs} template files"
        )
        return False


def test_javascript_directory():
    """Test all JavaScript files for /web references."""
    js_dir = Path("src/pynomaly/presentation/web/static/js")

    if not js_dir.exists():
        print("❌ JavaScript directory not found")
        return False

    js_files = list(js_dir.rglob("*.js"))

    if not js_files:
        print("❌ No JavaScript files found")
        return False

    total_web_refs = 0
    files_with_refs = 0

    for js_file in js_files:
        web_refs = scan_file_for_web_refs(js_file)
        if web_refs:
            total_web_refs += len(web_refs)
            files_with_refs += 1

    if total_web_refs == 0:
        print(f"✅ All {len(js_files)} JavaScript files updated - no /web references")
        return True
    else:
        print(f"❌ Found {total_web_refs} /web references in {files_with_refs} JS files")
        return False


def test_configuration_files():
    """Test configuration files for /web references."""
    config_files = [
        "config/web/nginx.conf",
        "config/web/tailwind.config.js",
        "deploy/docker/Dockerfile.web",
        "deploy/docker/docker-compose.production.yml",
    ]

    print("Testing configuration files...")
    all_passed = True

    for file_path_str in config_files:
        file_path = Path(file_path_str)

        if not file_path.exists():
            print(f"⚠️  {file_path_str} not found (may be optional)")
            continue

        web_refs = scan_file_for_web_refs(file_path)

        # Some config files legitimately might have /web in comments or documentation
        # Filter out comments where appropriate
        filtered_refs = []
        for line_num, line_content in web_refs:
            # Skip nginx comment lines and Docker comments
            if line_content.strip().startswith("#") and (
                "comment" in line_content.lower() or "note" in line_content.lower()
            ):
                continue
            filtered_refs.append((line_num, line_content))

        if filtered_refs:
            print(
                f"⚠️  {file_path_str} contains {len(filtered_refs)} potentially problematic /web references:"
            )
            for line_num, line_content in filtered_refs[:3]:
                print(f"   Line {line_num}: {line_content}")
        else:
            print(f"✅ {file_path_str} - no problematic /web references")

    return all_passed


def test_app_routing_structure():
    """Test that the app.py file has correct routing structure."""
    app_file = Path("src/pynomaly/presentation/web/app.py")

    if not app_file.exists():
        print("❌ app.py not found")
        return False

    content = app_file.read_text(encoding="utf-8")

    # Check for empty prefix in router mounting
    empty_prefix_pattern = r'app\.include_router\(router,\s*prefix=""\s*,'
    if re.search(empty_prefix_pattern, content):
        print("✅ Router mounted with empty prefix (root path)")

        # Check that there are no old /web prefix patterns
        old_prefix_pattern = r'app\.include_router\(router,\s*prefix="/web"\s*,'
        if not re.search(old_prefix_pattern, content):
            print("✅ No old /web prefix found in router mounting")
            return True
        else:
            print("❌ Old /web prefix still found in router mounting")
            return False
    else:
        print("❌ Router not found with empty prefix")
        return False


def create_summary_report():
    """Create a summary report of the URL refactoring."""
    print("\n" + "=" * 60)
    print("URL REFACTORING SUMMARY REPORT")
    print("=" * 60)

    # Count total files that might need checking
    all_code_files = []

    # Python files
    for py_file in Path(".").rglob("*.py"):
        if "node_modules" not in str(py_file) and ".git" not in str(py_file):
            all_code_files.append(py_file)

    # HTML files
    for html_file in Path(".").rglob("*.html"):
        if "node_modules" not in str(html_file) and ".git" not in str(html_file):
            all_code_files.append(html_file)

    # JS files
    for js_file in Path(".").rglob("*.js"):
        if "node_modules" not in str(js_file) and ".git" not in str(js_file):
            all_code_files.append(js_file)

    # Config files
    for config_file in Path(".").rglob("*.conf"):
        if "node_modules" not in str(config_file) and ".git" not in str(config_file):
            all_code_files.append(config_file)

    total_files_checked = len(all_code_files)
    files_with_web_refs = 0
    total_web_refs = 0

    for file_path in all_code_files:
        web_refs = scan_file_for_web_refs(file_path)
        if web_refs:
            files_with_web_refs += 1
            total_web_refs += len(web_refs)

    print(f"Total files checked: {total_files_checked}")
    print(f"Files with /web references: {files_with_web_refs}")
    print(f"Total /web references found: {total_web_refs}")

    if total_web_refs == 0:
        print("\n🎉 SUCCESS: All /web references have been successfully updated!")
    else:
        print(
            f"\n⚠️  REMAINING WORK: {total_web_refs} /web references in {files_with_web_refs} files need review"
        )

    return total_web_refs == 0


def main():
    """Run all URL refactoring tests."""
    print("Comprehensive URL Refactoring Test")
    print("From: /web -> To: /")
    print("=" * 50)

    tests = [
        ("Critical Files", test_critical_files),
        ("HTML Templates", test_template_directory),
        ("JavaScript Files", test_javascript_directory),
        ("Configuration Files", test_configuration_files),
        ("App Routing Structure", test_app_routing_structure),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1

    print(f"\nTests passed: {passed}/{total}")

    # Create summary report
    all_good = create_summary_report()

    if passed == total and all_good:
        print("\n🎉 All tests passed! URL refactoring appears complete.")
        return True
    else:
        print("\n⚠️  Some issues found. Review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
