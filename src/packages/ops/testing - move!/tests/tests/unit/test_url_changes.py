#!/usr/bin/env python3
"""
Test script to verify URL routing changes from /web to /
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_app_imports():
    """Test that the app imports correctly after URL changes."""
    try:
        from monorepo.presentation.web.app import create_web_app, mount_web_ui, router

        print("✅ App imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False


def test_router_prefix():
    """Test that the router prefix has been updated."""
    try:
        from fastapi import FastAPI

        from monorepo.presentation.web.app import mount_web_ui

        # Create a test FastAPI app
        app = FastAPI()

        # Mount the web UI
        mount_web_ui(app)

        # Check if the web routes are mounted with empty prefix
        web_routes = [route for route in app.routes if hasattr(route, "path_regex")]
        web_paths = [getattr(route, "path", "") for route in web_routes]

        # Look for routes that start with root instead of /web
        root_routes = [
            path for path in web_paths if path and not path.startswith("/web")
        ]

        if root_routes:
            print(f"✅ Found {len(root_routes)} routes using root path structure")
            print(f"   Sample routes: {root_routes[:5]}")
            return True
        else:
            print("❌ No root-level routes found")
            return False

    except Exception as e:
        print(f"❌ Router test error: {e}")
        return False


def test_template_updates():
    """Test that template files have been updated."""
    templates_dir = Path("src/pynomaly/presentation/web/templates")

    if not templates_dir.exists():
        print("❌ Templates directory not found")
        return False

    # Check base.html for /web references
    base_template = templates_dir / "base.html"
    if base_template.exists():
        content = base_template.read_text(encoding="utf-8")
        web_refs = content.count("/web/")

        if web_refs == 0:
            print("✅ base.html updated - no /web/ references found")
            return True
        else:
            print(f"❌ base.html still contains {web_refs} /web/ references")
            return False
    else:
        print("❌ base.html not found")
        return False


def test_js_updates():
    """Test that JavaScript files have been updated."""
    js_dir = Path("src/pynomaly/presentation/web/static/js")

    if not js_dir.exists():
        print("❌ JavaScript directory not found")
        return False

    js_files = list(js_dir.rglob("*.js"))

    if not js_files:
        print("❌ No JavaScript files found")
        return False

    web_refs_found = 0
    total_files = 0

    for js_file in js_files:
        try:
            content = js_file.read_text(encoding="utf-8")
            web_refs = content.count("/web/")
            web_refs_found += web_refs
            total_files += 1
        except Exception as e:
            print(f"⚠️  Could not read {js_file}: {e}")

    if web_refs_found == 0:
        print(f"✅ {total_files} JavaScript files updated - no /web/ references found")
        return True
    else:
        print(
            f"❌ Found {web_refs_found} /web/ references in {total_files} JavaScript files"
        )
        return False


def main():
    """Run all tests."""
    print("Testing URL scheme changes from /web to /")
    print("=" * 50)

    tests = [
        ("App imports", test_app_imports),
        ("Router prefix", test_router_prefix),
        ("Template updates", test_template_updates),
        ("JavaScript updates", test_js_updates),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! URL scheme changes appear successful.")
        return True
    else:
        print("⚠️  Some tests failed. Manual review may be needed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
