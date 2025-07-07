#!/usr/bin/env python3
"""
Simple test to verify the router prefix change
"""

import re
from pathlib import Path

def test_app_router_prefix():
    """Test that the router prefix has been changed in app.py"""
    app_file = Path("src/pynomaly/presentation/web/app.py")
    
    if not app_file.exists():
        print("❌ app.py not found")
        return False
    
    content = app_file.read_text(encoding='utf-8')
    
    # Look for the include_router call with empty prefix
    pattern = r'app\.include_router\(router,\s*prefix=""\s*,'
    if re.search(pattern, content):
        print("✅ Router prefix successfully changed to empty string")
        return True
    
    # Check if old prefix still exists
    old_pattern = r'app\.include_router\(router,\s*prefix="/web"\s*,'
    if re.search(old_pattern, content):
        print("❌ Old /web prefix still found in app.py")
        return False
    
    print("⚠️  Router prefix pattern not found")
    return False

def test_redirect_urls():
    """Test that redirect URLs have been updated"""
    app_file = Path("src/pynomaly/presentation/web/app.py")
    
    if not app_file.exists():
        print("❌ app.py not found")
        return False
    
    content = app_file.read_text(encoding='utf-8')
    
    # Count old /web redirects
    web_redirects = content.count('url="/web/')
    web_redirects += content.count("url='/web/")
    
    if web_redirects == 0:
        print("✅ All redirect URLs updated from /web/ to /")
        return True
    else:
        print(f"❌ Found {web_redirects} old /web/ redirect URLs")
        return False

def main():
    print("Simple URL routing test")
    print("=" * 30)
    
    tests = [
        test_app_router_prefix,
        test_redirect_urls,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    return passed == len(tests)

if __name__ == "__main__":
    main()
