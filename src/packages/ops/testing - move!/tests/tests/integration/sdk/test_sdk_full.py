#!/usr/bin/env python3
"""Test script for Pynomaly SDK across different environments."""

import os
import sys


def test_sdk():
    """Test the Pynomaly SDK functionality."""
    try:
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"🛤️  Python path: {sys.path[:3]}...")

        # Test basic import

        print("✅ Package import successful")

        # Test SDK import
        from pynomaly.presentation.sdk import PynomalySDK

        print("✅ SDK class import successful")

        # Test SDK initialization
        sdk = PynomalySDK()
        print("✅ SDK initialization successful")

        # List available methods
        methods = [method for method in dir(sdk) if not method.startswith("_")]
        print(f"✅ Available SDK methods: {len(methods)}")
        if methods:
            print(f"   First 10 methods: {', '.join(methods[:10])}")

        # Test version if available
        if hasattr(sdk, "get_version"):
            try:
                version = sdk.get_version()
                print(f"✅ Version method test: {version}")
            except Exception as e:
                print(f"⚠️  Version method failed: {e}")

        print("🎉 SDK test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ SDK test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sdk()
    sys.exit(0 if success else 1)
