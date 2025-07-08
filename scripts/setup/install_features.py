#!/usr/bin/env python3
"""
Interactive installer for Pynomaly optional features.

This script helps users install only the features they need.
"""

import subprocess
import sys


def run_command(cmd: list[str]) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False


def install_feature_group(group_name: str) -> bool:
    """Install a feature group using pip."""
    cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{group_name}]"]
    return run_command(cmd)


def main():
    """Interactive feature installer."""
    print("🚀 Pynomaly Feature Installer")
    print("=" * 50)

    # Feature definitions
    features: dict[str, dict[str, str]] = {
        "Basic Features": {
            "minimal": "Core anomaly detection only",
            "standard": "Core + data formats (Parquet, Excel)",
        },
        "Interfaces": {
            "cli": "Command-line interface",
            "api": "REST API server",
            "server": "Complete server (API + CLI + data formats)",
        },
        "Advanced ML": {
            "automl": "AutoML with Optuna and auto-sklearn",
            "explainability": "SHAP and LIME model explanation",
            "torch": "PyTorch deep learning",
            "tensorflow": "TensorFlow neural networks",
            "jax": "JAX high-performance computing",
            "graph": "Graph anomaly detection (PyGOD)",
        },
        "Production": {
            "auth": "JWT authentication",
            "monitoring": "Prometheus monitoring",
            "production": "Full production stack",
        },
        "Development": {
            "test": "Testing dependencies",
            "dev": "Development tools",
            "lint": "Code quality tools",
        },
    }

    # Show available features
    print("Available feature groups:")
    print()

    for category, group_features in features.items():
        print(f"📂 {category}")
        for feature, description in group_features.items():
            print(f"   {feature:15} - {description}")
        print()

    # Get user selection
    print("Installation options:")
    print("1. 🎯 Quick Start (server) - Complete server with CLI and API")
    print("2. 🧪 Researcher (automl + explainability) - ML research tools")
    print("3. 🚀 Production (production) - Full production deployment")
    print("4. 🛠️ Developer (dev + test + lint) - Development environment")
    print("5. 🔧 Custom - Choose specific features")
    print("6. ❓ Help - Show detailed feature information")
    print()

    choice = input("Select option (1-6): ").strip()

    if choice == "1":
        # Quick Start
        print("📦 Installing Quick Start package...")
        success = install_feature_group("server")
        if success:
            print("✅ Quick Start installation complete!")
            print("🚀 Try: python scripts/run/cli.py --help")
            print("🌐 Try: python scripts/run/run_api.py")

    elif choice == "2":
        # Researcher
        print("📦 Installing Researcher package...")
        success = all(
            [
                install_feature_group("server"),
                install_feature_group("automl"),
                install_feature_group("explainability"),
            ]
        )
        if success:
            print("✅ Researcher installation complete!")
            print("🧪 AutoML: python scripts/run/cli.py automl --help")
            print("🔍 Explainability: python scripts/run/cli.py explainability --help")

    elif choice == "3":
        # Production
        print("📦 Installing Production package...")
        success = install_feature_group("production")
        if success:
            print("✅ Production installation complete!")
            print("🚀 Production ready with monitoring and authentication")

    elif choice == "4":
        # Developer
        print("📦 Installing Developer package...")
        success = all(
            [
                install_feature_group("dev"),
                install_feature_group("test"),
                install_feature_group("lint"),
            ]
        )
        if success:
            print("✅ Developer installation complete!")
            print("🧪 Run tests: pytest")
            print("🔍 Check code: ruff check src/")

    elif choice == "5":
        # Custom
        print("📦 Custom installation")
        print("Enter feature names separated by spaces:")
        print("Example: automl explainability torch")

        feature_input = input("Features: ").strip()
        if feature_input:
            features_to_install = feature_input.split()
            print(f"Installing: {', '.join(features_to_install)}")

            success = True
            for feature in features_to_install:
                if not install_feature_group(feature):
                    success = False

            if success:
                print("✅ Custom installation complete!")
            else:
                print("❌ Some installations failed")

    elif choice == "6":
        # Help
        print("📚 Feature Information")
        print("=" * 50)

        print("🎯 Recommended combinations:")
        print("   • server: Complete anomaly detection platform")
        print("   • automl + explainability: ML research and interpretability")
        print("   • torch + explainability: Deep learning with explanations")
        print("   • production: Enterprise deployment")
        print()

        print("💡 Installation examples:")
        print("   pip install -e .[server]")
        print("   pip install -e .[automl,explainability]")
        print("   pip install -e .[torch,tensorflow,jax]")
        print("   pip install -e .[production]")
        print()

        print("⚠️  Notes:")
        print("   • Some features require additional system dependencies")
        print("   • PyTorch/TensorFlow are large downloads (~2GB each)")
        print("   • AutoML features may take time to install")

    else:
        print("❌ Invalid option")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
