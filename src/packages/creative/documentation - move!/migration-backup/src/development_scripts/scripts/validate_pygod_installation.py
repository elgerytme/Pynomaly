#!/usr/bin/env python3
"""
PyGOD Installation Validation Script

This script validates that PyGOD dependencies are properly installed
and the integration is working correctly.
"""

import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version compatibility."""
    logger.info("🐍 Checking Python version...")

    version = sys.version_info
    if version < (3, 11):
        logger.error(
            f"❌ Python {version.major}.{version.minor} detected. Python 3.11+ required."
        )
        return False

    logger.info(
        f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
    )
    return True


def check_core_dependencies():
    """Check core dependencies."""
    logger.info("📦 Checking core dependencies...")

    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("networkx", "networkx"),
        ("sklearn", "scikit-learn"),
        ("pydantic", "pydantic"),
    ]

    all_available = True
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name}")
        except ImportError:
            logger.error(f"❌ {package_name} - not installed")
            all_available = False

    return all_available


def check_pygod_dependencies():
    """Check PyGOD-specific dependencies."""
    logger.info("🔥 Checking PyGOD dependencies...")

    dependencies = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("pygod", "PyGOD"),
    ]

    all_available = True
    for import_name, package_name in dependencies:
        try:
            module = __import__(import_name)
            if hasattr(module, "__version__"):
                version = module.__version__
                logger.info(f"✅ {package_name} v{version}")
            else:
                logger.info(f"✅ {package_name}")
        except ImportError as e:
            logger.error(f"❌ {package_name} - {e}")
            all_available = False

    return all_available


def check_gpu_support():
    """Check GPU support if available."""
    logger.info("🖥️  Checking GPU support...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda

            logger.info(f"✅ CUDA available: v{cuda_version}")
            logger.info(f"✅ GPUs available: {gpu_count}")
            logger.info(f"✅ Primary GPU: {gpu_name}")

            # Test GPU memory
            try:
                device = torch.device("cuda:0")
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                memory_cached = torch.cuda.memory_reserved(device) / 1024**2
                logger.info(
                    f"✅ GPU memory: {memory_allocated:.1f}MB allocated, {memory_cached:.1f}MB cached"
                )
            except Exception as e:
                logger.warning(f"⚠️  GPU memory check failed: {e}")

            return True
        else:
            logger.info("ℹ️  CUDA not available - CPU-only mode")
            return False

    except ImportError:
        logger.warning("⚠️  Cannot check GPU support - PyTorch not available")
        return False


def validate_pygod_adapter():
    """Validate PyGOD adapter functionality."""
    logger.info("🔧 Validating PyGOD adapter...")

    try:
        from monorepo.infrastructure.adapters.pygod_adapter import PyGODAdapter

        logger.info("✅ PyGOD adapter import successful")

        # Check available algorithms
        algorithms = PyGODAdapter.get_supported_algorithms()
        logger.info(f"✅ Available algorithms: {len(algorithms)}")

        if not algorithms:
            logger.error("❌ No algorithms available")
            return False

        # List algorithms by category
        deep_learning = [
            "DOMINANT",
            "GCNAE",
            "ANOMALOUS",
            "MLPAE",
            "ANOMALYDAE",
            "GAAN",
            "GUIDE",
            "CONAD",
            "GADNR",
        ]
        statistical = ["SCAN", "RADAR"]

        dl_available = [algo for algo in deep_learning if algo in algorithms]
        stat_available = [algo for algo in statistical if algo in algorithms]

        logger.info(f"  📊 Deep Learning: {dl_available}")
        logger.info(f"  📈 Statistical: {stat_available}")

        # Test algorithm info retrieval
        test_algo = algorithms[0]
        try:
            info = PyGODAdapter.get_algorithm_info(test_algo)
            logger.info(
                f"✅ Algorithm info for {test_algo}: {info.get('type', 'Unknown type')}"
            )
        except Exception as e:
            logger.warning(f"⚠️  Algorithm info error: {e}")

        # Test adapter creation
        try:
            adapter = PyGODAdapter(algorithm_name=test_algo)
            logger.info(f"✅ Created {test_algo} adapter: {adapter.name}")
        except Exception as e:
            logger.error(f"❌ Adapter creation failed: {e}")
            return False

        return True

    except ImportError as e:
        logger.error(f"❌ PyGOD adapter import failed: {e}")
        return False


def validate_with_sample_data():
    """Validate with actual sample data."""
    logger.info("📊 Testing with sample data...")

    try:
        import numpy as np
        import pandas as pd

        from monorepo.domain.entities import Dataset
        from monorepo.domain.value_objects import ContaminationRate
        from monorepo.infrastructure.adapters.pygod_adapter import PyGODAdapter

        # Create simple test graph
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "source": [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                "target": [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
                "feature_0": np.random.normal(0, 1, 10),
                "feature_1": np.random.normal(0, 1, 10),
            }
        )

        dataset = Dataset(
            id="test_graph",
            name="Test Graph",
            data=data,
            metadata={
                "is_graph": True,
                "edge_columns": ["source", "target"],
                "feature_columns": ["feature_0", "feature_1"],
            },
        )

        # Test with a statistical algorithm (faster, no GPU required)
        algorithms = PyGODAdapter.get_supported_algorithms()
        test_algo = "SCAN" if "SCAN" in algorithms else algorithms[0]

        logger.info(f"Testing {test_algo} with sample data...")

        if test_algo == "SCAN":
            adapter = PyGODAdapter(
                algorithm_name=test_algo,
                contamination_rate=ContaminationRate(0.2),
                eps=0.5,
                mu=2,
            )
        else:
            adapter = PyGODAdapter(
                algorithm_name=test_algo, contamination_rate=ContaminationRate(0.2)
            )

        # Test training
        logger.info("  Training model...")
        adapter.fit(dataset)
        logger.info(f"  ✅ Training completed, fitted: {adapter.is_fitted}")

        # Test prediction
        logger.info("  Predicting anomalies...")
        result = adapter.predict(dataset)

        anomaly_count = sum(result.labels)
        logger.info(
            f"  ✅ Detected {anomaly_count} anomalies out of {len(result.labels)} samples"
        )
        logger.info(
            f"  ✅ Average anomaly score: {np.mean([s.value for s in result.scores]):.3f}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Sample data test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def generate_installation_report():
    """Generate comprehensive installation report."""
    logger.info("📋 Generating installation report...")

    report = {
        "python_version": check_python_version(),
        "core_dependencies": check_core_dependencies(),
        "pygod_dependencies": check_pygod_dependencies(),
        "gpu_support": check_gpu_support(),
        "adapter_validation": validate_pygod_adapter(),
        "sample_data_test": False,
    }

    # Only run sample data test if basic validation passes
    if all(
        [
            report["python_version"],
            report["core_dependencies"],
            report["pygod_dependencies"],
            report["adapter_validation"],
        ]
    ):
        report["sample_data_test"] = validate_with_sample_data()

    return report


def print_summary(report):
    """Print validation summary."""
    logger.info("\n" + "=" * 60)
    logger.info("📋 PYGOD VALIDATION SUMMARY")
    logger.info("=" * 60)

    checks = [
        ("Python Version", report["python_version"]),
        ("Core Dependencies", report["core_dependencies"]),
        ("PyGOD Dependencies", report["pygod_dependencies"]),
        ("GPU Support", report["gpu_support"]),
        ("Adapter Validation", report["adapter_validation"]),
        ("Sample Data Test", report["sample_data_test"]),
    ]

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name:<20}: {status}")

    # Overall status
    critical_checks = [
        "python_version",
        "core_dependencies",
        "pygod_dependencies",
        "adapter_validation",
    ]
    all_critical_pass = all(report[check] for check in critical_checks)

    logger.info("=" * 60)
    if all_critical_pass:
        logger.info("🎉 PyGOD integration is READY!")
        if report["sample_data_test"]:
            logger.info("🚀 Full functionality validated with sample data")
        else:
            logger.info("⚠️  Sample data test failed - check logs above")
    else:
        logger.info("❌ PyGOD integration is NOT READY")
        logger.info(
            "💡 Install missing dependencies with: pip install 'pynomaly[graph]'"
        )

    if report["gpu_support"]:
        logger.info("🖥️  GPU acceleration available for deep learning algorithms")
    else:
        logger.info("💻 CPU-only mode - statistical algorithms recommended")

    logger.info("=" * 60)


def main():
    """Main validation function."""
    logger.info("🎯 PyGOD Installation Validation")
    logger.info("=" * 60)

    try:
        report = generate_installation_report()
        print_summary(report)

        # Exit with appropriate code
        critical_checks = [
            "python_version",
            "core_dependencies",
            "pygod_dependencies",
            "adapter_validation",
        ]
        if all(report[check] for check in critical_checks):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

    except Exception as e:
        logger.error(f"❌ Validation script failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(2)  # Script error


if __name__ == "__main__":
    main()
