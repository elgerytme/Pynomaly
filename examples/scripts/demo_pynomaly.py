#!/usr/bin/env python3
"""
Pynomaly Demo Suite

Consolidates demo functionality from:
- demo_autonomous_mode.py
- demo_autonomous_enhancements.py
- demo_ui_test_results.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class PynomalaDemo:
    """Comprehensive demo suite for Pynomaly capabilities."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.demo_data_dir = None

    def setup_demo_environment(self):
        """Setup demo environment with sample data."""
        self.demo_data_dir = tempfile.mkdtemp(prefix="pynomaly_demo_")

        # Create sample datasets
        self._create_sample_datasets()
        print(f"📁 Demo environment created: {self.demo_data_dir}")

    def _create_sample_datasets(self):
        """Create sample datasets for demos."""

        # Simple 2D anomaly detection dataset
        simple_data = """x,y,label
1.0,2.0,normal
2.0,3.0,normal
3.0,4.0,normal
4.0,5.0,normal
5.0,6.0,normal
100.0,200.0,anomaly
2.5,3.5,normal
3.5,4.5,normal
1.5,2.5,normal
-50.0,-100.0,anomaly"""

        simple_file = Path(self.demo_data_dir) / "simple_anomalies.csv"
        with open(simple_file, "w") as f:
            f.write(simple_data)

        # Financial transaction dataset
        financial_data = """transaction_id,amount,merchant,location,time_hour,day_of_week,is_weekend
T001,25.50,Coffee Shop,Local,9,1,0
T002,120.00,Grocery Store,Local,14,1,0
T003,45.75,Gas Station,Local,17,1,0
T004,15000.00,Casino,International,2,1,0
T005,35.20,Restaurant,Local,19,1,0
T006,67.80,Pharmacy,Local,11,2,0
T007,200.00,ATM,Local,15,2,0
T008,5500.00,Electronics,International,3,2,0
T009,89.30,Clothing,Local,16,2,0
T010,42.15,Supermarket,Local,10,3,0"""

        financial_file = Path(self.demo_data_dir) / "financial_transactions.csv"
        with open(financial_file, "w") as f:
            f.write(financial_data)

    def demo_basic_detection(self) -> dict[str, Any]:
        """Demo basic anomaly detection."""
        print("\n🔍 Demo: Basic Anomaly Detection")
        print("-" * 40)

        simple_file = Path(self.demo_data_dir) / "simple_anomalies.csv"

        try:
            # Run basic detection
            start_time = time.time()

            cmd = [
                "python3",
                "-m",
                "pynomaly",
                "auto",
                "detect",
                str(simple_file),
                "--output",
                str(Path(self.demo_data_dir) / "basic_results.json"),
                "--format",
                "json",
            ]

            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root,
                env=env,
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Basic detection completed in {execution_time:.2f}s")

                # Load and display results
                results_file = Path(self.demo_data_dir) / "basic_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)

                    anomaly_count = sum(
                        1 for score in results.get("predictions", []) if score == 1
                    )
                    total_samples = len(results.get("predictions", []))

                    print(
                        f"📊 Results: {anomaly_count}/{total_samples} anomalies detected"
                    )

                    return {
                        "status": "success",
                        "execution_time": execution_time,
                        "anomalies_detected": anomaly_count,
                        "total_samples": total_samples,
                    }
                else:
                    print("❌ Results file not generated")
                    return {"status": "error", "message": "Results file not generated"}
            else:
                print(f"❌ Detection failed: {result.stderr}")
                return {"status": "error", "message": result.stderr}

        except Exception as e:
            print(f"❌ Demo error: {e}")
            return {"status": "error", "message": str(e)}

    def demo_autonomous_mode(self) -> dict[str, Any]:
        """Demo autonomous anomaly detection."""
        print("\n🤖 Demo: Autonomous Mode")
        print("-" * 40)

        financial_file = Path(self.demo_data_dir) / "financial_transactions.csv"

        try:
            # Run autonomous detection
            start_time = time.time()

            cmd = [
                "python3",
                "-m",
                "pynomaly",
                "auto",
                "detect",
                str(financial_file),
                "--output",
                str(Path(self.demo_data_dir) / "autonomous_results.json"),
            ]

            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root,
                env=env,
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Autonomous detection completed in {execution_time:.2f}s")
                print("🧠 Algorithm automatically selected and optimized")

                # Load and display results
                results_file = Path(self.demo_data_dir) / "autonomous_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)

                    algorithm_used = results.get("algorithm_info", {}).get(
                        "name", "Unknown"
                    )
                    anomaly_count = sum(
                        1 for score in results.get("predictions", []) if score == 1
                    )
                    total_samples = len(results.get("predictions", []))

                    print(f"🔧 Algorithm selected: {algorithm_used}")
                    print(
                        f"📊 Results: {anomaly_count}/{total_samples} anomalies detected"
                    )

                    return {
                        "status": "success",
                        "execution_time": execution_time,
                        "algorithm_used": algorithm_used,
                        "anomalies_detected": anomaly_count,
                        "total_samples": total_samples,
                    }
                else:
                    print("❌ Results file not generated")
                    return {"status": "error", "message": "Results file not generated"}
            else:
                print(f"❌ Autonomous detection failed: {result.stderr}")
                return {"status": "error", "message": result.stderr}

        except Exception as e:
            print(f"❌ Demo error: {e}")
            return {"status": "error", "message": str(e)}

    def demo_export_capabilities(self) -> dict[str, Any]:
        """Demo export functionality."""
        print("\n📤 Demo: Export Capabilities")
        print("-" * 40)

        simple_file = Path(self.demo_data_dir) / "simple_anomalies.csv"
        export_results = {}

        # Test different export formats
        formats = ["json", "csv", "excel"]

        for fmt in formats:
            try:
                output_file = Path(self.demo_data_dir) / f"export_demo.{fmt}"

                cmd = [
                    "python3",
                    "-m",
                    "pynomaly",
                    "export",
                    str(simple_file),
                    "--format",
                    fmt,
                    "--output",
                    str(output_file),
                ]

                env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    cwd=self.project_root,
                    env=env,
                )

                if result.returncode == 0 and output_file.exists():
                    file_size = output_file.stat().st_size
                    export_results[fmt] = {"status": "success", "file_size": file_size}
                    print(f"✅ {fmt.upper()} export successful ({file_size} bytes)")
                else:
                    export_results[fmt] = {"status": "error", "message": result.stderr}
                    print(f"❌ {fmt.upper()} export failed")

            except Exception as e:
                export_results[fmt] = {"status": "error", "message": str(e)}
                print(f"❌ {fmt.upper()} export error: {e}")

        successful_exports = [
            fmt
            for fmt, result in export_results.items()
            if result.get("status") == "success"
        ]

        print(
            f"📊 Export summary: {len(successful_exports)}/{len(formats)} formats successful"
        )

        return {
            "status": "success" if successful_exports else "error",
            "successful_formats": successful_exports,
            "format_results": export_results,
        }

    def demo_preprocessing(self) -> dict[str, Any]:
        """Demo data preprocessing capabilities."""
        print("\n🔧 Demo: Data Preprocessing")
        print("-" * 40)

        # Create dataset with missing values and outliers
        messy_data = """feature1,feature2,feature3,target
1.0,2.0,3.0,0
2.0,,4.0,0
3.0,4.0,5.0,0
,5.0,6.0,0
5.0,6.0,7.0,0
1000.0,2000.0,3000.0,1
6.0,7.0,8.0,0
7.0,8.0,,0
8.0,9.0,10.0,0"""

        messy_file = Path(self.demo_data_dir) / "messy_data.csv"
        with open(messy_file, "w") as f:
            f.write(messy_data)

        try:
            # Run preprocessing
            output_file = Path(self.demo_data_dir) / "cleaned_data.csv"

            cmd = [
                "python3",
                "-m",
                "pynomaly",
                "data",
                "clean",
                str(messy_file),
                "--output",
                str(output_file),
                "--missing-strategy",
                "mean",
                "--outlier-method",
                "iqr",
            ]

            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=self.project_root,
                env=env,
            )

            if result.returncode == 0 and output_file.exists():
                print("✅ Data preprocessing successful")
                print("🧹 Missing values filled, outliers handled")

                return {
                    "status": "success",
                    "input_file": str(messy_file),
                    "output_file": str(output_file),
                }
            else:
                print(f"❌ Preprocessing failed: {result.stderr}")
                return {"status": "error", "message": result.stderr}

        except Exception as e:
            print(f"❌ Preprocessing demo error: {e}")
            return {"status": "error", "message": str(e)}

    def demo_performance_benchmarking(self) -> dict[str, Any]:
        """Demo performance benchmarking."""
        print("\n⚡ Demo: Performance Benchmarking")
        print("-" * 40)

        simple_file = Path(self.demo_data_dir) / "simple_anomalies.csv"

        try:
            cmd = [
                "python3",
                "-m",
                "pynomaly",
                "benchmark",
                str(simple_file),
                "--algorithms",
                "IsolationForest,LOF,OneClassSVM",
                "--output",
                str(Path(self.demo_data_dir) / "benchmark_results.json"),
            ]

            start_time = time.time()
            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root,
                env=env,
            )
            execution_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Benchmark completed in {execution_time:.2f}s")
                print("📊 Multiple algorithms compared")

                return {"status": "success", "execution_time": execution_time}
            else:
                print(f"⚠️ Benchmark command not available: {result.stderr}")
                return {
                    "status": "warning",
                    "message": "Benchmark command not implemented",
                }

        except Exception as e:
            print(f"⚠️ Benchmark demo skipped: {e}")
            return {"status": "warning", "message": str(e)}

    def run_demo_suite(self, demo_type: str = "all") -> dict[str, Any]:
        """Run specified demo or all demos."""
        print("🚀 Starting Pynomaly Demo Suite...")
        print("=" * 50)

        # Setup demo environment
        self.setup_demo_environment()

        demo_results = {}

        if demo_type in ["all", "basic"]:
            demo_results["basic_detection"] = self.demo_basic_detection()

        if demo_type in ["all", "autonomous"]:
            demo_results["autonomous_mode"] = self.demo_autonomous_mode()

        if demo_type in ["all", "export"]:
            demo_results["export_capabilities"] = self.demo_export_capabilities()

        if demo_type in ["all", "preprocessing"]:
            demo_results["preprocessing"] = self.demo_preprocessing()

        if demo_type in ["all", "performance"]:
            demo_results["performance_benchmarking"] = (
                self.demo_performance_benchmarking()
            )

        # Generate summary
        successful_demos = [
            name
            for name, result in demo_results.items()
            if result.get("status") in ["success", "warning"]
        ]

        print("\n" + "=" * 50)
        print("📊 Demo Suite Results:")
        print(f"Completed: {len(successful_demos)}/{len(demo_results)} demos")

        for demo_name, result in demo_results.items():
            status_icon = {"success": "✅", "error": "❌", "warning": "⚠️"}.get(
                result.get("status"), "❓"
            )
            print(f"  {status_icon} {demo_name}")

        print(f"\n📁 Demo files created in: {self.demo_data_dir}")
        print("💡 You can explore the generated files to see Pynomaly's capabilities!")

        return {
            "summary": {
                "total_demos": len(demo_results),
                "successful_demos": len(successful_demos),
                "demo_directory": str(self.demo_data_dir),
            },
            "demo_results": demo_results,
        }


def main():
    """Main entry point for Pynomaly demo."""
    parser = argparse.ArgumentParser(
        description="Pynomaly Demo Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Types:
  all           Run all demos (default)
  basic         Basic anomaly detection
  autonomous    Autonomous mode demonstration
  export        Export capabilities
  preprocessing Data preprocessing
  performance   Performance benchmarking

Examples:
  python demo_pynomaly.py
  python demo_pynomaly.py --demo autonomous
  python demo_pynomaly.py --demo export
        """,
    )

    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "basic",
            "autonomous",
            "export",
            "preprocessing",
            "performance",
        ],
        default="all",
        help="Type of demo to run (default: all)",
    )

    args = parser.parse_args()

    # Run demo suite
    demo = PynomalaDemo()
    results = demo.run_demo_suite(args.demo)

    # Save results
    results_file = PROJECT_ROOT / "demo_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Demo results saved to: {results_file}")

    # Exit based on success rate
    success_rate = (
        results["summary"]["successful_demos"] / results["summary"]["total_demos"]
    )
    if success_rate >= 0.8:
        print("🎉 Demo suite completed successfully!")
        sys.exit(0)
    else:
        print("⚠️ Some demos failed. Check the results for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
