#!/usr/bin/env python3
"""Update performance baselines from benchmark results."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class BaselineManager:
    """Manage performance baselines for regression testing."""
    
    def __init__(self, baselines_dir: Path = Path("tests/performance/baselines")):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        
        # Baseline metadata
        self.metadata_file = self.baselines_dir / "baseline_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load baseline metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            "created": datetime.now().isoformat(),
            "baselines": {},
            "update_history": []
        }
    
    def _save_metadata(self) -> None:
        """Save baseline metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_baseline_from_benchmark(self, 
                                     benchmark_data: Dict[str, Any],
                                     suite_name: str,
                                     commit_hash: str = None) -> bool:
        """Create baseline from benchmark results.
        
        Args:
            benchmark_data: Benchmark results data
            suite_name: Name of the test suite
            commit_hash: Git commit hash for tracking
            
        Returns:
            True if baseline was created successfully
        """
        try:
            benchmarks = benchmark_data.get('benchmarks', [])
            if not benchmarks:
                print(f"⚠️  No benchmarks found for {suite_name}")
                return False
            
            # Extract baseline data
            baseline_data = {
                "suite": suite_name,
                "created": datetime.now().isoformat(),
                "commit_hash": commit_hash,
                "machine_info": benchmark_data.get('machine_info', {}),
                "commit_info": benchmark_data.get('commit_info', {}),
                "tests": {}
            }
            
            # Process each benchmark
            for benchmark in benchmarks:
                test_name = benchmark.get('name', '')
                if test_name:
                    stats = benchmark.get('stats', {})
                    baseline_data["tests"][test_name] = {
                        "mean": stats.get('mean', 0),
                        "min": stats.get('min', 0),
                        "max": stats.get('max', 0),
                        "stddev": stats.get('stddev', 0),
                        "median": stats.get('median', 0),
                        "iqr": stats.get('iqr', 0),
                        "outliers": stats.get('outliers', ''),
                        "rounds": stats.get('rounds', 0)
                    }
            
            # Save baseline file
            baseline_file = self.baselines_dir / f"{suite_name}_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            # Update metadata
            self.metadata["baselines"][suite_name] = {
                "file": str(baseline_file),
                "created": baseline_data["created"],
                "commit_hash": commit_hash,
                "test_count": len(baseline_data["tests"])
            }
            
            self.metadata["update_history"].append({
                "suite": suite_name,
                "action": "created",
                "timestamp": datetime.now().isoformat(),
                "commit_hash": commit_hash,
                "test_count": len(baseline_data["tests"])
            })
            
            self._save_metadata()
            
            print(f"✅ Created baseline for {suite_name} with {len(baseline_data['tests'])} tests")
            return True
            
        except Exception as e:
            print(f"❌ Error creating baseline for {suite_name}: {e}")
            return False
    
    def update_baseline_from_benchmark(self,
                                     benchmark_data: Dict[str, Any],
                                     suite_name: str,
                                     commit_hash: str = None,
                                     force: bool = False) -> bool:
        """Update existing baseline with new benchmark data.
        
        Args:
            benchmark_data: New benchmark results
            suite_name: Name of the test suite
            commit_hash: Git commit hash for tracking
            force: Force update even if baseline is recent
            
        Returns:
            True if baseline was updated successfully
        """
        baseline_file = self.baselines_dir / f"{suite_name}_baseline.json"
        
        # Check if baseline exists
        if not baseline_file.exists():
            print(f"📝 No existing baseline for {suite_name}, creating new one")
            return self.create_baseline_from_benchmark(benchmark_data, suite_name, commit_hash)
        
        # Load existing baseline
        try:
            with open(baseline_file, 'r') as f:
                existing_baseline = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ Error loading existing baseline: {e}")
            return False
        
        # Check if update is needed
        if not force:
            # Don't update if baseline is less than 24 hours old
            created_time = datetime.fromisoformat(existing_baseline.get('created', '1970-01-01'))
            age_hours = (datetime.now() - created_time).total_seconds() / 3600
            
            if age_hours < 24:
                print(f"ℹ️  Baseline for {suite_name} is only {age_hours:.1f} hours old, skipping update")
                return True
        
        # Backup existing baseline
        backup_file = self.baselines_dir / f"{suite_name}_baseline_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(baseline_file, backup_file)
        
        # Create updated baseline
        success = self.create_baseline_from_benchmark(benchmark_data, suite_name, commit_hash)
        
        if success:
            self.metadata["update_history"].append({
                "suite": suite_name,
                "action": "updated",
                "timestamp": datetime.now().isoformat(),
                "commit_hash": commit_hash,
                "backup_file": str(backup_file),
                "force": force
            })
            self._save_metadata()
            
            print(f"✅ Updated baseline for {suite_name} (backup: {backup_file.name})")
        else:
            # Restore backup if update failed
            shutil.copy2(backup_file, baseline_file)
            print(f"❌ Update failed, restored from backup")
        
        return success
    
    def get_baseline(self, suite_name: str) -> Optional[Dict[str, Any]]:
        """Get baseline data for a test suite.
        
        Args:
            suite_name: Name of the test suite
            
        Returns:
            Baseline data or None if not found
        """
        baseline_file = self.baselines_dir / f"{suite_name}_baseline.json"
        
        if not baseline_file.exists():
            return None
        
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all available baselines.
        
        Returns:
            List of baseline information
        """
        baselines = []
        
        for suite_name, info in self.metadata.get("baselines", {}).items():
            baseline_data = self.get_baseline(suite_name)
            if baseline_data:
                baselines.append({
                    "suite": suite_name,
                    "created": info.get("created"),
                    "commit_hash": info.get("commit_hash"),
                    "test_count": info.get("test_count", 0),
                    "file": info.get("file"),
                    "age_hours": (
                        datetime.now() - datetime.fromisoformat(info.get("created", "1970-01-01"))
                    ).total_seconds() / 3600
                })
        
        return sorted(baselines, key=lambda x: x["created"], reverse=True)
    
    def clean_old_baselines(self, keep_days: int = 30) -> int:
        """Clean up old baseline backups.
        
        Args:
            keep_days: Number of days to keep backups
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        # Find backup files
        backup_files = list(self.baselines_dir.glob("*_backup_*.json"))
        
        for backup_file in backup_files:
            try:
                file_time = backup_file.stat().st_mtime
                if file_time < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
                    print(f"🗑️  Cleaned up old backup: {backup_file.name}")
            except OSError as e:
                print(f"⚠️  Could not clean {backup_file.name}: {e}")
        
        return cleaned_count
    
    def generate_baseline_report(self) -> str:
        """Generate a report of current baselines.
        
        Returns:
            Markdown report of baseline status
        """
        baselines = self.list_baselines()
        
        report = [
            "# Performance Baselines Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Baselines Directory**: {self.baselines_dir}",
            "",
            "## Current Baselines",
            ""
        ]
        
        if baselines:
            report.extend([
                "| Suite | Created | Age | Tests | Commit |",
                "|-------|---------|-----|-------|--------|"
            ])
            
            for baseline in baselines:
                age_str = f"{baseline['age_hours']:.1f}h" if baseline['age_hours'] < 48 else f"{baseline['age_hours']/24:.1f}d"
                commit_short = baseline['commit_hash'][:8] if baseline['commit_hash'] else "N/A"
                
                report.append(
                    f"| {baseline['suite']} | {baseline['created'][:10]} | "
                    f"{age_str} | {baseline['test_count']} | {commit_short} |"
                )
            
            report.append("")
        else:
            report.append("No baselines found.")
            report.append("")
        
        # Update history
        history = self.metadata.get("update_history", [])
        if history:
            report.extend([
                "## Recent Updates",
                "",
                "| Suite | Action | Date | Commit |",
                "|-------|--------|------|--------|"
            ])
            
            for update in history[-10:]:  # Last 10 updates
                commit_short = update.get('commit_hash', 'N/A')[:8] if update.get('commit_hash') else "N/A"
                report.append(
                    f"| {update['suite']} | {update['action']} | "
                    f"{update['timestamp'][:10]} | {commit_short} |"
                )
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for baseline management."""
    parser = argparse.ArgumentParser(description="Update performance baselines")
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Directory containing benchmark results")
    parser.add_argument("--commit-hash", type=str,
                       help="Git commit hash for tracking")
    parser.add_argument("--force", action="store_true",
                       help="Force update even if baselines are recent")
    parser.add_argument("--baselines-dir", type=Path, default="tests/performance/baselines",
                       help="Directory to store baselines")
    parser.add_argument("--list", action="store_true",
                       help="List current baselines")
    parser.add_argument("--clean", type=int, metavar="DAYS",
                       help="Clean backup files older than DAYS")
    parser.add_argument("--report", action="store_true",
                       help="Generate baseline report")
    
    args = parser.parse_args()
    
    # Create baseline manager
    manager = BaselineManager(args.baselines_dir)
    
    # Handle list command
    if args.list:
        baselines = manager.list_baselines()
        if baselines:
            print("📊 Current Performance Baselines:")
            for baseline in baselines:
                age_str = f"{baseline['age_hours']:.1f}h" if baseline['age_hours'] < 48 else f"{baseline['age_hours']/24:.1f}d"
                print(f"  - {baseline['suite']}: {baseline['test_count']} tests, {age_str} old")
        else:
            print("No baselines found")
        return 0
    
    # Handle clean command
    if args.clean:
        cleaned = manager.clean_old_baselines(args.clean)
        print(f"🗑️  Cleaned {cleaned} old backup files")
        return 0
    
    # Handle report command
    if args.report:
        report = manager.generate_baseline_report()
        print(report)
        return 0
    
    # Main update functionality
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1
    
    print(f"🔄 Updating baselines from {args.input_dir}")
    
    # Find all benchmark files
    benchmark_files = list(args.input_dir.glob("**/benchmark-*.json"))
    
    if not benchmark_files:
        print("❌ No benchmark files found")
        return 1
    
    updated_count = 0
    failed_count = 0
    
    for benchmark_file in benchmark_files:
        try:
            # Extract suite name
            suite_name = benchmark_file.stem.replace("benchmark-", "")
            
            # Load benchmark data
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            # Update baseline
            if manager.update_baseline_from_benchmark(
                benchmark_data, suite_name, args.commit_hash, args.force
            ):
                updated_count += 1
            else:
                failed_count += 1
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ Error processing {benchmark_file}: {e}")
            failed_count += 1
    
    print(f"\n📊 Baseline Update Summary:")
    print(f"  - Updated: {updated_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Total processed: {len(benchmark_files)}")
    
    if updated_count > 0:
        print(f"\n✅ Successfully updated {updated_count} baselines")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())