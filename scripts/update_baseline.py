import json
import sys


def update_baseline(benchmark_file, baseline_file, git_commit):
    try:
        with open(benchmark_file) as bf:
            benchmark_results = json.load(bf)

        baseline = {
            "timestamp": "2025-07-08T18:15:33Z",
            "performance_metrics": {metric['name']: metric['stats']['mean'] for metric in benchmark_results["benchmarks"]},
            "git_ref": "main",
            "git_commit": git_commit,
            "description": "Updated baseline"
        }

        with open(baseline_file, 'w') as bf:
            json.dump(baseline, bf, indent=2)

        print("Baseline updated.")

    except Exception as e:
        print(f"Error updating baseline: {e}")


if __name__ == "__main__":
    update_baseline(
        benchmark_file=sys.argv[1],
        baseline_file=sys.argv[2],
        git_commit=sys.argv[3],
    )
