import json
import sys


def compare_benchmarks(baseline_file, current_file, output_file, threshold=0.1):
    try:
        # Load baseline metrics
        with open(baseline_file) as bl_file:
            baseline_data = json.load(bl_file)

        # Load current benchmark results
        with open(current_file) as cf_file:
            current_data = json.load(cf_file)

        # Open the output for writing
        with open(output_file, 'w') as out_file:
            out_file.write("Performance Comparison Report\n")
            out_file.write("============================\n\n")

            # Compare each metric
            for metric in current_data["benchmarks"]:
                name = metric['name']
                current_value = metric['stats']['mean']

                # Get the baseline value
                baseline_value = baseline_data["performance_metrics"].get(name)

                if baseline_value is not None:
                    difference = current_value - baseline_value
                    percentage_change = (difference / baseline_value) * 100

                    out_file.write(f"{name}: {current_value:.2f} (Baseline: {baseline_value:.2f})\n")
                    out_file.write(f"  Change: {difference:.2f} ({percentage_change:.2f}%)\n")

                    # Check if regression
                    if percentage_change > threshold * 100:
                        out_file.write("  ⚠️ Performance regression detected!\n")

                else:
                    out_file.write(f"{name}: {current_value:.2f} (No Baseline)\n")

            out_file.write("\nComparison complete.\n")

        print("Comparison complete. See the comparison report for details.")

    except Exception as e:
        print(f"Error during benchmark comparison: {e}")


if __name__ == "__main__":
    # Example usage:
    compare_benchmarks(
        baseline_file=sys.argv[1],
        current_file=sys.argv[2],
        output_file=sys.argv[3],
        threshold=float(sys.argv[4]),
    )
