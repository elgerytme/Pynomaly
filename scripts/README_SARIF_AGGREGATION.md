# SARIF Aggregation Script

This directory contains the `aggregate_sarif.py` script that merges multiple SARIF (Static Analysis Results Interchange Format) files into a single combined file.

## Overview

The SARIF aggregation script is designed to:
- Accept multiple SARIF files as input
- Merge their `runs` arrays into a single combined SARIF file
- Provide robust error handling for missing or invalid files
- Work with both GitHub Actions and local development workflows

## Usage

### Command Line Interface

```bash
# Basic usage
python scripts/aggregate_sarif.py file1.sarif file2.sarif file3.sarif

# Specify output file
python scripts/aggregate_sarif.py --output combined.sarif *.sarif

# Using short option
python scripts/aggregate_sarif.py -o security-results/combined.sarif reports/*.sarif
```

### Makefile Target

```bash
# Aggregate specific files
make aggregate-sarif SARIF_FILES="bandit.sarif pip-audit.sarif"

# Aggregate with custom output
make aggregate-sarif SARIF_FILES="*.sarif" OUTPUT="security-combined.sarif"
```

### GitHub Actions Integration

The script is automatically used in the `security-summary` job of the GitHub Actions workflow:

```yaml
- name: Aggregate SARIF files
  run: |
    python scripts/aggregate_sarif.py --output security-results/combined-security.sarif $SARIF_FILES
```

## Features

- **Robust Error Handling**: Gracefully handles missing files and invalid JSON
- **SARIF 2.1.0 Compliance**: Generates output compliant with SARIF 2.1.0 schema
- **Flexible Input**: Accepts any number of SARIF files as input
- **Verbose Output**: Provides detailed information about processing progress
- **Cross-platform**: Works on Windows, macOS, and Linux

## Output Format

The script generates a combined SARIF file with the following structure:

```json
{
  "version": "2.1.0",
  "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
  "runs": [
    // All runs from input files merged here
  ]
}
```

## Testing

Run the test suite to verify functionality:

```bash
python scripts/test_aggregate_sarif.py
```

## Integration with Security Workflows

### GitHub Actions

The script is integrated into the `security-summary` job in `.github/workflows/security.yml`:

1. Downloads all security scanning artifacts
2. Finds all SARIF files in the artifacts
3. Aggregates them into a single `combined-security.sarif` file
4. Uploads the combined file to GitHub's Security tab

### Local Development

Use the Makefile target for local SARIF aggregation:

```bash
# After running security scans
make aggregate-sarif SARIF_FILES="artifacts/security/*.sarif"
```

## Error Handling

The script handles various error conditions:

- **Missing files**: Warns and continues processing other files
- **Invalid JSON**: Reports JSON parsing errors and skips invalid files
- **Empty runs**: Processes files with empty runs arrays
- **File access errors**: Reports permission or I/O errors

## Security Considerations

- The script only reads and writes local files
- No network operations are performed
- Input validation ensures only valid SARIF structures are processed
- Output is always valid JSON with proper escaping

## Supported Tools

The aggregation script works with SARIF output from various security tools:

- **Bandit**: Python security linter
- **pip-audit**: Python package vulnerability scanner
- **Semgrep**: Static analysis tool
- **CodeQL**: GitHub's code analysis
- **Trivy**: Container security scanner
- Any tool that produces SARIF 2.1.0 compliant output

## Troubleshooting

### Common Issues

1. **"No input files specified"**: Ensure you provide at least one SARIF file
2. **"File not found"**: Check that the specified SARIF files exist
3. **"Invalid JSON"**: Verify that input files are valid JSON format
4. **"Permission denied"**: Ensure you have read access to input files and write access to output directory

### Debug Mode

For detailed debugging information, examine the script output which includes:
- Files being processed
- Number of runs in each file
- Total runs in the combined output
- Warnings for any issues encountered

## Contributing

When modifying the aggregation script:

1. Update the test suite in `test_aggregate_sarif.py`
2. Run tests to ensure functionality is preserved
3. Update this documentation if adding new features
4. Consider backward compatibility with existing workflows
