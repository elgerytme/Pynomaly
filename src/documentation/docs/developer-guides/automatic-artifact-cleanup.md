# Automatic Artifact Cleanup

The Pynomaly project includes an automatic artifact cleanup system that helps maintain a clean repository structure by automatically identifying and removing build artifacts, temporary files, and other clutter.

## Configuration File

The cleanup system is configured via the `.pyno-org.yaml` file in the project root. This file defines:

- **Delete patterns**: Files and directories that should be automatically deleted
- **Allowlist**: Exceptions to the delete patterns
- **Move patterns**: Rules for organizing files into appropriate directories
- **Safety settings**: Confirmation requirements and backup options

## Delete Patterns

The following patterns are automatically marked for deletion:

### Build Artifacts
- `dist/` - Distribution build directory
- `build/` - General build directory
- `*.egg-info` - Python egg info directories
- `__pycache__/` - Python bytecode cache
- `*.pyc`, `*.pyo`, `*.pyd` - Python bytecode files
- `.pytest_cache/` - pytest cache directory

### Log Files
- `*.log` - Log files
- `*.log.*` - Rotated log files
- `logs/` - Log directories

### Temporary Files
- `*.tmp` - Temporary files
- `*.temp` - Temporary files
- `*~` - Backup files (editor generated)
- `*.bak` - Backup files
- `*.backup` - Backup files

### Editor Swap Files
- `.*.swp` - Vim swap files
- `.*.swo` - Vim swap files
- `.*.swn` - Vim swap files
- `*.swp` - Swap files
- `*.swo` - Swap files
- `.#*` - Emacs lock files
- `#*#` - Emacs auto-save files

### Development Artifacts
- `.coverage` - Coverage data
- `.tox/` - Tox virtual environments
- `.mypy_cache/` - MyPy cache
- `htmlcov/` - HTML coverage reports
- `coverage.xml` - Coverage XML reports
- `*.cover` - Coverage files
- `.hypothesis/` - Hypothesis database

### Virtual Environments
- `venv/` - Virtual environment directory
- `.venv/` - Virtual environment directory
- `env/` - Environment directory
- `.env` - Environment file

## Allowlist

The allowlist protects specific files and directories from deletion even if they match delete patterns:

- `docs/build/` - Documentation build directory
- `examples/logs/` - Example log files
- `templates/build/` - Template build files
- `*.log.config` - Log configuration files
- `logging.yaml` - Logging configuration
- `*.tmp.example` - Example temporary files

## Usage

### Command Line Interface

The `pyno-org` script provides commands to validate and organize files:

```bash
# Validate current file organization
python scripts/pyno_org.py validate

# Show what would be organized (dry run)
python scripts/pyno_org.py organize --dry

# Execute file organization
python scripts/pyno_org.py organize --fix

# Execute without confirmation
python scripts/pyno_org.py organize --fix --force
```

### Output Examples

**Validation Output:**
```
[*] Validating file organization...
[-] File organization validation FAILED

[!] Found 39 violations:
  • Stray file in root: test.log
  • Stray directory in root: build/
  • Stray file in root: .main.py.swp
  ...

[!] Suggested fixes:
  • DELETE test.log (artifacts)
  • DELETE build/ (artifacts)
  • DELETE .main.py.swp (artifacts)
  ...
```

**Organization Output:**
```
[*] Planned Operations (5 total):
   1. DELETE: test.log -> DELETED
   2. DELETE: build/ -> DELETED
   3. DELETE: .main.py.swp -> DELETED
   4. MOVE: test_file.py -> tests/test_file.py
   5. MOVE: report.md -> reports/report.md
```

## Categorization Logic

The system categorizes files using a hierarchical approach:

1. **Configuration-based patterns**: Check if file matches configured delete patterns
2. **Allowlist protection**: Skip deletion if file is in allowlist
3. **Move patterns**: Apply configured move patterns for organization
4. **Fallback categorization**: Use built-in categorization rules

### Categories

- `artifacts_for_deletion` - Files/directories marked for automatic deletion
- `testing` - Test files (moved to `tests/`)
- `documentation` - Documentation files (moved to `docs/`)
- `reports` - Report files (moved to `reports/`)
- `scripts` - Script files (moved to `scripts/`)
- `configuration` - Config files (moved to `config/`)
- `miscellaneous` - Requires manual review

## Safety Features

### Confirmation
- Deletion operations require confirmation by default
- Use `--force` flag to skip confirmation
- Dry run mode shows planned operations without executing

### Backup
- Files can be backed up before deletion (configurable)
- Backup directory: `.pyno-org-backup`
- Maximum auto-delete size limit (10MB by default)

### Error Handling
- Graceful handling of permission errors
- Detailed error reporting
- Rollback capability for failed operations

## Configuration Examples

### Adding Custom Delete Patterns

```yaml
delete_patterns:
  - "*.log"
  - "build/"
  - "*.custom-temp"
  - "my-artifacts/"
```

### Adding Allowlist Exceptions

```yaml
allowlist:
  - "docs/build/"
  - "important.log"
  - "scripts/*.tmp"
```

### Custom Move Patterns

```yaml
move_patterns:
  custom_category:
    patterns:
      - "*.custom"
      - "custom-*"
    target: "custom-files/"
    excludes:
      - "custom-important.txt"
```

## Testing

Run the test suite to verify artifact cleanup functionality:

```bash
python tests/test_artifact_cleanup.py
```

The test suite covers:
- Configuration loading
- Pattern matching
- Artifact categorization
- Allowlist protection
- Integration with file organization system

## Best Practices

1. **Review before deletion**: Always run with `--dry` first to see what will be deleted
2. **Backup important files**: Ensure important files are in the allowlist
3. **Test configuration**: Use the test script to validate configuration changes
4. **Regular cleanup**: Run artifact cleanup regularly to maintain clean repository
5. **Team coordination**: Ensure team members understand the cleanup rules

## Troubleshooting

### Common Issues

**Configuration not loading**:
- Ensure `.pyno-org.yaml` is in the project root
- Check YAML syntax with a validator
- Verify PyYAML is installed

**Files not being deleted**:
- Check if files are in the allowlist
- Verify pattern matching with test script
- Ensure correct file path format

**Permission errors**:
- Run with appropriate permissions
- Check file/directory ownership
- Verify write access to target directories

### Debug Mode

For debugging, check the detailed analysis output:

```bash
python scripts/pyno_org.py organize --dry --output debug-report.json
```

This generates a detailed JSON report with categorization details and planned operations.

## Integration with CI/CD

The artifact cleanup can be integrated into CI/CD pipelines:

```yaml
- name: Validate file organization
  run: python scripts/pyno_org.py validate
  
- name: Clean up artifacts
  run: python scripts/pyno_org.py organize --fix --force
```

## Future Enhancements

Planned improvements include:
- Git integration for better change tracking
- Advanced pattern matching (regex support)
- Configurable backup retention policies
- Integration with .gitignore patterns
- Automatic cleanup scheduling
- Team-specific configuration profiles
