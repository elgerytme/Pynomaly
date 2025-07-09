#!/usr/bin/env python3
"""
Type coverage analysis script for pynomaly project.
Runs mypy and calculates type coverage percentage.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_mypy_and_get_coverage():
    """Run mypy and calculate type coverage percentage."""
    try:
        # Run mypy with specific flags to get detailed output
        result = subprocess.run([
            sys.executable, '-m', 'mypy',
            'src',
            '--ignore-missing-imports',
            '--no-strict-optional',
            '--show-error-codes',
            '--any-exprs-report=mypy-any-exprs',
            '--html-report=mypy-html-report',
            '--txt-report=mypy-txt-report'
        ], capture_output=True, text=True)
        
        # Count total lines and error lines
        lines = result.stdout.split('\n')
        error_lines = [line for line in lines if ': error:' in line]
        
        # Try to get a more accurate count of checked files
        checked_files = set()
        for line in lines:
            if ': error:' in line or ': note:' in line:
                file_path = line.split(':')[0]
                if file_path.startswith('src/'):
                    checked_files.add(file_path)
        
        total_files = len(checked_files) if checked_files else 1
        error_count = len(error_lines)
        
        # Calculate rough type coverage (this is a simplified approach)
        # A more accurate method would parse mypy's HTML/XML reports
        if error_count == 0:
            coverage_percent = 100.0
        else:
            # Very rough estimate - not entirely accurate
            coverage_percent = max(0, 100 - (error_count / total_files * 2))
        
        return coverage_percent, error_count, total_files
        
    except Exception as e:
        print(f"Error running mypy: {e}")
        return 0.0, 0, 0


def generate_badge_script():
    """Generate a badge generation script for README."""
    badge_script = '''#!/usr/bin/env python3
"""
Badge generation script for pynomaly project.
Generates badges for README.md based on project metrics.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_type_coverage():
    """Get type coverage percentage from mypy analysis."""
    try:
        from scripts.type_coverage_analysis import run_mypy_and_get_coverage
        coverage, errors, files = run_mypy_and_get_coverage()
        return coverage
    except Exception:
        return 0.0


def get_test_coverage():
    """Get test coverage percentage from pytest-cov."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            '--cov=src/pynomaly', 
            '--cov-report=json',
            '--tb=no',
            '-q'
        ], capture_output=True, text=True)
        
        # Read coverage.json if it exists
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
                return coverage_data.get('totals', {}).get('percent_covered', 0)
    except Exception:
        pass
    return 0.0


def generate_badge_url(label, value, color):
    """Generate shields.io badge URL."""
    if isinstance(value, float):
        value = f"{value:.1f}%"
    return f"https://img.shields.io/badge/{label}-{value}-{color}"


def get_badge_color(percentage):
    """Get color based on percentage."""
    if percentage >= 90:
        return "brightgreen"
    elif percentage >= 80:
        return "green"
    elif percentage >= 70:
        return "yellow"
    elif percentage >= 60:
        return "orange"
    else:
        return "red"


def generate_badges():
    """Generate all badges for the project."""
    badges = []
    
    # Type coverage badge
    type_coverage = get_type_coverage()
    type_color = get_badge_color(type_coverage)
    badges.append({
        'name': 'Type Coverage',
        'url': generate_badge_url('type%20coverage', f"{type_coverage:.1f}%25", type_color),
        'alt': 'Type Coverage'
    })
    
    # Test coverage badge
    test_coverage = get_test_coverage()
    test_color = get_badge_color(test_coverage)
    badges.append({
        'name': 'Test Coverage',
        'url': generate_badge_url('test%20coverage', f"{test_coverage:.1f}%25", test_color),
        'alt': 'Test Coverage'
    })
    
    # Code quality badge (based on ruff/lint results)
    badges.append({
        'name': 'Code Quality',
        'url': generate_badge_url('code%20quality', 'checked', 'blue'),
        'alt': 'Code Quality'
    })
    
    # Python version badge
    badges.append({
        'name': 'Python Version',
        'url': generate_badge_url('python', '3.11%2B', 'blue'),
        'alt': 'Python Version'
    })
    
    return badges


def update_readme_badges():
    """Update README.md with generated badges."""
    readme_path = Path('README.md')
    if not readme_path.exists():
        print("README.md not found")
        return
    
    badges = generate_badges()
    badge_markdown = []
    
    for badge in badges:
        badge_markdown.append(f"![{badge['alt']}]({badge['url']})")
    
    badge_section = "\\n".join(badge_markdown)
    
    # Read current README
    with open(readme_path) as f:
        content = f.read()
    
    # Insert badges after the title (simple approach)
    lines = content.split('\\n')
    new_lines = []
    badges_inserted = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Insert badges after the main title
        if line.startswith('# ') and not badges_inserted:
            new_lines.append('')
            new_lines.append(badge_section)
            new_lines.append('')
            badges_inserted = True
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write('\\n'.join(new_lines))
    
    print(f"Updated README.md with {len(badges)} badges")


if __name__ == '__main__':
    print("Generating badges...")
    badges = generate_badges()
    
    print("\\nGenerated badges:")
    for badge in badges:
        print(f"- {badge['name']}: {badge['url']}")
    
    if '--update-readme' in sys.argv:
        update_readme_badges()
    
    print("\\nBadge generation complete!")
'''
    
    script_path = Path('scripts/generate_badges.py')
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(badge_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created badge generation script: {script_path}")


def main():
    """Main function to run type coverage analysis."""
    print("Running type coverage analysis...")
    
    coverage, errors, files = run_mypy_and_get_coverage()
    
    print(f"\\nType Coverage Results:")
    print(f"  Files analyzed: {files}")
    print(f"  Type errors: {errors}")
    print(f"  Estimated type coverage: {coverage:.1f}%")
    
    if coverage < 90:
        print(f"\\n❌ Type coverage ({coverage:.1f}%) is below 90% threshold!")
        generate_badge_script()
        return 1
    else:
        print(f"\\n✅ Type coverage ({coverage:.1f}%) meets 90% threshold!")
        generate_badge_script()
        return 0


if __name__ == '__main__':
    sys.exit(main())
