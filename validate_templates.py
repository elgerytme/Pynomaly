#!/usr/bin/env python3
"""
Validate Jinja2 templates for syntax errors
"""
import os
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError, meta

def validate_template(template_path):
    """Validate a single template file."""
    try:
        # Read the template content
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Create a Jinja2 environment
        env = Environment(loader=FileSystemLoader(template_path.parent))
        
        # Try to parse the template
        try:
            env.parse(template_content)
            return True, None
        except TemplateSyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
            
    except Exception as e:
        return False, f"Failed to read file: {e}"

def main():
    """Main function to validate all templates."""
    if len(sys.argv) != 2:
        print("Usage: python validate_templates.py <templates_directory>")
        sys.exit(1)
    
    templates_dir = Path(sys.argv[1])
    
    if not templates_dir.exists():
        print(f"Directory {templates_dir} does not exist")
        sys.exit(1)
    
    # Find all HTML template files
    template_files = list(templates_dir.glob("**/*.html"))
    
    if not template_files:
        print("No HTML template files found")
        sys.exit(0)
    
    print(f"Validating {len(template_files)} template files...")
    
    errors = []
    
    for template_file in template_files:
        is_valid, error = validate_template(template_file)
        
        if is_valid:
            print(f"✓ {template_file.name}")
        else:
            print(f"✗ {template_file.name}: {error}")
            errors.append((template_file.name, error))
    
    if errors:
        print(f"\nFound {len(errors)} template(s) with errors:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
        sys.exit(1)
    else:
        print(f"\nAll {len(template_files)} templates are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()
