#!/usr/bin/env python3
"""
Explore registered_groups and their contents for diagnostic purposes.
"""

from pynomaly.presentation.cli.app import app

# Check type and contents of registered_groups
print(f"registered_groups type: {type(app.registered_groups)}")

for idx, group in enumerate(app.registered_groups):
    print(f"Group {idx}: {group}")
    if hasattr(group, 'name'):
        print(f"  Name: {group.name}")
    if hasattr(group, 'typer_instance'):
        print(f"  Typer instance type: {type(group.typer_instance)}")
        if hasattr(group.typer_instance, 'registered_commands'):
            print(f"  Registered commands: {len(group.typer_instance.registered_commands)}")
    else:
        print("  ❌ No Typer instance attribute!")
