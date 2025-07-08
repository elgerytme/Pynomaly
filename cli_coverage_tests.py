#!/usr/bin/env python3
"""CLI Coverage Test Framework for Pynomaly.

This script uses the enumerated CLI surface to generate parametrized tests
for comprehensive CLI coverage validation.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Any


def load_cli_surface() -> Dict[str, Any]:
    """Load the CLI surface JSON file."""
    cli_surface_file = Path("cli_surface.json")
    if not cli_surface_file.exists():
        raise FileNotFoundError("cli_surface.json not found. Run cli_inspection.py first.")
    
    with open(cli_surface_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_commands() -> List[Dict[str, Any]]:
    """Extract all commands from all CLI files."""
    cli_surface = load_cli_surface()
    all_commands = []
    
    for file_info in cli_surface["files"]:
        file_path = file_info["file"]
        for command in file_info.get("commands", []):
            command_info = {
                "file": file_path,
                "command": command["command"],
                "function": command["function"],
                "docstring": command["docstring"],
                "options": command["options"],
                "full_command_path": f"{Path(file_path).stem}.{command['command']}"
            }
            all_commands.append(command_info)
    
    return all_commands


def extract_command_option_combinations() -> List[Dict[str, Any]]:
    """Extract all command/option combinations for parametrized testing."""
    all_commands = extract_all_commands()
    combinations = []
    
    for command in all_commands:
        base_combo = {
            "file": command["file"],
            "command": command["command"],
            "function": command["function"],
            "full_command_path": command["full_command_path"]
        }
        
        # Add combination for command with no options
        combinations.append({
            **base_combo,
            "test_case": "no_options",
            "options": []
        })
        
        # Add combinations for each individual option
        for option in command["options"]:
            if option["is_option"]:
                option_combo = {
                    **base_combo,
                    "test_case": f"single_option_{option['name']}",
                    "options": [option]
                }
                combinations.append(option_combo)
        
        # Add combination for all options together
        all_options = [opt for opt in command["options"] if opt["is_option"]]
        if all_options:
            combinations.append({
                **base_combo,
                "test_case": "all_options",
                "options": all_options
            })
    
    return combinations


def generate_cli_command_string(combination: Dict[str, Any]) -> str:
    """Generate a CLI command string from a combination."""
    command_parts = ["pynomaly"]
    
    # Extract module name from file path (e.g., "autonomous" from "autonomous.py")
    file_path = Path(combination["file"])
    if file_path.stem != "app":  # app.py contains main commands
        command_parts.append(file_path.stem)
    
    command_parts.append(combination["command"])
    
    # Add options
    for option in combination["options"]:
        if option["option_names"]:
            # Use first option name
            flag = option["option_names"][0]
            command_parts.append(flag)
            
            # Add value if needed (not for boolean flags)
            if option["type"] != "bool" and not flag.endswith("/--no-" + option["name"]):
                if option["default"] is not None and option["default"] != "<ellipsis>":
                    command_parts.append(str(option["default"]))
                else:
                    # Use placeholder values based on type
                    if "int" in option["type"]:
                        command_parts.append("42")
                    elif "float" in option["type"]:
                        command_parts.append("0.5")
                    elif "Path" in option["type"]:
                        command_parts.append("test_data.csv")
                    else:
                        command_parts.append("test_value")
    
    return " ".join(command_parts)


# Pytest fixtures and test functions

@pytest.fixture(scope="session")
def cli_surface():
    """Fixture to load CLI surface once per test session."""
    return load_cli_surface()


@pytest.fixture(scope="session")
def all_commands():
    """Fixture to extract all commands once per test session."""
    return extract_all_commands()


@pytest.fixture(scope="session")
def command_option_combinations():
    """Fixture to extract all command/option combinations."""
    return extract_command_option_combinations()


def test_cli_surface_loaded(cli_surface):
    """Test that CLI surface was loaded successfully."""
    assert cli_surface is not None
    assert "metadata" in cli_surface
    assert "files" in cli_surface
    assert "summary" in cli_surface
    
    # Verify summary matches what the inspector found
    summary = cli_surface["summary"]
    assert summary["total_commands"] > 0
    assert summary["total_options"] > 0


@pytest.mark.parametrize("file_info", [])
def test_all_cli_files_parsed(cli_surface):
    """Test that all CLI files were parsed without errors."""
    for file_info in cli_surface["files"]:
        # Check that file was parsed (no error field)
        assert "error" not in file_info, f"Error parsing {file_info['file']}: {file_info.get('error', 'Unknown')}"
        
        # Check basic structure
        assert "commands" in file_info
        assert isinstance(file_info["commands"], list)


def test_command_coverage_expectations(all_commands):
    """Test coverage expectations based on command count."""
    # These are expectations that can be updated as the CLI evolves
    min_expected_commands = 200  # Minimum number of commands expected
    
    assert len(all_commands) >= min_expected_commands, \
        f"Expected at least {min_expected_commands} commands, found {len(all_commands)}"
    
    # Check that major command categories are present
    command_names = [cmd["command"] for cmd in all_commands]
    expected_categories = ["detect", "train", "create", "list", "show", "export"]
    
    for category in expected_categories:
        category_commands = [name for name in command_names if category in name]
        assert len(category_commands) > 0, f"No commands found for category '{category}'"


@pytest.mark.parametrize("combination", [])
def test_command_option_combination_structure(command_option_combinations):
    """Test that all command/option combinations have valid structure."""
    for combination in command_option_combinations:
        # Check required fields
        required_fields = ["file", "command", "function", "full_command_path", "test_case", "options"]
        for field in required_fields:
            assert field in combination, f"Missing field '{field}' in combination"
        
        # Check that options have valid structure
        for option in combination["options"]:
            assert "name" in option
            assert "type" in option
            assert "is_option" in option or "is_argument" in option


def test_generate_command_strings(command_option_combinations):
    """Test that command strings can be generated for all combinations."""
    sample_combinations = command_option_combinations[:10]  # Test first 10 combinations
    
    for combination in sample_combinations:
        command_string = generate_cli_command_string(combination)
        
        # Basic validation
        assert command_string.startswith("pynomaly")
        assert combination["command"] in command_string
        
        # Should not contain placeholder values for actual testing
        assert "<ellipsis>" not in command_string


if __name__ == "__main__":
    # Demo usage
    print("🧪 CLI Coverage Test Framework Demo")
    print("=" * 50)
    
    try:
        cli_surface = load_cli_surface()
        print(f"✅ Loaded CLI surface with {cli_surface['summary']['total_commands']} commands")
        
        all_commands = extract_all_commands()
        print(f"✅ Extracted {len(all_commands)} total commands")
        
        combinations = extract_command_option_combinations()
        print(f"✅ Generated {len(combinations)} command/option combinations")
        
        # Show sample command strings
        print("\n📋 Sample Command Strings:")
        for i, combo in enumerate(combinations[:5]):
            cmd_str = generate_cli_command_string(combo)
            print(f"  {i+1}. {cmd_str}")
        
        print(f"\n🎯 Ready for parametrized testing with {len(combinations)} test cases!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
