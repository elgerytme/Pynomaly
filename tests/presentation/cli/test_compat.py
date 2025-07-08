"""
Unit tests for CLI compatibility layer.

This module tests the compatibility layer functions that handle
different Typer versions' command attribute access patterns.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import typer

from pynomaly.presentation.cli._compat import (
    count_commands,
    get_command,
    get_command_names,
    has_commands,
    list_commands,
)


class TestListCommands:
    """Test cases for the list_commands function."""

    def test_list_commands_with_registered_commands_attribute(self):
        """Test list_commands with registered_commands attribute (newer Typer)."""
        # Create a mock Typer group with registered_commands as dict
        group = Mock()
        hello_cmd = Mock()
        world_cmd = Mock()
        group.registered_commands = {"hello": hello_cmd, "world": world_cmd}
        group.commands = {"old_hello": Mock()}  # Should be ignored
        
        result = list_commands(group)
        
        assert "hello" in result
        assert "world" in result
        assert result["hello"] == hello_cmd
        assert result["world"] == world_cmd
        assert "old_hello" not in result

    def test_list_commands_with_commands_attribute(self):
        """Test list_commands falling back to commands attribute (older Typer)."""
        # Create a mock Typer group without registered_commands
        group = Mock()
        del group.registered_commands  # Remove the attribute
        hello_cmd = Mock()
        world_cmd = Mock()
        group.commands = {"hello": hello_cmd, "world": world_cmd}
        
        result = list_commands(group)
        
        assert "hello" in result
        assert "world" in result
        assert result["hello"] == hello_cmd
        assert result["world"] == world_cmd

    def test_list_commands_with_no_commands(self):
        """Test list_commands with no commands attribute."""
        # Create a mock Typer group without any command attributes
        group = Mock()
        del group.registered_commands
        del group.commands
        
        result = list_commands(group)
        
        assert result == {}

    def test_list_commands_with_registered_commands_list(self):
        """Test list_commands with registered_commands as list (Typer >= 0.15.1)."""
        # Create mock CommandInfo objects
        hello_cmd_info = Mock()
        hello_cmd_info.name = "hello"
        world_cmd_info = Mock()
        world_cmd_info.name = "world"
        
        group = Mock()
        group.registered_commands = [hello_cmd_info, world_cmd_info]
        
        result = list_commands(group)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "hello" in result
        assert "world" in result
        assert result["hello"] == hello_cmd_info
        assert result["world"] == world_cmd_info

    def test_list_commands_with_real_typer_app(self):
        """Test list_commands with a real Typer app."""
        app = typer.Typer()
        
        @app.command("hello")
        def hello_cmd():
            """Test command."""
            pass
        
        @app.command("world")
        def world_cmd():
            """Another test command."""
            pass
        
        result = list_commands(app)
        
        assert result is not None
        assert isinstance(result, dict)
        # Command names should be present
        command_names = list(result.keys())
        assert "hello" in command_names
        assert "world" in command_names

    def test_list_commands_with_empty_typer_app(self):
        """Test list_commands with an empty Typer app."""
        app = typer.Typer()
        
        result = list_commands(app)
        
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 0


class TestGetCommandNames:
    """Test cases for the get_command_names function."""

    def test_get_command_names_with_dict_commands(self):
        """Test get_command_names with dictionary commands."""
        group = Mock()
        group.registered_commands = {"hello": Mock(), "world": Mock(), "test": Mock()}
        
        result = get_command_names(group)
        
        assert set(result) == {"hello", "world", "test"}

    def test_get_command_names_with_no_commands(self):
        """Test get_command_names with no commands."""
        group = Mock()
        del group.registered_commands
        del group.commands
        
        result = get_command_names(group)
        
        assert result == []

    def test_get_command_names_with_real_typer_app(self):
        """Test get_command_names with a real Typer app."""
        app = typer.Typer()
        
        @app.command("hello")
        def hello_cmd():
            pass
        
        @app.command("world")
        def world_cmd():
            pass
        
        result = get_command_names(app)
        
        assert "hello" in result
        assert "world" in result

    def test_get_command_names_with_non_dict_commands(self):
        """Test get_command_names with non-dictionary commands."""
        group = Mock()
        group.registered_commands = ["hello", "world"]  # Not a dict, no .name attribute
        
        result = get_command_names(group)
        
        # Since these aren't CommandInfo objects with .name attributes,
        # they won't be processed into the dict, so result should be empty
        assert result == []


class TestGetCommand:
    """Test cases for the get_command function."""

    def test_get_command_with_existing_command(self):
        """Test get_command with an existing command."""
        hello_cmd = Mock()
        group = Mock()
        group.registered_commands = {"hello": hello_cmd, "world": Mock()}
        
        result = get_command(group, "hello")
        
        assert result == hello_cmd

    def test_get_command_with_non_existing_command(self):
        """Test get_command with a non-existing command."""
        group = Mock()
        group.registered_commands = {"hello": Mock(), "world": Mock()}
        
        result = get_command(group, "non_existing")
        
        assert result is None

    def test_get_command_with_no_commands(self):
        """Test get_command with no commands."""
        group = Mock()
        del group.registered_commands
        del group.commands
        
        result = get_command(group, "hello")
        
        assert result is None

    def test_get_command_with_real_typer_app(self):
        """Test get_command with a real Typer app."""
        app = typer.Typer()
        
        @app.command("hello")
        def hello_cmd():
            pass
        
        result = get_command(app, "hello")
        
        assert result is not None

    def test_get_command_with_non_dict_commands(self):
        """Test get_command with non-dictionary commands."""
        group = Mock()
        group.registered_commands = ["hello", "world"]  # Not a dict
        
        result = get_command(group, "hello")
        
        assert result is None


class TestHasCommands:
    """Test cases for the has_commands function."""

    def test_has_commands_with_commands(self):
        """Test has_commands with existing commands."""
        group = Mock()
        group.registered_commands = {"hello": Mock(), "world": Mock()}
        
        result = has_commands(group)
        
        assert result is True

    def test_has_commands_with_empty_commands(self):
        """Test has_commands with empty commands."""
        group = Mock()
        group.registered_commands = {}
        
        result = has_commands(group)
        
        assert result is False

    def test_has_commands_with_no_commands(self):
        """Test has_commands with no commands attribute."""
        group = Mock()
        del group.registered_commands
        del group.commands
        
        result = has_commands(group)
        
        assert result is False

    def test_has_commands_with_real_typer_app(self):
        """Test has_commands with a real Typer app."""
        app = typer.Typer()
        
        # Initially no commands
        assert has_commands(app) is False
        
        @app.command("hello")
        def hello_cmd():
            pass
        
        # Now has commands
        assert has_commands(app) is True

    def test_has_commands_with_non_dict_commands(self):
        """Test has_commands with non-dictionary commands."""
        group = Mock()
        group.registered_commands = ["hello", "world"]  # Not a dict, no .name attribute
        
        result = has_commands(group)
        
        # Since these aren't CommandInfo objects with .name attributes,
        # they won't be added to the dict, so result should be False
        assert result is False


class TestCountCommands:
    """Test cases for the count_commands function."""

    def test_count_commands_with_commands(self):
        """Test count_commands with existing commands."""
        group = Mock()
        group.registered_commands = {"hello": Mock(), "world": Mock(), "test": Mock()}
        
        result = count_commands(group)
        
        assert result == 3

    def test_count_commands_with_empty_commands(self):
        """Test count_commands with empty commands."""
        group = Mock()
        group.registered_commands = {}
        
        result = count_commands(group)
        
        assert result == 0

    def test_count_commands_with_no_commands(self):
        """Test count_commands with no commands attribute."""
        group = Mock()
        del group.registered_commands
        del group.commands
        
        result = count_commands(group)
        
        assert result == 0

    def test_count_commands_with_real_typer_app(self):
        """Test count_commands with a real Typer app."""
        app = typer.Typer()
        
        # Initially no commands
        assert count_commands(app) == 0
        
        @app.command("hello")
        def hello_cmd():
            pass
        
        @app.command("world")
        def world_cmd():
            pass
        
        # Now has 2 commands
        assert count_commands(app) == 2

    def test_count_commands_with_non_dict_commands(self):
        """Test count_commands with non-dictionary commands."""
        group = Mock()
        group.registered_commands = ["hello", "world"]  # Not a dict, no .name attribute
        
        result = count_commands(group)
        
        # Since these aren't CommandInfo objects with .name attributes,
        # they won't be added to the dict, so result should be 0
        assert result == 0


class TestCompatibilityScenarios:
    """Test cases for various compatibility scenarios."""

    def test_newer_typer_version_simulation(self):
        """Test simulation of newer Typer version behavior."""
        # Simulate newer Typer version with registered_commands
        app = typer.Typer()
        
        @app.command("hello")
        def hello_cmd():
            pass
        
        # Mock the newer attribute structure
        with patch.object(app, 'registered_commands', {"hello": Mock(), "world": Mock()}):
            result = list_commands(app)
            assert "hello" in result
            assert "world" in result

    def test_older_typer_version_simulation(self):
        """Test simulation of older Typer version behavior."""
        # Simulate older Typer version with commands only
        app = typer.Typer()
        
        # Mock the older attribute structure
        with patch.object(app, 'registered_commands', None):
            with patch.object(app, 'commands', {"hello": Mock(), "world": Mock()}, create=True):
                result = list_commands(app)
                assert "hello" in result
                assert "world" in result

    def test_fallback_behavior(self):
        """Test fallback behavior when both attributes are missing."""
        app = typer.Typer()
        
        # Mock both attributes as missing
        with patch.object(app, 'registered_commands', None):
            with patch.object(app, 'commands', None, create=True):
                result = list_commands(app)
                assert result == {}

    def test_edge_case_with_none_commands(self):
        """Test edge case where commands attribute is None."""
        group = Mock()
        group.registered_commands = None
        group.commands = None
        
        result = list_commands(group)
        assert result == {}

    def test_edge_case_with_non_standard_commands(self):
        """Test edge case with non-standard commands structure."""
        group = Mock()
        group.registered_commands = Mock()
        group.registered_commands.keys = lambda: ["hello", "world"]
        group.registered_commands.get = lambda name: Mock() if name in ["hello", "world"] else None
        
        names = get_command_names(group)
        assert "hello" in names
        assert "world" in names
        
        cmd = get_command(group, "hello")
        assert cmd is not None


class TestIntegrationWithRealTyper:
    """Integration tests with real Typer instances."""

    def test_integration_with_complex_typer_app(self):
        """Test integration with a complex Typer app structure."""
        app = typer.Typer()
        
        @app.command("hello")
        def hello_cmd(name: str = "World"):
            """Say hello."""
            print(f"Hello {name}!")
        
        @app.command("goodbye")
        def goodbye_cmd(name: str = "World"):
            """Say goodbye."""
            print(f"Goodbye {name}!")
        
        # Test all compatibility functions
        assert has_commands(app) is True
        assert count_commands(app) == 2
        
        command_names = get_command_names(app)
        assert "hello" in command_names
        assert "goodbye" in command_names
        
        hello_cmd_obj = get_command(app, "hello")
        assert hello_cmd_obj is not None
        
        commands = list_commands(app)
        assert commands is not None
        assert len(commands) == 2

    def test_integration_with_nested_typer_apps(self):
        """Test integration with nested Typer apps."""
        main_app = typer.Typer()
        sub_app = typer.Typer()
        
        @sub_app.command("sub-hello")
        def sub_hello():
            """Sub command."""
            pass
        
        @main_app.command("main-hello")
        def main_hello():
            """Main command."""
            pass
        
        main_app.add_typer(sub_app, name="sub")
        
        # Test main app
        assert has_commands(main_app) is True
        main_commands = get_command_names(main_app)
        assert "main-hello" in main_commands
        
        # Test sub app
        assert has_commands(sub_app) is True
        sub_commands = get_command_names(sub_app)
        assert "sub-hello" in sub_commands
