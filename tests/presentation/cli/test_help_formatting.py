"""Tests for CLI help formatting improvements."""

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app
from pynomaly.presentation.cli.detectors import app as detector_app
from pynomaly.presentation.cli.help_formatter import (
    STANDARD_HELP_TEXTS,
    STANDARD_OPTION_HELP,
    HelpFormatter,
    get_option_help,
    get_standard_help,
)


class TestHelpFormatter:
    """Test the HelpFormatter utility class."""

    def test_format_command_help_basic(self):
        """Test basic command help formatting."""
        help_text = HelpFormatter.format_command_help(
            "This is a test command description."
        )
        assert "This is a test command description." in help_text

    def test_format_command_help_with_examples(self):
        """Test command help formatting with examples."""
        examples = ["example command 1", "example command 2"]
        help_text = HelpFormatter.format_command_help(
            "Test description", usage_examples=examples
        )

        assert "Test description" in help_text
        assert "Examples:" in help_text
        assert "example command 1" in help_text
        assert "example command 2" in help_text

    def test_format_command_help_with_notes(self):
        """Test command help formatting with notes."""
        help_text = HelpFormatter.format_command_help(
            "Test description", notes="This is a note"
        )

        assert "Test description" in help_text
        assert "Note: This is a note" in help_text

    def test_format_option_help(self):
        """Test option help formatting."""
        help_text = HelpFormatter.format_option_help(
            "This is a very long option description that should be wrapped to the specified width"
        )
        # Should wrap text
        assert len(help_text.split("\n")[0]) <= 60

    def test_create_examples_table(self):
        """Test examples table creation."""
        examples = {"command1": "Description 1", "command2": "Description 2"}
        table = HelpFormatter.create_examples_table(examples)
        assert table.title == "Usage Examples"

    def test_create_tips_panel(self):
        """Test tips panel creation."""
        tips = ["Tip 1", "Tip 2"]
        panel = HelpFormatter.create_tips_panel(tips)
        assert "💡 Tips" in str(panel)


class TestStandardHelpTexts:
    """Test standardized help text functions."""

    def test_get_standard_help_existing_command(self):
        """Test getting help for existing command."""
        help_info = get_standard_help("detector")
        assert "help" in help_info
        assert "description" in help_info
        assert "examples" in help_info
        assert "🔧" in help_info["help"]

    def test_get_standard_help_non_existing_command(self):
        """Test getting help for non-existing command."""
        help_info = get_standard_help("nonexistent")
        assert "help" in help_info
        assert "description" in help_info
        assert "examples" in help_info
        assert "nonexistent" in help_info["description"]

    def test_get_option_help_existing_option(self):
        """Test getting help for existing option."""
        help_text = get_option_help("verbose")
        assert "detailed output" in help_text.lower()

    def test_get_option_help_non_existing_option(self):
        """Test getting help for non-existing option."""
        help_text = get_option_help("nonexistent")
        assert "nonexistent" in help_text

    def test_standard_help_texts_structure(self):
        """Test that standard help texts have required structure."""
        for command, help_info in STANDARD_HELP_TEXTS.items():
            assert "help" in help_info
            assert "description" in help_info
            assert "examples" in help_info
            assert isinstance(help_info["examples"], list)
            assert len(help_info["examples"]) > 0

    def test_standard_option_help_coverage(self):
        """Test that standard option help covers common options."""
        common_options = [
            "verbose",
            "quiet",
            "output",
            "format",
            "force",
            "algorithm",
            "contamination",
            "cv",
            "folds",
        ]
        for option in common_options:
            assert option in STANDARD_OPTION_HELP
            assert len(STANDARD_OPTION_HELP[option]) > 10


class TestCLIHelpIntegration:
    """Test CLI help formatting integration."""

    def setUp(self):
        self.runner = CliRunner()

    def test_main_app_help_formatting(self):
        """Test main app help formatting."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        # Should not fail
        assert result.exit_code == 0

        # Should contain standardized help text
        assert "Pynomaly - State-of-the-art anomaly detection" in result.output
        assert "🔧 Create and manage anomaly detectors" in result.output
        assert "📊 Load and manage datasets" in result.output

    def test_detector_subcommand_help_formatting(self):
        """Test detector subcommand help formatting."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["--help"])

        # Should not fail
        assert result.exit_code == 0

        # Should contain standardized help text
        assert "🔧 Create and manage anomaly detectors" in result.output

        # Should have examples command
        assert "examples" in result.output.lower()

    def test_detector_list_help_formatting(self):
        """Test detector list command help formatting."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["list", "--help"])

        # Should not fail
        assert result.exit_code == 0

        # Should contain examples
        assert "Examples:" in result.output
        assert "pynomaly detector list" in result.output

        # Should have organized option groups
        assert "Filters" in result.output
        assert "Display Options" in result.output

    def test_detector_examples_command(self):
        """Test detector examples command."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["examples"])

        # Should not fail
        assert result.exit_code == 0

        # Should contain formatted examples
        assert "Detector Management Examples" in result.output
        assert "pynomaly detector create" in result.output

    def test_help_consistency_across_commands(self):
        """Test that help formatting is consistent across commands."""
        runner = CliRunner()

        # Test multiple subcommands
        subcommands = ["detector", "dataset", "auto"]

        for subcmd in subcommands:
            result = runner.invoke(app, [subcmd, "--help"])
            assert result.exit_code == 0

            # All should have emojis in help text (standardized format)
            lines = result.output.split("\n")
            help_line = next(
                (line for line in lines if subcmd in line and "─" not in line), ""
            )

            # Most standardized commands should have emojis
            if subcmd in ["detector", "dataset"]:
                # These have emojis in standardized help
                assert any(char in help_line for char in ["🔧", "📊", "🤖", "🧠"])


class TestRichFormattingFeatures:
    """Test Rich formatting features."""

    def test_rich_markup_in_help(self):
        """Test that Rich markup is properly handled."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["list", "--help"])

        # Should contain Rich markup
        assert "Examples:" in result.output
        assert "pynomaly detector list" in result.output

    def test_option_grouping(self):
        """Test that options are properly grouped."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["list", "--help"])

        # Should have grouped options
        assert "Filters" in result.output
        assert "Display Options" in result.output
        assert "--algorithm" in result.output
        assert "--format" in result.output

    def test_help_panels_formatting(self):
        """Test that help panels are properly formatted."""
        runner = CliRunner()
        result = runner.invoke(detector_app, ["--help"])

        # Should have proper panel formatting
        assert "Commands" in result.output
        assert "Options" in result.output

        # Commands should be organized
        assert "create" in result.output
        assert "list" in result.output
        assert "examples" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
