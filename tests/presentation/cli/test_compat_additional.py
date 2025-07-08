"""Additional tests for CLI compatibility module to achieve 100% coverage."""

import pytest
from unittest.mock import Mock, patch

from pynomaly.presentation.cli._compat import (
    list_commands,
    get_command_names,
    get_command,
    has_commands,
    count_commands,
)


class TestAdditionalCoverage:
    """Additional test cases to cover missing lines in _compat module."""

    def test_list_commands_returns_none_case(self):
        """Test case where list_commands should return None."""
        # Test with a mock that returns None for commands
        mock_app = Mock()
        mock_app.registered_commands = None
        # Force commands to return None (line 59)
        mock_app.commands = None
        
        result = list_commands(mock_app)
        assert result == {}  # Should return {} when commands is None
    
    def test_get_command_names_with_none_from_list_commands(self):
        """Test get_command_names when list_commands returns None."""
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=None):
            result = get_command_names(Mock())
            assert result == []
    
    def test_get_command_names_with_keys_method(self):
        """Test get_command_names with object that has keys method but isn't dict."""
        mock_obj = Mock()
        mock_obj.keys.return_value = ['cmd1', 'cmd2']
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = get_command_names(Mock())
            assert result == ['cmd1', 'cmd2']
    
    def test_get_command_names_with_other_object(self):
        """Test get_command_names with object that doesn't have keys method."""
        mock_obj = Mock()
        del mock_obj.keys  # Remove keys method
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = get_command_names(Mock())
            assert result == []
    
    def test_get_command_with_none_from_list_commands(self):
        """Test get_command when list_commands returns None."""
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=None):
            result = get_command(Mock(), "test")
            assert result is None
    
    def test_get_command_with_get_method(self):
        """Test get_command with object that has get method but isn't dict."""
        mock_obj = Mock()
        mock_obj.get.return_value = "test_command"
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = get_command(Mock(), "test")
            assert result == "test_command"
    
    def test_get_command_with_other_object(self):
        """Test get_command with object that doesn't have get method."""
        mock_obj = Mock()
        del mock_obj.get  # Remove get method
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = get_command(Mock(), "test")
            assert result is None
    
    def test_has_commands_with_none_from_list_commands(self):
        """Test has_commands when list_commands returns None."""
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=None):
            result = has_commands(Mock())
            assert result is False
    
    def test_has_commands_with_len_method(self):
        """Test has_commands with object that has __len__ method."""
        mock_obj = Mock()
        mock_obj.__len__ = Mock(return_value=5)
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = has_commands(Mock())
            assert result is True
    
    def test_has_commands_with_empty_len(self):
        """Test has_commands with object that has __len__ method returning 0."""
        mock_obj = Mock()
        mock_obj.__len__ = Mock(return_value=0)
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = has_commands(Mock())
            assert result is False
    
    def test_has_commands_with_no_len_method(self):
        """Test has_commands with object that doesn't have __len__ method."""
        mock_obj = Mock()
        del mock_obj.__len__  # Remove __len__ method
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = has_commands(Mock())
            assert result is False
    
    def test_count_commands_with_none_from_list_commands(self):
        """Test count_commands when list_commands returns None."""
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=None):
            result = count_commands(Mock())
            assert result == 0
    
    def test_count_commands_with_len_method(self):
        """Test count_commands with object that has __len__ method."""
        mock_obj = Mock()
        mock_obj.__len__ = Mock(return_value=3)
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = count_commands(Mock())
            assert result == 3
    
    def test_count_commands_with_no_len_method(self):
        """Test count_commands with object that doesn't have __len__ method."""
        mock_obj = Mock()
        del mock_obj.__len__  # Remove __len__ method
        
        with patch('pynomaly.presentation.cli._compat.list_commands', return_value=mock_obj):
            result = count_commands(Mock())
            assert result == 0
    
    def test_list_commands_with_command_without_name(self):
        """Test list_commands with command object that doesn't have name attribute."""
        mock_cmd = Mock()
        del mock_cmd.name  # Remove name attribute
        
        mock_app = Mock()
        mock_app.registered_commands = [mock_cmd]
        
        result = list_commands(mock_app)
        assert result == {}  # Should be empty dict since no valid commands
    
    def test_list_commands_with_command_with_none_name(self):
        """Test list_commands with command object that has None name."""
        mock_cmd = Mock()
        mock_cmd.name = None
        
        mock_app = Mock()
        mock_app.registered_commands = [mock_cmd]
        
        result = list_commands(mock_app)
        assert result == {}  # Should be empty dict since name is None
    
    def test_list_commands_with_command_with_empty_name(self):
        """Test list_commands with command object that has empty name."""
        mock_cmd = Mock()
        mock_cmd.name = ""
        
        mock_app = Mock()
        mock_app.registered_commands = [mock_cmd]
        
        result = list_commands(mock_app)
        assert result == {}  # Should be empty dict since name is empty
