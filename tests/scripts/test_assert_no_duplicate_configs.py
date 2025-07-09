import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the script to test by modifying the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/validation')))
import assert_no_duplicate_configs

class TestAssertNoDuplicateConfigs(unittest.TestCase):

    def test_script_exists(self):
        """Test that the script file exists and can be imported."""
        self.assertTrue(hasattr(assert_no_duplicate_configs, 'main'))
        
    def test_no_duplicates_simple(self):
        """Test with a simple case of no duplicates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create unique configuration files
            config1 = Path(temp_dir, 'config1.yml')
            config2 = Path(temp_dir, 'config2.yaml')
            config1.write_text('key: value')
            config2.write_text('key: different_value')

            # Run the script with output capture
            original_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Patch the ROOT variable to point to our temp directory
                with patch.object(assert_no_duplicate_configs, 'Path') as mock_path:
                    # Create a mock instance that when called with __file__ returns a path
                    mock_instance = MagicMock()
                    mock_instance.resolve.return_value.parents = [None, None, Path(temp_dir)]
                    mock_path.return_value = mock_instance
                    
                    # Call the main function
                    assert_no_duplicate_configs.main()
                    
                    # Check the output
                    output = captured_output.getvalue()
                    self.assertIn("✅ No duplicate configuration files detected.", output)
                    
            finally:
                sys.stdout = original_stdout

    def test_basic_functionality(self):
        """Test that the main function can be called without errors."""
        # This is a basic smoke test that verifies the function exists and can be called
        with patch('builtins.print') as mock_print:
            try:
                assert_no_duplicate_configs.main()
            except SystemExit:
                # SystemExit is expected when duplicates are found
                pass
        
        # Verify that print was called (either success or failure message)
        self.assertTrue(mock_print.called)

if __name__ == '__main__':
    unittest.main()

