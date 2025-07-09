"""
Unit tests for the structure enforcer scanner module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from pynomaly.structure_enforcer.scanner import RepositoryScanner, scan_repository
from pynomaly.structure_enforcer.models import DirectoryNode, FileNode, Model


class TestRepositoryScanner:
    """Test the RepositoryScanner class."""
    
    def test_init(self):
        """Test scanner initialization."""
        root_path = Path("/test/path")
        scanner = RepositoryScanner(root_path)
        
        assert scanner.root_path == root_path
        assert scanner.total_files == 0
        assert scanner.total_directories == 0
        assert scanner.max_depth == 0
        assert scanner.layers == {}
        assert scanner.dependencies == {}
    
    def test_scan_nonexistent_directory(self):
        """Test scanning a non-existent directory."""
        root_path = Path("/nonexistent")
        scanner = RepositoryScanner(root_path)
        
        with pytest.raises(FileNotFoundError):
            scanner.scan()
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_scan_empty_directory(self, mock_iterdir, mock_exists):
        """Test scanning an empty directory."""
        root_path = Path("/test/empty")
        mock_exists.return_value = True
        mock_iterdir.return_value = []
        
        scanner = RepositoryScanner(root_path)
        model = scanner.scan()
        
        assert isinstance(model, Model)
        assert model.root_path == root_path
        assert model.total_files == 0
        assert model.total_directories == 1  # Root directory
        assert model.max_depth == 0
        assert len(model.layers) == 0
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.is_dir')
    @patch('pathlib.Path.stat')
    def test_scan_directory_with_files(self, mock_stat, mock_is_dir, mock_is_file, mock_iterdir, mock_exists):
        """Test scanning a directory with files."""
        root_path = Path("/test/with_files")
        mock_exists.return_value = True
        
        # Mock files
        file1 = Mock()
        file1.name = "test.py"
        file1.is_file.return_value = True
        file1.is_dir.return_value = False
        file1.suffix = ".py"
        
        file2 = Mock()
        file2.name = "README.md"
        file2.is_file.return_value = True
        file2.is_dir.return_value = False
        file2.suffix = ".md"
        
        mock_iterdir.return_value = [file1, file2]
        mock_is_file.side_effect = lambda: True
        mock_is_dir.side_effect = lambda: False
        
        # Mock file stats
        mock_stat.return_value = Mock(st_size=100, st_ctime=1000, st_mtime=1000)
        
        scanner = RepositoryScanner(root_path)
        model = scanner.scan()
        
        assert model.total_files == 2
        assert model.total_directories == 1
        assert len(model.root_directory.files) == 2
        assert len(model.root_directory.subdirectories) == 0
    
    def test_scan_file_creates_file_node(self):
        """Test that _scan_file creates proper FileNode."""
        root_path = Path("/test")
        scanner = RepositoryScanner(root_path)
        
        file_path = Path("/test/example.py")
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=200, st_ctime=2000, st_mtime=2000)
            
            file_node = scanner._scan_file(file_path)
            
            assert isinstance(file_node, FileNode)
            assert file_node.path == file_path
            assert file_node.name == "example.py"
            assert file_node.size == 200
            assert file_node.is_python == True
    
    def test_identify_layers(self):
        """Test layer identification."""
        root_path = Path("/test")
        scanner = RepositoryScanner(root_path)
        
        # Mock directory structure
        with patch.object(scanner, '_find_directory_node') as mock_find:
            pynomaly_node = Mock()
            pynomaly_node.subdirectories = [
                Mock(name="domain"),
                Mock(name="application"),
                Mock(name="infrastructure"),
                Mock(name="presentation"),
            ]
            mock_find.return_value = pynomaly_node
            
            scanner._identify_layers()
            
            assert len(scanner.layers) == 4
            assert "domain" in scanner.layers
            assert "application" in scanner.layers
            assert "infrastructure" in scanner.layers
            assert "presentation" in scanner.layers
    
    def test_analyze_file_dependencies(self):
        """Test analyzing dependencies in a Python file."""
        root_path = Path("/test")
        scanner = RepositoryScanner(root_path)
        
        file_node = Mock()
        file_node.is_python = True
        file_node.path = Path("/test/example.py")
        
        python_code = '''
from pynomaly.domain.entities import Entity
from pynomaly.application.use_cases import UseCase
import external_library
'''
        
        with patch('builtins.open', mock_open(read_data=python_code)):
            deps = scanner._analyze_file_dependencies(file_node)
            
            assert "domain" in deps
            assert "application" in deps
            assert "external_library" not in deps  # Only pynomaly imports tracked


class TestScanRepositoryFunction:
    """Test the scan_repository function."""
    
    def test_scan_repository_with_path(self):
        """Test scanning with explicit path."""
        test_path = Path("/test/path")
        
        with patch.object(RepositoryScanner, 'scan') as mock_scan:
            mock_model = Mock(spec=Model)
            mock_scan.return_value = mock_model
            
            result = scan_repository(test_path)
            
            assert result == mock_model
    
    def test_scan_repository_without_path(self):
        """Test scanning with default current directory."""
        with patch.object(RepositoryScanner, 'scan') as mock_scan:
            with patch('pathlib.Path.cwd') as mock_cwd:
                mock_cwd.return_value = Path("/current/dir")
                mock_model = Mock(spec=Model)
                mock_scan.return_value = mock_model
                
                result = scan_repository()
                
                assert result == mock_model
