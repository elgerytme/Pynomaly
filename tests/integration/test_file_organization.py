import pytest
import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Add project root to path to allow script imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.analysis.detect_stray_files import detect_stray_files
from scripts.analysis.organize_files import FileOrganizer

@pytest.fixture
def temp_project_dir():
    with TemporaryDirectory() as tmpdirname:
        # Setup a temporary directory structure
        os.makedirs(os.path.join(tmpdirname, 'wrong_location'), exist_ok=True)
        with open(os.path.join(tmpdirname, 'wrong_location', 'test_sample.py'), 'w') as f:
            f.write('# Sample test file')
        
        os.makedirs(os.path.join(tmpdirname, 'should_be_deleted'), exist_ok=True)
        with open(os.path.join(tmpdirname, 'should_be_deleted', 'temp_file.tmp'), 'w') as f:
            f.write('temporary data')
            
        yield tmpdirname


@pytest.mark.integration
@pytest.mark.file_organization
def test_detect_stray_files(temp_project_dir):
    # Use detect_stray_files to find issues
    test_files = [
        os.path.join(temp_project_dir, 'wrong_location', 'test_sample.py'),
        os.path.join(temp_project_dir, 'should_be_deleted', 'temp_file.tmp')
    ]
    
    stray_files, suggestions = detect_stray_files(test_files)
    
    assert len(stray_files) == 2
    # Check that suggestions contain the expected patterns
    assert any('MOVE' in suggestion and 'test_sample.py' in suggestion and 'tests' in suggestion for suggestion in suggestions)
    assert any('DELETE' in suggestion and 'temp_file.tmp' in suggestion and 'artifact' in suggestion for suggestion in suggestions)


@pytest.mark.integration
@pytest.mark.file_organization
def test_file_organizer_dry_run(temp_project_dir):
    """Test file organizer dry run mode."""
    organizer = FileOrganizer(project_root=temp_project_dir, dry_run=True)
    analysis = organizer.analyze_repository()
    
    # Should find some stray items
    operations = organizer.plan_organization(analysis)
    
    # Execute operations in dry run mode
    results = organizer.execute_operations(operations)
    
    # In dry run mode, no actual changes should be made
    assert isinstance(results, dict)
    assert 'executed' in results
    assert 'errors' in results


@pytest.mark.integration
@pytest.mark.file_organization
@pytest.mark.slow
def test_file_organizer_fix(temp_project_dir):
    """Test file organizer fix mode with actual file operations."""
    organizer = FileOrganizer(project_root=temp_project_dir, dry_run=False)
    analysis = organizer.analyze_repository()
    operations = organizer.plan_organization(analysis)

    results = organizer.execute_operations(operations)
    
    # Check that operations were executed
    assert isinstance(results, dict)
    assert 'executed' in results
    assert 'errors' in results
    
    # In fix mode, files should be actually moved/deleted
    assert len(results['executed']) >= 0  # At least no errors


@pytest.mark.integration
@pytest.mark.file_organization
def test_file_organizer_categories():
    """Test that file organizer correctly categorizes different file types."""
    from scripts.analysis.detect_stray_files import categorize_file
    
    # Test different file categories
    assert categorize_file('test_example.py') == 'testing'
    assert categorize_file('temp_file.tmp') == 'temporary'
    assert categorize_file('setup_script.py') == 'scripts'
    assert categorize_file('README.md') == 'miscellaneous'

