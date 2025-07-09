import pytest
from scripts.analysis.detect_stray_files import detect_stray_files
from scripts.analysis.organize_files import FileOrganizer
from tempfile import TemporaryDirectory
import os

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


def test_detect_stray_files(temp_project_dir):
    # Use detect_stray_files to find issues
    stray_files, suggestions = detect_stray_files([os.path.join(temp_project_dir, 'wrong_location', 'test_sample.py'),
                                                   os.path.join(temp_project_dir, 'should_be_deleted', 'temp_file.tmp')])
    
    assert len(stray_files) == 2
    assert suggestions == [
        f"MOVE wrong_location/test_sample.py from wrong_location to tests",
        f"DELETE should_be_deleted/temp_file.tmp (temporary/artifact file)"
    ]


def test_file_organizer_dry_run(temp_project_dir):
    organizer = FileOrganizer(project_root=temp_project_dir, dry_run=True)
    analysis = organizer.analyze_repository()
    operations = organizer.plan_organization(analysis)
    assert len(operations) == 2

    results = organizer.execute_operations(operations)
    assert results['executed'] == operations


def test_file_organizer_fix(temp_project_dir):
    organizer = FileOrganizer(project_root=temp_project_dir, dry_run=False)
    analysis = organizer.analyze_repository()
    operations = organizer.plan_organization(analysis)

    results = organizer.execute_operations(operations)
    assert len(results['executed']) == 2
    # Check the file was moved
    assert os.path.exists(os.path.join(temp_project_dir, 'tests', 'test_sample.py'))
    # Check the temporary file was deleted
    assert not os.path.exists(os.path.join(temp_project_dir, 'should_be_deleted', 'temp_file.tmp'))

