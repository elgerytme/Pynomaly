import pytest
from pathlib import Path
from scripts.validation.auto_fix_root_directory import RootDirectoryFixer

@pytest.fixture
def setup_tmp_path(tmp_path):
    # Set up initial directory structure and files
    (tmp_path / "baseline_outputs").mkdir()
    (tmp_path / "developer-guides").mkdir()
    (tmp_path / "script_example.sh").touch()
    (tmp_path / "Dockerfile").touch()
    (tmp_path / "coverage.xml").touch()
    (tmp_path / "README.md").touch()
    return tmp_path

def test_documentation_files_moved(setup_tmp_path):
    fixer = RootDirectoryFixer(project_root=setup_tmp_path, dry_run=True)
    fixer._fix_documentation_files()

    expected_docs = setup_tmp_path / "docs" / "project" / "README.md"
    assert expected_docs.exists()


def test_script_files_moved(setup_tmp_path):
    fixer = RootDirectoryFixer(project_root=setup_tmp_path, dry_run=True)
    fixer._fix_script_files()

    expected_script = setup_tmp_path / "scripts" / "deployment" / "Dockerfile"
    assert expected_script.exists()


def test_config_files_moved(setup_tmp_path):
    fixer = RootDirectoryFixer(project_root=setup_tmp_path, dry_run=True)
    fixer._fix_configuration_files()

    expected_config = setup_tmp_path / "config" / "coverage.xml"
    assert expected_config.exists()


def test_directories_handled(setup_tmp_path):
    fixer = RootDirectoryFixer(project_root=setup_tmp_path, dry_run=True)
    fixer._fix_directories()

    expected_baseline_output = setup_tmp_path / "baseline_outputs"
    assert expected_baseline_output.exists()

    expected_developer_guide = setup_tmp_path / "developer-guides"
    assert expected_developer_guide.exists()

