"""Basic tests for Data Transfer Objects (DTOs) without external dependencies."""

from __future__ import annotations

def test_dto_structure():
    """Test that DTO files exist and have basic structure."""
    import os
    
    dto_path = "/mnt/c/Users/andre/Pynomaly/src/pynomaly/application/dto"
    
    # Check that DTO files exist
    expected_files = [
        "__init__.py",
        "detector_dto.py", 
        "dataset_dto.py",
        "result_dto.py",
        "experiment_dto.py",
        "automl_dto.py",
        "explainability_dto.py"
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(dto_path, file_name)
        assert os.path.exists(file_path), f"Missing DTO file: {file_name}"
    
    print("✓ All expected DTO files exist")

def test_dto_imports_structure():
    """Test DTO import structure without actually importing."""
    import ast
    
    init_file = "/mnt/c/Users/andre/Pynomaly/src/pynomaly/application/dto/__init__.py"
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Parse the AST to check imports
    tree = ast.parse(content)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names:
                for alias in node.names:
                    imports.append(alias.name)
    
    # Check that key DTOs are being imported
    expected_dtos = [
        "CreateDetectorDTO",
        "DetectorResponseDTO", 
        "DetectionRequestDTO",
        "DatasetDTO",
        "CreateDatasetDTO",
        "DetectionResultDTO",
        "AnomalyDTO",
        "CreateExperimentDTO",
        "AutoMLRequestDTO",
        "AutoMLResponseDTO",
        "FeatureContributionDTO",
        "LocalExplanationDTO"
    ]
    
    for expected_dto in expected_dtos:
        assert expected_dto in imports, f"Missing DTO in __init__.py: {expected_dto}"
    
    print(f"✓ Found {len(imports)} DTO imports in __init__.py")
    print(f"✓ All {len(expected_dtos)} expected core DTOs are imported")

def test_dto_file_structure():
    """Test that each DTO file has proper structure."""
    import ast
    
    dto_files = {
        "detector_dto.py": ["CreateDetectorDTO", "DetectorResponseDTO", "DetectionRequestDTO"],
        "dataset_dto.py": ["DatasetDTO", "CreateDatasetDTO"],
        "result_dto.py": ["DetectionResultDTO", "AnomalyDTO"], 
        "experiment_dto.py": ["CreateExperimentDTO", "ExperimentResponseDTO"],
        "automl_dto.py": ["AutoMLRequestDTO", "AutoMLResponseDTO"],
        "explainability_dto.py": ["FeatureContributionDTO", "LocalExplanationDTO"]
    }
    
    base_path = "/mnt/c/Users/andre/Pynomaly/src/pynomaly/application/dto"
    
    for file_name, expected_classes in dto_files.items():
        file_path = f"{base_path}/{file_name}"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find class definitions
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        # Check that expected classes exist
        for expected_class in expected_classes:
            assert expected_class in classes, f"Missing class {expected_class} in {file_name}"
        
        print(f"✓ {file_name}: Found {len(classes)} classes, expected classes present")

def test_pydantic_usage():
    """Test that DTOs use Pydantic BaseModel."""
    import ast
    
    base_path = "/mnt/c/Users/andre/Pynomaly/src/pynomaly/application/dto"
    dto_files = ["detector_dto.py", "dataset_dto.py", "result_dto.py"]
    
    for file_name in dto_files:
        file_path = f"{base_path}/{file_name}"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for pydantic imports
        assert "from pydantic import BaseModel" in content, f"Missing Pydantic import in {file_name}"
        assert "BaseModel" in content, f"BaseModel not used in {file_name}"
        
        print(f"✓ {file_name}: Uses Pydantic BaseModel")

def test_type_annotations():
    """Test that DTOs have proper type annotations."""
    import ast
    
    base_path = "/mnt/c/Users/andre/Pynomaly/src/pynomaly/application/dto"
    
    file_path = f"{base_path}/detector_dto.py"
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for from __future__ import annotations
    assert "from __future__ import annotations" in content, "Missing future annotations import"
    
    # Check for type imports
    type_imports = ["typing", "UUID", "datetime"]
    for type_import in type_imports:
        assert type_import in content, f"Missing type import: {type_import}"
    
    print("✓ DTOs have proper type annotations and imports")

def main():
    """Run all basic DTO tests."""
    print("Running basic DTO structure tests...\n")
    
    try:
        test_dto_structure()
        test_dto_imports_structure() 
        test_dto_file_structure()
        test_pydantic_usage()
        test_type_annotations()
        
        print("\n=== Summary ===")
        print("✓ All basic DTO structure tests passed")
        print("✓ DTO files exist and have proper structure")
        print("✓ DTOs use Pydantic and proper type annotations")
        print("✓ Import structure is correct")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)