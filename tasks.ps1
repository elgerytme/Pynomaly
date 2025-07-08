function Build { poetry run python -m build }
function Clean { Remove-Item -Recurse -Force dist,build,.pytest_cache }
function Start-Dev { poetry run uvicorn pynomaly.api:app --reload }
