import pandas as pd
from typing import Any

class FileAdapter:
    def load(self, path: str) -> pd.DataFrame:
        """Load data from the given path into a pandas DataFrame."""
        raise NotImplementedError("load must be implemented by subclasses")

class CSVAdapter(FileAdapter):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

class JSONAdapter(FileAdapter):
    def load(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)

class ParquetAdapter(FileAdapter):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

def get_file_adapter(path: str) -> FileAdapter:
    """Return an appropriate FileAdapter based on file extension."""
    lower = path.lower()
    if lower.endswith('.csv'):
        return CSVAdapter()
    if lower.endswith('.json'):
        return JSONAdapter()
    if lower.endswith('.parquet'):
        return ParquetAdapter()
    raise ValueError(f"Unsupported file format: {path}")