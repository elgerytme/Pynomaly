
import pandas as pd
from typing import Any, Dict

from .data_source_adapter import DataSourceAdapter


class PandasCSVAdapter(DataSourceAdapter):
    """Data source adapter for reading CSV files using Pandas."""

    def read_data(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Reads data from a CSV file.

        Args:
            source_config: A dictionary containing 'file_path' for the CSV file.

        Returns:
            A Pandas DataFrame containing the data from the CSV file.

        Raises:
            ValueError: If 'file_path' is not provided in source_config.
        """
        file_path = source_config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in source_config for PandasCSVAdapter.")

        try:
            df = pd.read_csv(file_path, **{k: v for k, v in source_config.items() if k != "file_path"})
            return df
        except Exception as e:
            raise IOError(f"Error reading CSV file {file_path}: {e}")
