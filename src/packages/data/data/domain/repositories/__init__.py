"""Data domain repository interfaces."""

from .data_origin_repository import DataOriginRepository
from .data_source_repository import DataSourceRepository
from .data_asset_repository import DataAssetRepository
from .data_set_repository import DataSetRepository
from .data_element_repository import DataElementRepository

__all__ = [
    "DataOriginRepository",
    "DataSourceRepository",
    "DataAssetRepository",
    "DataSetRepository",
    "DataElementRepository",
]