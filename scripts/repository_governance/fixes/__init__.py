"""
Repository governance automated fixes module.
"""

from .auto_fixer import AutoFixer
from .backup_file_fixer import BackupFileFixer
from .domain_leakage_fixer import DomainLeakageFixer
from .structure_fixer import StructureFixer

__all__ = [
    "AutoFixer",
    "BackupFileFixer",
    "DomainLeakageFixer",
    "StructureFixer",
]