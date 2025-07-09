from sqlalchemy import Integer, Column
from sqlalchemy.orm import declared_attr, version_id_col

class VersionedMixin:
    @declared_attr.directive
    def version(cls):
        return Column(Integer, nullable=False, default=1)

    __mapper_args__ = {"version_id_col": version_id_col('version'), "version_id_generator": False}

