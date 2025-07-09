"""0002_add_versioning

Revision ID: 9c17f4999523
Revises: 
Create Date: 2025-07-09 10:17:47.374203

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c17f4999523'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite' and conn.dialect.server_version_info < (3, 35):
        # Logic for recreating tables
        # Implement the detailed recreation logic here, considering alembic operations
        op.drop_table('your_table_name')  # Replace with your table's name
        op.create_table('your_table_name',  # Replace with creation logic
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('version', sa.Integer, nullable=False, server_default=sa.text("1"))
        )
    else:
        op.add_column('your_table_name', sa.Column('version', sa.Integer, nullable=False, server_default=sa.text("1")))
    op.execute('UPDATE your_table_name SET version = 1')  # Set default version for existing rows


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    if conn.dialect.name == 'sqlite' and conn.dialect.server_version_info < (3, 35):
        # Logic for recreating tables
        # Implement the detailed recreation logic here, considering alembic operations
        op.drop_table('your_table_name')  # Replace with your table's name
        op.create_table('your_table_name',  # Replace with your table's creation logic
            sa.Column('id', sa.Integer, primary_key=True)  # and other existing columns
        )
    else:
        op.drop_column('your_table_name', 'version')
