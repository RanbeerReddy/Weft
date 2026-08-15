"""Merge multiple heads

Revision ID: b591f26ac495
Revises: c780f73d4748, c8a3d4f5e6b7
Create Date: 2026-08-15 22:09:46.177478

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b591f26ac495'
down_revision: Union[str, Sequence[str], None] = ('c780f73d4748', 'c8a3d4f5e6b7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
