"""changing vector column in Embedding table

Revision ID: 49515a1fc02d
Revises: 5ead58e39909
Create Date: 2026-06-09 22:52:13.460249

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "49515a1fc02d"
down_revision: Union[str, Sequence[str], None] = "5ead58e39909"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    No-op: the initial migration (3ee3c270d2c3) already creates
    embedding_vector as pgvector Vector(384).
    """
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
