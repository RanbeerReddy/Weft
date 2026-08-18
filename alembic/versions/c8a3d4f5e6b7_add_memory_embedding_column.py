"""add memory embedding column

Revision ID: c8a3d4f5e6b7
Revises: b52e3df2f91a
Create Date: 2026-08-15 21:55:00.000000

"""

from typing import Sequence, Union

import pgvector
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8a3d4f5e6b7"
down_revision: Union[str, Sequence[str], None] = "b52e3df2f91a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "c780f73d4748"


def upgrade() -> None:
    # Add embedding_vector to memories
    op.add_column(
        "memories",
        sa.Column(
            "embedding_vector",
            pgvector.sqlalchemy.vector.VECTOR(dim=384),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("memories", "embedding_vector")
