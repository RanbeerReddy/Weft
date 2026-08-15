"""add idempotency and tsvector trigger

Revision ID: b52e3df2f91a
Revises: ac1aec0d8776
Create Date: 2026-08-15 21:50:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b52e3df2f91a'
down_revision: Union[str, Sequence[str], None] = 'ac1aec0d8776'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add unique constraints to enforce idempotency
    op.create_unique_constraint('uq_chunk_msg_order', 'chunks', ['message_id', 'chunk_order'])
    op.create_unique_constraint('uq_embedding_chunk_id', 'embeddings', ['chunk_order'])

    # Create the trigger for chunk_tsvector updates
    op.execute("""
        CREATE TRIGGER tsvectorupdate 
        BEFORE INSERT OR UPDATE ON chunks 
        FOR EACH ROW EXECUTE FUNCTION 
        tsvector_update_trigger(chunk_tsvector, 'pg_catalog.english', chunk_text);
    """)

    # Backfill missing tsvectors (for chunks inserted between migrations)
    op.execute("UPDATE chunks SET chunk_text = chunk_text WHERE chunk_tsvector IS NULL;")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tsvectorupdate ON chunks;")
    op.drop_constraint('uq_embedding_chunk_id', 'embeddings', type_='unique')
    op.drop_constraint('uq_chunk_msg_order', 'chunks', type_='unique')
