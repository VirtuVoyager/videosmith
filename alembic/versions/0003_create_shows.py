"""create shows

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-16

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shows",
        sa.Column("show_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("style_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_shows_created_at", "shows", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_shows_created_at", table_name="shows")
    op.drop_table("shows")
