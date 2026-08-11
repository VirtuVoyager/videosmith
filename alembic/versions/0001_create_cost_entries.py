"""create cost_entries

Revision ID: 0001
Revises:
Create Date: 2026-08-11

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cost_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("item", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
    )
    op.create_index("ix_cost_entries_project_id", "cost_entries", ["project_id"])
    op.create_index("ix_cost_entries_at", "cost_entries", ["at"])


def downgrade() -> None:
    op.drop_index("ix_cost_entries_at", table_name="cost_entries")
    op.drop_index("ix_cost_entries_project_id", table_name="cost_entries")
    op.drop_table("cost_entries")
