"""add show_id to projects

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-16

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("show_id", sa.String(), nullable=True))
    op.create_index("ix_projects_show_id", "projects", ["show_id"])


def downgrade() -> None:
    op.drop_index("ix_projects_show_id", table_name="projects")
    op.drop_column("projects", "show_id")
