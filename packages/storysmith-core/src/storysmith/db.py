from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import Float, String, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from storysmith.models import CostEntry


class Base(DeclarativeBase):
    pass


class CostEntryRow(Base):
    """Postgres-persisted mirror of CostEntry, in addition to state.cost_ledger
    (§8) -- queryable across runs/processes, unlike the in-memory ledger a
    single VideoProject carries."""

    __tablename__ = "cost_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String, index=True)
    at: Mapped[datetime] = mapped_column(index=True)
    item: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    cost_usd: Mapped[float] = mapped_column(Float)


class ProjectRow(Base):
    """Cheap queryable snapshot of the latest known VideoProject state, kept
    in sync from the same node-wrapper choke point as CostEntryRow (§7/§8).

    The FastAPI review console (apps/api) lists/looks up projects from this
    table rather than reaching into LangGraph's checkpoint tables directly --
    those are keyed by thread_id/checkpoint_id and aren't meant to be queried
    ad hoc (e.g. "all projects with status=review"); this table is.
    """

    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    mode: Mapped[str] = mapped_column(String)
    brief: Mapped[str] = mapped_column(String)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    total_cost_usd: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(index=True)


def to_psycopg_dsn(db_url: str) -> str:
    """AsyncPostgresSaver (psycopg3) wants a bare postgresql:// DSN; our own
    SQLAlchemy engine uses the postgresql+psycopg:// dialect string. Both
    settings.db_url values are the same underlying DSN, just spelled
    differently for the two libraries -- this is the one place that
    difference is bridged."""
    return db_url.replace("postgresql+psycopg://", "postgresql://", 1)


_engine_cache: dict[str, AsyncEngine] = {}


def _get_engine(db_url: str) -> AsyncEngine:
    if db_url not in _engine_cache:
        _engine_cache[db_url] = create_async_engine(db_url, pool_pre_ping=True)
    return _engine_cache[db_url]


async def ensure_schema(db_url: str) -> None:
    # Dev/test convenience (matches WP1's stub-adapter zero-setup philosophy):
    # CREATE TABLE IF NOT EXISTS via the ORM model, safe to call every run().
    # `alembic upgrade head` (alembic/versions/0001_create_cost_entries.py) is
    # the real migration path for shared/production databases.
    engine = _get_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def record_cost_entries(db_url: str, *, project_id: str, entries: list[CostEntry]) -> None:
    if not entries:
        return
    engine = _get_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add_all(
            [
                CostEntryRow(
                    project_id=project_id,
                    at=entry.at,
                    item=entry.item,
                    provider=entry.provider,
                    cost_usd=entry.cost_usd,
                )
                for entry in entries
            ]
        )
        await session.commit()


async def upsert_project_snapshot(
    db_url: str,
    *,
    project_id: str,
    thread_id: str,
    status: str,
    mode: str,
    brief: str,
    title: str | None,
    total_cost_usd: float,
) -> None:
    engine = _get_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    values = {
        "project_id": project_id,
        "thread_id": thread_id,
        "status": status,
        "mode": mode,
        "brief": brief,
        "title": title,
        "total_cost_usd": total_cost_usd,
        "updated_at": datetime.now(UTC),
    }
    async with session_factory() as session:
        stmt = pg_insert(ProjectRow).values(**values)
        stmt = stmt.on_conflict_do_update(index_elements=[ProjectRow.project_id], set_=values)
        await session.execute(stmt)
        await session.commit()


async def list_projects(db_url: str) -> list[ProjectRow]:
    engine = _get_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        result = await session.execute(select(ProjectRow).order_by(ProjectRow.updated_at.desc()))
        return list(result.scalars().all())


async def get_project(db_url: str, *, project_id: str) -> ProjectRow | None:
    engine = _get_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        result = await session.execute(
            select(ProjectRow).where(ProjectRow.project_id == project_id)
        )
        return result.scalar_one_or_none()


async def sum_cost_for_day(db_url: str, *, day: date) -> float:
    engine = _get_engine(db_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    start = datetime.combine(day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    async with session_factory() as session:
        result = await session.execute(
            select(func.coalesce(func.sum(CostEntryRow.cost_usd), 0.0)).where(
                CostEntryRow.at >= start, CostEntryRow.at < end
            )
        )
        return float(result.scalar_one())
