# Local/OrbStack use (§13's AWS/ECS deployment scaffolding is still out of
# scope -- this just needs to run under `docker compose up`, not be
# production-hardened). Built from the repo root so it has access to every
# workspace member's pyproject.toml (uv sync needs the whole workspace).
FROM python:3.12-slim

# ffmpeg: POST /runs executes the real graph end-to-end, including the
# Editor (WP5), which shells out to ffmpeg directly -- not needed for
# /healthz or /projects, but required the moment a run actually executes.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/storysmith-core/pyproject.toml packages/storysmith-core/pyproject.toml
COPY packages/storysmith-adapters/pyproject.toml packages/storysmith-adapters/pyproject.toml
COPY apps/worker/pyproject.toml apps/worker/pyproject.toml
COPY apps/api/pyproject.toml apps/api/pyproject.toml
RUN uv sync --frozen --no-install-project --all-packages

COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/ apps/api/
COPY configs/ configs/
RUN uv sync --frozen --all-packages

EXPOSE 8000
CMD ["uv", "run", "--package", "api", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
