"""Dump apps/api's OpenAPI schema to repo-root openapi.json (§7).

    uv run python scripts/gen_openapi.py
    cd apps/ui && npm run gen-api-types   # regenerates lib/api-types.ts

Run both whenever a route in apps/api/src/api/main.py changes shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from api.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    schema = app.openapi()
    out_path = REPO_ROOT / "openapi.json"
    out_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
