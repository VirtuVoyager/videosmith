from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.wp1

_CORE_SRC = Path(__file__).resolve().parent.parent.parent / "packages" / "storysmith-core" / "src"
_FORBIDDEN = re.compile(r"os\.environ|os\.getenv\(")


def test_no_env_reads_in_core() -> None:
    offenders = [
        path
        for path in _CORE_SRC.rglob("*.py")
        if _FORBIDDEN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"os.environ/os.getenv used directly in storysmith-core: {offenders}"
