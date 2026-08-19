from __future__ import annotations

from pathlib import Path

import pytest
from storysmith.settings import Settings
from storysmith_adapters.storage_local import LocalStorage

pytestmark = pytest.mark.wp1


async def test_put_then_get_roundtrips(tmp_path: Path) -> None:
    storage = LocalStorage(Settings(_env_file=None, output_dir=str(tmp_path)))
    uri = await storage.put(key="a/b/c.png", data=b"DATA", content_type="image/png")
    assert await storage.get(uri=uri) == b"DATA"


async def test_regression_uri_is_relative_not_absolute(tmp_path: Path) -> None:
    """put() used to bake the resolved absolute path into the URI
    (file:///app/out/...), which only resolves inside whichever filesystem
    wrote it -- a real bug hit live: scene_stills ran inside a Docker
    container (writing under /app/out), and resuming the same project from
    the host process (different absolute root, same bind-mounted directory)
    got FileNotFoundError trying to open that literal container path. The
    URI must carry only the relative key, resolved against each process's
    own settings.output_dir."""
    storage = LocalStorage(Settings(_env_file=None, output_dir=str(tmp_path)))
    uri = await storage.put(key="scene_0/still.png", data=b"X", content_type="image/png")
    assert uri == "local://scene_0/still.png"
    assert str(tmp_path) not in uri


async def test_get_resolves_against_this_processs_own_output_dir(tmp_path: Path) -> None:
    """The same relative-key URI must resolve correctly against two
    different output_dir roots pointing at the same underlying directory
    (the host-path vs. container-bind-mount-path scenario)."""
    writer = LocalStorage(Settings(_env_file=None, output_dir=str(tmp_path)))
    uri = await writer.put(key="k.bin", data=b"BYTES", content_type="application/octet-stream")

    other_root = tmp_path  # simulates a different process, same bind-mounted dir
    reader = LocalStorage(Settings(_env_file=None, output_dir=str(other_root)))
    assert await reader.get(uri=uri) == b"BYTES"


async def test_get_rejects_non_local_scheme(tmp_path: Path) -> None:
    storage = LocalStorage(Settings(_env_file=None, output_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="non-local"):
        await storage.get(uri="file:///absolute/old/style.png")
