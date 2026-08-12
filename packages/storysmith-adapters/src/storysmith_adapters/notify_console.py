from __future__ import annotations

import structlog

_log = structlog.get_logger()


class ConsoleNotify:
    """NotifyPort fallback for when Telegram isn't configured yet (§7) --
    logs the review-gate message instead of failing the run. Approve/reject
    still works through the UI console regardless, since it reads/writes
    Postgres directly rather than depending on the Telegram message itself.
    """

    async def send(self, *, text: str, link: str | None = None) -> None:
        _log.info("notify", text=text, link=link)
