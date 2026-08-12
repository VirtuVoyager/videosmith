from __future__ import annotations

import httpx
from storysmith.errors import TransientError
from storysmith.settings import Settings
from storysmith.util.retry import with_retries

API_BASE = "https://api.telegram.org"
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class TelegramNotify:
    """NotifyPort via the Telegram Bot API's sendMessage."""

    def __init__(self, settings: Settings) -> None:
        self._bot_token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id

    async def send(self, *, text: str, link: str | None = None) -> None:
        body = f"{text}\n{link}" if link else text

        async def _call() -> None:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{API_BASE}/bot{self._bot_token}/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": body,
                        "disable_web_page_preview": False,
                    },
                )
            if response.status_code in _TRANSIENT_STATUS_CODES:
                raise TransientError(
                    f"telegram sendMessage transient failure: {response.status_code}"
                )
            response.raise_for_status()

        await with_retries(_call)
