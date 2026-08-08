from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from storysmith.errors import TransientError


def exponential_backoff(attempt: int) -> float:
    return float(2**attempt)


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    backoff: Callable[[int], float] = exponential_backoff,
    retry_on: tuple[type[Exception], ...] = (TransientError,),
) -> T:
    """Call fn, retrying on retry_on exceptions with backoff(attempt) seconds between tries."""
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(backoff(attempt))
    raise AssertionError("unreachable")
