from __future__ import annotations


class StorySmithError(Exception):
    """Base class for all StorySmith domain errors."""


class TransientError(StorySmithError):
    """Retryable failure from an external call (rate limit, timeout, 5xx)."""


class ContentRejectedError(StorySmithError):
    """Provider refused the request on content-policy grounds. Not retried."""


class LLMStructuredOutputError(StorySmithError):
    """LLM failed to produce schema-valid output after the repair round."""


class BudgetExceededError(StorySmithError):
    """Raised when a caller needs to hard-fail on budget overrun outside the graph."""
