"""Centralised LLM client with tenacity retry and random exponential jitter.

All nodes import `call_llm` / `call_llm_json` from here instead of
building their own OpenAI client.  The singleton client provides
connection pooling; the tenacity decorator provides resilience against
transient NVIDIA NIM failures (network blips, 429 rate-limits, 5xx
server errors).

Retry policy
------------
- wait:   wait_random_exponential(multiplier=1, max=60)  — true random
          jitter eliminates the thundering-herd /惊群效应 when multiple
          agents retry concurrently.
- stop:   stop_after_attempt(4)  — up to 4 total attempts.
- retry:  on openai.APIConnectionError, APITimeoutError, RateLimitError,
          and APIStatusError with status >= 500.
"""

import logging
import os

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
    retry_if_exception,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_api_key = os.getenv("NVIDIA_API_KEY", "")
_model = os.getenv("NIM_DEEP_MODEL", "meta/llama-3.1-70b-instruct")

# ---------------------------------------------------------------------------
# Singleton client (created lazily on first call)
# ---------------------------------------------------------------------------
_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if not _api_key:
            raise ValueError("NVIDIA_API_KEY not set")
        _client = openai.OpenAI(base_url=_base_url, api_key=_api_key)
    return _client


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on connection/timeout/rate-limit errors and 5xx server errors."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm(messages: list, model: str | None = None) -> str:
    """Send a chat-completion request and return the assistant message text.

    Retries automatically on transient failures with random exponential
    backoff (true jitter — no thundering herd).
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model or _model,
        messages=messages,
    )
    return response.choices[0].message.content


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm_json(messages: list, json_schema: dict, model: str | None = None) -> str:
    """Send a chat-completion request with JSON-schema-constrained output.

    Used by the D-agent for structured ClaimGraphDraft generation.
    Retries automatically on transient failures.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model or _model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": json_schema,
        },
    )
    return response.choices[0].message.content
