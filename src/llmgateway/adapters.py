"""Provider adapters: translate one neutral request into each vendor's HTTP shape and back.

A neutral request is a list of messages [{"role": "system"|"user"|"assistant", "content": str}].
Adapters never perform I/O: they build (url, headers, body) and parse a decoded JSON response.
A transport function does the actual HTTP, so everything here is testable offline.

The request and response shapes follow each vendor's public chat/messages documentation. They
are verified in this repo against recorded-shape fakes, not against live calls, so check model
names and fields against current vendor docs before production use.
"""
from __future__ import annotations

from dataclasses import dataclass


class ProviderError(Exception):
    def __init__(self, status: int, message: str, retryable: bool):
        super().__init__(f"{status}: {message}")
        self.status, self.retryable = status, retryable


def classify_status(status: int) -> bool:
    """Retryable? Rate limits, timeouts and server errors are; auth and bad requests are not."""
    return status in (408, 409, 425, 429) or status >= 500


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class OpenAIAdapter:
    name = "openai"

    def build(self, model: str, messages: list[dict], max_tokens: int, api_key: str):
        return ("https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                {"model": model, "messages": messages, "max_completion_tokens": max_tokens})

    def parse(self, data: dict) -> tuple[str, Usage]:
        try:
            text = data["choices"][0]["message"]["content"] or ""
            u = data.get("usage", {})
            return text, Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(502, f"unexpected response shape: {exc!r}", True) from None


class AnthropicAdapter:
    name = "anthropic"

    def build(self, model: str, messages: list[dict], max_tokens: int, api_key: str):
        system, rest = _split_system(messages)
        body = {"model": model, "max_tokens": max_tokens, "messages": rest}
        if system:
            body["system"] = system
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body)

    def parse(self, data: dict) -> tuple[str, Usage]:
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            u = data.get("usage", {})
            return text, Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(502, f"unexpected response shape: {exc!r}", True) from None


class GeminiAdapter:
    name = "gemini"

    def build(self, model: str, messages: list[dict], max_tokens: int, api_key: str):
        system, rest = _split_system(messages)
        body = {"contents": [{"role": "model" if m["role"] == "assistant" else "user",
                              "parts": [{"text": m["content"]}]} for m in rest],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": api_key, "Content-Type": "application/json"}, body)

    def parse(self, data: dict) -> tuple[str, Usage]:
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            u = data.get("usageMetadata", {})
            return text, Usage(int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0)))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(502, f"unexpected response shape: {exc!r}", True) from None


ADAPTERS = {a.name: a for a in (OpenAIAdapter(), AnthropicAdapter(), GeminiAdapter())}
