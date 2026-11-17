"""Demo with a fake transport: shows failover, caching and cost tracking (no network, no keys)."""
from __future__ import annotations

import sys

from .gateway import Gateway, Provider


def fake_transport(url, headers, body):
    if "openai" in url:
        return 429, {"error": {"message": "rate limited"}}
    if "anthropic" in url:
        return 200, {"content": [{"type": "text", "text": "Hello from the fallback."}],
                     "usage": {"input_tokens": 12, "output_tokens": 6}}
    return 200, {"candidates": [{"content": {"parts": [{"text": "gemini"}]}}], "usageMetadata": {}}


def main(argv: list[str] | None = None) -> int:
    gw = Gateway([Provider("openai", "model-a", "k1", 2.0, 8.0),
                  Provider("anthropic", "model-b", "k2", 3.0, 15.0),
                  Provider("gemini", "model-c", "k3", 1.0, 4.0)],
                 fake_transport, cache_ttl=60, max_retries=0, sleep=lambda s: None)
    msgs = [{"role": "user", "content": "Say hello."}]
    r = gw.complete(msgs)
    print(r.text, "| via", r.provider, "| attempts", r.attempts, "| cost", round(r.cost, 6))
    r2 = gw.complete(msgs)
    print("second call cached:", r2.cached, "| cost", r2.cost)
    print("breakers:", {k: b.state for k, b in gw.breakers.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
