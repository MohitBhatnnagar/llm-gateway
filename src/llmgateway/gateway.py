"""The gateway: routing, retries, failover, circuit breaker, cache, budget and cost tracking.

transport(url, headers, body) -> (status:int, data:dict) performs the HTTP call (injected).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .adapters import ADAPTERS, ProviderError, Usage, classify_status

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s\-().]?){9,14}\d(?!\d)")


def redact(text: str) -> str:
    """Remove obvious personal identifiers before text leaves the organisation."""
    return PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", text))


@dataclass
class Provider:
    name: str                    # key into ADAPTERS
    model: str
    api_key: str
    in_price: float              # price per 1M input tokens (you supply current prices)
    out_price: float             # price per 1M output tokens
    tags: frozenset = frozenset()  # capabilities, e.g. {"long_context", "vision"}


@dataclass
class Response:
    text: str
    provider: str
    model: str
    usage: Usage
    cost: float
    cached: bool = False
    attempts: list[str] = field(default_factory=list)  # one entry per provider tried


class BudgetExceeded(RuntimeError):
    pass


class AllProvidersFailed(RuntimeError):
    pass


class Breaker:
    """Closed -> open after N consecutive failures -> half-open after cooldown (one trial)."""

    def __init__(self, threshold: int, cooldown: float, clock):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        return self.clock() - self.opened_at >= self.cooldown  # half-open: let one request test it

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self.opened_at >= self.cooldown else "open"


class Gateway:
    def __init__(self, providers: list[Provider], transport: Callable, *, policy: str = "ordered",
                 max_retries: int = 1, breaker_threshold: int = 3, breaker_cooldown: float = 30.0,
                 cache_ttl: float = 0.0, daily_budget: float | None = None,
                 do_redact: bool = False, clock=time.time, sleep=time.sleep):
        if policy not in ("ordered", "cheapest"):
            raise ValueError("policy must be 'ordered' or 'cheapest'")
        if not providers:
            raise ValueError("at least one provider is required")
        for p in providers:
            if p.name not in ADAPTERS:
                raise ValueError(f"unknown provider {p.name!r}")
        self.providers, self.transport, self.policy = providers, transport, policy
        self.max_retries, self.cache_ttl = max_retries, cache_ttl
        self.daily_budget, self.do_redact = daily_budget, do_redact
        self.clock, self.sleep = clock, sleep
        self.breakers = {p.model + "@" + p.name: Breaker(breaker_threshold, breaker_cooldown, clock) for p in providers}
        self._cache: dict[str, tuple[float, Response]] = {}
        self.spend: dict[str, float] = {}
        self.log: list[dict] = []

    # -- helpers -------------------------------------------------------------
    def _key(self, p: Provider) -> str:
        return p.model + "@" + p.name

    def _order(self, require: set[str]) -> list[Provider]:
        eligible = [p for p in self.providers if require <= set(p.tags)]
        if self.policy == "cheapest":
            return sorted(eligible, key=lambda p: (p.in_price + p.out_price, p.name))
        return eligible

    def _day(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    @staticmethod
    def _cost(p: Provider, u: Usage) -> float:
        return (u.input_tokens * p.in_price + u.output_tokens * p.out_price) / 1_000_000

    # -- main entry ----------------------------------------------------------
    def complete(self, messages: list[dict], max_tokens: int = 512, require: set[str] | None = None) -> Response:
        require = require or set()
        if self.do_redact:
            messages = [{**m, "content": redact(m["content"])} for m in messages]
        if self.daily_budget is not None and self.spend.get(self._day(), 0.0) >= self.daily_budget:
            raise BudgetExceeded(f"daily budget of {self.daily_budget} reached")

        ck = hashlib.sha256(json.dumps([messages, max_tokens, sorted(require)], sort_keys=True).encode()).hexdigest()
        if self.cache_ttl:
            hit = self._cache.get(ck)
            if hit and self.clock() - hit[0] < self.cache_ttl:
                self.log.append({"event": "cache_hit", "key": ck[:12]})
                return Response(**{**hit[1].__dict__, "cached": True, "cost": 0.0, "attempts": []})

        order = self._order(require)
        if not order:
            raise AllProvidersFailed(f"no provider has the required capabilities {sorted(require)}")
        attempts: list[str] = []
        for p in order:
            br = self.breakers[self._key(p)]
            if not br.allow():
                attempts.append(f"{p.name}:circuit_open")
                continue
            for n in range(self.max_retries + 1):
                try:
                    resp = self._call(p, messages, max_tokens)
                    br.success()
                    attempts.append(f"{p.name}:ok")
                    resp.attempts = attempts
                    self.spend[self._day()] = self.spend.get(self._day(), 0.0) + resp.cost
                    self.log.append({"event": "ok", "provider": p.name, "model": p.model, "cost": resp.cost,
                                     "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
                                     "prompt_hash": ck[:12]})  # a hash, never the prompt itself
                    if self.cache_ttl:
                        self._cache[ck] = (self.clock(), resp)
                    return resp
                except ProviderError as exc:
                    attempts.append(f"{p.name}:{exc.status}")
                    if not exc.retryable:
                        br.failure()
                        break  # a 400/401 will not improve on retry; try the next provider
                    br.failure()
                    if n < self.max_retries:
                        self.sleep(min(0.5 * 2 ** n, 4.0))
        raise AllProvidersFailed("all providers failed: " + ", ".join(attempts))

    def _call(self, p: Provider, messages: list[dict], max_tokens: int) -> Response:
        adapter = ADAPTERS[p.name]
        url, headers, body = adapter.build(p.model, messages, max_tokens, p.api_key)
        try:
            status, data = self.transport(url, headers, body)
        except TimeoutError:
            raise ProviderError(408, "timeout", True) from None
        if status != 200:
            msg = data.get("error", {}) if isinstance(data, dict) else {}
            detail = msg.get("message", "") if isinstance(msg, dict) else str(msg)
            raise ProviderError(status, detail or "request failed", classify_status(status))
        text, usage = adapter.parse(data)
        return Response(text, p.name, p.model, usage, self._cost(p, usage))
