import unittest

from llmgateway import (ADAPTERS, AllProvidersFailed, BudgetExceeded, Gateway, Provider, ProviderError,
                        classify_status, redact)

MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Again"}]

OPENAI_OK = {"choices": [{"message": {"content": "o"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
ANTH_OK = {"content": [{"type": "text", "text": "a"}], "usage": {"input_tokens": 10, "output_tokens": 5}}
GEM_OK = {"candidates": [{"content": {"parts": [{"text": "g"}]}}],
          "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


class AdapterTests(unittest.TestCase):
    def test_openai_request_shape(self):
        url, h, b = ADAPTERS["openai"].build("m", MSGS, 100, "KEY")
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(h["Authorization"], "Bearer KEY")
        self.assertEqual((b["model"], b["max_completion_tokens"], len(b["messages"])), ("m", 100, 4))

    def test_anthropic_moves_system_out_of_messages(self):
        url, h, b = ADAPTERS["anthropic"].build("m", MSGS, 100, "KEY")
        self.assertTrue(url.endswith("/v1/messages"))
        self.assertEqual(h["x-api-key"], "KEY")
        self.assertIn("anthropic-version", h)
        self.assertEqual(b["system"], "Be brief.")
        self.assertEqual([m["role"] for m in b["messages"]], ["user", "assistant", "user"])

    def test_gemini_maps_roles_and_system(self):
        url, h, b = ADAPTERS["gemini"].build("gm", MSGS, 100, "KEY")
        self.assertIn("models/gm:generateContent", url)
        self.assertEqual(h["x-goog-api-key"], "KEY")
        self.assertEqual([c["role"] for c in b["contents"]], ["user", "model", "user"])
        self.assertEqual(b["systemInstruction"]["parts"][0]["text"], "Be brief.")
        self.assertEqual(b["generationConfig"]["maxOutputTokens"], 100)

    def test_no_system_field_when_no_system_message(self):
        _, _, b = ADAPTERS["anthropic"].build("m", [{"role": "user", "content": "x"}], 10, "k")
        self.assertNotIn("system", b)

    def test_parse_each_provider(self):
        for name, data in (("openai", OPENAI_OK), ("anthropic", ANTH_OK), ("gemini", GEM_OK)):
            text, usage = ADAPTERS[name].parse(data)
            self.assertEqual((text, usage.input_tokens, usage.output_tokens), (text, 10, 5))

    def test_anthropic_joins_text_blocks_and_ignores_others(self):
        text, _ = ADAPTERS["anthropic"].parse({"content": [{"type": "text", "text": "a"},
                                                         {"type": "tool_use", "name": "x"},
                                                         {"type": "text", "text": "b"}]})
        self.assertEqual(text, "ab")

    def test_malformed_responses_are_retryable_provider_errors(self):
        for name in ADAPTERS:
            with self.assertRaises(ProviderError) as cm:
                ADAPTERS[name].parse({"unexpected": True})
            self.assertTrue(cm.exception.retryable)

    def test_status_classification(self):
        for s in (408, 429, 500, 503):
            self.assertTrue(classify_status(s))
        for s in (400, 401, 403, 404):
            self.assertFalse(classify_status(s))


def providers(prices=((2, 8), (3, 15), (1, 4))):
    return [Provider("openai", "m1", "k1", *prices[0]), Provider("anthropic", "m2", "k2", *prices[1]),
            Provider("gemini", "m3", "k3", *prices[2])]


def transport_for(script):
    """script: provider-name -> list of (status, data) returned in order (last one repeats)."""
    calls = []

    def t(url, headers, body):
        name = "openai" if "openai" in url else "anthropic" if "anthropic" in url else "gemini"
        calls.append(name)
        seq = script[name]
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item
    t.calls = calls
    return t


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def gw(script, **kw):
    t = transport_for(script)
    kw.setdefault("sleep", lambda s: None)
    return Gateway(kw.pop("provs", providers()), t, **kw), t


class RoutingTests(unittest.TestCase):
    def test_ordered_uses_first_provider(self):
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]})
        r = g.complete(MSGS)
        self.assertEqual((r.provider, r.text, t.calls), ("openai", "o", ["openai"]))

    def test_cheapest_policy_prefers_lowest_combined_price(self):
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  policy="cheapest")
        self.assertEqual(g.complete(MSGS).provider, "gemini")

    def test_capability_requirement_filters_providers(self):
        provs = providers()
        provs[1] = Provider("anthropic", "m2", "k2", 3, 15, frozenset({"long_context"}))
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]}, provs=provs)
        self.assertEqual(g.complete(MSGS, require={"long_context"}).provider, "anthropic")

    def test_unsatisfiable_capability(self):
        g, _ = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]})
        with self.assertRaises(AllProvidersFailed):
            g.complete(MSGS, require={"vision"})

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            Gateway([], lambda *a: (200, {}))
        with self.assertRaises(ValueError):
            Gateway([Provider("nobody", "m", "k", 1, 1)], lambda *a: (200, {}))
        with self.assertRaises(ValueError):
            Gateway(providers(), lambda *a: (200, {}), policy="random")


class FailoverTests(unittest.TestCase):
    def test_rate_limit_fails_over_to_next_provider(self):
        g, t = gw({"openai": [(429, {"error": {"message": "slow down"}})], "anthropic": [(200, ANTH_OK)],
                   "gemini": [(200, GEM_OK)]}, max_retries=0)
        r = g.complete(MSGS)
        self.assertEqual(r.provider, "anthropic")
        self.assertEqual(r.attempts, ["openai:429", "anthropic:ok"])

    def test_retries_same_provider_before_failing_over(self):
        g, t = gw({"openai": [(500, {}), (200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  max_retries=1)
        r = g.complete(MSGS)
        self.assertEqual((r.provider, t.calls), ("openai", ["openai", "openai"]))

    def test_backoff_sleeps_between_retries(self):
        slept = []
        g, _ = gw({"openai": [(500, {}), (500, {}), (200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)],
                   "gemini": [(200, GEM_OK)]}, max_retries=2, sleep=slept.append)
        g.complete(MSGS)
        self.assertEqual(slept, [0.5, 1.0])

    def test_auth_error_is_not_retried_but_fails_over(self):
        g, t = gw({"openai": [(401, {"error": {"message": "bad key"}})], "anthropic": [(200, ANTH_OK)],
                   "gemini": [(200, GEM_OK)]}, max_retries=3)
        r = g.complete(MSGS)
        self.assertEqual((r.provider, t.calls), ("anthropic", ["openai", "anthropic"]))

    def test_timeout_is_treated_as_retryable(self):
        g, t = gw({"openai": [TimeoutError()], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]}, max_retries=0)
        self.assertEqual(g.complete(MSGS).attempts[0], "openai:408")

    def test_all_fail(self):
        err = (500, {})
        g, _ = gw({"openai": [err], "anthropic": [err], "gemini": [err]}, max_retries=0)
        with self.assertRaises(AllProvidersFailed) as cm:
            g.complete(MSGS)
        self.assertIn("openai:500", str(cm.exception))

    def test_malformed_200_response_fails_over(self):
        g, _ = gw({"openai": [(200, {"weird": 1})], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  max_retries=0)
        self.assertEqual(g.complete(MSGS).provider, "anthropic")


class BreakerTests(unittest.TestCase):
    def test_opens_after_threshold_and_skips_provider(self):
        clock = Clock()
        g, t = gw({"openai": [(500, {})], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  max_retries=0, breaker_threshold=2, clock=clock)
        g.complete(MSGS)
        g.complete(MSGS)  # second failure opens the breaker
        t.calls.clear()
        r = g.complete(MSGS)
        self.assertNotIn("openai", t.calls)
        self.assertEqual(r.attempts[0], "openai:circuit_open")

    def test_half_open_after_cooldown_then_recovers(self):
        clock = Clock()
        script = {"openai": [(500, {}), (500, {}), (200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]}
        g, t = gw(script, max_retries=0, breaker_threshold=2, breaker_cooldown=30, clock=clock)
        g.complete(MSGS); g.complete(MSGS)
        key = "m1@openai"
        self.assertEqual(g.breakers[key].state, "open")
        clock.t += 31
        self.assertEqual(g.breakers[key].state, "half-open")
        self.assertEqual(g.complete(MSGS).provider, "openai")
        self.assertEqual(g.breakers[key].state, "closed")


class CacheBudgetCostTests(unittest.TestCase):
    def test_cost_from_usage_and_prices(self):
        g, _ = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]})
        r = g.complete(MSGS)
        self.assertAlmostEqual(r.cost, (10 * 2 + 5 * 8) / 1_000_000)

    def test_cache_hit_is_free_and_skips_the_network(self):
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]}, cache_ttl=60)
        g.complete(MSGS)
        r = g.complete(MSGS)
        self.assertEqual((r.cached, r.cost, len(t.calls)), (True, 0.0, 1))

    def test_cache_expires(self):
        clock = Clock()
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  cache_ttl=60, clock=clock)
        g.complete(MSGS)
        clock.t += 61
        g.complete(MSGS)
        self.assertEqual(len(t.calls), 2)

    def test_different_prompts_do_not_share_cache(self):
        g, t = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]}, cache_ttl=60)
        g.complete(MSGS)
        g.complete([{"role": "user", "content": "different"}])
        self.assertEqual(len(t.calls), 2)

    def test_daily_budget_blocks_further_calls(self):
        g, _ = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  daily_budget=0.00005)
        g.complete(MSGS)  # costs 0.00006, over budget
        with self.assertRaises(BudgetExceeded):
            g.complete([{"role": "user", "content": "another"}])

    def test_budget_resets_next_day(self):
        clock = Clock()
        g, _ = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]},
                  daily_budget=0.00005, clock=clock)
        g.complete(MSGS)
        clock.t += 86400
        g.complete([{"role": "user", "content": "tomorrow"}])

    def test_log_stores_prompt_hash_not_prompt(self):
        g, _ = gw({"openai": [(200, OPENAI_OK)], "anthropic": [(200, ANTH_OK)], "gemini": [(200, GEM_OK)]})
        g.complete([{"role": "user", "content": "SECRET-CONTENT-123"}])
        self.assertNotIn("SECRET-CONTENT-123", str(g.log))
        self.assertIn("prompt_hash", g.log[-1])


class RedactTests(unittest.TestCase):
    def test_redacts_email_and_phone(self):
        out = redact("Mail jane.doe@example.com or call +91 98765 43210 today")
        self.assertNotIn("jane.doe", out)
        self.assertNotIn("98765", out)
        self.assertIn("[EMAIL]", out)
        self.assertIn("[PHONE]", out)

    def test_plain_numbers_are_kept(self):
        self.assertEqual(redact("We shipped 1200 units in 2026"), "We shipped 1200 units in 2026")

    def test_gateway_redacts_before_sending(self):
        seen = []

        def t(url, headers, body):
            seen.append(str(body))
            return 200, OPENAI_OK
        g = Gateway(providers(), t, do_redact=True)
        g.complete([{"role": "user", "content": "contact bob@corp.example"}])
        self.assertNotIn("bob@corp.example", seen[0])


if __name__ == "__main__":
    unittest.main()
