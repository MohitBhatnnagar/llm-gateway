# llm-gateway

A **provider-agnostic LLM gateway**: one interface in front of OpenAI (ChatGPT), Anthropic (Claude) and Google (Gemini) style APIs, with routing, retries, failover, a circuit breaker, caching, a daily budget and cost tracking.

Teams that call a model API directly from every service end up with scattered keys, no failover, surprise bills and prompts in logs. The gateway is the one place those concerns live.

Standard library only. HTTP is an injected `transport`, so the whole thing is tested offline. 32 tests.

```bash
pip install -e .
llmgateway                    # demo: rate-limited provider -> failover -> cached repeat
python -m unittest discover -s tests -v
```

## What it does

| Concern | Behaviour |
|---|---|
| **Adapters** | One neutral message format is translated to each vendor's request shape (system prompt handling, role names, token limit field) and each response is parsed back to text plus token usage |
| **Routing** | `ordered` (priority list) or `cheapest` (by combined token price). Optional capability tags, for example require `long_context` |
| **Retries** | Exponential backoff on retryable errors (429, 408, 5xx, timeouts, malformed responses) |
| **Failover** | After retries, move to the next provider. Auth and bad-request errors are not retried (they will not improve) but still fail over |
| **Circuit breaker** | Opens after N consecutive failures so a dead provider is skipped instantly, then half-opens after a cooldown to test recovery |
| **Cache** | Exact-match with TTL on (messages, token limit, capabilities). A cache hit costs nothing and makes no network call |
| **Budget** | A daily spend ceiling that raises `BudgetExceeded` (resets by UTC day) |
| **Cost** | Computed from token usage and the prices you supply |
| **Privacy** | Optional redaction of emails and phone numbers before text leaves; the call log stores a prompt **hash**, never the prompt |

## Example

```python
from llmgateway import Gateway, Provider

gw = Gateway(
    [Provider("openai", "your-model", API_KEY_1, in_price=2.0, out_price=8.0),
     Provider("anthropic", "your-model", API_KEY_2, in_price=3.0, out_price=15.0),
     Provider("gemini", "your-model", API_KEY_3, in_price=1.0, out_price=4.0)],
    transport=my_http_post,            # (url, headers, body) -> (status, json)
    policy="ordered", cache_ttl=300, daily_budget=25.0, do_redact=True)

r = gw.complete([{"role": "user", "content": "Summarise this ticket..."}])
print(r.text, r.provider, r.cost, r.attempts)
```

## Important caveats

- **Not live-tested.** Request and response shapes follow each vendor's public documentation, but the tests use fake transports. Verify model names, field names and API versions against current vendor docs before relying on it.
- **Prices are inputs.** Vendor prices change; the gateway has no built-in price table by design.
- No streaming, tool-calling passthrough or per-tenant quotas yet. Those are the natural next steps.

## Design decisions

1. **Adapters do no I/O.** They build a request and parse a response, so they are pure and trivially testable; the transport is the only thing that touches the network.
2. **Error classification drives behaviour.** Retrying a 401 wastes time; failing over on a 429 is correct. Encoding that once is the main value of a gateway.
3. **A malformed 200 is a failure, not data.** Unexpected response shapes are retryable provider errors rather than crashes or silent empty answers.
4. **Cache and budget sit in front of the providers,** so they protect cost regardless of which vendor answers.

Related: [`multi-agent-orchestrator`](https://github.com/MohitBhatnnagar/multi-agent-orchestrator), [`agentic-ai-patterns`](https://github.com/MohitBhatnnagar/agentic-ai-patterns), [`ai-governance-framework`](https://github.com/MohitBhatnnagar/ai-governance-framework) (why redaction and audit matter).

## License

MIT
