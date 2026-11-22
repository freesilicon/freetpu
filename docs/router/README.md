# FreeTPU Router v0

Zero-dependency, OpenAI-compatible HTTP router (`Personal-AI-Router` equivalent) for heterogeneous TPU silicon: wired Google Pixel Tensor G5 `llama-server` / `ggml-rpc` clusters, local CPU/GPU/Mac hosts, and Cloud TPU (`v5e` / `v6e`) Spot and on-demand workers.

Built exclusively on the Python 3 standard library (`http.server`, `http.client`, `threading`), so any developer can launch it in one command with zero `pip` dependencies.

---

## 60-Second Quickstart

```bash
# 1. Run the offline end-to-end self-test suite (zero TPU hardware required)
python3 router/freetpu_router.py --self-test

# 2. Validate an example multi-backend config
python3 router/freetpu_router.py --check-config --config router/examples/hybrid-pixel-cloud.json

# 3. Start the router on http://127.0.0.1:8080/v1
python3 router/freetpu_router.py --config router/examples/hybrid-pixel-cloud.json --policy local_first

# 4. Point any OpenAI SDK, curl, or coding agent at the router
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=local-dev

# Inspect discovered models (when using llama-server, pass `--alias <id>` to advertise
# a fixed model ID, or pick the discovered ID directly from /v1/models):
curl -s http://localhost:8080/v1/models | jq .
MODEL_ID=$(curl -s http://localhost:8080/v1/models | python3 -c 'import sys, json; data = json.load(sys.stdin).get("data") or []; print(data[0]["id"] if data else "")')
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(MODEL_ID="$MODEL_ID" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL_ID"], "messages": [{"role": "user", "content": "Hello from FreeTPU Router!"}], "stream": True}))')"
```

---

## API Surface

| Endpoint | Method | Behavior |
| --- | --- | --- |
| `/v1/chat/completions` | `POST` | OpenAI Chat Completions (non-streaming JSON and live SSE `stream: true`) |
| `/v1/completions` | `POST` | OpenAI Legacy Text Completions (non-streaming JSON and SSE streaming) |
| `/v1/models` | `GET` | Dynamically aggregates models across currently healthy, non-overloaded backends |
| `/health` | `GET` | Per-backend readiness, EWMA latency, cooldown state, and scraped `tpu-exporter` HBM/duty-cycle telemetry |
| `/metrics` | `GET` | Prometheus text exposition `0.0.4` (requests, in-flight, failovers, tokens, TTFT histogram, USD cost) |

Inference responses include observability headers:

- `X-FreeTPU-Backend`: ID of the backend that served the request
- `X-FreeTPU-Failovers`: number of pre-first-token retries performed for the request
- `X-FreeTPU-Request-Id`: router-assigned request identifier

---

## Sample Multi-Backend Configuration

See [`router/examples/hybrid-pixel-cloud.json`](../../router/examples/hybrid-pixel-cloud.json) and [`router/examples/local-pixel-cluster.json`](../../router/examples/local-pixel-cluster.json).

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "policy": "local_first",
  "health_poll_interval_s": 10.0,
  "failure_cooldown_s": 15.0,
  "backends": [
    {
      "id": "pixel-g5-cluster",
      "type": "pixel-tpu",
      "local": true,
      "base_url": "http://127.0.0.1:8081/v1",
      "models": ["qwen2.5-1.5b-q8_0", "qwen2.5-32b-q4_0"],
      "cost_per_1m_prompt_usd": 0.0,
      "cost_per_1m_completion_usd": 0.0,
      "priority": 10
    },
    {
      "id": "cloud-tpu-v5e-spot",
      "type": "cloud-tpu-v5e",
      "local": false,
      "spot": true,
      "base_url": "http://127.0.0.1:8083/v1",
      "tpu_exporter_url": "http://127.0.0.1:9400/metrics",
      "max_hbm_utilization": 0.90,
      "models": ["qwen2.5-32b-q4_0"],
      "cost_per_1m_prompt_usd": 0.15,
      "cost_per_1m_completion_usd": 0.30,
      "api_key_env": "FREETPU_CLOUD_BACKEND_TOKEN",
      "priority": 30
    }
  ]
}
```

---

## Routing Policies

Set the default policy via `"policy"` in JSON config or `--policy`, and optionally override per request with the `X-FreeTPU-Policy` HTTP header:

| Policy | Selection Order |
| --- | --- |
| `local_first` | Healthy local nodes (`local: true`, e.g., wired Pixel Tensor G5 cluster or Mac host) first; falls back to Cloud TPU workers when local is down, overloaded, or lacks the requested model. |
| `cheapest` | Lowest blended USD per 1M tokens (`(cost_per_1m_prompt_usd + cost_per_1m_completion_usd) / 2`), breaking ties by local preference, lowest EWMA latency, and lowest HBM utilization. |
| `lowest_latency` | Lowest observed EWMA request latency / probe RTT, breaking ties by active in-flight count and HBM utilization. |
| `round_robin` | Evenly rotates the primary backend per model across healthy candidates while preserving remaining candidates as failover targets. |

---

## `tpu-exporter` Telemetry & Spot Preemption Failover

### Prometheus `tpu-exporter` Polling

When a backend sets `tpu_exporter_url`, the router scrapes its `/metrics` endpoint on startup and every `health_poll_interval_s` using the exact Prometheus metric names exposed by `tpu-exporter`:

- `freetpu_exporter_up`
- `freetpu_tpu_devices_discovered`
- `freetpu_tpu_health_status`
- `freetpu_tpu_hbm_used_bytes`
- `freetpu_tpu_hbm_total_bytes`
- `freetpu_tpu_hbm_utilization_ratio`
- `freetpu_tpu_duty_cycle_percent`
- `freetpu_tpu_active_pids`

A TPU backend is automatically marked unhealthy (`healthy: false`) and excluded from routing if:

1. `freetpu_exporter_up == 0` or the exporter endpoint is unreachable (`tpu_exporter_down` / `tpu_exporter_unreachable`),
2. All reported chips have `freetpu_tpu_health_status == 0` (`tpu_unhealthy`),
3. Peak `freetpu_tpu_hbm_utilization_ratio >= max_hbm_utilization` (default `0.95`, `tpu_hbm_oom_risk`), or
4. Optional `max_duty_cycle_percent` is exceeded (`tpu_duty_cycle_overloaded`).

### Automatic Spot-TPU Preemption Failover

For both non-streaming JSON and SSE (`stream: true`) requests:

- If a selected backend returns HTTP `502`, `503`, or `504`, raises a connection/timeout error, or drops an SSE stream **before delivering the first chunk/token**, the router places that backend into a bounded cooldown (`failure_cooldown_s`), increments `freetpu_router_failovers_total`, and seamlessly retries the next healthy backend holding the requested model.
- For SSE streams, downstream `200 OK` headers are held until the first complete upstream SSE data event arrives, preventing partial-stream corruption on pre-first-token preemption or keepalive-only aborts.
- Browser cross-origin access is disabled by default (`cors_allowed_origins: []`); untrusted `Origin` headers on inference `POST`s are rejected with `403`. Opt in explicitly via `"cors_allowed_origins"` in config or `--cors-origin`.
- Inbound request bodies are bounded by `max_request_body_bytes` (default 10 MiB) and `client_read_timeout_s` (default 30s).
- Authenticated non-loopback backends (`api_key` or `api_key_env` on a non-loopback host) require `https://` unless `"allow_insecure_http": true` is explicitly set on the backend.

---

## Prometheus `/metrics`

`GET /metrics` emits Prometheus text exposition `0.0.4` metrics:

- `freetpu_router_up`
- `freetpu_router_requests_total{model,backend_id,status}`
- `freetpu_router_active_inflight{backend_id}`
- `freetpu_router_failovers_total{model,from_backend,to_backend,reason}`
- `freetpu_router_prompt_tokens_total{model,backend_id}`
- `freetpu_router_completion_tokens_total{model,backend_id}`
- `freetpu_router_estimated_cost_usd_total{model,backend_id}`
- `freetpu_router_ttft_seconds_{bucket,sum,count}{model,backend_id}`
- `freetpu_router_request_duration_seconds_{bucket,sum,count}{model,backend_id}`
- `freetpu_router_backend_healthy{backend_id,type,local,spot}`
- `freetpu_router_backend_latency_seconds{backend_id}`
- `freetpu_router_backend_tpu_hbm_utilization_ratio{backend_id}`
- `freetpu_router_backend_tpu_duty_cycle_percent{backend_id}`
- `freetpu_router_backend_tpu_healthy_chips{backend_id}`

---

## Extension Hook API

`router/hooks.py` defines three clean Python interfaces so external wrappers and control planes can extend v0 without forking:

1. **`AuthHook`** (`AllowAllAuthHook`, `BearerTokenAuthHook`)
   - `authenticate(headers, path, method) -> AuthContext`
2. **`UsageMeterHook`** (`NoOpUsageMeterHook`, `JsonlUsageMeterHook`, `CallbackUsageMeterHook`)
   - `record_usage(event: UsageEvent) -> None`
   - `UsageEvent` fields: `request_id`, `model`, `backend_id`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `estimated_cost_usd`, `timestamp_unix`, `ttft_ms`, `stream`, `failovers`, `principal`, `status_code`.
3. **`ScalerHook`** (`NoOpScalerHook`)
   - `on_queue_pressure(event: QueuePressureEvent) -> None`
   - `on_backend_idle(event: BackendIdleEvent) -> None`

Hooks can be passed directly in Python via `create_router_server(config, auth_hook=..., usage_hook=..., scaler_hook=...)` or configured via `"hook": "my_package.hooks:MyHook"` in JSON config.
