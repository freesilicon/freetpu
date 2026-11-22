# Headless Google Colab TPU Tooling (Track B -> Track C)

Clean-room operator guide and harness documentation for driving single-chip Cloud TPUs (`v6e-1` Trillium, `32 GiB` HBM and `v5e-1`, `16 GiB` HBM) on Google Colab via Google's **sanctioned headless control planes**, supporting FreeTPU Track B research prior to Track C GCP Cloud TPU VM / TPU Research Cloud (TRC) graduation.

---

## 1. Sanctioned Headless Control Planes

Google provides two official, supported headless automation surfaces for Colab runtimes—no browser cookie extraction or anti-bot bypassing is required or permitted:

| Control Plane | Endpoint / Binary | Auth & Scopes | Best Use Case |
| --- | --- | --- | --- |
| **`google-colab-cli` (`colab`)** | `colab` CLI (`uv tool install google-colab-cli`) | `--auth=adc` or `--auth=oauth2` (`openid`, `cloud-platform`, `userinfo.email`, `colaboratory`) | Zero-waste `colab run` ephemeral TPU probes, persistent kernel `colab exec`, `/api/contents/` file sync, `colab log -o proof.ipynb`, `colab usage` |
| **Official Colab MCP & Public REST `v1beta`** | `https://colaboratory.googleapis.com/mcp` & `https://colaboratory.googleapis.com/v1beta` | OAuth2 / ADC with `colaboratory` scope + allowlisted `X-Goog-User-Project` | IDE/agent MCP orchestration (`list_runtime_specs`, `create_runtime`, `execute_code`, `delete_runtime`) with UUID `requestId` & `executionId` reattachment |

### `google-colab-cli` Dual-Path & `--no-public-api` Behavior

- Released `google-colab-cli` (`0.8.0`) talks to Colab's session tunnel (`https://colab.research.google.com/tun/m/{ccu-info,assignments,assign,unassign}`) directly, which works with standard ADC (`userinfo.email` + `cloud-platform` + `openid`) for `colab usage` and `colab sessions` even when a GCP quota project is not allowlisted for `colaboratory.googleapis.com/v1beta`.
- CLI builds that default to the Public API expose `--no-public-api` to fall back to the `/tun/m/*` session tunnel.
- Always pass global flags (`--auth`, `--config`, and `--no-public-api` when supported) **before** the subcommand:

```bash
colab --auth=adc --config /tmp/freetpu-colab.json usage
colab --auth=adc --config /tmp/freetpu-colab.json sessions
```

> **Note:** Do not confuse `--auth={adc,oauth2}` (CLI control-plane authentication) with `colab auth` (an interactive command that injects VM-side GCP credentials into a running kernel).

### Official Colab MCP (`https://colaboratory.googleapis.com/mcp`) & Public REST `v1beta`

The remote Streamable HTTP MCP server at `https://colaboratory.googleapis.com/mcp` exposes 10 tools mirroring `https://colaboratory.googleapis.com/v1beta`:

- **Account & Discovery**: `get_subscription`, `list_runtime_specs` (pre-flight TPU eligibility)
- **Runtime Lifecycle**: `list_runtimes`, `get_runtime`, `create_runtime` (idempotent UUID `requestId`), `delete_runtime`
- **Kernel & Execution**: `create_session`, `list_sessions`, `delete_session`, `execute_code` (600s RPC cutoff; pass a client UUID `executionId` to reattach without re-executing)

**Allowlist & Scope Behavior:**

- Unauthenticated `initialize` and `tools/list` probes on `https://colaboratory.googleapis.com/mcp` return HTTP `200` and advertise all 10 tool schemas.
- Tool invocations (`tools/call`) and REST endpoints (`GET /v1beta/subscription`, `GET /v1beta/runtimespecs`, `POST /v1beta/runtimes`) require the `https://www.googleapis.com/auth/colaboratory` OAuth scope and an allowlisted GCP quota project (`X-Goog-User-Project`).
- Unallowlisted or missing-scope ADC tokens receive HTTP `403` (`ACCESS_TOKEN_SCOPE_INSUFFICIENT` or `PERMISSION_DENIED`). In that case, re-mint ADC with the `colaboratory` scope:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

If your GCP project still receives `403` on `colaboratory.googleapis.com/v1beta`, use `google-colab-cli` (`--no-public-api` / `/tun/m/*` tunnel) instead.

#### Antigravity MCP Configuration (`~/.gemini/config/mcp_config.json`)

```json
{
  "mcpServers": {
    "colaboratory": {
      "authProviderType": "google_credentials",
      "headers": {
        "X-Goog-User-Project": "<your-gcp-project-id>"
      },
      "serverUrl": "https://colaboratory.googleapis.com/mcp"
    }
  }
}
```

---

## 2. Operational Patterns for Zero-Waste TPU R&D

1. **Pre-Flight Compute Unit (CU) Guard**: Run `colab usage` before provisioning to confirm positive `paidComputeUnitsBalance` and check active assignments. Never leak idle VMs.
2. **State Isolation (`--config`)**: Pass `--config <run-dir>/colab-sessions.json` per worker so concurrent Track A/B agents never overwrite `~/.config/colab-cli/sessions.json`.
3. **Ephemeral Runs (`colab run`)**: Use `colab run --tpu V6E1 script.py` for single-shot sweeps—it provisions, executes, propagates exit codes, and automatically tears down the VM on completion or error.
4. **Graceful `v6e1 -> v5e1` Fallback**: HTTP `412` (`FAILED_PRECONDITION`, `REASON_NO_BACKENDS`, `REASON_TOO_MANY_ACTIVE_ASSIGNMENTS`, `REASON_QUOTA_EXCEEDED_USAGE_TIME`) or `400` indicates capacity/quota limits—fall back cleanly from `v6e1` (`32 GiB` HBM) to `v5e1` (`16 GiB` HBM).
5. **Prefer `colab exec` Over `colab ssh`**: Runtime `/colab/ssh` enforces a `5`-session concurrency cap (`HTTP 429`) and a minimal `PATH`. Jupyter WebSocket `colab exec` keeps imports, XLA/PJRT/Pallas caches, and TPU device handles warm in memory across calls without hitting the SSH cap.
6. **Deterministic Artifact Sync**: Use `colab upload` / `colab download` (`/api/contents/...`) and `colab log -s <name> -o proof.ipynb` for replayable execution notebooks instead of `drive.mount` (which blocks headless agents on interactive prompts and hits DriveFS `Errno 5` quotas).

---

## 3. Using `tools/colab_tpu_bench.py`

[`tools/colab_tpu_bench.py`](../../tools/colab_tpu_bench.py) wraps `google-colab-cli` with CU guards, isolated `--config` state, `v6e1 -> v5e1` fallback, structured probe JSON capture, and `.ipynb` proof export:

```bash
# Offline CI self-test (no Colab credentials or hardware required)
python3 tools/colab_tpu_bench.py --self-test

# Inspect planned commands without provisioning a VM
python3 tools/colab_tpu_bench.py --dry-run --out-dir /tmp/freetpu-colab-dry

# Run a live single-shot TPU probe (v6e1 with automatic v5e1 fallback)
python3 tools/colab_tpu_bench.py \
  --auth adc \
  --min-cu-balance 10 \
  --out-dir /tmp/freetpu-colab-proof
```

Outputs in `--out-dir`:
- `summary.json` — CU balance before run, chosen TPU (`v6e1` or `v5e1`), attempt diagnostics, and parsed probe metrics (`jax` device topology, `/dev/vfio/*`, `tpu-info`, `bf16` GEMM microbenchmark).
- `<session>.stdout.txt` — captured remote script output.
- `<session>.ipynb` — replayable Jupyter notebook exported via `colab log -o`.

---

## 4. Colab ToS & Track C Graduation Boundaries

- **Allowed on Colab (Track B)**: Personal interactive and automated R&D, single-chip `v6e-1` / `v5e-1` Pallas/Mosaic/PJRT/`llama.cpp` benchmarks, and artifact download within paid Compute Unit balance (e.g., Google AI Ultra `2,000 CUs/month` with 24-hour continuous background execution while CUs > 0).
- **Strictly Prohibited (Hard Stop)**:
  - Hosting public inference APIs, Gradio/web UIs bypassing notebooks, or multi-tenant serving.
  - Reselling compute, cryptocurrency mining, or sharing account credentials.
  - Bypassing BotGuard, reCAPTCHA, or rate limits (`412`/`429`), or extracting browser cookies.
- **When to Graduate to Track C (GCP Cloud TPU VMs / TRC)**:
  - Multi-chip TPU pod slices and `400+ GB/s` ICI collective networking.
  - Uncontainerized kernel / `/dev/vfio/*` bare-metal driver work (`libopenbuenos` datacenter backend).
  - Persistent or public serving workloads.
