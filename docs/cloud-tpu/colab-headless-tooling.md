# Headless Google Colab TPU Tooling (Track B -> Track C)

Clean-room operator guide and harness documentation for driving single-chip Cloud TPUs (`v6e-1` Trillium, `32 GiB` HBM and `v5e-1`, `16 GiB` HBM) on Google Colab via Google's **sanctioned headless control planes**, supporting FreeTPU Track B research prior to Track C GCP Cloud TPU VM / TPU Research Cloud (TRC) graduation.

---

## 1. Sanctioned Headless Control Planes

Google provides two official, supported headless automation surfaces for Colab runtimes—no browser cookie extraction or anti-bot bypassing is required or permitted:

| Control Plane | Endpoint / Binary | Auth & Scopes | Best Use Case |
| --- | --- | --- | --- |
| **`google-colab-cli` (`colab`)** | `colab` CLI (`uv tool install google-colab-cli`) | `--auth=adc` or `--auth=oauth2` (`openid`, `cloud-platform`, `userinfo.email`, `colaboratory`) | Zero-waste `colab run` ephemeral TPU probes, persistent kernel `colab exec`, `/api/contents/` file sync, `colab log -o proof.ipynb`, `colab usage` |
| **Official Colab MCP & Public REST `v1beta`** | `https://colaboratory.googleapis.com/mcp` & `https://colaboratory.googleapis.com/v1beta` | OAuth2 / ADC with `colaboratory` (or `colaboratory.readonly` for read-only REST discovery/inspection) + allowlisted `X-Goog-User-Project` | IDE/agent MCP orchestration (`list_runtime_specs`, `create_runtime`, `execute_code`, `delete_runtime`) with UUID `requestId` & `executionId` reattachment; Public REST `v1beta` covers subscription, runtime-spec discovery, and runtime lifecycle |

### `google-colab-cli` Dual-Path & `--no-public-api` Behavior

- Released `google-colab-cli` (`0.8.0`) talks to Colab's session tunnel (`https://colab.research.google.com/tun/m/{ccu-info,assignments,assign,unassign}`) directly, which works with standard ADC (`userinfo.email` + `cloud-platform` + `openid`) for `colab usage` and `colab sessions` even when a GCP quota project is not allowlisted for `colaboratory.googleapis.com/v1beta`.
- CLI builds that default to the Public API expose `--no-public-api` to fall back to the `/tun/m/*` session tunnel.
- Always pass global flags (`--auth`, `--config`, and `--no-public-api` when supported) **before** the subcommand:

```bash
mkdir -p -m 700 ~/.cache/freetpu-colab
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json usage
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json sessions
```

> **Note:** Do not confuse `--auth={adc,oauth2}` (CLI control-plane authentication) with `colab auth` (an interactive command that injects VM-side GCP credentials into a running kernel). For `--auth=oauth2` first-time setup, run `colab --auth=oauth2 whoami` interactively so the OAuth consent URL and code prompt are visible before scripted runs.

### Official Colab MCP (`https://colaboratory.googleapis.com/mcp`) & Public REST `v1beta`

The remote Streamable HTTP MCP server at `https://colaboratory.googleapis.com/mcp` exposes 10 tools: 6 account/discovery and runtime-lifecycle tools corresponding to Public REST `v1beta` (`https://colaboratory.googleapis.com/v1beta`), plus 4 **MCP-only** kernel session and code-execution tools that are **not** exposed by Public REST `v1beta`:

- **Account & Discovery (MCP & Public REST `v1beta`)**: `get_subscription` (`GET /v1beta/subscription`), `list_runtime_specs` (`GET /v1beta/runtimespecs`, pre-flight TPU eligibility)
- **Runtime Lifecycle (MCP & Public REST `v1beta`)**: `list_runtimes` (`GET /v1beta/runtimes`), `get_runtime` (`GET /v1beta/runtimes/*`), `create_runtime` (`POST /v1beta/runtimes`, idempotent UUID `requestId`), `delete_runtime` (`DELETE /v1beta/runtimes/*`)
- **Kernel & Execution (MCP-only — not in Public REST `v1beta`)**: `create_session`, `list_sessions`, `delete_session`, `execute_code` (600s RPC cutoff; pass a client UUID `executionId` **against the same named session** to reattach to an in-flight call—runtime-targeted calls create a fresh session and cannot reattach, and reusing an `executionId` after its result has been returned re-executes the code). Public REST `v1beta` alone does not expose kernel session or code-execution endpoints; use Official Colab MCP or `google-colab-cli` for headless code execution.

**Allowlist & Scope Behavior:**

- Unauthenticated `initialize` and `tools/list` probes on `https://colaboratory.googleapis.com/mcp` return HTTP `200` and advertise all 10 tool schemas.
- Read-only REST discovery/inspection endpoints (`GET /v1beta/subscription`, `GET /v1beta/runtimespecs`, `GET /v1beta/runtimes`, `GET /v1beta/runtimes/*`) accept the narrower `https://www.googleapis.com/auth/colaboratory.readonly` OAuth scope (as well as the full `https://www.googleapis.com/auth/colaboratory` scope) plus an allowlisted GCP quota project (`X-Goog-User-Project`). Prefer `colaboratory.readonly` for discovery-only credentials so they cannot create or delete runtimes unnecessarily.
- Mutating REST calls (`POST /v1beta/runtimes`, `DELETE /v1beta/runtimes/*`) and full MCP `tools/call` orchestration require the read/write `https://www.googleapis.com/auth/colaboratory` OAuth scope and an allowlisted GCP quota project (`X-Goog-User-Project`).
- Unallowlisted or missing-scope ADC tokens receive HTTP `403` (`ACCESS_TOKEN_SCOPE_INSUFFICIENT` or `PERMISSION_DENIED`). Re-mint ADC with the least-privilege scope required (substitute `colaboratory.readonly` for discovery-only REST callers):

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

1. **Pre-Flight Compute Unit (CU) Guard**: Run `colab usage` before provisioning to parse the signed `Current balance` and check active assignments via `colab sessions`. Never leak idle VMs.
2. **State Isolation (`--config`)**: Pass `--config <private-run-dir>/colab-sessions.json` per worker so concurrent Track A/B agents never overwrite `~/.config/colab-cli/sessions.json`. Use an absolute path or normalize relative names to avoid empty-dirname directory creation errors.
3. **Ephemeral Runs (`colab run -- ...`)**: Use `colab run --tpu V6E1 -- script.py [args...]` for single-shot sweeps—it provisions, executes, propagates exit codes, and attempts teardown on completion or error. Verify release afterwards with `colab sessions`.
4. **Conservative `v6e1 -> v5e1` Fallback**: Only confirmed pre-allocation refusals (e.g. HTTP `412` `FAILED_PRECONDITION`, `REASON_NO_BACKENDS`, `REASON_TOO_MANY_ACTIVE_ASSIGNMENTS`, `REASON_QUOTA_EXCEEDED_USAGE_TIME`, or `400`) fall back cleanly from `v6e1` (`32 GiB` HBM) to `v5e1` (`16 GiB` HBM). Treat `ReadTimeout` as an unknown allocation outcome and reconcile with `colab sessions` rather than blindly starting a second TPU.
5. **Prefer `colab exec` Over `colab ssh`**: Runtime `/colab/ssh` enforces a `5`-session concurrency cap (`HTTP 429`) and a minimal `PATH`. Jupyter WebSocket `colab exec` keeps imports, XLA/PJRT/Pallas caches, and TPU device handles warm in memory across calls without hitting the SSH cap.
6. **Deterministic Artifact Sync**: Use `colab upload` / `colab download` (`/api/contents/...`) and `colab log -s <name> -o proof.ipynb` (with unique per-run session names) for replayable execution notebooks instead of `drive.mount` (which blocks headless agents on interactive prompts and hits DriveFS `Errno 5` quotas).

---

## 3. Using `tools/colab_tpu_bench.py`

[`tools/colab_tpu_bench.py`](../../tools/colab_tpu_bench.py) wraps `google-colab-cli` with private `0700` output/config checks, signed CU guards, isolated `--config` normalization, conservative `v6e1 -> v5e1` fallback, post-run `colab sessions` teardown verification, TPU-validated probe JSON capture, and `.ipynb` proof export:

```bash
# Offline CI self-test (no Colab credentials or hardware required)
python3 tools/colab_tpu_bench.py --self-test

# Inspect planned commands without provisioning a VM (private temp dir)
OUT_DIR="$(mktemp -d)" && chmod 700 "$OUT_DIR"
python3 tools/colab_tpu_bench.py --dry-run --out-dir "$OUT_DIR"

# Run a live single-shot TPU probe (v6e1 with automatic v5e1 fallback on refusal)
mkdir -p -m 700 ~/.cache/freetpu-colab
mkdir -p -m 700 ~/.cache/freetpu-colab/proof
python3 tools/colab_tpu_bench.py \
  --auth adc \
  --min-cu-balance 10 \
  --out-dir ~/.cache/freetpu-colab/proof
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
