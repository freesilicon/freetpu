# Playbook 02: Cloud TPU `v6e-1` / `v5e-1` Headless Colab 60-Second Benchmark

Run `nvbench`-style compute and `nvbandwidth`-style HBM2e streaming benchmarks headlessly on Google Colab Cloud TPU (`v6e-1` Trillium 32 GB HBM with automatic `v5e-1` 16 GB HBM fallback) and verify session teardown via `colab sessions` / `colab stop -s` to prevent idle Compute Unit burn.

---

## Prerequisites

- `python3` + `numpy` locally.
- [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) **`>= 0.7.6`** (`colab`) installed and authenticated (`--auth=adc` or `--auth=oauth2`) with Colab Pro+ / Compute Units.
  - **Security minimum**: Releases `< 0.7.6` (including `0.7.5`) log authorization headers and runtime-proxy tokens into `~/.config/colab-cli/colab.log` under the default umask (`--colab-config` only relocates the session file). Install or upgrade with `uv tool install 'google-colab-cli>=0.7.6'` or `pip install -U 'google-colab-cli>=0.7.6'` so logs are redacted and owner-only (`0600`).

---

## 60-Second Copy-Paste Commands

### 1. Version-Gated Headless Run (`v6e-1` -> `v5e-1`)

```bash
# Isolate Colab session metadata in a private 0700 directory (never /tmp)
mkdir -p -m 700 ~/.cache/freetpu-colab

# Check installed CLI version first (unauthenticated; must be >= 0.7.6)
colab version

# Run compute + bandwidth remotely on Colab TPU, plus local comms, with isolated state
# (`freetpu_bench.py` verifies `colab version >= 0.7.6` before running `colab usage` or provisioning)
python3 bench/freetpu_bench.py \
  --suite all \
  --backend colab_tpu \
  --colab-auth adc \
  --colab-tpu auto \
  --colab-config ~/.cache/freetpu-colab/sessions.json \
  --json runs/colab-tpu.json \
  --markdown runs/colab-tpu.md

cat runs/colab-tpu.md
```

### 2. Force a Specific Chip (`v5e1` or `v6e1`) or Inspect Planned Commands

```bash
# Dry-run: inspect exact commands without allocating a TPU VM (prints plan directly to stdout; add --json - for JSON)
python3 bench/freetpu_bench.py --suite compute,bandwidth --backend colab_tpu --dry-run

# Target v5e-1 directly (skips v6e-1 capacity probe)
python3 bench/freetpu_bench.py \
  --suite compute,bandwidth \
  --backend colab_tpu \
  --colab-tpu v5e1 \
  --colab-config ~/.cache/freetpu-colab/v5e1.json \
  --markdown runs/colab-v5e1.md
```

### 3. Direct `colab run` Invocation (Zero Wrapper)

```bash
(
  # 1. Fail closed BEFORE any authenticated call if `colab` on PATH is < 0.7.6
  python3 -c '
import re, subprocess, sys
out = subprocess.check_output(["colab", "version"], text=True, stderr=subprocess.STDOUT)
m = re.search(r"(?i)\b(\d+)\.(\d+)\.(\d+)(?:[-.]?(a|b|rc|dev|alpha|beta)[0-9]*)?", out)
ver = tuple(int(m.group(i)) for i in (1, 2, 3)) if m else (0, 0, 0)
if not m or ver < (0, 7, 6) or (ver == (0, 7, 6) and m.group(4)):
    sys.exit(f"Refusing to authenticate: need google-colab-cli >= 0.7.6 (got {out.strip()!r})")
' || exit 1

  mkdir -p -m 700 "$HOME/.cache/freetpu-colab"
  SESSION="freetpu-bench-direct-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
  COLAB_CFG="$HOME/.cache/freetpu-colab/${SESSION}.json"
  RUN_RC=0
  RAN_COLAB=0

  cleanup() {
    trap - EXIT INT TERM HUP
    [ "$RAN_COLAB" -eq 1 ] || exit "$RUN_RC"
    BEFORE=$(colab --auth=adc --config "$COLAB_CFG" sessions 2>&1 || true)
    EP=$(printf '%s\n' "$BEFORE" | awk -v s="[$SESSION]" '$1 == s {print $2; exit}')
    STOP_OUT=$(colab --auth=adc --config "$COLAB_CFG" stop -s "$SESSION" 2>&1)
    STOP_RC=$?
    AFTER=$(colab --auth=adc --config "$COLAB_CFG" sessions 2>&1)
    AFTER_RC=$?
    VERIFIED=1
    if [ "$AFTER_RC" -ne 0 ] || printf '%s\n' "$AFTER" | grep -Fq "[$SESSION]"; then
      VERIFIED=0
    elif [ -n "$EP" ]; then
      printf '%s\n' "$AFTER" | grep -Fq "$EP" && VERIFIED=0
    elif [ "$STOP_RC" -ne 0 ] || ! printf '%s\n' "$STOP_OUT" | grep -qi "session terminated"; then
      VERIFIED=0
    fi
    if [ "$VERIFIED" -ne 1 ]; then
      printf 'ERROR: Colab teardown unverified for session=%s endpoint=%s.\n' "$SESSION" "${EP:-unknown}" >&2
      printf 'Recovery config preserved at %s; inspect and recover with:\n' "$COLAB_CFG" >&2
      printf '  colab --auth=adc --config %s sessions\n' "$COLAB_CFG" >&2
      printf '  colab --auth=adc --config %s stop -s %s\n' "$COLAB_CFG" "$SESSION" >&2
      exit 1
    fi
    exit "$RUN_RC"
  }
  trap 'RUN_RC=130; cleanup' INT
  trap 'RUN_RC=143; cleanup' TERM
  trap 'RUN_RC=129; cleanup' HUP
  trap 'cleanup' EXIT

  RAN_COLAB=1
  COLAB_CLI_ASSIGN_RETRIES=0 colab --auth=adc --config "$COLAB_CFG" run \
    -s "$SESSION" --tpu V5E1 --keep --timeout 300 \
    bench/freetpu_bench.py --suite compute,bandwidth --backend jax_tpu --emit-markers || RUN_RC=$?
)
```

The standalone recipe checks `colab version >= 0.7.6` before making any authenticated call, generates a unique per-run `$SESSION` name and matching `$COLAB_CFG` file so reruns never overwrite an earlier cleanup handle, and fails with a non-zero exit code (printing the preserved `$COLAB_CFG` recovery commands) if the runtime endpoint cannot be verified gone after `colab stop -s "$SESSION"`. Only remove `$COLAB_CFG` after confirming teardown succeeded.

---

## What Good Looks Like

| Metric | Cloud TPU `v5e-1` (Verified Live Silicon) | Cloud TPU `v6e-1` (Published Spec) | Notes |
| --- | ---: | ---: | --- |
| **BF16 GEMM (`8192^3`)** | **`179.36 TFLOP/s`** peak (`91.0%` of `197 TF/s`) | `918 TFLOP/s` peak | MXU saturates at `>=4096^3` |
| **INT8 GEMM (`8192^3`)** | **`350.69 TOPS`** peak (`89.2%` of `394 TOPS`) | `1,836 TOPS` peak | `2x` BF16 throughput |
| **HBM2e Triad BW (`1 GiB`)** | **`655.55 GB/s`** (`80.0%` of `819 GB/s`) | `1,638 GB/s` peak | 2 reads + 1 write BF16 triad |
| **Small-Op Sync Floor (`1x64x32`)** | **`117.6–135.0 us`** (`32.0 us` pipelined async) | `~100 us` PJRT est. | Motivates fused/persistent kernels |
| **Public On-Demand Rate** | **`$1.20 / chip-hr`** | **`$2.70 / chip-hr`** | GCP `us-east5` / `us-central1` |

---

## Operational Guardrails

- **Require `google-colab-cli >= 0.7.6`**: Older releases (`<= 0.7.5`) write authorization headers and runtime-proxy tokens to `~/.config/colab-cli/colab.log` under a default umask; `0.7.6+` redacts credentials and locks log permissions to the owner.
- **Always isolate `--colab-config` in a private `0700` dir**: Passing `--colab-config ~/.cache/freetpu-colab/...` keeps parallel benchmark runs separate from `~/.config/colab-cli/sessions.json` and avoids shared `/tmp` directories.
- **Automatic `v6e1 -> v5e1` Fallback & Endpoint-Verified Teardown**: If Colab refuses allocation before `[colab] Session READY`, `freetpu_bench.py` falls back to `v5e1` automatically (never retrying after `[colab] Session READY` to avoid duplicate CU spend). It invokes `colab run --keep`, records the runtime endpoint from `colab sessions`, runs `colab stop -s <session>` (including in a `finally` block on `KeyboardInterrupt`, `SIGTERM`, and `SIGHUP`), and verifies that the backend endpoint disappeared (preserving the recovery session file and failing verification if the endpoint persists even as an orphaned `[?]` entry).
- **Scope Boundary**: Colab is strictly for single-chip R&D and reproducible benchmarking—never deploy public inference APIs or multi-tenant proxies on Colab runtimes.
