---
name: freetpu
description: >-
  Write, verify, and benchmark custom TPU kernels on Google Pixel Tensor G5
  (buenos_a0) via libopenbuenos/VLIW ISA tools and on Cloud TPU (v5e-1 / v6e-1)
  via Pallas/Mosaic and headless google-colab-cli. Use when asked to author,
  fuse, port, debug, or benchmark Edge TPU or Cloud TPU kernels, run offline
  host-mock parity checks, or execute zero-waste Colab TPU runs.
---

# FreeTPU Kernel & Benchmark Skill

Write, verify, and benchmark custom TPU kernels across **Pixel Tensor G5 (`buenos_a0`)** and **Cloud TPU (`v5e-1` / `v6e-1`)** with zero proprietary vendor binaries in source control.

Load [references/silicon-cheatsheet.md](references/silicon-cheatsheet.md) when choosing tile shapes, debugging VMEM/SRAM overflows, or working around Tensor G5 `INT4` / `W8A16` / DVFS errata.

## Pre-Flight Rules

1. **Clean-room boundary**: Never commit or redistribute proprietary `.so`, `.ko`, `.tflite`, `.dgc0`, or firmware blobs (e.g., `dive_interp_rv64.bin`). Physical Tensor G5 runs extract `ext[1]` locally on-device from the user's own `.dgc0` via `OPENBUENOS_DIVE_INTERP_PATH` or `ob_gemm_extract_dive_interp_from_dgc0`.
2. **Route by target**:
   - **Mobile / bare-metal Edge TPU (`buenos_a0`)**: Use Workflow 1 (`libopenbuenos`, `tools/emit_custom_gemm.py`, `tools/spline_0x640.py`, `tpu_baremetal_runner`).
   - **Datacenter Cloud TPU (`v5e-1` / `v6e-1`)**: Use Workflow 2 (`google-colab-cli` / `tools/colab_tpu_bench.py`) + Workflow 3 (Pallas/Mosaic).
3. **Parity before timing**: Always check numerical parity against NumPy `cpu_ref` (and XLA on Cloud TPU) before reporting latencies or TFLOP/s.
4. **Resolve the FreeTPU checkout root**: Workflows 1, 2, and 3 all invoke repository-relative paths (`tools/...`, `skills/freetpu/templates/...`, `libopenbuenos/...`). Whenever your working directory differs from the FreeTPU root — whether this skill was loaded via a global symlink (`~/.claude/skills/freetpu`, `~/.codex/skills/freetpu`, or `~/.gemini/config/skills/freetpu`) or via project-local parent-directory discovery (e.g., started inside `libopenbuenos/`) — resolve the exact `SKILL.md` path your agent loaded back to the checkout root and `cd` there before running any workflow below:

```bash
# Set SKILL_MD to the exact SKILL.md path loaded by your agent
# (project-local e.g. ../skills/freetpu/SKILL.md, or global symlink):
SKILL_MD="<path-to-loaded-SKILL.md>"
REPO_ROOT="$(python3 -c '
import pathlib, sys
p = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
root = p.parents[2]
assert (root / "tools" / "emit_custom_gemm.py").is_file(), f"Not a FreeTPU checkout: {root}"
print(root)
' "$SKILL_MD")" && [ -n "$REPO_ROOT" ] && cd "$REPO_ROOT"
```

---

## Workflow 1: Pixel Tensor G5 (`buenos_a0`) Bare-Metal & Host Mock

Use this workflow for zero-Google-`.so` C/C++ kernels, closed-form `(M, K, N)` VLIW GEMMs, and `0x640` minimax spline fusion.

1. **Verify host mock runtime & C11/C++17 headers offline** (macOS/Linux, no phone required):
   ```bash
   cmake -S libopenbuenos -B libopenbuenos/build/cmake-host -DCMAKE_BUILD_TYPE=Release
   cmake --build libopenbuenos/build/cmake-host -j
   ctest --test-dir libopenbuenos/build/cmake-host --output-on-failure
   ```
   Set `ob_config_t::mock_mode = 1` (or `OPENBUENOS_MOCK=1`) in host binaries to exercise the DMA-BUF slab pool, LRU graph cache, 16-credit VII ring, and CPU reference parity without `/dev/edgetpu-soc`.

2. **Synthesize & verify VLIW GEMM + `0x640` minimax splines offline**:
   ```bash
   python3 tools/emit_custom_gemm.py --self-test
   python3 tools/spline_0x640.py --self-test --no-emit
   python3 skills/freetpu/templates/buenos_gemm_probe.py --self-test
   ```
   - `tools/emit_custom_gemm.py` synthesizes `chunk[0..6]` and packed `ext[2]` (`int32` bias + `[K/4, N_blk, 4]` `uint8` `w + 128`) using IEEE-754 **FP32 Round-Half-to-Even (RNE)** (`np.rint`), matching `buenos_a0` hardware rather than fixed-point RHU.
   - `tools/spline_0x640.py` synthesizes 208-byte `0x640` (`LoadCoefficientTables`) 8-segment degree-4 Remez minimax splines (`SiLU`, `GELU`, `exp`) with zero-overhead Mode-A fusion.
   - To emit a runnable `.dgc0` + `.in.bin` + `.ref.bin` bundle locally, supply both a user-extracted shell (`--shell` or `FREETPU_SHELL_DGC0`) and `--out-prefix` inside a fresh private directory:
     ```bash
     PROBE_DIR="$(umask 077 && mktemp -d)"
     python3 skills/freetpu/templates/buenos_gemm_probe.py \
       --shell /path/to/local_shell.dgc0 \
       --out-prefix "$PROBE_DIR/freetpu_probe"
     ```

3. **Cross-build and prove on physical Tensor G5 silicon** (unlocked Android ARM64 device):
   ```bash
   make -C libopenbuenos verify-header android verify-android
   make -C runtime verify-baremetal
   ```
   Push `tpu_baremetal_runner` and the emitted artifacts (`$PROBE_DIR/freetpu_probe.*`) via `adb push` to `/data/local/tmp/`, run with `--ref`, and verify both bit-exact parity (`max_abs_err == 0`) and clean-room ELF mapping (`maps_has_google_tpu_libs == false`):
   ```bash
   adb shell "/data/local/tmp/tpu_baremetal_runner \
     --dgc0 /data/local/tmp/freetpu_probe.dgc0 \
     --in /data/local/tmp/freetpu_probe.in.bin \
     --ref /data/local/tmp/freetpu_probe.ref.bin \
     --iters 10 --json"
   ```
   - **Errata guard**: Never pack 4-bit weights as nibbles for `0x980` (`LiteRT#12178`); unpack `Q4_0` at load time to `uint8` (`w + 8`) or use `W8A16 (BF16)` with per-channel `FP32` scales. See [references/silicon-cheatsheet.md](references/silicon-cheatsheet.md).

---

## Workflow 2: Headless Cloud TPU (`v5e-1` / `v6e-1`) via `google-colab-cli`

Use this workflow with `google-colab-cli >= 0.7.6` (redacted logs) to run Pallas/Mosaic or JAX/PJRT benchmarks on Colab TPUs without leaking Compute Units (CUs).

1. **Check CU balance & isolate session state** in a fresh private temp directory (always pass global flags before the subcommand):
   ```bash
   WORK_DIR="$(umask 077 && mktemp -d)"
   COLAB_STATE="$WORK_DIR/colab-sessions.json"
   colab --auth=adc --config "$COLAB_STATE" usage
   ```
   - Abort if `Current balance` is below your run budget (e.g., `< 5.0` CUs).
   - Always keep `--config "$COLAB_STATE"` inside your private `WORK_DIR` (`0700`) so parallel workers never clobber `~/.config/colab-cli/sessions.json` and shared `/tmp` state files cannot pre-exist under another owner.
   - `google-colab-cli v0.8.0` already uses `/tun/m` directly and does not accept `--no-public-api`; omit `--no-public-api` on releases that lack that option (pass it only on older releases exposing the Public API selector when targeting `/tun/m/*` on unallowlisted GCP quota projects).

2. **Run ephemeral probe with verified teardown and `v6e1 -> v5e1` fallback**:
   - Preferred wrapper (`tools/colab_tpu_bench.py`):
     ```bash
     # Offline check & command preview
     python3 tools/colab_tpu_bench.py --self-test
     python3 tools/colab_tpu_bench.py --dry-run --out-dir "$WORK_DIR/dry"

     # Live single-shot run (v6e1 with automatic v5e1 fallback on 412/400)
     python3 tools/colab_tpu_bench.py \
       --auth adc \
       --min-cu-balance 10 \
       --out-dir "$WORK_DIR/proof" \
       --script skills/freetpu/templates/tile_fused_gemm_silu.py
     ```
   - Direct CLI pattern (automatic `v6e1 -> v5e1` fallback is provided by `tools/colab_tpu_bench.py`; the direct CLI recipe runs a single selected `TPU_TYPE`, defaulting to `V6E1` — set `TPU_TYPE=V5E1` when `V6E1` quota or capacity is unavailable. Do **not** assume default `colab run` teardown succeeded: if unassign fails, `colab run` suppresses the error, removes the local session record, and still prints `Session terminated.` while leaving the TPU billing CUs. Instead, build a local launcher in `$WORK_DIR`, insulate `colab new` and `cleanup()` from `INT`/`TERM` so cancellation waits for `$SESSION_ID` to persist and finishes teardown uninterrupted, execute via `colab exec --file` under a local wall-clock watchdog, require `__FREETPU_EXEC_OK__`, and release by name via `colab stop` with ADC-propagating `colab usage` + `colab sessions` verification):
     ```bash
     (
       SESSION_ID="freetpu-probe-$(date +%Y%m%d-%H%M%S)-$$"
       TPU_TYPE="${TPU_TYPE:-V6E1}"
       BACKUP_STATE="$WORK_DIR/colab-state-backup.json"
       RUN_RC=0; ALLOC_PID=""; RUN_PID=""; WD_PID=""; ENDPOINT=""; CANCELLED=0; CLEANED=0

       # Build local launcher before allocating a TPU; abort on any write error:
       LAUNCHER="$WORK_DIR/$SESSION_ID.launch.py"
       if ! {
         printf 'import sys\nsys.argv = ["tile_swiglu.py"]\n' \
           && cat skills/freetpu/templates/tile_swiglu.py \
           && printf '\nprint("__FREETPU_EXEC_OK__")\n'
       } > "$LAUNCHER"; then
         echo "ERROR: failed to assemble launcher $LAUNCHER; aborting before TPU allocation." >&2
         exit 1
       fi

       BASE_SESS="$(colab --auth=adc --config "$COLAB_STATE" sessions 2>/dev/null)" || BASE_SESS=""
       BASE_ENDPOINTS="$(printf '%s\n' "$BASE_SESS" | awk '$1 ~ /^\[/ && $2 != "" {print $2}')"

       cleanup() {
         [ "$CLEANED" -eq 1 ] && return
         CLEANED=1
         trap '' INT TERM
         trap - EXIT

         [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null || true
         [ -n "$WD_PID" ] && wait "$WD_PID" 2>/dev/null || true
         WD_PID=""
         [ -n "$ALLOC_PID" ] && wait "$ALLOC_PID" 2>/dev/null || true
         [ ! -f "$BACKUP_STATE" ] && [ -f "$COLAB_STATE" ] && ( umask 077; cp -p "$COLAB_STATE" "$BACKUP_STATE" ) || true

         # Fall back to $BACKUP_STATE if `colab exec` pruned $COLAB_STATE on a 401/404 kernel error:
         STOP_STATE="$COLAB_STATE"
         PRE_SESS="$(colab --auth=adc --config "$COLAB_STATE" sessions 2>/dev/null)" || PRE_SESS=""
         if ! printf '%s\n' "$PRE_SESS" | awk -v s="[$SESSION_ID]" '$1 == s {found=1} END {exit !found}' && [ -f "$BACKUP_STATE" ]; then
           STOP_STATE="$BACKUP_STATE"
           PRE_SESS="$(colab --auth=adc --config "$BACKUP_STATE" sessions 2>/dev/null)" || PRE_SESS=""
         fi
         [ -z "$ENDPOINT" ] && ENDPOINT="$(printf '%s\n' "$PRE_SESS" | awk -v s="[$SESSION_ID]" '$1 == s {print $2; exit}')"

         colab --auth=adc --config "$STOP_STATE" log -s "$SESSION_ID" -o "$WORK_DIR/$SESSION_ID.ipynb" || true
         STOP_RC=0
         colab --auth=adc --config "$STOP_STATE" stop -s "$SESSION_ID" || STOP_RC=$?

         SESS_ERR="$WORK_DIR/sessions.err"
         NEW_UNNAMED=""
         TEARDOWN_RC=0
         if ! colab --auth=adc --config "$STOP_STATE" usage >/dev/null; then
           TEARDOWN_RC=1
         elif ! SESS_OUT="$(colab --auth=adc --config "$STOP_STATE" sessions 2>"$SESS_ERR")"; then
           TEARDOWN_RC=1
         elif grep -Eiq 'No valid default credentials|Failed to refresh' "$SESS_ERR"; then
           TEARDOWN_RC=1
         elif [ -n "$ENDPOINT" ]; then
           if [ "$STOP_RC" -ne 0 ] || printf '%s\n' "$SESS_OUT" | awk -v s="[$SESSION_ID]" -v e="$ENDPOINT" '$1 == s || $2 == e {found=1} END {exit !found}'; then
             TEARDOWN_RC=1
           fi
         else
           NEW_UNNAMED="$(printf '%s\n' "$SESS_OUT" | awk -v base="$BASE_ENDPOINTS" '
             BEGIN { n = split(base, arr, "\n"); for (i = 1; i <= n; i++) if (arr[i] != "") seen[arr[i]] = 1 }
             $1 == "[?]" && $2 != "" && !($2 in seen) { print $2 }
           ' | tr '\n' ' ' | sed 's/ *$//')"
           if [ "$STOP_RC" -ne 0 ] || [ "$RUN_RC" -eq 0 ] || [ "$CANCELLED" -eq 1 ] || [ -n "$NEW_UNNAMED" ] \
             || printf '%s\n' "$SESS_OUT" | awk -v s="[$SESSION_ID]" '$1 == s {found=1} END {exit !found}'; then
             TEARDOWN_RC=1
           fi
         fi

         ( umask 077; printf 'COLAB_STATE=%q\nBACKUP_STATE=%q\nSTOP_STATE=%q\nSESSION_ID=%q\nENDPOINT=%q\nORPHAN_ENDPOINTS=%q\n' \
             "$COLAB_STATE" "$BACKUP_STATE" "$STOP_STATE" "$SESSION_ID" "$ENDPOINT" "$NEW_UNNAMED" > "$WORK_DIR/recovery.env" )
         if [ "$TEARDOWN_RC" -ne 0 ]; then
           printf 'ERROR: TPU teardown unconfirmed (session=%s endpoint=%s orphan_endpoints=%s).\nRecovery: colab --auth=adc --config %q stop -s %q && colab --auth=adc --config %q sessions\nIdentifiers saved in %s (for unnamed [?] orphans, terminate in Colab Manage sessions and verify they disappear).\n' \
             "$SESSION_ID" "${ENDPOINT:-unknown}" "${NEW_UNNAMED:-none}" "$STOP_STATE" "$SESSION_ID" "$STOP_STATE" "$WORK_DIR/recovery.env" >&2
           exit 1
         fi
         exit "$RUN_RC"
       }

       on_signal() {
         trap '' INT TERM
         CANCELLED=1
         RUN_RC="$1"
         [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null || true
         [ -n "$RUN_PID" ] && kill -TERM "$RUN_PID" 2>/dev/null || true
         [ -n "$RUN_PID" ] && wait "$RUN_PID" 2>/dev/null || true
         cleanup
       }

       trap 'on_signal 130' INT
       trap 'on_signal 143' TERM
       trap 'cleanup' EXIT

       ( trap '' INT TERM; exec colab --auth=adc --config "$COLAB_STATE" new -s "$SESSION_ID" --tpu "$TPU_TYPE" ) &
       ALLOC_PID=$!
       wait "$ALLOC_PID" || RUN_RC=$?
       ALLOC_PID=""

       # Back up allocation state and capture ENDPOINT before `colab exec` (which prunes state on 401/404):
       [ -f "$COLAB_STATE" ] && ( umask 077; cp -p "$COLAB_STATE" "$BACKUP_STATE" ) || true
       if [ "$RUN_RC" -eq 0 ] && [ "$CANCELLED" -eq 0 ]; then
         ALLOC_SESS="$(colab --auth=adc --config "$COLAB_STATE" sessions 2>/dev/null)" || ALLOC_SESS=""
         ENDPOINT="$(printf '%s\n' "$ALLOC_SESS" | awk -v s="[$SESSION_ID]" '$1 == s {print $2; exit}')"
         EXEC_LOG="$WORK_DIR/$SESSION_ID.exec.log"
         colab --auth=adc --config "$COLAB_STATE" exec \
           -s "$SESSION_ID" --timeout 300 --file "$LAUNCHER" \
           >"$EXEC_LOG" 2>&1 &
         RUN_PID=$!
         python3 -c 'import os, signal, sys, time; time.sleep(float(sys.argv[1])); os.kill(int(sys.argv[2]), signal.SIGTERM)' \
           "${EXEC_DEADLINE_SECS:-360}" "$RUN_PID" >/dev/null 2>&1 &
         WD_PID=$!
         wait "$RUN_PID" || RUN_RC=$?
         RUN_PID=""
         kill "$WD_PID" 2>/dev/null || true
         wait "$WD_PID" 2>/dev/null || true
         WD_PID=""
         sed -E 's/([?&][^=&]*[Tt][Oo][Kk][Ee][Nn]=)[^&[:space:]]+/\1[REDACTED]/g' "$EXEC_LOG"
         if [ "$RUN_RC" -eq 0 ] && ! grep -Fxq "__FREETPU_EXEC_OK__" "$EXEC_LOG"; then
           echo "ERROR: remote TPU benchmark did not emit completion marker __FREETPU_EXEC_OK__." >&2
           RUN_RC=1
         fi
       fi
     )
     ```
   - **Teardown verification**:
     - **Wrapper (`tools/colab_tpu_bench.py`)**: Inspect `$WORK_DIR/proof/summary.json` (`session_name`, `created_endpoint`, `teardown_verified`, `recovery_config`, and `recovery_command`). If `teardown_verified` is not `true`, run the recorded `recovery_command` (or stop `<session_name>` against `<recovery_config>` and verify `<created_endpoint>` is absent from `colab --auth=adc --config <recovery_config> sessions` alongside a passing `colab usage`), or terminate `<created_endpoint>` in the Colab UI (`Runtime -> Manage sessions`) if local state is no longer present.
     - **Direct CLI (`$WORK_DIR/recovery.env`)**: After the subshell exits, `source "$WORK_DIR/recovery.env"` (or run the printed `Recovery:` command) to load `COLAB_STATE`, `BACKUP_STATE`, `STOP_STATE`, `SESSION_ID`, `ENDPOINT`, and `ORPHAN_ENDPOINTS`, then confirm `colab --auth=adc --config "$STOP_STATE" usage >/dev/null` and that `ENDPOINT`, `SESSION_ID`, and every entry in `ORPHAN_ENDPOINTS` are absent from `colab --auth=adc --config "$STOP_STATE" sessions`. If a named runtime survives, stop it against `$STOP_STATE`; when `ENDPOINT` is empty and `ORPHAN_ENDPOINTS` is non-empty, terminate each matching `[?]` runtime in Colab (`https://colab.research.google.com/` -> `Runtime -> Manage sessions`) and re-run `colab sessions` to confirm its endpoint disappears.
   - For multi-step interactive sessions, prefer `colab exec` (keeps XLA/PJRT caches warm) over `colab ssh` (capped at 5 sessions / HTTP 429). Use `colab download` instead of `drive.mount`.

---

## Workflow 3: Authoring & Benchmarking Tile-Based TPU Kernels

Start from the templates in `templates/`—each supports `--self-test` offline on CPU (NumPy) and full `tpu_custom_call` execution on Cloud TPU:

| Target | Template | Purpose |
| --- | --- | --- |
| Cloud TPU (`v5e-1` / `v6e-1`) | [templates/tile_fused_gemm_silu.py](templates/tile_fused_gemm_silu.py) | Double-buffered `GEMM + SiLU` via `pl.pallas_call` + `pltpu.make_async_copy` vs XLA & NumPy `cpu_ref` |
| Cloud TPU (`v5e-1` / `v6e-1`) | [templates/tile_swiglu.py](templates/tile_swiglu.py) | Bit-exact `SwiGLU` in `FP32 -> BF16` + HBM streaming bandwidth (`GB/s`) probe |
| Pixel Tensor G5 (`buenos_a0`) | [templates/buenos_gemm_probe.py](templates/buenos_gemm_probe.py) | Offline INT8 FP32-RNE GEMM + `0x640` minimax spline synthesis & optional `.dgc0` emission |

### Authoring Rules

- **Cloud TPU (Pallas / Mosaic)**:
  1. Tile `M, N, K` in multiples of MXU/vector lanes (`(8, 128)` minimum; `(512, 512, 256)` to `(1024, 1024, 256)` for large `4096^3` GEMMs).
  2. Keep per-kernel scratch inside the **`16 MiB` VMEM** budget (leave `vmem_limit_bytes` unset in `CompilerParams` unless `LIBTPU_INIT_ARGS` sets a strictly larger `--xla_tpu_scoped_vmem_limit_kib`).
  3. Accumulate in `jnp.float32` (`preferred_element_type=jnp.float32`) and cast to `jnp.bfloat16` only on final VMEM -> HBM store.
  4. Overlap HBM DMA with MXU compute via `pltpu.emit_pipeline` or explicit 2-slot `pltpu.make_async_copy(..., pltpu.SemaphoreType.DMA((2,)))`.
  5. Confirm native Mosaic lowering (`"tpu_custom_call" in fn.lower(...).as_text()`) before benchmarking.
  6. Note the **`~115-135 us` sync PJRT dispatch floor**: fuse epilogues (SiLU, SwiGLU, RMSNorm) or pipeline async dispatches rather than chaining tiny kernels.

- **Tensor G5 (`buenos_a0`)**:
  1. Respect `768 KiB` NM SRAM and default `libopenbuenos` tile limits (`64 x 512 x 256`); align `K` to 4 (or 32 for `Q8_0`/`Q4_0` blocks) and `N` to 8/32.
  2. Match hardware **IEEE-754 FP32 RNE** (`np.rint`) when computing CPU reference tensors.
  3. Fuse activations (`SiLU`, `GELU`, `exp`) into `0x640` minimax splines or `0xa40` chained graphs to avoid LPDDR round-trips (`27.59 GB/s` ceiling).
  4. Benchmark with `PM=5` (`1,286.4 MHz`) + Firmware WakeLock (`perf_mode >= 1`), never default `PM=0` (`38.4 MHz`).

### Verification & Reporting Checklist

1. Run offline syntax & parity self-test locally:
   ```bash
   python3 tools/validate_skill.py --run-self-tests
   ```
2. On Cloud TPU, verify `"tpu_custom_call": true`, `max_abs_err` vs `cpu_ref`/XLA, warmup + `jax.block_until_ready()`, `median_us`, and `TFLOP/s` or `GB/s`, and save `<session>.ipynb` via `colab log -o`.
3. On Tensor G5, report `hw_us`, `wall_ms`, `max_abs_err == 0`, and `maps_has_google_tpu_libs == false`.
