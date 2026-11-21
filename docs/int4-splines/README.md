# Tensor G5 (`buenos_a0`) `INT4` (`W4A8`) & VICA `0x640` LLM Splines (`tpu-int4-splines`)

100% offline preparation package for the Open Pixel TPU Stack, delivering the two highest-leverage LLM primitives for `ggml-edgetpu` on Tensor G5 (`buenos_a0`):

1. **Part A — Clean-Room `0x640` Minimax Degree-4 Splines (`SiLU`, `GELU`, `exp(x)`) + Live-Silicon Probe Blobs** ([`splines_silu_gelu_exp.md`](splines_silu_gelu_exp.md))
2. **Part B — Native `INT4` (`W4A8` / `W4A16` / Blockwise) Compiler RE + 19-Model `.tflite` Sweep Pack** ([`int4_w4a8_isa.md`](int4_w4a8_isa.md))

---

## 1. Artifact Inventory

### Code & Generated Artifacts (`work/int4-splines/`)

| Artifact | Purpose | Verification Status |
|---|---|---|
| [`spline_0x640.py`](../../tools/spline_0x640.py) | `0x640` codec, Stage 1 / `stream125` patcher, pure-NumPy Remez minimax synthesizer, FP32-RNE Horner hardware simulator | `--self-test` PASS: `0x640` 3×208B round-trip exact; Stage 1 transform 5520/5520 B exact; `logistic_scale_zp` 4096/4096 exact vs `rango` silicon |
| `synthesized_splines.json` | Machine-readable clean-room 8-segment degree-4 tables for `SiLU`, `GELU`, `exp(x)` | Max quantized error: `0.00234` / `0.00186` / `0.00045` LSB (`<< 0.5` LSB) |
| `spline_probes/manifest.json` | 3 ready-to-run patched `.dgc0` probes (`silu_probe`, `gelu_probe`, `exp_probe`) + `.in.bin` + `.ref.bin` | All 3 parse cleanly with `dgc0_dump.py` (`663,296 B`, `buenos_a0`, `chunk[3]=5520 B`) |
| [`gen_int4_sweep.py`](../../tools/gen_int4_sweep.py) | Pure-Python TFLite v3 `INT4` + `BlockwiseQuantization` + `W4A16` + `DEQUANTIZE` builder & validator | Emits 19 validated `.tflite` + `.in.bin` + `.ref.bin` + `manifest.json` |
| `sweep_int4/manifest.json` | 19-model `run_oracle_batch.py`-compatible sweep manifest | Strict TFL3 schema & `ceil(N*K/2)` INT4 byte-length validated; `fc_i4_pt_1x16x16_id.ref.bin == fc_i8_pt_1x16x16_id_ref.ref.bin` |

### Probe SHA-256 Digest Table (`work/int4-splines/spline_probes/`)

| Probe | `dgc0` SHA-256 (prefix) | `ref.bin` SHA-256 (prefix) | Max Quantized LSB Err | Offline Exact / 256 |
|---|---|---|---:|---:|
| `silu_probe` | `3dd9430f58a0c34e` | `536adf93626017f9` | `0.002338` | `255 / 256` (`max_diff=1`) |
| `gelu_probe` | `409d6d032cf84d88` | `b76e34a08dad03a0` | `0.001858` | `256 / 256` (`max_diff=0`) |
| `exp_probe` | `e467188cfaae99c1` | `1bb112ab1b2796ec` | `0.000449` | `256 / 256` (`max_diff=0`) |

---

## 2. Consolidated `[OBSERVED]` / `[INFERRED]` / `[UNKNOWN]` Ledger

- **`[OBSERVED]`** `0x640` 208-byte wire format, `0x200@60` `stream125` (`mul[44:13]`, `fclip_max[77:46]`, `fclip_min[114:83]`), and 4-site Stage 1 patcher verified bit-exact (`5520/5520` B) against `vec_logistic_1x16x16x16.dgc0`; FP32-RNE Horner simulator matches live `rango` `logistic_scale_zp.out.bin` `4096/4096`.
- **`[OBSERVED]`** All 34 embedded `*_spline.pb` LUTs decoded from `libedgetpu_tflite_compiler.so`; clean-room Remez minimax splines for `SiLU`, `GELU`, and `exp(x)` achieve `4.69e-5`, `3.72e-5`, and `1.88e-6` max FP32 error (`< 0.0024` quantized LSB).
- **`[OBSERVED]`** `libedgetpu_tflite_compiler.so` and `libedgetpu_litert.so` accept `TensorType_INT4` (`0x11`) weights on `FULLY_CONNECTED` with `INT8`/`INT16` activations (rejecting `INT4` inputs and `< 4 bit` subbyte conversion), pack `FIXED_POINT4` as unsigned `[0, 15]` (`+8` offset) in little-endian nibble order (`even -> [3:0], odd -> [7:4]`), enforce `kAllowedParameterSubChannelBlockSize == 32`, and have `0` string references to TFL3 `BlockwiseQuantization`.
- **`[INFERRED]`** `silu_probe.dgc0`, `gelu_probe.dgc0`, and `exp_probe.dgc0` will match `.ref.bin` on `rango` within `<= 1` LSB (`0` LSB expected if hardware uses standard FP32 Horner).
- **`[INFERRED]`** Per-tensor and per-channel `INT4` `FULLY_CONNECTED` models (#2–#14) will compile to `buenos_a0` `.dgc0` on `rango`, whereas TFL3 `BlockwiseQuantization` (#15–#17) may reject at `GetQuantizedType` and require explicit `go/odml-subchannel-quant` lowering.
- **`[UNKNOWN]`** Live `rango` execution of `silu_probe` / `gelu_probe` / `exp_probe` and the exact `0x040`/`0x440`/`0x500` bitfield delta between `fc_i4_pt_1x16x16_id` and `fc_i8_pt_1x16x16_id_ref` (both ready for 1-command `rango-runner` execution below).

---

## 3. Ready-to-Run `rango` Verification Commands (For `rango-runner`)

> **Do NOT run from offline lanes.** Execute from the USB-owning `rango-runner` lane once `rango` is free.

### A. Run the 19-Model `INT4` (`W4A8` / `W4A16` / Blockwise) Compile + Execute Sweep

```bash
python3 <corpus-dir>/work/device-runner/run_oracle_batch.py \
  --manifest <corpus-dir>/work/int4-splines/sweep_int4/manifest.json \
  --out-dir <corpus-dir>/work/device-runner/phase4_sweep_int4
```

Then diff the paired INT8 vs INT4 identity twins immediately:

```bash
python3 tools/dgc0/dgc0_dump.py \
  --diff \
  <corpus-dir>/work/device-runner/phase4_sweep_int4/fc_i8_pt_1x16x16_id_ref.dgc0 \
  <corpus-dir>/work/device-runner/phase4_sweep_int4/fc_i4_pt_1x16x16_id.dgc0
```

### B. Execute the 3 Patched `0x640` `SiLU` / `GELU` / `exp` Live-Silicon Probes

```bash
PROBE_DIR=<corpus-dir>/work/int4-splines/spline_probes
OUT_DIR=<corpus-dir>/work/device-runner/phase4_spline_probes
mkdir -p "$OUT_DIR"
adb shell "mkdir -p /data/local/tmp/tpu_splines"
adb push "$PROBE_DIR"/*.dgc0 "$PROBE_DIR"/*.in.bin /data/local/tmp/tpu_splines/

for p in silu_probe gelu_probe exp_probe; do
  adb shell "/data/local/tmp/tpu_direct_runner --run-dgc0 /data/local/tmp/tpu_splines/${p}.dgc0 /data/local/tmp/tpu_splines/${p}.in.bin /data/local/tmp/tpu_splines/${p}.out.bin"
  adb pull "/data/local/tmp/tpu_splines/${p}.out.bin" "$OUT_DIR/${p}.out.bin"
  python3 -c "import numpy as np; r=np.fromfile('$PROBE_DIR/${p}.ref.bin',np.int8); o=np.fromfile('$OUT_DIR/${p}.out.bin',np.int8); d=np.abs(r.astype(int)-o.astype(int)); print('${p}: exact=%d/4096 max_diff=%d' % (int((d==0).sum()), int(d.max())))"
done
```
