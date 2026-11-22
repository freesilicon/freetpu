# FreeTPU Tile DSL (`@freetpu.tile_kernel`)

Write **one** tiled kernel in Python and lower it cleanly to **both** Edge TPUs (Pixel Tensor G5 `buenos_a0`) and Cloud TPUs (`v5e-1` / `v6e-1` via Pallas/Mosaic `tpu_custom_call`), with deterministic pure-NumPy parity verification (`cpu_ref`).

---

## 60-Second Quickstart

```bash
# 1. Build libopenbuenos static + shared library for Python ctypes
make -C libopenbuenos host host-shared test

# 2. Run the full offline unit & parity suite (zero TPU/Colab credentials needed)
OPENBUENOS_MOCK=1 python3 -m freetpu_tile --self-test

# 3. Run single-source examples across cpu_ref, buenos_a0 (mock), and cloud_tpu_pallas (dry-run)
OPENBUENOS_MOCK=1 python3 examples/tile_dsl/fused_gemm_silu.py --target all
OPENBUENOS_MOCK=1 python3 examples/tile_dsl/swiglu.py --target all
```

---

## Single-Source Kernel Authoring (TileGym Style)

```python
import freetpu_tile as freetpu

@freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
def fused_gemm_silu(a, b, out):
    pid_m = freetpu.program_id(0)
    pid_n = freetpu.program_id(1)
    a_tile = freetpu.load(a, (pid_m, None), shape=(64, 512))
    b_tile = freetpu.load(b, (None, pid_n), shape=(512, 512))
    acc = freetpu.dot(a_tile, b_tile)
    freetpu.store(out, freetpu.silu(acc), (pid_m, pid_n))
```

Execute or inspect the exact same kernel on any target:

```python
# 1. Pure NumPy reference
y_ref = fused_gemm_silu(a, q_w, target="cpu_ref", scales={"b": s_w})

# 2. Edge TPU Tensor G5 (buenos_a0) — offline mock or live ADB rango
y_edge = fused_gemm_silu(a, q_w, target="buenos_a0", scales={"b": s_w})
# Live on rooted Pixel 10 Pro Fold (rango):
#   fused_gemm_silu(a, q_w, target="buenos_a0", scales={"b": s_w}, adb_serial="<serial>")

# 3. Cloud TPU (v5e-1 / v6e-1) — local Pallas lowering or live Colab TPU
lowered = fused_gemm_silu.lower(a, b, target="cloud_tpu_pallas")
print(lowered.source_code)  # Emits pl.pallas_call + pl.BlockSpec + pltpu.VMEM
```

---

## Supported Ops, Dtypes & Backend Lowering

| Surface | Supported | Lowering Target |
| --- | --- | --- |
| **Core Ops** | `matmul`, `fused_matmul_silu`, `fused_matmul_gelu`, `swiglu` | Pattern-fused from symbolic `@tile_kernel` DAG (`dot`, `silu`, `gelu`, `mul`, `store`) |
| **Dtypes** | `bf16`, `w8a16`, `int8` | Shared `QuantParams` & per-channel `W8A16` scale handling |
| **`cpu_ref`** | Pure NumPy | IEEE BF16-rounded / FP32-RNE deterministic parity reference |
| **`buenos_a0`** | Edge TPU Tensor G5 | `tools/emit_custom_gemm.py` + `tools/multitile_synth.py` VLIW chunks, Mode-A `+0 B` `0x640` minimax splines (`tools/spline_0x640.py`), `libopenbuenos` `ctypes` C-API (`OPENBUENOS_MOCK=1` or `/dev/edgetpu-soc`) |
| **`cloud_tpu_pallas`** | Cloud TPU `v5e-1` / `v6e-1` | JAX Pallas/Mosaic (`pl.pallas_call`, `pl.BlockSpec`, `pltpu.VMEM`, `CompilerParams` -> `tpu_custom_call`) |

---

## Headless Execution on Live Silicon

### Edge TPU (`buenos_a0` / Pixel Tensor G5 via ADB)

Build the zero-Google-`.so` Android ARM64 runner and run the single-source example against `/dev/edgetpu-soc` (`PM=5` @ `1,286.4 MHz` + FW WakeLock):

```bash
make -C runtime tile-runner verify-tile-runner
python3 examples/tile_dsl/fused_gemm_silu.py --target buenos_a0 --adb-serial <ADB_SERIAL> --dtype w8a16 --M 64 --K 512 --N 512
```

### Cloud TPU (`v6e-1` / `v5e-1` via `google-colab-cli`)

Run the exact same unmodified example script headlessly with isolated session state, CU budget pre-check, `v6e1 -> v5e1` fallback, auto-stop, and `.ipynb` log capture:

```bash
python3 -m freetpu_tile.colab_runner examples/tile_dsl/fused_gemm_silu.py \
  --out-dir runs/colab_proof \
  -- --target cloud_tpu_pallas --dtype w8a16 --M 64 --K 512 --N 512 --iters 20
```

Dry-run offline without Colab credentials:

```bash
python3 -m freetpu_tile.colab_runner examples/tile_dsl/fused_gemm_silu.py --dry-run
```

---

## Agent Authoring Checklist (TileGym Pattern)

1. Keep the kernel pure inside `@freetpu.tile_kernel(block_m=..., block_n=..., block_k=..., dtype=...)` using `freetpu.program_id`, `load`, `dot`, `silu`/`gelu`, `mul`, and `store`.
2. Choose `block_k` and `block_n` as multiples of `128` (e.g. `512`) and `block_m >= 8` (e.g. `64`) so one kernel source satisfies both Cloud TPU Mosaic `(8, 128)` VREG/MXU alignment and Tensor G5 `lt_64x` (`64 x K x N`) single-tile execution.
3. Validate offline first with `OPENBUENOS_MOCK=1` and `kernel.parity_check(...)` before dispatching to live silicon.
