#!/usr/bin/env python3
"""Starter template: Pallas/Mosaic bit-exact SwiGLU + HBM bandwidth probe for Cloud TPU.

Computes `SwiGLU(gate, up) = (gate * sigmoid(gate)) * up` in FP32 and stores BF16.
Supports `--self-test` offline on any host with pure NumPy, and full `tpu_custom_call`
parity + HBM streaming bandwidth measurement (`GB/s`) on Cloud TPU (`v5e-1` / `v6e-1`).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Tuple

import numpy as np

V5E_PEAK_HBM_GBS = 820.0


def cpu_ref_swiglu(gate_f32: np.ndarray, up_f32: np.ndarray) -> np.ndarray:
    """Clean NumPy FP32 reference for SwiGLU."""
    g = gate_f32.astype(np.float32)
    u = up_f32.astype(np.float32)
    sig = np.where(g >= 0.0, 1.0 / (1.0 + np.exp(-g)), np.exp(g) / (1.0 + np.exp(g)))
    return ((g * sig) * u).astype(np.float32)


def tiled_cpu_swiglu(
    gate_f32: np.ndarray, up_f32: np.ndarray, block_m: int, block_n: int
) -> np.ndarray:
    """Reference tiled SwiGLU verifying block spec partitioning."""
    m, n = gate_f32.shape
    if up_f32.shape != (m, n):
        raise ValueError(f"Shape mismatch: {gate_f32.shape} vs {up_f32.shape}")
    if m % block_m or n % block_n:
        raise ValueError(f"Shape ({m}, {n}) must be divisible by ({block_m}, {block_n})")
    out = np.empty((m, n), dtype=np.float32)
    for i in range(0, m, block_m):
        for j in range(0, n, block_n):
            out[i : i + block_m, j : j + block_n] = cpu_ref_swiglu(
                gate_f32[i : i + block_m, j : j + block_n],
                up_f32[i : i + block_m, j : j + block_n],
            )
    return out


def build_pallas_swiglu(m: int, n: int, block_m: int, block_n: int):
    """Build a Pallas/Mosaic SwiGLU kernel lowering to `tpu_custom_call`."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if m % block_m or n % block_n:
        raise ValueError(f"Shape ({m}, {n}) must be divisible by ({block_m}, {block_n})")
    CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams")

    def kernel(gate_ref, up_ref, out_ref):
        g = gate_ref[...].astype(jnp.float32)
        u = up_ref[...].astype(jnp.float32)
        out_ref[...] = ((g * jax.nn.sigmoid(g)) * u).astype(jnp.bfloat16)

    @jax.jit
    def run(gate, up):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
            grid=(m // block_m, n // block_n),
            in_specs=[
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                disable_bounds_checks=True,
            ),
        )(gate, up)

    return run


def run_self_test() -> Dict[str, Any]:
    """Offline NumPy parity check (no JAX/TPU hardware required)."""
    rng = np.random.default_rng(7)
    m, n = 64, 1024
    block_m, block_n = 32, 256
    gate = rng.standard_normal((m, n), dtype=np.float32)
    up = rng.standard_normal((m, n), dtype=np.float32)

    ref = cpu_ref_swiglu(gate, up)
    tiled = tiled_cpu_swiglu(gate, up, block_m, block_n)
    max_abs_err = float(np.max(np.abs(ref - tiled)))
    assert max_abs_err == 0.0, f"Tiled SwiGLU mismatch: {max_abs_err}"

    return {
        "status": "ok",
        "mode": "cpu_self_test",
        "shape": [m, n],
        "tile": [block_m, block_n],
        "max_abs_err": max_abs_err,
    }


def run_tpu_benchmark(m: int, n: int, block_m: int, block_n: int, iters: int) -> Dict[str, Any]:
    """Benchmark Pallas SwiGLU vs XLA and report HBM streaming bandwidth."""
    import jax
    import jax.numpy as jnp

    devs = jax.devices()
    if devs[0].platform != "tpu":
        raise RuntimeError(f"Expected TPU device, got {devs[0]} (use --self-test for offline CPU check)")

    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    gate = jax.random.normal(k1, (m, n), dtype=jnp.bfloat16)
    up = jax.random.normal(k2, (m, n), dtype=jnp.bfloat16)
    jax.block_until_ready((gate, up))

    @jax.jit
    def xla_swiglu(g, u):
        gf = g.astype(jnp.float32)
        uf = u.astype(jnp.float32)
        return ((gf * jax.nn.sigmoid(gf)) * uf).astype(jnp.bfloat16)

    pallas_fn = build_pallas_swiglu(m, n, block_m, block_n)
    has_tpu_custom_call = "tpu_custom_call" in pallas_fn.lower(gate, up).as_text()

    ref_xla = xla_swiglu(gate, up)
    out_pallas = pallas_fn(gate, up)
    jax.block_until_ready((ref_xla, out_pallas))
    max_abs_err = float(jnp.max(jnp.abs(out_pallas.astype(jnp.float32) - ref_xla.astype(jnp.float32))))

    def _bench(fn) -> Tuple[float, float]:
        for _ in range(5):
            fn(gate, up).block_until_ready()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(gate, up).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.min(ts))

    p_med_s, p_min_s = _bench(pallas_fn)
    x_med_s, _ = _bench(xla_swiglu)

    # 2 BF16 reads (gate, up) + 1 BF16 write (out) = 6 bytes per element.
    total_bytes = float(m * n * 6)
    med_gbs = (total_bytes / p_med_s) / 1e9
    peak_gbs = (total_bytes / p_min_s) / 1e9

    return {
        "status": "ok",
        "mode": "cloud_tpu_pallas",
        "device_kind": devs[0].device_kind,
        "shape": [m, n],
        "tile": [block_m, block_n],
        "tpu_custom_call": has_tpu_custom_call,
        "max_abs_err_vs_xla": max_abs_err,
        "total_hbm_mb": round(total_bytes / 1e6, 2),
        "median_us": round(p_med_s * 1e6, 2),
        "min_us": round(p_min_s * 1e6, 2),
        "median_gb_s": round(med_gbs, 2),
        "peak_gb_s": round(peak_gbs, 2),
        "pct_of_v5e_peak_bw": round(100.0 * med_gbs / V5E_PEAK_HBM_GBS, 2),
        "xla_median_us": round(x_med_s * 1e6, 2),
        "pct_of_xla": round(100.0 * (x_med_s / p_med_s), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pallas/Mosaic SwiGLU + HBM bandwidth probe template")
    ap.add_argument("--self-test", action="store_true", help="Run offline NumPy parity check")
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return
    print(json.dumps(run_tpu_benchmark(args.M, args.N, args.bm, args.bn, args.iters), indent=2))


if __name__ == "__main__":
    main()
