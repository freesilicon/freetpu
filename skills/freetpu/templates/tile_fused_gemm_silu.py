#!/usr/bin/env python3
"""Starter template: Pallas/Mosaic double-buffered BF16 GEMM + SiLU for Cloud TPU (v5e-1 / v6e-1).

Features:
  - Double-buffered HBM -> VMEM DMA via `pltpu.make_async_copy` inside the 16 MiB VMEM ceiling.
  - FP32 accumulation + fused SiLU epilogue (`x * sigmoid(x)`) before BF16 writeback.
  - Verifies `tpu_custom_call` Mosaic lowering and numerical parity vs XLA & NumPy `cpu_ref`.
  - Supports `--self-test` offline on any machine using pure NumPy (zero JAX/TPU required).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, Tuple

import numpy as np

VMEM_LIMIT_BYTES = 16 * 1024 * 1024


def estimate_vmem_bytes(bm: int, bn: int, bk: int) -> int:
    """Return scratch VMEM footprint for 2-slot BF16 A/B double buffers + FP32 accumulator."""
    a_db = 2 * bm * bk * 2
    b_db = 2 * bk * bn * 2
    acc = bm * bn * 4
    return a_db + b_db + acc


def cpu_ref_gemm_silu(a_f32: np.ndarray, b_f32: np.ndarray) -> np.ndarray:
    """Clean NumPy FP32 reference for GEMM + SiLU."""
    acc = a_f32.astype(np.float32) @ b_f32.astype(np.float32)
    sig = np.where(acc >= 0.0, 1.0 / (1.0 + np.exp(-acc)), np.exp(acc) / (1.0 + np.exp(acc)))
    return (acc * sig).astype(np.float32)


def tiled_cpu_gemm_silu(a_f32: np.ndarray, b_f32: np.ndarray, bm: int, bn: int, bk: int) -> np.ndarray:
    """Reference tiled execution verifying block decomposition against `cpu_ref_gemm_silu`."""
    m, k = a_f32.shape
    k2, n = b_f32.shape
    if k != k2:
        raise ValueError(f"Inner dim mismatch: {k} != {k2}")
    if m % bm or n % bn or k % bk:
        raise ValueError(f"Shape ({m}, {k}, {n}) must be divisible by tile ({bm}, {bk}, {bn})")
    out = np.empty((m, n), dtype=np.float32)
    for i in range(0, m, bm):
        for j in range(0, n, bn):
            acc = np.zeros((bm, bn), dtype=np.float32)
            for kk in range(0, k, bk):
                acc += a_f32[i : i + bm, kk : kk + bk] @ b_f32[kk : kk + bk, j : j + bn]
            sig = np.where(acc >= 0.0, 1.0 / (1.0 + np.exp(-acc)), np.exp(acc) / (1.0 + np.exp(acc)))
            out[i : i + bm, j : j + bn] = acc * sig
    return out


def build_pallas_gemm_silu(m: int, k: int, n: int, bm: int, bn: int, bk: int):
    """Build a double-buffered Pallas/Mosaic `GEMM + SiLU` kernel for Cloud TPU."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if m % bm or n % bn or k % bk:
        raise ValueError(f"Shape ({m}, {k}, {n}) must be divisible by ({bm}, {bk}, {bn})")
    vmem_est = estimate_vmem_bytes(bm, bn, bk)
    if vmem_est > VMEM_LIMIT_BYTES:
        raise ValueError(f"Tile ({bm}, {bn}, {bk}) uses {vmem_est} B > {VMEM_LIMIT_BYTES} B VMEM budget")

    num_k = k // bk
    if num_k < 2:
        raise ValueError("Double-buffered K-loop requires K // bk >= 2")

    CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams")

    def outer_kernel(a_hbm_ref, b_hbm_ref, o_ref, a_bufs, b_bufs, acc_ref, a_sems, b_sems):
        i = pl.program_id(0)
        j = pl.program_id(1)
        m_off = i * bm
        n_off = j * bn

        def start_copy(k_idx: int, slot: int):
            k_off = k_idx * bk
            a_cp = pltpu.make_async_copy(
                a_hbm_ref.at[pl.ds(m_off, bm), pl.ds(k_off, bk)],
                a_bufs.at[slot],
                a_sems.at[slot],
            )
            b_cp = pltpu.make_async_copy(
                b_hbm_ref.at[pl.ds(k_off, bk), pl.ds(n_off, bn)],
                b_bufs.at[slot],
                b_sems.at[slot],
            )
            a_cp.start()
            b_cp.start()
            return a_cp, b_cp

        pending = {0: start_copy(0, 0)}
        for k_idx in range(num_k):
            cur = k_idx % 2
            if k_idx + 1 < num_k:
                nxt = (k_idx + 1) % 2
                pending[nxt] = start_copy(k_idx + 1, nxt)
            a_cp, b_cp = pending[cur]
            a_cp.wait()
            b_cp.wait()
            prod = jnp.dot(a_bufs[cur], b_bufs[cur], preferred_element_type=jnp.float32)
            if k_idx == 0:
                acc_ref[...] = prod
            else:
                acc_ref[...] += prod

        v = acc_ref[...]
        o_ref[...] = (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)

    @jax.jit
    def run(a, b):
        return pl.pallas_call(
            outer_kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
            grid=(m // bm, n // bn),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((2, bm, bk), jnp.bfloat16),
                pltpu.VMEM((2, bk, bn), jnp.bfloat16),
                pltpu.VMEM((bm, bn), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
        )(a, b)

    return run


def run_self_test() -> Dict[str, Any]:
    """Offline CPU parity and VMEM budget check (no JAX/TPU hardware required)."""
    rng = np.random.default_rng(0)
    m, k, n = 128, 256, 128
    bm, bn, bk = 64, 64, 64
    a = (rng.standard_normal((m, k)) / math.sqrt(k)).astype(np.float32)
    b = (rng.standard_normal((k, n)) / math.sqrt(k)).astype(np.float32)

    ref = cpu_ref_gemm_silu(a, b)
    tiled = tiled_cpu_gemm_silu(a, b, bm, bn, bk)
    max_abs_err = float(np.max(np.abs(ref - tiled)))
    vmem_4096_tile = estimate_vmem_bytes(1024, 1024, 256)
    assert max_abs_err < 1e-6, f"CPU tiled parity failed: max_abs_err={max_abs_err}"
    assert vmem_4096_tile <= VMEM_LIMIT_BYTES

    return {
        "status": "ok",
        "mode": "cpu_self_test",
        "shape": [m, k, n],
        "tile": [bm, bn, bk],
        "max_abs_err": max_abs_err,
        "vmem_bytes_1024x1024x256": vmem_4096_tile,
        "vmem_limit_bytes": VMEM_LIMIT_BYTES,
    }


def run_tpu_benchmark(m: int, k: int, n: int, bm: int, bn: int, bk: int, iters: int) -> Dict[str, Any]:
    """Execute and benchmark Pallas/Mosaic GEMM+SiLU on Cloud TPU vs XLA + NumPy cpu_ref."""
    import jax
    import jax.numpy as jnp

    devs = jax.devices()
    if devs[0].platform != "tpu":
        raise RuntimeError(f"Expected TPU device, got {devs[0]} (use --self-test for offline CPU check)")

    key = jax.random.PRNGKey(0)
    a = (jax.random.normal(key, (m, k), dtype=jnp.bfloat16) / math.sqrt(k))
    b = (jax.random.normal(jax.random.PRNGKey(1), (k, n), dtype=jnp.bfloat16) / math.sqrt(k))
    jax.block_until_ready((a, b))

    @jax.jit
    def xla_fn(x, y):
        v = jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        return (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)

    pallas_fn = build_pallas_gemm_silu(m, k, n, bm, bn, bk)
    has_tpu_custom_call = "tpu_custom_call" in pallas_fn.lower(a, b).as_text()

    ref_xla = xla_fn(a, b)
    out_pallas = pallas_fn(a, b)
    jax.block_until_ready((ref_xla, out_pallas))
    max_abs_err_xla = float(jnp.max(jnp.abs(out_pallas.astype(jnp.float32) - ref_xla.astype(jnp.float32))))

    # Check a corner slice against NumPy FP32 cpu_ref to guard against global scale bugs.
    sl = min(64, m, n)
    cpu_slice = cpu_ref_gemm_silu(np.asarray(a[:sl], dtype=np.float32), np.asarray(b[:, :sl], dtype=np.float32))
    max_abs_err_cpu = float(np.max(np.abs(np.asarray(out_pallas[:sl, :sl], dtype=np.float32) - cpu_slice)))

    def _bench(fn) -> Tuple[float, float]:
        for _ in range(4):
            fn(a, b).block_until_ready()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(a, b).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.min(ts))

    p_med_s, p_min_s = _bench(pallas_fn)
    x_med_s, _ = _bench(xla_fn)
    flops = 2.0 * m * k * n

    return {
        "status": "ok",
        "mode": "cloud_tpu_pallas",
        "device_kind": devs[0].device_kind,
        "shape": [m, k, n],
        "tile": [bm, bn, bk],
        "vmem_bytes": estimate_vmem_bytes(bm, bn, bk),
        "tpu_custom_call": has_tpu_custom_call,
        "max_abs_err_vs_xla": max_abs_err_xla,
        "max_abs_err_vs_cpu_slice": max_abs_err_cpu,
        "median_us": round(p_med_s * 1e6, 2),
        "min_us": round(p_min_s * 1e6, 2),
        "median_tflops": round(flops / (p_med_s * 1e12), 2),
        "peak_tflops": round(flops / (p_min_s * 1e12), 2),
        "xla_median_us": round(x_med_s * 1e6, 2),
        "pct_of_xla": round(100.0 * (x_med_s / p_med_s), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pallas/Mosaic double-buffered GEMM + SiLU starter template")
    ap.add_argument("--self-test", action="store_true", help="Run offline NumPy parity & VMEM budget check")
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--bm", type=int, default=1024)
    ap.add_argument("--bn", type=int, default=1024)
    ap.add_argument("--bk", type=int, default=256)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2))
        return
    print(json.dumps(run_tpu_benchmark(args.M, args.K, args.N, args.bm, args.bn, args.bk, args.iters), indent=2))


if __name__ == "__main__":
    main()
