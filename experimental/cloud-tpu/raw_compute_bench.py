#!/usr/bin/env python3
"""Raw BF16/INT8 GEMM TFLOP/s & HBM streaming bandwidth benchmark for Cloud TPU."""

import importlib.metadata
import json
import time

import jax
from jax.extend import backend as jax_backend
import jax.numpy as jnp
import numpy as np


def main() -> None:
  print("=== RAW TPU COMPUTE & HBM BANDWIDTH BENCHMARK ===")
  pkgs = {}
  for name in ("jax", "jaxlib", "libtpu", "tpu-info"):
    try:
      pkgs[name] = importlib.metadata.version(name)
    except Exception:
      pass
  print("VERSIONS:", json.dumps(pkgs))
  be = jax_backend.get_backend()
  print("PJRT_PLATFORM:", be.platform, "VERSION:", be.platform_version.replace("\n", " | "))
  for d in jax.devices():
    ms = {k: int(v) for k, v in d.memory_stats().items()}
    print(
        "DEVICE:",
        json.dumps({
            "str": str(d),
            "kind": d.device_kind,
            "hbm_gib": round(ms["bytes_limit"] / (1024**3), 3),
            "hbm_gb": round(ms["bytes_limit"] / 1e9, 3),
        }),
    )

  key = jax.random.PRNGKey(0)

  def bench_matmul(m, k, n, dtype_in, dtype_out, warmup=5, iters=25):
    if dtype_in == jnp.int8:
      a = jax.random.randint(key, (m, k), -8, 8, dtype=jnp.int8)
      b = jax.random.randint(key, (k, n), -8, 8, dtype=jnp.int8)
    else:
      a = jax.random.normal(key, (m, k), dtype=dtype_in)
      b = jax.random.normal(key, (k, n), dtype=dtype_in)
    a.block_until_ready()
    b.block_until_ready()

    @jax.jit
    def fn(x, y):
      return jax.lax.dot_general(
          x, y, (((1,), (0,)), ((), ())), preferred_element_type=dtype_out
      )

    for _ in range(warmup):
      fn(a, b).block_until_ready()
    ts = []
    for _ in range(iters):
      t0 = time.perf_counter()
      fn(a, b).block_until_ready()
      ts.append(time.perf_counter() - t0)
    med_s = float(np.median(ts))
    min_s = float(np.min(ts))
    flops = 2.0 * m * k * n
    return {
        "shape": f"{m}x{k}x{n}",
        "dtype": f"{jnp.dtype(dtype_in).name}->{jnp.dtype(dtype_out).name}",
        "median_us": round(med_s * 1e6, 2),
        "min_us": round(min_s * 1e6, 2),
        "median_tflops": round(flops / (med_s * 1e12), 2),
        "peak_tflops": round(flops / (min_s * 1e12), 2),
    }

  cases = [
      (1, 64, 32, jnp.bfloat16, jnp.float32),
      (16, 128, 128, jnp.bfloat16, jnp.float32),
      (1, 576, 1536, jnp.bfloat16, jnp.float32),
      (16, 576, 576, jnp.bfloat16, jnp.float32),
      (1024, 1024, 1024, jnp.bfloat16, jnp.float32),
      (2048, 2048, 2048, jnp.bfloat16, jnp.float32),
      (4096, 4096, 4096, jnp.bfloat16, jnp.float32),
      (8192, 8192, 8192, jnp.bfloat16, jnp.float32),
      (4096, 4096, 4096, jnp.bfloat16, jnp.bfloat16),
      (8192, 8192, 8192, jnp.bfloat16, jnp.bfloat16),
      (16384, 16384, 16384, jnp.bfloat16, jnp.bfloat16),
      (4096, 4096, 4096, jnp.int8, jnp.int32),
      (8192, 8192, 8192, jnp.int8, jnp.int32),
      (16384, 16384, 16384, jnp.int8, jnp.int32),
  ]
  bf16_ok = 0
  int8_ok = 0
  bw_ok = 0
  for case in cases:
    try:
      print("MATMUL:", json.dumps(bench_matmul(*case)))
      if case[3] == jnp.int8:
        int8_ok += 1
      else:
        bf16_ok += 1
    except Exception as e:
      print("MATMUL_ERR:", case[:3], repr(e)[:200])

  for n_elems in (32 * 1024 * 1024, 128 * 1024 * 1024, 512 * 1024 * 1024):
    try:
      x = jax.random.normal(key, (n_elems,), dtype=jnp.bfloat16)
      y = jax.random.normal(key, (n_elems,), dtype=jnp.bfloat16)
      s = jnp.array(1.25, dtype=jnp.bfloat16)
      x.block_until_ready()
      y.block_until_ready()

      @jax.jit
      def triad(a, b, alpha):
        return a + alpha * b

      for _ in range(5):
        triad(x, y, s).block_until_ready()
      ts = []
      for _ in range(25):
        t0 = time.perf_counter()
        triad(x, y, s).block_until_ready()
        ts.append(time.perf_counter() - t0)
      med_s = float(np.median(ts))
      min_s = float(np.min(ts))
      total_b = 3.0 * n_elems * 2
      print(
          "HBM_BW:",
          json.dumps({
              "tensor_mb": round(n_elems * 2 / 1e6, 1),
              "total_mb": round(total_b / 1e6, 1),
              "median_us": round(med_s * 1e6, 2),
              "median_gb_s": round(total_b / (med_s * 1e9), 2),
              "peak_gb_s": round(total_b / (min_s * 1e9), 2),
          }),
      )
      bw_ok += 1
      del x, y, s
    except Exception as e:
      print("HBM_BW_ERR:", n_elems, repr(e)[:200])

  if bf16_ok < 1 or int8_ok < 1 or bw_ok < 1:
    raise RuntimeError(
        f"Raw TPU benchmark incomplete: bf16_ok={bf16_ok}, int8_ok={int8_ok}, bw_ok={bw_ok}"
    )
  print("=== RAW_BENCH_COMPLETE ===")


if __name__ == "__main__":
  main()
