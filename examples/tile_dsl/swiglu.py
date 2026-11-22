#!/usr/bin/env python3
"""Single-source FreeTPU Tile DSL example: Fused SwiGLU MLP block.

Computes:
    out = (SiLU(x @ w_gate) * (x @ w_up)) @ w_down

Lowers cleanly from the exact same `@freetpu.tile_kernel` source to:
  1. `cpu_ref`:          NumPy reference
  2. `buenos_a0`:        Batched shared-A `Gate(SiLU) + Up` + `Down` via `libopenbuenos`
  3. `cloud_tpu_pallas`: Pallas/Mosaic `pl.pallas_call` -> `tpu_custom_call`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import freetpu_tile as freetpu  # noqa: E402
from freetpu_tile.backends.cpu_ref import quantize_w8a16_weights  # noqa: E402


def build_swiglu_kernel(
    *,
    block_m: int = 64,
    block_n: int = 512,
    block_k: int = 512,
    dtype: str = "w8a16",
) -> freetpu.TileKernel:
    """Construct single-source @freetpu.tile_kernel for 3-projection SwiGLU."""

    @freetpu.tile_kernel(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        dtype=dtype,
    )
    def swiglu_mlp(x, w_gate, w_up, w_down, out):
        gate = freetpu.silu(freetpu.dot(x, w_gate))
        up = freetpu.dot(x, w_up)
        hidden = freetpu.mul(gate, up)
        y = freetpu.dot(hidden, w_down)
        freetpu.store(out, y)

    return swiglu_mlp


def make_inputs(
    m_dim: int,
    k_dim: int,
    n_inter: int,
    dtype: str,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    if dtype == "int8":
        x = rng.integers(-16, 17, size=(m_dim, k_dim), dtype=np.int8)
        wg = rng.integers(-16, 17, size=(k_dim, n_inter), dtype=np.int8)
        wu = rng.integers(-16, 17, size=(k_dim, n_inter), dtype=np.int8)
        wd = rng.integers(-16, 17, size=(n_inter, k_dim), dtype=np.int8)
        return x, wg, wu, wd, {}

    x = (rng.standard_normal((m_dim, k_dim)) * 0.35).astype(np.float32)
    wg_f = (rng.standard_normal((k_dim, n_inter)) / math.sqrt(k_dim)).astype(np.float32)
    wu_f = (rng.standard_normal((k_dim, n_inter)) / math.sqrt(k_dim)).astype(np.float32)
    wd_f = (rng.standard_normal((n_inter, k_dim)) / math.sqrt(n_inter)).astype(np.float32)
    if dtype == "w8a16":
        qg, sg = quantize_w8a16_weights(wg_f, rhs_transposed=False)
        qu, su = quantize_w8a16_weights(wu_f, rhs_transposed=False)
        qd, sd = quantize_w8a16_weights(wd_f, rhs_transposed=False)
        return x, qg, qu, qd, {"w_gate": sg, "w_up": su, "w_down": sd}
    return x, wg_f, wu_f, wd_f, {}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-source FreeTPU Tile DSL SwiGLU MLP")
    ap.add_argument("--target", default="all", choices=("cpu_ref", "buenos_a0", "cloud_tpu_pallas", "all"))
    ap.add_argument("--dtype", default="w8a16", choices=("bf16", "w8a16"))
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N-inter", type=int, default=512)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--adb-serial", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    x, wg, wu, wd, scales = make_inputs(args.M, args.K, args.N_inter, args.dtype)
    kernel = build_swiglu_kernel(block_m=64, block_n=512, block_k=512, dtype=args.dtype)

    if args.target == "all":
        dry_targets: List[str] = ["buenos_a0", "cloud_tpu_pallas"] if args.dry_run else []
        if not args.dry_run:
            try:
                import jax  # type: ignore

                if jax.default_backend() != "tpu":
                    dry_targets.append("cloud_tpu_pallas")
            except ImportError:
                dry_targets.append("cloud_tpu_pallas")
        report = kernel.parity_check(
            x,
            wg,
            wu,
            wd,
            targets=["cpu_ref", "buenos_a0", "cloud_tpu_pallas"],
            dry_run_targets=dry_targets,
            scales=scales,
            adb_serial=args.adb_serial,
            warmup=args.warmup,
            iters=args.iters,
        )
        payload: Dict[str, Any] = report.to_dict()
        payload["ir"] = kernel.ir.to_dict()
        ok = report.passed
    else:
        tgt = freetpu.Target.parse(args.target)
        dry = args.dry_run and tgt != freetpu.Target.CPU_REF
        if tgt == freetpu.Target.CPU_REF:
            res = kernel.run(x, wg, wu, wd, target=tgt, scales=scales)
            payload = {"ir": kernel.ir.to_dict(), "result": res.summary_dict()}
            ok = True
        else:
            report = kernel.parity_check(
                x,
                wg,
                wu,
                wd,
                targets=["cpu_ref", tgt.value],
                dry_run_targets=[tgt.value] if dry else [],
                scales=scales,
                adb_serial=args.adb_serial,
                warmup=args.warmup,
                iters=args.iters,
            )
            payload = report.to_dict()
            payload["ir"] = kernel.ir.to_dict()
            ok = report.passed

    out_text = json.dumps(payload, indent=2)
    print(out_text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(out_text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
