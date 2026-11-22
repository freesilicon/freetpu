#!/usr/bin/env python3
"""Starter template: Tensor G5 (buenos_a0) offline GEMM synthesis + 0x640 spline probe.

Exercises FreeTPU's clean-room ISA tools (`tools/emit_custom_gemm.py` and
`tools/spline_0x640.py`) offline on any host:
  1. Synthesizes multi-tile VLIW chunks + `ext[2]` params for `(M, K, N)`.
  2. Verifies IEEE-754 FP32-RNE (`np.rint`) parity against an independent NumPy `cpu_ref`.
  3. Synthesizes an 8-segment degree-4 minimax `SiLU` spline and verifies its
     208-byte `0x640` packet round-trip and FP32 Horner max error.
  4. Optionally emits `.dgc0` / `.in.bin` / `.ref.bin` when `--shell` is supplied
     (using a locally extracted user DGC0 shell; never bundled in the repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def resolve_repo_root(explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "tools" / "emit_custom_gemm.py").is_file():
            return parent
    raise FileNotFoundError(
        "Could not locate FreeTPU repo root (tools/emit_custom_gemm.py). Pass --repo-root <path>."
    )


def numpy_fp32_rne_gemm(
    x_mk: np.ndarray,
    w_nk: np.ndarray,
    bias_n: np.ndarray,
    s_in: float,
    zp_in: int,
    s_w: float,
    s_out: float,
    zp_out: int,
    act: int,
) -> np.ndarray:
    """Independent NumPy reference matching Tensor G5 IEEE-754 FP32-RNE semantics."""
    x64 = x_mk.astype(np.int64) - np.int64(zp_in)
    w64 = w_nk.astype(np.int64)
    acc = x64 @ w64.T + bias_n.astype(np.int64)[None, :]

    qmin, qmax = -128, 127
    if act == 1:  # ACT_RELU
        qmin = max(-128, min(127, zp_out))
    elif act == 2:  # ACT_RELU6
        qmin = max(-128, min(127, zp_out))
        qmax = max(-128, min(127, int(round(6.0 / float(np.float32(s_out)))) + zp_out))

    real_mul = np.float32(np.float32(s_in) * np.float32(s_w) / np.float32(s_out))
    scaled = acc.astype(np.float64).astype(np.float32) * real_mul
    rounded = np.rint(scaled).astype(np.float64)
    clamped = np.clip(rounded, float(qmin - zp_out), float(qmax - zp_out))
    q = clamped.astype(np.int64) + np.int64(zp_out)
    return np.clip(q, qmin, qmax).astype(np.int8)


def resolve_emission_paths(shell_path: Path, out_prefix: Path) -> Tuple[Path, Path, Path, Path]:
    """Return output artifact paths after verifying none aliases `shell_path`."""
    dgc0_out = Path(f"{out_prefix}.dgc0")
    in_out = Path(f"{out_prefix}.in.bin")
    ref_out = Path(f"{out_prefix}.ref.bin")
    meta_out = Path(f"{out_prefix}.meta.json")
    out_paths = (dgc0_out, in_out, ref_out, meta_out)

    shell_resolved = shell_path.resolve()
    shell_exists = shell_path.exists()
    for out_path in out_paths:
        out_resolved = out_path.resolve()
        same_target = out_resolved == shell_resolved
        if not same_target and shell_exists and out_path.exists():
            same_target = os.path.samefile(shell_path, out_path)
        if same_target:
            raise ValueError(
                f"Output path {out_path} refers to input shell {shell_path}; "
                "choose a different --out-prefix so the input shell is not overwritten."
            )
    return out_paths


def _verify_edge_cases() -> None:
    """Verify overflow-safe FP32-RNE clamping and input-shell overwrite protection."""
    # Accumulator=-1048576, real_mul=2048.0, zp_out=-1 -> scaled=-2^31; must clamp to -128 (never wrap in int32).
    x_neg = np.array([[-128] * 64], dtype=np.int8)
    w_neg = np.array([[127] * 64], dtype=np.int8)
    b_neg = np.array([-1048576 - (-128 * 127 * 64)], dtype=np.int32)
    out_neg = numpy_fp32_rne_gemm(x_neg, w_neg, b_neg, s_in=64.0, zp_in=0, s_w=32.0, s_out=1.0, zp_out=-1, act=0)
    assert int(out_neg[0, 0]) == -128, f"Expected -128 on negative saturation, got {int(out_neg[0, 0])}"

    # Positive saturation well beyond INT32_MAX must clamp to +127.
    b_pos = np.array([1048576 - (-128 * 127 * 64)], dtype=np.int32)
    out_pos = numpy_fp32_rne_gemm(x_neg, w_neg, b_pos, s_in=128.0, zp_in=0, s_w=32.0, s_out=1.0, zp_out=1, act=0)
    assert int(out_pos[0, 0]) == 127, f"Expected 127 on positive saturation, got {int(out_pos[0, 0])}"

    # Same-path emission target must be rejected before any write occurs.
    try:
        resolve_emission_paths(Path("probe.dgc0"), Path("probe"))
    except ValueError:
        pass
    else:
        raise AssertionError("resolve_emission_paths failed to reject overwriting input shell")


def run_probe(
    repo_root: Path,
    m: int,
    k: int,
    n: int,
    s_in: float,
    zp_in: int,
    s_w: float,
    s_out: float,
    zp_out: int,
    act_name: str,
    shell_path: Optional[Path],
    out_prefix: Optional[Path],
) -> Dict[str, Any]:
    if (shell_path is None) != (out_prefix is None):
        raise ValueError(
            "Both --shell (or FREETPU_SHELL_DGC0) and --out-prefix are required together to emit .dgc0 artifacts."
        )

    out_paths: Optional[Tuple[Path, Path, Path, Path]] = None
    if shell_path is not None and out_prefix is not None:
        out_paths = resolve_emission_paths(shell_path, out_prefix)

    tools_dir = repo_root / "tools"
    for p in (str(tools_dir / "dgc0"), str(tools_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import emit_custom_gemm as ecg  # noqa: E402
    import multitile_synth as ms  # noqa: E402
    import spline_0x640 as sp  # noqa: E402

    act_id = ecg.ACT_NAMES[act_name]
    w, b, x = ecg.synth_weights_bias_input(
        m, k, n, seed=7, s_in=s_in, zp_in=zp_in, s_w=s_w, s_out=s_out, zp_out=zp_out, act=act_id
    )

    chunks, ext2 = ms.synthesize_gemm_chunks(
        M=m,
        K=k,
        N=n,
        weights_nk=w,
        bias_n=b,
        s_in=s_in,
        zp_in=zp_in,
        s_w=s_w,
        s_out=s_out,
        zp_out=zp_out,
        act=act_id,
    )
    ref_rows, _ = ecg.gemm_int8_reference_fp32_rne(x, w, b, s_in, zp_in, s_w, s_out, zp_out, act_id)
    np_ref = numpy_fp32_rne_gemm(
        np.array(x, dtype=np.int8),
        np.array(w, dtype=np.int8),
        np.array(b, dtype=np.int32),
        s_in,
        zp_in,
        s_w,
        s_out,
        zp_out,
        act_id,
    )
    max_abs_err = int(np.max(np.abs(np.array(ref_rows, dtype=np.int32) - np_ref.astype(np.int32))))
    assert max_abs_err == 0, f"FP32-RNE reference mismatch: max_abs_err={max_abs_err}"

    # Synthesize 8-segment degree-4 minimax SiLU spline and verify 208 B 0x640 codec round-trip.
    segs_raw, bps, _ = sp.optimize_8seg_minimax(
        sp.fn_silu, (-11.1, 11.1), init_bps=[-6.5, -3.5, -1.6, 0.0, 1.6, 3.5, 6.5]
    )
    pkt = sp.SplinePacket0x640(
        header_76b=(0xFFFF << 12) | 0x640,
        coeffs=segs_raw,
        breakpoints=bps,
        tail_84b=0x7804,
    )
    raw_208b = sp.encode_0x640(pkt)
    dec_pkt = sp.decode_0x640(raw_208b)
    assert sp.encode_0x640(dec_pkt) == raw_208b
    xs = np.linspace(-11.1, 11.1, 4001, dtype=np.float32)
    silu_err = float(np.max(np.abs(sp.fn_silu(xs.astype(np.float64)) - sp.eval_0x640_horner_fp32(xs, pkt, -11.1, 11.1))))
    assert silu_err < 1e-4, f"Unexpected 0x640 SiLU error: {silu_err}"

    result: Dict[str, Any] = {
        "status": "ok",
        "target": "buenos_a0",
        "shape": [m, k, n],
        "act": act_name,
        "chunk_sizes_bytes": [len(c) for c in chunks],
        "ext2_bytes": len(ext2),
        "fp32_rne_max_abs_err": max_abs_err,
        "spline_0x640_bytes": len(raw_208b),
        "spline_silu_max_abs_err": round(silu_err, 7),
        "dgc0_emitted": False,
    }

    if shell_path is not None and out_prefix is not None and out_paths is not None:
        dgc0_out, in_out, ref_out, meta_out = out_paths
        blob, in_b, ref_b, info = ecg.emit_custom_gemm_dgc0(
            weights_nk=w,
            bias_n=b,
            x_mk=x,
            s_in=s_in,
            zp_in=zp_in,
            s_w=s_w,
            s_out=s_out,
            zp_out=zp_out,
            act=act_id,
            shell_path=shell_path,
        )
        dgc0_out.parent.mkdir(parents=True, exist_ok=True)
        dgc0_out.write_bytes(blob)
        in_out.write_bytes(in_b)
        ref_out.write_bytes(ref_b)
        meta_out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        result["dgc0_emitted"] = True
        result["dgc0_bytes"] = len(blob)
        result["out_prefix"] = str(out_prefix)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor G5 (buenos_a0) offline GEMM + 0x640 spline probe")
    ap.add_argument("--self-test", action="store_true", help="Run offline VLIW + spline parity check")
    ap.add_argument("--repo-root", type=Path, default=None, help="Path to FreeTPU repo root")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--s-in", type=float, default=0.03125)
    ap.add_argument("--zp-in", type=int, default=-11)
    ap.add_argument("--s-w", type=float, default=0.015625)
    ap.add_argument("--s-out", type=float, default=0.05)
    ap.add_argument("--zp-out", type=int, default=19)
    ap.add_argument("--act", choices=("NONE", "RELU", "RELU6"), default="RELU")
    ap.add_argument(
        "--shell",
        type=Path,
        default=None,
        help="User-extracted local template DGC0 shell (or set FREETPU_SHELL_DGC0)",
    )
    ap.add_argument("--out-prefix", type=Path, default=None, help="Prefix for emitted .dgc0/.in.bin/.ref.bin")
    args = ap.parse_args()

    env_shell = os.environ.get("FREETPU_SHELL_DGC0", "").strip()
    shell_path = args.shell if args.shell is not None else (
        Path(env_shell) if env_shell and not args.self_test else None
    )
    if (shell_path is None) != (args.out_prefix is None):
        ap.error("Both --shell (or FREETPU_SHELL_DGC0) and --out-prefix must be supplied together.")
    if shell_path is not None and args.out_prefix is not None:
        try:
            resolve_emission_paths(shell_path, args.out_prefix)
        except ValueError as exc:
            ap.error(str(exc))
    if args.self_test:
        _verify_edge_cases()

    repo_root = resolve_repo_root(args.repo_root)
    report = run_probe(
        repo_root=repo_root,
        m=args.M,
        k=args.K,
        n=args.N,
        s_in=args.s_in,
        zp_in=args.zp_in,
        s_w=args.s_w,
        s_out=args.s_out,
        zp_out=args.zp_out,
        act_name=args.act,
        shell_path=shell_path,
        out_prefix=args.out_prefix,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
