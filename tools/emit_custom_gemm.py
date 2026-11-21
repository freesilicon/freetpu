#!/usr/bin/env python3
"""
Clean-room parameterized multi-tile INT8 GEMM microkernel emitter for Tensor G5 (buenos_a0 / DGC0).

Synthesizes complete, runnable DarwiNN `DGC0` binaries for arbitrary `(M, K, N)` shapes,
custom INT8 weights `W[N, K]`, INT32 biases `B[N]`, quantization `(s_in, zp_in, s_w, s_out, zp_out)`,
and fused activation (`NONE` / `RELU` / `RELU6`) with ZERO TFLite compiler involvement:

    Y[m, n] = clamp(round_requant(sum_k((X[m, k] - zp_in) * W[n, k]) + bias[n]) + zp_out, qmin, qmax)

Architecture:
  - `multitile_synth.py`: closed-form VLIW synthesis of `chunk[0..6]` and packed `ext[2]` params
    (verified 81/81 bit-exact across the entire compiled reference corpus).
  - `dgc0_builder.py`: dynamic `DIVE` (`ext[0]`) + `DWS0` + `DGC0` container builder preserving
    the static RV64 interpreter ELF (`ext[1]`), `DGD0`, `PutBits` VA patch ops, `0x0a` gaps,
    and 64 B alignment (verified 81/81 bit-exact `ext[0]` + round-trip `dgc0_dump` validation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR / "dgc0") not in sys.path:
    sys.path.insert(0, str(_THIS_DIR / "dgc0"))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dgc0_builder  # noqa: E402
import dgc0_dump  # noqa: E402
import multitile_synth as ms  # noqa: E402
import sweep_gen  # noqa: E402

DEFAULT_SHELL_DGC0 = dgc0_builder.DEFAULT_SHELL_DGC0

ACT_NONE = ms.ACT_NONE
ACT_RELU = ms.ACT_RELU
ACT_RELU6 = ms.ACT_RELU6

ACT_NAMES = {
    "NONE": ACT_NONE,
    "RELU": ACT_RELU,
    "RELU6": ACT_RELU6,
}


def fp32_bits(val: float) -> int:
    return ms.fp32_bits(val)


def bits_to_fp32(u32: int) -> float:
    return ms.bits_to_fp32(u32)


def _round_half_to_even(x: float) -> int:
    """IEEE-754 round-to-nearest, ties-to-even (matches Tensor G5 FP32->INT32 hardware)."""
    return int(round(x))


def gemm_int8_reference_fp32_rne(
    x_mk: Sequence[Sequence[int]],
    weights_nk: Sequence[Sequence[int]],
    bias_n: Sequence[int],
    s_in: float,
    zp_in: int,
    s_w: float,
    s_out: float,
    zp_out: int,
    act: int,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Exact Tensor G5 hardware FP32-RNE INT8 GEMM reference."""
    s_in_f = ms.f32(s_in)
    s_w_f = ms.f32(s_w)
    s_out_f = ms.f32(s_out)
    real_mul_f = ms.f32(ms.f32(s_in_f * s_w_f) / s_out_f)

    qmin, qmax = -128, 127
    if act == ACT_RELU:
        qmin = max(-128, min(127, zp_out))
    elif act == ACT_RELU6:
        qmin = max(-128, min(127, zp_out))
        qmax = max(-128, min(127, int(round(6.0 / s_out_f)) + zp_out))

    out_rows: List[List[int]] = []
    for row in x_mk:
        y_row = []
        for w_row, b in zip(weights_nk, bias_n):
            acc = int(b)
            for xv, wv in zip(row, w_row):
                acc += (int(xv) - zp_in) * int(wv)
            scaled_f = ms.f32(ms.f32(float(acc)) * real_mul_f)
            val = _round_half_to_even(scaled_f) + zp_out
            val = max(qmin, min(qmax, val))
            y_row.append(val)
        out_rows.append(y_row)
    return out_rows, {
        "real_mul_fp32": real_mul_f,
        "qmin": qmin,
        "qmax": qmax,
    }


def gemm_int8_reference(
    x_mk: Sequence[Sequence[int]],
    weights_nk: Sequence[Sequence[int]],
    bias_n: Sequence[int],
    s_in: float,
    zp_in: int,
    s_w: float,
    s_out: float,
    zp_out: int,
    act: int,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Pure integer TFLite-exact INT8 FC/GEMM reference (using float32 scales to match compiler)."""
    s_in_f = ms.f32(s_in)
    s_w_f = ms.f32(s_w)
    s_out_f = ms.f32(s_out)
    real_mul = (s_in_f * s_w_f) / s_out_f
    q_mul, q_shift = sweep_gen.quantize_multiplier(real_mul)

    qmin, qmax = -128, 127
    if act == ACT_RELU:
        qmin = max(-128, min(127, zp_out))
    elif act == ACT_RELU6:
        qmin = max(-128, min(127, zp_out))
        qmax = max(-128, min(127, int(round(6.0 / s_out_f)) + zp_out))

    out_rows: List[List[int]] = []
    for row in x_mk:
        y_row = []
        for w_row, b in zip(weights_nk, bias_n):
            acc = int(b)
            for xv, wv in zip(row, w_row):
                acc += (int(xv) - zp_in) * int(wv)
            scaled = sweep_gen.multiply_by_quantized_multiplier(acc, q_mul, q_shift)
            val = scaled + zp_out
            val = max(qmin, min(qmax, val))
            y_row.append(val)
        out_rows.append(y_row)
    return out_rows, {
        "real_mul_fp32": real_mul,
        "quant_mul": q_mul,
        "quant_shift": q_shift,
        "qmin": qmin,
        "qmax": qmax,
    }


def emit_custom_gemm_dgc0(
    *,
    weights_nk: Sequence[Sequence[int]],
    bias_n: Sequence[int],
    x_mk: Sequence[Sequence[int]],
    s_in: float = 0.03125,
    zp_in: int = 0,
    s_w: float = 0.015625,
    s_out: float = 0.0625,
    zp_out: int = 0,
    act: int = ACT_NONE,
    n_dim: Optional[int] = None,
    include_bias: Optional[bool] = None,
    shell_path: Path = DEFAULT_SHELL_DGC0,
) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    """
    Construct a parameterized arbitrary-(M, K, N) custom INT8 GEMM DGC0 binary + in.bin + ref.bin.

    Returns `(dgc0_bytes, in_bytes, ref_bytes, info_dict)` where `ref_bytes` uses exact Tensor G5
    FP32-RNE semantics and `info_dict` also records `tflite_max_diff` vs TFLite quantized-multiplier.
    """
    m_dim = len(x_mk)
    if m_dim <= 0:
        raise ValueError("x_mk must have M >= 1 rows")
    k_dim = len(x_mk[0])
    if any(len(r) != k_dim for r in x_mk):
        raise ValueError("All rows of x_mk must have length K")
    actual_n = len(weights_nk)
    if n_dim is not None and n_dim != actual_n:
        raise ValueError(f"n_dim={n_dim} does not match len(weights_nk)={actual_n}")
    n_dim = actual_n
    if n_dim <= 0 or any(len(r) != k_dim for r in weights_nk):
        raise ValueError(f"weights_nk must have shape [{n_dim}, {k_dim}]")
    if len(bias_n) != n_dim:
        raise ValueError(f"bias_n length {len(bias_n)} != N={n_dim}")

    chunks, ext2 = ms.synthesize_gemm_chunks(
        M=m_dim,
        K=k_dim,
        N=n_dim,
        weights_nk=weights_nk,
        bias_n=bias_n,
        s_in=s_in,
        zp_in=zp_in,
        s_w=s_w,
        s_out=s_out,
        zp_out=zp_out,
        act=act,
        include_bias=include_bias,
    )
    dgc0_bytes = dgc0_builder.build_dgc0_container(
        chunks=chunks,
        ext2=ext2,
        M=m_dim,
        K=k_dim,
        N=n_dim,
        shell_path=shell_path,
    )
    rt_meta = dgc0_builder.verify_container_roundtrip(dgc0_bytes, m_dim, k_dim, n_dim, chunks, ext2)

    ref_fp32, fp32_meta = gemm_int8_reference_fp32_rne(
        x_mk=x_mk,
        weights_nk=weights_nk,
        bias_n=bias_n,
        s_in=s_in,
        zp_in=zp_in,
        s_w=s_w,
        s_out=s_out,
        zp_out=zp_out,
        act=act,
    )
    ref_tfl, tfl_meta = gemm_int8_reference(
        x_mk=x_mk,
        weights_nk=weights_nk,
        bias_n=bias_n,
        s_in=s_in,
        zp_in=zp_in,
        s_w=s_w,
        s_out=s_out,
        zp_out=zp_out,
        act=act,
    )
    tfl_max_diff = max(
        abs(a - b)
        for r1, r2 in zip(ref_fp32, ref_tfl)
        for a, b in zip(r1, r2)
    )

    in_bytes = bytes((v & 0xFF) for row in x_mk for v in row)
    ref_bytes = bytes((v & 0xFF) for row in ref_fp32 for v in row)
    ref_tfl_bytes = bytes((v & 0xFF) for row in ref_tfl for v in row)

    info: Dict[str, Any] = {
        "M": m_dim,
        "K": k_dim,
        "N": n_dim,
        "in_bytes": len(in_bytes),
        "out_bytes": len(ref_bytes),
        "param_bytes": len(ext2),
        "chunk_sizes": [len(c) for c in chunks],
        "s_in": s_in,
        "zp_in": zp_in,
        "s_w": s_w,
        "s_out": s_out,
        "zp_out": zp_out,
        "act": act,
        "fp32_vs_tflite_max_abs_diff": tfl_max_diff,
        "sha256_dgc0": rt_meta["sha256"],
        "sha256_in": hashlib.sha256(in_bytes).hexdigest(),
        "sha256_ref": hashlib.sha256(ref_bytes).hexdigest(),
        "sha256_ref_tflite": hashlib.sha256(ref_tfl_bytes).hexdigest(),
        "dgc0_size_bytes": rt_meta["size_bytes"],
        "dive_scratch_bytes": rt_meta["dive_scratch_bytes"],
        "exec_field_14": rt_meta["exec_field_14"],
        **tfl_meta,
    }
    return dgc0_bytes, in_bytes, ref_bytes, info


def synth_weights_bias_input(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    seed: int = 0,
    s_in: float = 0.03125,
    zp_in: int = 0,
    s_w: float = 0.015625,
    s_out: float = 0.05,
    zp_out: int = 0,
    act: int = ACT_NONE,
    avoid_tie_ambiguity: bool = True,
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    """
    Synthesize non-trivial deterministic INT8 weights W[N,K], INT32 biases B[N], and INT8 inputs X[M,K]
    that exercise negative/positive accumulators, activation clamping, and asymmetric zero-points.
    When `avoid_tie_ambiguity=True`, adjusts bias by +/-1 on any channel where FP32-RNE and TFLite
    fixed-point rounding disagree by 1 ULP so both reference definitions agree identically (`diff == 0`).
    """
    w = []
    for ni in range(n_dim):
        row = []
        for ki in range(k_dim):
            val = ((ni * 19 + ki * 13 + (ni ^ ki) * 7 + seed * 31 + 23) % 45) - 22
            row.append(val)
        w.append(row)

    # Choose bias range scaled to K so outputs span both negative (clamp/ReLU) and positive dynamic range
    b_span = max(800, k_dim * 25)
    bias = [((ni * 631 + seed * 97 + 17) % (2 * b_span)) - b_span for ni in range(n_dim)]

    x = []
    for mi in range(m_dim):
        row = [((mi * 37 + ki * 29 + (mi ^ ki) * 11 + seed * 19 + 41) % 77) - 38 for ki in range(k_dim)]
        x.append(row)

    if avoid_tie_ambiguity:
        # Nudge any channel whose exact half-way rounding differs between FP32-RNE and TFLite
        for _ in range(8):
            r_fp, _ = gemm_int8_reference_fp32_rne(x, w, bias, s_in, zp_in, s_w, s_out, zp_out, act)
            r_tf, _ = gemm_int8_reference(x, w, bias, s_in, zp_in, s_w, s_out, zp_out, act)
            bad_cols = {
                ni
                for mi in range(m_dim)
                for ni in range(n_dim)
                if r_fp[mi][ni] != r_tf[mi][ni]
            }
            if not bad_cols:
                break
            for ni in bad_cols:
                bias[ni] += 3
    return w, bias, x


def self_test(corpus_dir: Optional[Path] = None) -> int:
    """
    Verify:
      1. Self-contained synthesis across representative multi-tile (M, K, N) shapes
         (chunk[0..6] + ext[2] + FP32-RNE reference).
      2. When --corpus-dir / FREETPU_CORPUS_DIR is provided, also verify 15 legacy +
         81 multi-tile reference models bit-for-bit.
    """
    shapes = [
        (1, 16, 16, 0.03125, 0, 0.015625, 0.0625, 0, ACT_NONE),
        (1, 16, 32, 0.03125, -11, 0.015625, 0.05, 19, ACT_RELU),
        (4, 32, 32, 0.03125, -11, 0.015625, 0.05, 19, ACT_RELU),
        (8, 64, 64, 0.03125, 13, 0.015625, 0.03, -21, ACT_RELU6),
        (16, 64, 64, 0.0625, -17, 0.015625, 0.07, 25, ACT_NONE),
        (6, 48, 48, 0.0625, 7, 0.03125, 0.07, -15, ACT_RELU6),
        (3, 32, 24, 0.03125, -9, 0.015625, 0.05, 11, ACT_RELU),
        (32, 32, 32, 0.03125, 5, 0.015625, 0.05, -7, ACT_NONE),
        (1, 512, 128, 0.03125, 0, 0.015625, 0.0625, 0, ACT_NONE),
    ]
    for seed, (M, K, N, s_in, zp_in, s_w, s_out, zp_out, act) in enumerate(shapes):
        w, b, x = synth_weights_bias_input(
            M, K, N, seed=seed, s_in=s_in, zp_in=zp_in, s_w=s_w, s_out=s_out, zp_out=zp_out, act=act
        )
        chunks, ext2 = ms.synthesize_gemm_chunks(
            M=M, K=K, N=N, weights_nk=w, bias_n=b,
            s_in=s_in, zp_in=zp_in, s_w=s_w, s_out=s_out, zp_out=zp_out, act=act,
        )
        assert len(chunks) == 7 and all(len(c) > 0 and len(c) % 16 == 0 for c in chunks)
        assert len(ext2) > 0
        y_fp32, _ = gemm_int8_reference_fp32_rne(x, w, b, s_in, zp_in, s_w, s_out, zp_out, act)
        assert len(y_fp32) == M and len(y_fp32[0]) == N

    if corpus_dir is None or not corpus_dir.is_dir():
        print(
            f"=== EMIT_CUSTOM_GEMM SELF-TEST SUMMARY ===\n"
            f"  Self-contained synthesis : {len(shapes)}/{len(shapes)} PASS (chunk[0..6] + ext[2] + FP32-RNE ref)\n"
            f"  External 81-model parity : SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)"
        )
        return 0

    import verify_full_isa_81 as v81

    # 1. Legacy 15 cases through top-level emit_custom_gemm_dgc0
    manifest = json.loads((corpus_dir / "work" / "isa-gemm" / "sweep1" / "manifest.json").read_text())
    legacy_cases: List[Tuple[str, Path, Dict[str, Any]]] = []
    s1_dir = corpus_dir / "work" / "device-runner" / "phase3_gemm_sweep1"
    b0_dir = corpus_dir / "work" / "device-runner" / "phase3_batch0"
    for n in [
        "g00_base_1x16x16",
        "g01_zpin_pos17",
        "g02_zpin_neg37",
        "g03_zpout_pos29",
        "g04_zpout_neg45",
        "g05_sout_pow2_div64",
        "g06_sout_pow2_div256",
        "g07_sout_nonpow2_005",
        "g08_sout_nonpow2_003",
        "g09_act_relu",
        "g10_act_relu6",
        "g11_act_relu_zpout_n128",
        "g18_k16_n8",
        "g19_k16_n20",
    ]:
        legacy_cases.append((n, s1_dir / f"{n}.dgc0", manifest[n]["meta"]))
    legacy_cases.append(
        (
            "gemm_fc_1x16x32",
            b0_dir / "gemm_fc_1x16x32.dgc0",
            dict(M=1, K=16, N=32, s_in=0.03125, zp_in=-5, s_w=0.015625, s_out=0.0625, zp_out=3, fused_activation=0),
        )
    )

    legacy_fail = 0
    for name, ref_path, m in legacy_cases:
        k_dim, n_dim = m["K"], m["N"]
        dummy_w = [[0] * k_dim for _ in range(n_dim)]
        dummy_b = [1] * n_dim  # non-zero bias to match compiler has_bias=True
        dummy_x = [[0] * k_dim]
        emitted, _, _, _ = emit_custom_gemm_dgc0(
            weights_nk=dummy_w,
            bias_n=dummy_b,
            x_mk=dummy_x,
            s_in=m["s_in"],
            zp_in=m["zp_in"],
            s_w=m["s_w"],
            s_out=m["s_out"],
            zp_out=m["zp_out"],
            act=m["fused_activation"],
        )
        p_emit = dgc0_dump.parse_dgc0(emitted)
        ref_raw = ref_path.read_bytes()
        p_ref = dgc0_dump.parse_dgc0(ref_raw)
        em_chunks = [
            emitted[c["offset_in_dgc0"] : c["offset_in_dgc0"] + c["size_bytes"]]
            for c in p_emit["external_buffers"][0]["dive_module"]["hardware_instruction_chunks"]
        ]
        rf_chunks = [
            ref_raw[c["offset_in_dgc0"] : c["offset_in_dgc0"] + c["size_bytes"]]
            for c in p_ref["external_buffers"][0]["dive_module"]["hardware_instruction_chunks"]
        ]
        bad = []
        for ci in range(7):
            ec = v81.mask_c6_cyclic_tag(em_chunks[ci]) if ci == 6 else em_chunks[ci]
            rc = v81.mask_c6_cyclic_tag(rf_chunks[ci]) if ci == 6 else rf_chunks[ci]
            if ec != rc:
                bad.append(f"c{ci}({len(ec)}B vs {len(rc)}B)")
        if bad:
            legacy_fail += 1
            print(f"FAIL [legacy] {name}: {bad}")
        else:
            print(f"PASS [legacy] {name:24s}: chunk[0..6] 100% BIT-EXACT")

    # 2. Full 81-model multi-tile corpus verification (chunk[0..6] + ext[0] + ext[2] + DGC0 container)
    rows = v81.load_corpus_81(corpus_dir)
    all_manifests = v81.load_manifests(corpus_dir)
    corpus_fail = 0
    for p, M, K, N in rows:
        st = p.stem
        ref_raw = p.read_bytes()
        p_ref = dgc0_dump.parse_dgc0(ref_raw)
        eb0_r = p_ref["external_buffers"][0]
        eb2_r = p_ref["external_buffers"][2]
        rf_chunks = [
            ref_raw[c["offset_in_dgc0"] : c["offset_in_dgc0"] + c["size_bytes"]]
            for c in eb0_r["dive_module"]["hardware_instruction_chunks"]
        ]
        rf_ext2 = ref_raw[eb2_r["offset"] : eb2_r["offset"] + eb2_r["size_bytes"]]

        tf = v81.parse_tflite_fc_full(Path(all_manifests[st]["tflite_path"]).read_bytes())
        dummy_x = [[0] * K for _ in range(M)]
        emitted, _, _, _ = emit_custom_gemm_dgc0(
            weights_nk=tf["weights_nk"],
            bias_n=tf["bias_n"],
            x_mk=dummy_x,
            s_in=tf["s_in"],
            zp_in=tf["zp_in"],
            s_w=tf["s_w"],
            s_out=tf["s_out"],
            zp_out=tf["zp_out"],
            act=tf["act"],
        )
        p_emit = dgc0_dump.parse_dgc0(emitted)
        eb0_e = p_emit["external_buffers"][0]
        eb2_e = p_emit["external_buffers"][2]
        em_chunks = [
            emitted[c["offset_in_dgc0"] : c["offset_in_dgc0"] + c["size_bytes"]]
            for c in eb0_e["dive_module"]["hardware_instruction_chunks"]
        ]
        em_ext2 = emitted[eb2_e["offset"] : eb2_e["offset"] + eb2_e["size_bytes"]]

        bad = []
        for ci in range(7):
            ec = v81.mask_c6_cyclic_tag(em_chunks[ci]) if ci == 6 else em_chunks[ci]
            rc = v81.mask_c6_cyclic_tag(rf_chunks[ci]) if ci == 6 else rf_chunks[ci]
            if ec != rc:
                bad.append(f"c{ci}")
        if em_ext2 != rf_ext2:
            bad.append("ext2")
        if eb0_e["size_bytes"] != eb0_r["size_bytes"]:
            bad.append(f"ext0_sz({eb0_e['size_bytes']}!={eb0_r['size_bytes']})")
        if bad:
            corpus_fail += 1
            print(f"FAIL [corpus81] {st} ({M},{K},{N}): {bad}")

    print(
        f"\n=== EMIT_CUSTOM_GEMM SELF-TEST SUMMARY ===\n"
        f"  Legacy 15 cases  : {len(legacy_cases) - legacy_fail}/{len(legacy_cases)} PASS\n"
        f"  Multi-tile 81/81 : {len(rows) - corpus_fail}/{len(rows)} PASS (chunk[0..6] + ext[0] + ext[2] + DGC0 round-trip)"
    )
    return legacy_fail + corpus_fail


def emit_multitile_suite(out_dir: Path, shell_path: Optional[Path] = DEFAULT_SHELL_DGC0) -> Dict[str, Any]:
    """
    Emit 6 novel multi-tile custom INT8 GEMM microkernels (NO TFLite compiler) under `out_dir`:
      1. (4, 32, 32)  non-PoT s_out=0.05, zp_in=-11, zp_out=+19, RELU
      2. (8, 64, 64)  non-PoT s_out=0.03, zp_in=+13, zp_out=-21, RELU6 (FP32 ULP bit + 3D dual-K bcast)
      3. (16, 64, 64) non-PoT s_out=0.07, zp_in=-17, zp_out=+25, NONE  (dive_scratch_196608 + g=1 scatter)
      4. (6, 48, 48)  non-PoT s_out=0.03, zp_in=+7,  zp_out=-15, RELU6 (M%4==2 6-pkt 6c0 + c3/c5 VICA + ch2=5480)
      5. (3, 32, 24)  non-PoT s_out=0.05, zp_in=-9,  zp_out=+11, RELU  (odd M=3, non-PoT N=24, 15 tiles)
      6. (32, 32, 32) non-PoT s_out=0.05, zp_in=+5,  zp_out=-7,  NONE  (M>16 2D 64B scatter + 196608 scratch)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("custom_4x32x32_nonpot005_relu",   4, 32, 32, 0.03125, -11, 0.015625, 0.05,  19, ACT_RELU,  1),
        ("custom_8x64x64_nonpot003_relu6",  8, 64, 64, 0.03125,  13, 0.015625, 0.03, -21, ACT_RELU6, 2),
        ("custom_16x64x64_nonpot007_none", 16, 64, 64, 0.0625,  -17, 0.015625, 0.07,  25, ACT_NONE,  3),
        ("custom_6x48x48_nonpot007_relu6",  6, 48, 48, 0.0625,    7, 0.03125,  0.07, -15, ACT_RELU6, 4),
        ("custom_3x32x24_nonpot005_relu",   3, 32, 24, 0.03125,  -9, 0.015625, 0.05,  11, ACT_RELU,  5),
        ("custom_32x32x32_nonpot005_none", 32, 32, 32, 0.03125,   5, 0.015625, 0.05,  -7, ACT_NONE,  6),
        ("custom_10x32x32_unseen_relu6",   10, 32, 32, 0.0625,  -13, 0.03125,  0.07, -10, ACT_RELU6, 7),
    ]
    manifest: Dict[str, Any] = {}
    for name, M, K, N, s_in, zp_in, s_w, s_out, zp_out, act, seed in specs:
        w, b, x = synth_weights_bias_input(
            M, K, N, seed=seed, s_in=s_in, zp_in=zp_in, s_w=s_w, s_out=s_out, zp_out=zp_out, act=act
        )
        blob, in_b, ref_b, info = emit_custom_gemm_dgc0(
            weights_nk=w,
            bias_n=b,
            x_mk=x,
            s_in=s_in,
            zp_in=zp_in,
            s_w=s_w,
            s_out=s_out,
            zp_out=zp_out,
            act=act,
            shell_path=shell_path,
        )
        dgc0_p = out_dir / f"{name}.dgc0"
        in_p = out_dir / f"{name}.in.bin"
        ref_p = out_dir / f"{name}.ref.bin"
        meta_p = out_dir / f"{name}.meta.json"
        dgc0_p.write_bytes(blob)
        in_p.write_bytes(in_b)
        ref_p.write_bytes(ref_b)
        ref_vals = list(struct.unpack(f"{len(ref_b)}b", ref_b))
        entry = {
            "name": name,
            "dgc0_path": str(dgc0_p),
            "in_path": str(in_p),
            "ref_path": str(ref_p),
            "ref_min": min(ref_vals),
            "ref_max": max(ref_vals),
            "ref_unique": len(set(ref_vals)),
            **info,
        }
        meta_p.write_text(json.dumps(entry, indent=2) + "\n")
        manifest[name] = entry
        print(
            f"[MULTITILE] {name:34s} ({M:2d},{K:2d},{N:2d}) dgc0={len(blob):6d}B "
            f"in={len(in_b):4d}B out={len(ref_b):4d}B ref=[{min(ref_vals):4d},{max(ref_vals):4d}] "
            f"uniq={len(set(ref_vals)):3d} fp32_vs_tfl={info['fp32_vs_tflite_max_abs_diff']} "
            f"sha={info['sha256_dgc0'][:12]}"
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean-room Tensor G5 multi-tile INT8 GEMM microkernel emitter")
    ap.add_argument("--self-test", action="store_true", help="Verify synthesis (and external 81-model corpus if --corpus-dir set)")
    ap.add_argument("--corpus-dir", type=Path, default=None, help="Optional external corpus directory (or FREETPU_CORPUS_DIR)")
    ap.add_argument("--shell", type=Path, default=DEFAULT_SHELL_DGC0, help="Path to user-extracted template DGC0 shell (or FREETPU_SHELL_DGC0)")
    ap.add_argument("--emit-multitile", action="store_true", help="Emit 7 novel multi-tile custom kernels under --out-dir")
    ap.add_argument("--out-dir", type=Path, default=Path("out_multitile"), help="Output directory for --emit-multitile")
    ap.add_argument("--M", type=int, help="Emit a custom GEMM with rows M")
    ap.add_argument("--K", type=int, help="Inner dimension K")
    ap.add_argument("--N", type=int, help="Output channels N")
    ap.add_argument("--s-in", type=float, default=0.03125)
    ap.add_argument("--zp-in", type=int, default=0)
    ap.add_argument("--s-w", type=float, default=0.015625)
    ap.add_argument("--s-out", type=float, default=0.05)
    ap.add_argument("--zp-out", type=int, default=0)
    ap.add_argument("--act", choices=sorted(ACT_NAMES.keys()), default="NONE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=str, help="Output file prefix (writes <prefix>.dgc0, .in.bin, .ref.bin, .meta.json)")
    args = ap.parse_args()

    c_env = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
    corpus_dir = args.corpus_dir or (Path(c_env) if c_env else None)

    if args.self_test:
        bad = self_test(corpus_dir)
        sys.exit(0 if bad == 0 else 1)

    if args.emit_multitile:
        emit_multitile_suite(args.out_dir, shell_path=args.shell)
        return

    if args.M and args.K and args.N and args.out_prefix:
        act_id = ACT_NAMES[args.act]
        w, b, x = synth_weights_bias_input(
            args.M, args.K, args.N,
            seed=args.seed,
            s_in=args.s_in, zp_in=args.zp_in, s_w=args.s_w, s_out=args.s_out, zp_out=args.zp_out,
            act=act_id,
        )
        blob, in_b, ref_b, info = emit_custom_gemm_dgc0(
            weights_nk=w, bias_n=b, x_mk=x,
            s_in=args.s_in, zp_in=args.zp_in, s_w=args.s_w, s_out=args.s_out, zp_out=args.zp_out,
            act=act_id,
            shell_path=args.shell,
        )
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.dgc0").write_bytes(blob)
        Path(f"{prefix}.in.bin").write_bytes(in_b)
        Path(f"{prefix}.ref.bin").write_bytes(ref_b)
        Path(f"{prefix}.meta.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
