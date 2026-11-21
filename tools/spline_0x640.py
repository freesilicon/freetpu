#!/usr/bin/env python3
"""Clean-room Tensor G5 (buenos_a0) VICA 0x640 spline codec, minimax synthesizer, and DGC0 probe patcher.

Deliverable Part A for tpu-int4-splines (100% offline):
- Exact 208-byte (1664-bit) LoadCoefficientTables (0x640) bit-level encoder/decoder.
- Exact Stage 1 (0x3c0, 0x640@36, 0x200@45 HDR-DATA, 0x200@60 stream125) patcher,
  verified 5520/5520 byte-exact against logistic_scale_zp.dgc0 -> vec_logistic_1x16x16x16.dgc0.
- Clean-room pure-NumPy 8-segment degree-4 minimax (Remez exchange + equibalanced breakpoints)
  synthesizer for:
    * SiLU(x) = x / (1 + exp(-x))
    * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    * exp(x)  on [-12, 0]
- FP32-RNE Horner hardware simulator verified 4096/4096 against live rango silicon output.
- Emits ready-to-run live silicon probe blobs:
    work/int4-splines/spline_probes/{silu_probe,gelu_probe,exp_probe}.{dgc0,in.bin,ref.bin}
    work/int4-splines/spline_probes/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
import dgc0_dump  # noqa: E402
from dgc0_patcher import Dgc0Image  # noqa: E402

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
BASE = Path(_CORPUS_ENV) if _CORPUS_ENV else _THIS_DIR

LOGISTIC_SCALE_ZP_DGC0 = BASE / "work/device-runner/phase3_isa_vec_b1b/logistic_scale_zp.dgc0"
LOGISTIC_SCALE_ZP_IN = BASE / "work/isa-vec/batch1b/logistic_scale_zp.in.bin"
LOGISTIC_SCALE_ZP_OUT = BASE / "work/device-runner/phase3_isa_vec_b1b/logistic_scale_zp.out.bin"
VEC_LOGISTIC_DGC0 = BASE / "work/device-runner/phase3_batch0/vec_logistic_1x16x16x16.dgc0"
OUT_DIR = Path("out_spline_probes")
TABLES_JSON = Path("synthesized_splines.json")

# Exact chunk[3] offsets inside logistic_scale_zp.dgc0 / vec_logistic_1x16x16x16.dgc0 (5520 B)
OFF_0x3C0 = 0
OFF_0x640_S1 = 3056
OFF_0x200_HDR = 3392
OFF_0x640_S2 = 4176
OFF_0x200_S2 = 4512
OFF_0x640_S3 = 5008
OFF_0x200_S3 = 5344


# ---------------------------------------------------------------------------
# Bit / float helpers
# ---------------------------------------------------------------------------


def f32_to_u32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def u32_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_dive_chunk(img: Dgc0Image, chunk_idx: int) -> bytes:
    dive = img.ext_slots[0].payload
    r = dgc0_dump.FBReader(dive)
    chunks = r.uoffset_vector(r.field_pos(r.root_offset(), 3))
    return r.byte_vector(r.field_pos(chunks[chunk_idx], 0))[2]


# ---------------------------------------------------------------------------
# Exact 0x640 LoadCoefficientTables (208 B = 1664 bits) codec
# ---------------------------------------------------------------------------


@dataclass
class SplinePacket0x640:
    header_76b: int  # bits [75:0]
    coeffs: List[List[float]]  # 8 segments x 5 float32 coeffs [c0, c1, c2, c3, c4]
    breakpoints: List[float]  # 7 ascending float32 breakpoints bp[0..6]
    tail_84b: int  # bits [1663:1580], e.g. 0x7804 (deg-4) or 0x7801 (deg-1)


def decode_0x640(pkt: bytes) -> SplinePacket0x640:
    """Decode an exact 208-byte (1664-bit) 0x640 LoadCoefficientTables packet."""
    if len(pkt) != 208:
        raise ValueError(f"0x640 packet must be 208 bytes, got {len(pkt)}")
    bits = int.from_bytes(pkt, "little")
    header_76b = bits & ((1 << 76) - 1)
    if (header_76b & 0xFFF) != 0x640:
        raise ValueError(f"Expected opcode 0x640, got 0x{header_76b & 0xFFF:03x}")

    floats: List[float] = []
    for i in range(47):
        u32 = (bits >> (76 + 32 * i)) & 0xFFFFFFFF
        floats.append(u32_to_f32(u32))

    coeffs = [floats[5 * s : 5 * s + 5] for s in range(8)]
    breakpoints = floats[40:47]
    tail_84b = bits >> (76 + 32 * 47)
    return SplinePacket0x640(
        header_76b=header_76b,
        coeffs=coeffs,
        breakpoints=breakpoints,
        tail_84b=tail_84b,
    )


def encode_0x640(pkt: SplinePacket0x640) -> bytes:
    """Encode a SplinePacket0x640 back into an exact 208-byte wire packet."""
    if len(pkt.coeffs) != 8 or any(len(seg) != 5 for seg in pkt.coeffs):
        raise ValueError("coeffs must be 8x5")
    if len(pkt.breakpoints) != 7:
        raise ValueError("breakpoints must have length 7")

    bits = pkt.header_76b & ((1 << 76) - 1)
    flat = [c for seg in pkt.coeffs for c in seg] + list(pkt.breakpoints)
    for i, fval in enumerate(flat):
        bits |= f32_to_u32(fval) << (76 + 32 * i)
    bits |= (pkt.tail_84b & ((1 << 84) - 1)) << (76 + 32 * 47)
    return bits.to_bytes(208, "little")


# ---------------------------------------------------------------------------
# Exact 0x200 CSR stream125 & Stage 1 (s_in, zp_in) patchers
# ---------------------------------------------------------------------------


def decode_stream125_from_0x200(chunk: bytes, insn_off: int, insn_len: int = 32) -> Dict[str, object]:
    """Decode CSRs 0x5B..0x5F (125-bit stream: mul, fclip_max, fclip_min) inside a 0x200 packet."""
    regs: Dict[int, int] = {}
    for off in range(insn_off + 8, insn_off + insn_len, 4):
        u32 = struct.unpack_from("<I", chunk, off)[0]
        if u32 != 0:
            reg = u32 & 0x7F
            regs[reg] = (u32 >> 7) & 0x1FFFFFF

    s125 = 0
    for idx, reg in enumerate(range(0x5B, 0x60)):
        s125 |= regs.get(reg, 0) << (25 * idx)

    mul_u32 = (s125 >> 13) & 0xFFFFFFFF
    fmax_u32 = (s125 >> 46) & 0xFFFFFFFF
    fmin_u32 = (s125 >> 83) & 0xFFFFFFFF
    return {
        "regs": regs,
        "stream125": s125,
        "mul": u32_to_f32(mul_u32),
        "fclip_max": u32_to_f32(fmax_u32),
        "fclip_min": u32_to_f32(fmin_u32),
    }


def patch_stream125_in_0x200(
    chunk: bytes,
    insn_off: int,
    insn_len: int,
    *,
    mul: float | None = None,
    fclip_max: float | None = None,
    fclip_min: float | None = None,
) -> bytes:
    """Patch mul / fclip_max / fclip_min inside CSRs 0x5B..0x5F of a 0x200 packet.

    Exact bit layout in 125-bit stream (0x5B..0x5F, 25 bits each):
      - mul       (fp32): bits [44:13]
      - fclip_max (fp32): bits [77:46]
      - fclip_min (fp32): bits [114:83]

    Note: A compiled 0x200 packet may omit 0x5B (or 0x5D) when its 25-bit CSR value is 0
    (e.g. exact powers of two / integers whose low 12 mantissa bits are 0). In that case,
    any new value must also keep omitted registers == 0 so packet length stays constant.
    """
    dec = decode_stream125_from_0x200(chunk, insn_off, insn_len)
    present_regs = set(dec["regs"].keys())
    s125 = int(dec["stream125"])

    def set_field(val: float, bit_lo: int) -> None:
        nonlocal s125
        s125 &= ~(0xFFFFFFFF << bit_lo)
        s125 |= f32_to_u32(val) << bit_lo

    if mul is not None:
        set_field(mul, 13)
    if fclip_max is not None:
        set_field(fclip_max, 46)
    if fclip_min is not None:
        set_field(fclip_min, 83)

    new_vals = {0x5B + i: (s125 >> (25 * i)) & 0x1FFFFFF for i in range(5)}
    for reg, val25 in new_vals.items():
        if reg not in present_regs and val25 != 0:
            raise ValueError(
                f"CSR 0x{reg:02x} is omitted in template 0x200@{insn_off} "
                f"but new float requires non-zero val25=0x{val25:x}; pick a float with clean low bits"
            )

    buf = bytearray(chunk)
    for off in range(insn_off + 8, insn_off + insn_len, 4):
        u32 = struct.unpack_from("<I", buf, off)[0]
        if u32 != 0:
            reg = u32 & 0x7F
            if reg in new_vals:
                struct.pack_into("<I", buf, off, (new_vals[reg] << 7) | reg)
    return bytes(buf)


def patch_stage1_input_quant(
    chunk: bytes,
    s_in: float,
    zp_in: int,
    *,
    fclip_min: float | None = None,
    fclip_max: float | None = None,
) -> bytes:
    """Patch Stage 1 input quantization (s_in, zp_in) and Stage 2 clip bounds in chunk[3].

    Updates all 4 hardware sites diff-verified between logistic_scale_zp.dgc0 and
    vec_logistic_1x16x16x16.dgc0:
      1) 0x3c0@0:       zp_in duplicated into 16 bits at w25[63:60], w26[11:0]
      2) 0x640@3056:    bp0 = -zp_in * s_in; seg0=[+128, 1/s_in], seg1..7=[-128, 1/s_in]
      3) 0x200@3392:    HDR-DATA (bytes +272..+320) zp+128, s_in, (127-zp)*s_in, (-128-zp)*s_in
      4) 0x200@4512:    stream125 mul=s_in, optional fclip_min/fclip_max
    """
    if not (-128 <= zp_in <= 127):
        raise ValueError(f"zp_in out of int8 range: {zp_in}")
    buf = bytearray(chunk)

    # 1) 0x3c0@0 (w25[63:60], w26[11:0])
    z8 = zp_in & 0xFF
    zp_dup = (z8 << 8) | z8
    w25, w26 = struct.unpack_from("<QQ", buf, 25 * 8)
    w25 = (w25 & 0x0FFFFFFFFFFFFFFF) | ((zp_dup & 0xF) << 60)
    w26 = (w26 & ~0xFFF) | ((zp_dup >> 4) & 0xFFF)
    struct.pack_into("<QQ", buf, 25 * 8, w25, w26)

    # 2) 0x640@3056 (Stage 1 dequant linear spline)
    p1 = decode_0x640(bytes(buf[OFF_0x640_S1 : OFF_0x640_S1 + 208]))
    inv_s = float(np.float32(1.0) / np.float32(s_in))
    bp0 = float(np.float32(-zp_in) * np.float32(s_in))
    p1.coeffs[0] = [128.0, inv_s, 0.0, 0.0, 0.0]
    for s in range(1, 8):
        p1.coeffs[s] = [-128.0, inv_s, 0.0, 0.0, 0.0]
    p1.breakpoints[0] = bp0
    buf[OFF_0x640_S1 : OFF_0x640_S1 + 208] = encode_0x640(p1)

    # 3) 0x200@3392 HDR-DATA (last 48 bytes = bytes 272..320)
    hdr_off = OFF_0x200_HDR + 272
    val = int.from_bytes(buf[hdr_off : hdr_off + 48], "little")
    zp_u8 = (zp_in + 128) & 0xFF
    hi = float(np.float32(127 - zp_in) * np.float32(s_in))
    lo = float(np.float32(-128 - zp_in) * np.float32(s_in))
    for bit_pos, width, fbits in [
        (80, 8, zp_u8),
        (328, 8, zp_u8),
        (189, 32, f32_to_u32(s_in)),
        (222, 32, f32_to_u32(hi)),
        (259, 32, f32_to_u32(lo)),
    ]:
        mask = ((1 << width) - 1) << bit_pos
        val = (val & ~mask) | ((fbits & ((1 << width) - 1)) << bit_pos)
    buf[hdr_off : hdr_off + 48] = val.to_bytes(48, "little")

    # 4) 0x200@4512 stream125
    patched = patch_stream125_in_0x200(
        bytes(buf),
        OFF_0x200_S2,
        32,
        mul=s_in,
        fclip_max=fclip_max,
        fclip_min=fclip_min,
    )
    return patched


# ---------------------------------------------------------------------------
# Hardware FP32-RNE Horner Simulator
# ---------------------------------------------------------------------------


def eval_0x640_horner_fp32(
    x_in: np.ndarray,
    pkt: SplinePacket0x640,
    fclip_min: float,
    fclip_max: float,
) -> np.ndarray:
    """Simulate Tensor G5 VICA Stage 2 FP32 Horner spline evaluation.

    Segment selection: first segment s in [0..6] with x < bp[s], else segment 7.
    """
    x = np.clip(x_in.astype(np.float32), np.float32(fclip_min), np.float32(fclip_max))
    bps = np.array(pkt.breakpoints, dtype=np.float32)
    seg_idx = np.full(x.shape, 7, dtype=np.int32)
    for s in range(6, -1, -1):
        seg_idx = np.where(x < bps[s], s, seg_idx)

    c_arr = np.array(pkt.coeffs, dtype=np.float32)  # (8, 5)
    c0 = c_arr[seg_idx, 0]
    c1 = c_arr[seg_idx, 1]
    c2 = c_arr[seg_idx, 2]
    c3 = c_arr[seg_idx, 3]
    c4 = c_arr[seg_idx, 4]

    # Exact FP32 Horner evaluation
    y = np.float32(c4)
    y = np.float32(y * x + c3)
    y = np.float32(y * x + c2)
    y = np.float32(y * x + c1)
    y = np.float32(y * x + c0)
    return y


def simulate_3stage_dgc0(q_in: np.ndarray, chunk3: bytes) -> np.ndarray:
    """End-to-end simulation of the 3-stage logistic_scale_zp-derived VICA pipeline."""
    # Extract Stage 1 params from HDR-DATA & Stage 2 stream125
    hdr_val = int.from_bytes(chunk3[OFF_0x200_HDR + 272 : OFF_0x200_HDR + 320], "little")
    zp_in = ((hdr_val >> 80) & 0xFF) - 128
    s_in = u32_to_f32((hdr_val >> 189) & 0xFFFFFFFF)

    s2_csr = decode_stream125_from_0x200(chunk3, OFF_0x200_S2, 32)
    pkt2 = decode_0x640(chunk3[OFF_0x640_S2 : OFF_0x640_S2 + 208])

    x = (q_in.astype(np.float32) - np.float32(zp_in)) * np.float32(s_in)
    y = eval_0x640_horner_fp32(x, pkt2, float(s2_csr["fclip_min"]), float(s2_csr["fclip_max"]))
    u = np.clip(np.rint(y), 0.0, 255.0).astype(np.int32)
    q_out = (u - 128).astype(np.int8)
    return q_out


# ---------------------------------------------------------------------------
# Clean-Room Pure-NumPy Degree-4 Remez Exchange Minimax Synthesizer
# ---------------------------------------------------------------------------


def _eval_poly_f64(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate c0 + c1*x + ... + cd*x^d in float64 via Horner."""
    y = np.zeros_like(x, dtype=np.float64)
    for c in reversed(coeffs):
        y = y * x + c
    return y


def remez_poly_deg4(
    fn: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_iters: int = 20,
    n_grid: int = 2001,
) -> Tuple[np.ndarray, float]:
    """Compute minimax degree-4 polynomial p(x) = sum_{k=0}^4 c_k x^k on [a, b] via Remez exchange.

    Works in centered/normalized coordinate t in [-1, 1] for numerical conditioning, then
    converts back to standard power basis c0..c4 in float64 (before final FP32 cast).
    """
    if b <= a:
        raise ValueError(f"Invalid interval [{a}, {b}]")
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    # Initial reference: 6 Chebyshev extrema in [-1, 1]: cos(k*pi/5), k=0..5 (ascending)
    t_ref = -np.cos(np.linspace(0.0, np.pi, 6))
    t_grid = np.linspace(-1.0, 1.0, n_grid)
    x_grid = mid + half * t_grid
    y_grid = fn(x_grid)

    best_t_coeffs = None
    best_err = float("inf")
    signs = np.array([(-1.0) ** k for k in range(6)], dtype=np.float64)

    for _ in range(n_iters):
        x_ref = mid + half * t_ref
        y_ref = fn(x_ref)
        # Solve [1, t, t^2, t^3, t^4, (-1)^k] * [c0..c4, E] = y_ref
        V = np.vander(t_ref, N=5, increasing=True)
        A = np.column_stack([V, signs])
        try:
            sol = np.linalg.solve(A, y_ref)
        except np.linalg.LinAlgError:
            break
        t_coeffs = sol[:5]
        err_grid = y_grid - _eval_poly_f64(t_coeffs, t_grid)
        max_abs = float(np.max(np.abs(err_grid)))
        if max_abs < best_err:
            best_err = max_abs
            best_t_coeffs = t_coeffs.copy()

        # Check equioscillation convergence
        E_est = abs(float(sol[5]))
        if E_est > 0 and (max_abs - E_est) / max(max_abs, 1e-15) < 1e-7:
            break

        # Find 6 alternating extrema via sign-change bracket decomposition
        # Locate roots of err_grid between reference nodes
        brackets = [0]
        for k in range(5):
            i_lo = int(np.searchsorted(t_grid, t_ref[k]))
            i_hi = int(np.searchsorted(t_grid, t_ref[k + 1]))
            i_lo = max(0, min(n_grid - 1, i_lo))
            i_hi = max(i_lo + 1, min(n_grid - 1, i_hi))
            sub = err_grid[i_lo : i_hi + 1]
            # Find index closest to zero crossing
            cross = np.where(np.signbit(sub[:-1]) != np.signbit(sub[1:]))[0]
            if len(cross) > 0:
                brackets.append(i_lo + int(cross[0]) + 1)
            else:
                brackets.append((i_lo + i_hi) // 2)
        brackets.append(n_grid - 1)

        new_t_ref = np.empty(6, dtype=np.float64)
        for k in range(6):
            lo, hi = brackets[k], brackets[k + 1]
            if hi <= lo:
                new_t_ref[k] = t_ref[k]
                continue
            sub = err_grid[lo : hi + 1]
            # Pick extrema with matching sign if possible, else max abs
            want_pos = (sol[5] * signs[k]) >= 0
            idx = int(np.argmax(sub) if want_pos else np.argmin(sub))
            new_t_ref[k] = t_grid[lo + idx]

        # Enforce strict ordering
        if not np.all(np.diff(new_t_ref) > 1e-9):
            break
        t_ref = new_t_ref

    assert best_t_coeffs is not None

    # Convert p(t) where t = (x - mid) / half = alpha * x + beta into standard power basis in x
    alpha = 1.0 / half
    beta = -mid / half
    x_coeffs = np.zeros(5, dtype=np.float64)
    # Binomial expansion of (alpha*x + beta)^k
    lin_pow = np.array([1.0], dtype=np.float64)
    lin = np.array([beta, alpha], dtype=np.float64)
    for k in range(5):
        x_coeffs[: len(lin_pow)] += best_t_coeffs[k] * lin_pow
        if k < 4:
            lin_pow = np.convolve(lin_pow, lin)

    # Measure true FP32 Horner max error on dense grid
    x_f32 = x_grid.astype(np.float32)
    c_f32 = x_coeffs.astype(np.float32)
    y_f32 = c_f32[4]
    y_f32 = np.float32(y_f32 * x_f32 + c_f32[3])
    y_f32 = np.float32(y_f32 * x_f32 + c_f32[2])
    y_f32 = np.float32(y_f32 * x_f32 + c_f32[1])
    y_f32 = np.float32(y_f32 * x_f32 + c_f32[0])
    fp32_err = float(np.max(np.abs(fn(x_f32.astype(np.float64)) - y_f32.astype(np.float64))))
    return c_f32.astype(np.float64), fp32_err


def optimize_8seg_minimax(
    fn: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[float, float],
    init_bps: Sequence[float] | None = None,
    fixed_left_poly: Sequence[float] | None = None,
    fixed_right_poly: Sequence[float] | None = None,
    n_balance_iters: int = 18,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Synthesize an 8-segment degree-4 minimax spline with equibalanced breakpoints.

    Optionally pins segment 0 (fixed_left_poly, e.g. constant 0) and/or segment 7
    (fixed_right_poly, e.g. linear x) for asymptotic tails like SiLU/GELU.
    """
    lo, hi = float(domain[0]), float(domain[1])
    if init_bps is None:
        bps = list(np.linspace(lo, hi, 9)[1:-1])
    else:
        bps = [float(b) for b in init_bps]
    assert len(bps) == 7

    def fit_all(cur_bps: List[float]) -> Tuple[List[List[float]], List[float]]:
        edges = [lo] + cur_bps + [hi]
        segs: List[List[float]] = []
        errs: List[float] = []
        for s in range(8):
            a, b = edges[s], edges[s + 1]
            if s == 0 and fixed_left_poly is not None:
                c = np.array(fixed_left_poly, dtype=np.float32)
                xs = np.linspace(a, b, 512, dtype=np.float32)
                ys = c[4]
                for k in (3, 2, 1, 0):
                    ys = np.float32(ys * xs + c[k])
                err = float(np.max(np.abs(fn(xs.astype(np.float64)) - ys.astype(np.float64))))
                segs.append([float(v) for v in c])
                errs.append(err)
            elif s == 7 and fixed_right_poly is not None:
                c = np.array(fixed_right_poly, dtype=np.float32)
                xs = np.linspace(a, b, 512, dtype=np.float32)
                ys = c[4]
                for k in (3, 2, 1, 0):
                    ys = np.float32(ys * xs + c[k])
                err = float(np.max(np.abs(fn(xs.astype(np.float64)) - ys.astype(np.float64))))
                segs.append([float(v) for v in c])
                errs.append(err)
            else:
                c, err = remez_poly_deg4(fn, a, b)
                segs.append([float(np.float32(v)) for v in c])
                errs.append(err)
        return segs, errs

    best_segs, best_errs = fit_all(bps)
    best_bps = list(bps)
    best_max = max(best_errs)

    # Active interior segment indices (all 8 unless tails are fixed)
    free_bp_indices = list(range(7))
    # Iterative error-equibalancing (adjacent breakpoint shifts using ~h^5 error scaling)
    cur_bps = list(bps)
    cur_errs = list(best_errs)
    for it in range(n_balance_iters):
        edges = [lo] + cur_bps + [hi]
        new_bps = list(cur_bps)
        step_damp = 0.55 * (0.88 ** it)
        for i in free_bp_indices:
            e_left = max(cur_errs[i], 1e-12)
            e_right = max(cur_errs[i + 1], 1e-12)
            # For deg-4 poly, error ~ h^5 -> ratio of desired widths ~ (e_right/e_left)^(1/5)
            w_left = edges[i + 1] - edges[i]
            w_right = edges[i + 2] - edges[i + 1]
            # If one side is a fixed tail, adjust gently in log-error space
            if (i == 0 and fixed_left_poly is not None) or (i == 6 and fixed_right_poly is not None):
                ratio = (e_right / e_left) ** 0.08
            else:
                ratio = (e_right / e_left) ** 0.20
            target_frac = (w_left * ratio) / (w_left * ratio + w_right)
            target_bp = edges[i] + target_frac * (w_left + w_right)
            cand = cur_bps[i] + step_damp * (target_bp - cur_bps[i])
            # Clamp strictly inside neighbors with safety margin
            margin = 0.05 * (edges[i + 2] - edges[i])
            cand = max(edges[i] + margin, min(edges[i + 2] - margin, cand))
            new_bps[i] = float(np.float32(cand))
        # Ensure strict monotonicity after float32 rounding
        if not all(new_bps[k] < new_bps[k + 1] for k in range(6)):
            break
        segs, errs = fit_all(new_bps)
        cur_bps, cur_errs = new_bps, errs
        if max(errs) < best_max:
            best_max = max(errs)
            best_segs = segs
            best_errs = errs
            best_bps = list(new_bps)

    return best_segs, [float(np.float32(b)) for b in best_bps], best_errs


# ---------------------------------------------------------------------------
# Target LLM Activations (exact float64 ground truth)
# ---------------------------------------------------------------------------


def fn_silu(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    # Numerically stable sigmoid
    sig = np.where(x64 >= 0, 1.0 / (1.0 + np.exp(-x64)), np.exp(x64) / (1.0 + np.exp(x64)))
    return x64 * sig


def fn_gelu_tanh(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * x64 * (1.0 + np.tanh(k * (x64 + 0.044715 * (x64 ** 3))))


def fn_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=np.float64))


def scale_spline_affine(
    coeffs_8x5: List[List[float]],
    scale: float,
    offset: float,
) -> List[List[float]]:
    """Fuse affine output quantization u(x) = scale * f(x) + offset into 8x5 polynomial coeffs."""
    out: List[List[float]] = []
    for seg in coeffs_8x5:
        c0 = float(np.float32(seg[0] * scale + offset))
        c1 = float(np.float32(seg[1] * scale))
        c2 = float(np.float32(seg[2] * scale))
        c3 = float(np.float32(seg[3] * scale))
        c4 = float(np.float32(seg[4] * scale))
        out.append([c0, c1, c2, c3, c4])
    return out


# ---------------------------------------------------------------------------
# Probe builder & self-test
# ---------------------------------------------------------------------------


def build_input_ramp_4096() -> np.ndarray:
    """Deterministic 4096-byte int8 input covering every single int8 value [-128..127] 16x."""
    return np.tile(np.arange(-128, 128, dtype=np.int8), 16)


def run_self_tests_and_emit(emit_probes: bool = True) -> Dict[str, object]:
    report: Dict[str, object] = {}
    has_corpus = LOGISTIC_SCALE_ZP_DGC0.is_file() and VEC_LOGISTIC_DGC0.is_file()
    c3_lsz: Optional[bytes] = None

    if has_corpus:
        # 1. Bit-exact 0x640 round-trip on all 3 packets of logistic_scale_zp.dgc0
        img_lsz = Dgc0Image(LOGISTIC_SCALE_ZP_DGC0.read_bytes())
        c3_lsz = get_dive_chunk(img_lsz, 3)
        for name, off in [("S1@36", OFF_0x640_S1), ("S2@51", OFF_0x640_S2), ("S3@66", OFF_0x640_S3)]:
            raw_pkt = c3_lsz[off : off + 208]
            dec = decode_0x640(raw_pkt)
            reenc = encode_0x640(dec)
            assert reenc == raw_pkt, f"0x640 round-trip mismatch at {name} (off={off})"
        report["roundtrip_0x640_3pkts"] = "PASS (bit-exact 208B x 3)"

        # 2. Bit-exact Stage 1 patcher proof: logistic_scale_zp -> vec_logistic
        img_vl = Dgc0Image(VEC_LOGISTIC_DGC0.read_bytes())
        c3_vl = get_dive_chunk(img_vl, 3)
        c3_transformed = patch_stage1_input_quant(c3_lsz, s_in=0.0625, zp_in=0)
        assert c3_transformed == c3_vl, "Stage 1 transform logistic_scale_zp -> vec_logistic did not match 5520/5520 bytes!"
        report["stage1_patcher_exact_proof"] = "PASS (5520/5520 bytes exact)"

        # 3. Live silicon FP32-RNE Horner simulator check
        q_in_lsz = np.frombuffer(LOGISTIC_SCALE_ZP_IN.read_bytes(), dtype=np.int8)
        hw_out_lsz = np.frombuffer(LOGISTIC_SCALE_ZP_OUT.read_bytes(), dtype=np.int8)
        sim_out_lsz = simulate_3stage_dgc0(q_in_lsz, c3_lsz)
        exact_matches = int(np.sum(sim_out_lsz == hw_out_lsz))
        assert exact_matches == 4096, f"Simulator mismatch vs hardware: {exact_matches}/4096"
        report["hw_sim_logistic_4096"] = f"PASS ({exact_matches}/4096 exact)"
    else:
        report["external_corpus_checks"] = "SKIPPED (set FREETPU_CORPUS_DIR to verify against compiled DGC0s)"
        emit_probes = False

    # 4. Synthesize clean-room 8-segment degree-4 minimax splines
    specs = [
        {
            "name": "silu",
            "formula": "x / (1 + exp(-x))",
            "fn": fn_silu,
            # Cover full [-11.1, +11.1] matching logistic_scale_zp Stage 2 fclip
            "domain": (-11.1, 11.1),
            "init_bps": [-6.5, -3.5, -1.6, 0.0, 1.6, 3.5, 6.5],
            "fixed_left": None,
            "fixed_right": None,
            # Probe keeps logistic_scale_zp Stage 1 untouched (s_in=0.03125, zp_in=-10 -> x in [-3.6875, +4.28125])
            "probe_s_in": 0.03125,
            "probe_zp_in": -10,
            "probe_fclip": (-11.1, 11.1),
            "probe_patch_stage1": False,
            # Output quant: u = 50*SiLU(x) + 16 in [2.1, 227.1] -> s_out=0.02, zp_out=-112
            "s_out": 0.02,
            "zp_out": -112,
        },
        {
            "name": "gelu",
            "formula": "0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
            "fn": fn_gelu_tanh,
            # Cover full [-11.1, +11.1] with exact asymptotic tails (< -4.2 -> 0, > +4.2 -> x)
            # and 6 minimax segments concentrated on the active curvature band [-4.2, +4.2]
            "domain": (-11.1, 11.1),
            "init_bps": [-4.2, -2.6, -1.3, 0.0, 1.3, 2.6, 4.2],
            "fixed_left": [0.0, 0.0, 0.0, 0.0, 0.0],
            "fixed_right": [0.0, 1.0, 0.0, 0.0, 0.0],
            "probe_s_in": 0.03125,
            "probe_zp_in": -10,
            "probe_fclip": (-11.1, 11.1),
            "probe_patch_stage1": False,
            "s_out": 0.02,
            "zp_out": -112,
        },
        {
            "name": "exp",
            "formula": "exp(x) on [-12, 0]",
            "fn": fn_exp,
            "domain": (-12.0, 0.0),
            # Exponentially tighter segments near 0 where curvature is largest
            "init_bps": [-9.5, -7.2, -5.2, -3.5, -2.1, -1.1, -0.45],
            "fixed_left": None,
            "fixed_right": None,
            # Probe patches Stage 1 to s_in=0.125, zp_in=0 (x in [-16, +15.875]) and Stage 2 fclip=[-12, 0]!
            "probe_s_in": 0.125,
            "probe_zp_in": 0,
            "probe_fclip": (-12.0, 0.0),
            "probe_patch_stage1": True,
            # Output quant: u = 256*exp(x) in [0, 255] -> s_out=1/256, zp_out=-128
            "s_out": 1.0 / 256.0,
            "zp_out": -128,
        },
    ]

    if c3_lsz is not None:
        pkt2_template = decode_0x640(c3_lsz[OFF_0x640_S2 : OFF_0x640_S2 + 208])
    else:
        pkt2_template = SplinePacket0x640(
            header_76b=(0xFFFF << 12) | 0x640,
            coeffs=[[0.0] * 5 for _ in range(8)],
            breakpoints=[-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
            tail_84b=0x7804,
        )
    q_ramp = build_input_ramp_4096()
    all_q_int8 = np.arange(-128, 128, dtype=np.int8)

    tables_summary: Dict[str, object] = {}
    probe_manifest: List[Dict[str, object]] = []
    if emit_probes:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sp in specs:
        name = str(sp["name"])
        fn = sp["fn"]
        lo, hi = sp["domain"]
        segs_raw, bps, seg_errs = optimize_8seg_minimax(
            fn,
            (lo, hi),
            init_bps=sp["init_bps"],
            fixed_left_poly=sp["fixed_left"],
            fixed_right_poly=sp["fixed_right"],
        )

        # Measure full-domain float32 Horner max error on 20,001 dense points
        pkt_raw = SplinePacket0x640(
            header_76b=pkt2_template.header_76b,
            coeffs=segs_raw,
            breakpoints=bps,
            tail_84b=pkt2_template.tail_84b,
        )
        xs_dense = np.linspace(lo, hi, 20001, dtype=np.float32)
        y_hat_dense = eval_0x640_horner_fp32(xs_dense, pkt_raw, lo, hi)
        y_true_dense = fn(xs_dense.astype(np.float64))
        max_abs_f32_err = float(np.max(np.abs(y_true_dense - y_hat_dense.astype(np.float64))))

        # Affine-fused quantization coefficients: u(x) = f(x) / s_out + (zp_out + 128)
        s_out = float(sp["s_out"])
        zp_out = int(sp["zp_out"])
        scale_u = 1.0 / s_out
        offset_u = float(zp_out + 128)
        segs_quant = scale_spline_affine(segs_raw, scale_u, offset_u)
        pkt_quant = SplinePacket0x640(
            header_76b=pkt2_template.header_76b,
            coeffs=segs_quant,
            breakpoints=bps,
            tail_84b=pkt2_template.tail_84b,
        )
        pkt_bytes = encode_0x640(pkt_quant)
        # Verify round-trip of synthesized packet
        assert encode_0x640(decode_0x640(pkt_bytes)) == pkt_bytes

        if c3_lsz is not None:
            # Patch chunk[3] & simulate 3-stage hardware pipeline
            c3_probe = bytearray(c3_lsz)
            if sp["probe_patch_stage1"]:
                fmin, fmax = sp["probe_fclip"]
                c3_probe = bytearray(
                    patch_stage1_input_quant(
                        bytes(c3_probe),
                        s_in=float(sp["probe_s_in"]),
                        zp_in=int(sp["probe_zp_in"]),
                        fclip_min=float(fmin),
                        fclip_max=float(fmax),
                    )
                )
            c3_probe[OFF_0x640_S2 : OFF_0x640_S2 + 208] = pkt_bytes
            c3_probe_bytes = bytes(c3_probe)
            q_sim_256 = simulate_3stage_dgc0(all_q_int8, c3_probe_bytes)
        else:
            c3_probe_bytes = b""
            q_sim_256 = None

        # Evaluate across ALL 256 int8 inputs: compare pre-rounding spline error in quantized LSBs
        s_in = float(sp["probe_s_in"])
        zp_in = int(sp["probe_zp_in"])
        fmin, fmax = sp["probe_fclip"]
        x_256 = (all_q_int8.astype(np.float64) - zp_in) * s_in
        x_256_clamped = np.clip(x_256, fmin, fmax)
        u_true_256 = np.clip(fn(x_256_clamped) * scale_u + offset_u, 0.0, 255.0)
        u_sim_256 = np.clip(
            eval_0x640_horner_fp32(x_256.astype(np.float32), pkt_quant, fmin, fmax).astype(np.float64),
            0.0,
            255.0,
        )
        max_lsb_err = float(np.max(np.abs(u_true_256 - u_sim_256)))

        # Integer quantized comparison (ideal float64 vs hardware FP32-RNE simulation)
        q_ideal_256 = (np.clip(np.rint(u_true_256), 0.0, 255.0).astype(np.int32) - 128).astype(np.int8)
        if q_sim_256 is None:
            q_sim_256 = (np.clip(np.rint(u_sim_256), 0.0, 255.0).astype(np.int32) - 128).astype(np.int8)
        max_int8_diff = int(np.max(np.abs(q_ideal_256.astype(np.int32) - q_sim_256.astype(np.int32))))
        exact_256 = int(np.sum(q_ideal_256 == q_sim_256))

        assert max_lsb_err < 0.5, f"{name}: max_lsb_err={max_lsb_err:.6f} >= 0.5 LSB!"
        assert max_int8_diff <= 1, f"{name}: max_int8_diff={max_int8_diff} > 1!"

        entry_summary = {
            "formula": sp["formula"],
            "domain": [lo, hi],
            "breakpoints": bps,
            "coeffs_raw_8x5": segs_raw,
            "coeffs_quant_8x5": segs_quant,
            "seg_max_abs_f32_err": seg_errs,
            "dense_max_abs_f32_err": max_abs_f32_err,
            "probe_s_in": s_in,
            "probe_zp_in": zp_in,
            "probe_fclip": [fmin, fmax],
            "probe_patch_stage1": bool(sp["probe_patch_stage1"]),
            "probe_s_out": s_out,
            "probe_zp_out": zp_out,
            "max_quantized_lsb_err": max_lsb_err,
            "int8_exact_matches_256": exact_256,
            "int8_max_abs_diff": max_int8_diff,
            "pkt_0x640_sha256": sha256_bytes(pkt_bytes),
        }
        tables_summary[name] = entry_summary

        if emit_probes:
            probe_img = Dgc0Image(LOGISTIC_SCALE_ZP_DGC0.read_bytes())
            probe_img.patch_vliw_chunk(3, c3_probe_bytes)
            dgc0_bytes = probe_img.repack()

            # Verify emitted DGC0 parses cleanly with dgc0_dump
            parsed = dgc0_dump.parse_dgc0(dgc0_bytes)
            assert parsed["external_buffers"][0]["dive_module"]["hardware_instruction_chunks"][3]["size_bytes"] == 5520

            ref_4096 = simulate_3stage_dgc0(q_ramp, c3_probe_bytes)
            dgc0_path = OUT_DIR / f"{name}_probe.dgc0"
            in_path = OUT_DIR / f"{name}_probe.in.bin"
            ref_path = OUT_DIR / f"{name}_probe.ref.bin"
            dgc0_path.write_bytes(dgc0_bytes)
            in_path.write_bytes(q_ramp.tobytes())
            ref_path.write_bytes(ref_4096.tobytes())

            probe_manifest.append(
                {
                    "name": f"{name}_probe",
                    "formula": sp["formula"],
                    "dgc0_path": str(dgc0_path),
                    "in_path": str(in_path),
                    "ref_path": str(ref_path),
                    "in_bytes": 4096,
                    "out_bytes": 4096,
                    "s_in": s_in,
                    "zp_in": zp_in,
                    "fclip": [fmin, fmax],
                    "s_out": s_out,
                    "zp_out": zp_out,
                    "patch_stage1": bool(sp["probe_patch_stage1"]),
                    "max_quantized_lsb_err": max_lsb_err,
                    "int8_exact_matches_256": exact_256,
                    "dgc0_sha256": sha256_bytes(dgc0_bytes),
                    "in_sha256": sha256_bytes(q_ramp.tobytes()),
                    "ref_sha256": sha256_bytes(ref_4096.tobytes()),
                }
            )

    if emit_probes:
        (OUT_DIR / "manifest.json").write_text(json.dumps(probe_manifest, indent=2) + "\n")
        TABLES_JSON.write_text(json.dumps(tables_summary, indent=2) + "\n")

    report["splines"] = {
        k: {
            "dense_max_abs_f32_err": v["dense_max_abs_f32_err"],
            "max_quantized_lsb_err": v["max_quantized_lsb_err"],
            "int8_exact_matches_256": v["int8_exact_matches_256"],
            "int8_max_abs_diff": v["int8_max_abs_diff"],
        }
        for k, v in tables_summary.items()
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor G5 VICA 0x640 spline codec & minimax synthesizer")
    ap.add_argument("--self-test", action="store_true", help="Run full verification & emit probe blobs")
    ap.add_argument("--no-emit", action="store_true", help="Run self-tests without writing probe blobs")
    ap.add_argument("--decode", type=Path, help="Decode all 0x640 packets from a DGC0 file")
    args = ap.parse_args()

    if args.decode:
        img = Dgc0Image(args.decode.read_bytes())
        c3 = get_dive_chunk(img, 3)
        for off in range(0, len(c3) - 208 + 1, 16):
            w0 = struct.unpack_from("<Q", c3, off)[0]
            if (w0 & 0xFFF) == 0x640 and ((w0 >> 12) & 0xFFFF) == 0xFFFF:
                pkt = decode_0x640(c3[off : off + 208])
                seq = (w0 >> 48) & 0xFFFF
                print(f"0x640@{seq} (off={off}, tail=0x{pkt.tail_84b:x}):")
                print(f"  bps = {[round(b, 6) for b in pkt.breakpoints]}")
                for s, c in enumerate(pkt.coeffs):
                    print(f"  seg{s} = {[round(v, 7) for v in c]}")
        return

    report = run_self_tests_and_emit(emit_probes=not args.no_emit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
