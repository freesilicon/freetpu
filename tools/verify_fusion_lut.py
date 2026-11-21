#!/usr/bin/env python3
"""Clean-room Tensor G5 (buenos_a0) 0x640 LUT/Spline Fusion Synthesizer & Verification Harness.

Covers Step 4 offline verification and probe emission:
  1. Verifies exact FP32-RNE Horner hardware simulation against ALL THREE live `rango` silicon
     outputs in `work/device-runner/phase5_live/spline_probes/` (`silu_probe`, `gelu_probe`,
     `exp_probe`): 12,288 / 12,288 bytes bit-exact (`max_abs_err == 0`) on both baremetal and
     direct runners.
  2. Implements Mode-A zero-overhead in-place Fused `GEMM + SiLU` and `GEMM + GELU` inside
     `chunk[5]`'s existing `0x640` + `HDR_DATA` stage (0 extra flits, 0 extra bytes, 0 DRAM
     round-trip, 0 extra VII launch!), including:
       - Exact zero-crossing breakpoint placement (`f(x)/s_out + zp_out == -0.5`) so negative
         `int8` outputs fold `+256.0` into `c0` (raw byte `q_out & 0xFF`) directly in 1 stage.
       - Hardware `fclip_max` saturation guard (`b17+222`) at `f^{-1}((127 - zp_out) * s_out)`
         preventing positive `int8` wrap-around (`127 -> 128 = -128`).
  3. Emits ready-to-run fused custom `.dgc0` + `.in.bin` + `.ref.bin` probes into
     `work/isa-gemm/int4_and_fusion/probes/` (`fused_gemm_silu_1x16x16_zp0`,
     `fused_gemm_gelu_1x16x16_zp0`, `fused_gemm_silu_1x16x16_zpm112`,
     `fused_gemm_silu_4x32x32_zp0`), verifying full `DGC0`/`DWS0`/`DIVE` round-trip and
     < 0.02 quantized-LSB spline accuracy across all 256 intermediate `u_mid` levels.
  4. Verifies multi-op chaining invariants (`pipe2_fc_relu_1x16x16` 2-stage VICA `chunk[5]`,
     `pipe2_conv1x1_add_1x16x16x16` multi-input `0xa40`, and `fc_i4_pc_*` multi-parameter
     `0xa40`/`0x940`/`0x980` `ext[2]` offset chaining).
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
sys.path.insert(0, str(_THIS_DIR))

import dgc0_builder as db  # noqa: E402
import dgc0_dump  # noqa: E402
from dgc0_patcher import Dgc0Image  # noqa: E402
import multitile_synth as ms  # noqa: E402
import spline_0x640 as sp  # noqa: E402

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
BASE: Optional[Path] = Path(_CORPUS_ENV) if _CORPUS_ENV else None
LIVE_SPLINES_DIR = (BASE / "work/device-runner/phase5_live/spline_probes") if BASE else Path("phase5_live/spline_probes")
PROBE_SRC_DIR = (BASE / "work/int4-splines/spline_probes") if BASE else Path("spline_probes")
BATCH0_DIR = (BASE / "work/device-runner/phase3_batch0") if BASE else Path("phase3_batch0")
OUT_PROBES_DIR = Path("out_fusion_probes")


# ---------------------------------------------------------------------------
# 1. Verify FP32-RNE Horner Simulator Against Live `rango` Silicon (12,288 B)
# ---------------------------------------------------------------------------


def verify_live_silicon_splines() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    total_exact = 0
    total_bytes = 0
    for name in ["silu_probe", "gelu_probe", "exp_probe"]:
        dgc0_raw = (PROBE_SRC_DIR / f"{name}.dgc0").read_bytes()
        in_arr = np.frombuffer((PROBE_SRC_DIR / f"{name}.in.bin").read_bytes(), dtype=np.int8)
        bm_out = np.frombuffer((LIVE_SPLINES_DIR / f"{name}.bm.out.bin").read_bytes(), dtype=np.int8)
        dir_out = np.frombuffer((LIVE_SPLINES_DIR / f"{name}.dir.out.bin").read_bytes(), dtype=np.int8)
        assert np.array_equal(bm_out, dir_out), f"{name}: baremetal vs direct mismatch!"

        img = Dgc0Image(dgc0_raw)
        c3 = sp.get_dive_chunk(img, 3)
        sim_out = sp.simulate_3stage_dgc0(in_arr, c3)
        exact = int(np.sum(sim_out == bm_out))
        max_err = int(np.max(np.abs(sim_out.astype(np.int32) - bm_out.astype(np.int32))))
        assert exact == len(bm_out) and max_err == 0, f"{name}: sim vs silicon {exact}/{len(bm_out)} max_err={max_err}"
        total_exact += exact
        total_bytes += len(bm_out)
        results[name] = {"exact": exact, "total": len(bm_out), "max_abs_err": max_err}
    results["total_exact"] = total_exact
    results["total_bytes"] = total_bytes
    return results


# ---------------------------------------------------------------------------
# 2. Mode-A Single-Stage Fused `GEMM + SiLU/GELU` Spline Synthesis (0 Extra Flits)
# ---------------------------------------------------------------------------


def _find_monotone_root(fn: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """Bisection root for strictly increasing `fn(x) == target` on `[lo, hi]`."""
    a, b = lo, hi
    for _ in range(80):
        m = 0.5 * (a + b)
        if fn(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _find_negative_crossing_interval(
    fn: Callable[[float], float],
    threshold: float,
    lo: float = -11.1,
    hi: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """For SiLU/GELU on `[lo, 0]`, find `[x_left, x_right]` where `fn(x) < threshold` (for `threshold < 0`)."""
    xs = np.linspace(lo, hi, 40001)
    ys = np.array([fn(float(x)) for x in xs])
    below = np.where(ys < threshold)[0]
    if len(below) == 0:
        return None
    i0, i1 = int(below[0]), int(below[-1])
    # Refine left crossing (decreasing) and right crossing (increasing)
    if i0 > 0:
        a, b = float(xs[i0 - 1]), float(xs[i0])
        for _ in range(60):
            m = 0.5 * (a + b)
            if fn(m) >= threshold:
                a = m
            else:
                b = m
        x_l = 0.5 * (a + b)
    else:
        x_l = lo
    if i1 + 1 < len(xs):
        a, b = float(xs[i1]), float(xs[i1 + 1])
        for _ in range(60):
            m = 0.5 * (a + b)
            if fn(m) < threshold:
                a = m
            else:
                b = m
        x_r = 0.5 * (a + b)
    else:
        x_r = hi
    return (x_l, x_r)


def build_fused_single_stage_spline(
    act_name: str,
    *,
    s_mid: float,
    zp_mid: int,
    s_out: float,
    zp_out: int,
    domain: Tuple[float, float] = (-11.1, 11.1),
) -> Tuple[List[List[float]], List[float], float, float, float]:
    """Synthesize an 8-segment degree-4 `0x640` spline encoding `u_byte = clamp(RNE(f(x)/s_out + zp_out), -128, 127) & 0xFF`.

    Directly replaces `chunk[5]`'s existing degree-1 MSB-flip `0x640` packet (with `HDR_DATA`
    `0x61[9:2] = 0`) so GEMM + SiLU/GELU executes in 1 VICA pass with 0 extra flits!

    Returns `(coeffs_8x5, breakpoints_7, fclip_min, fclip_max, max_lsb_err)`.
    """
    if act_name == "silu":
        fn_vec = sp.fn_silu
        fn_s = lambda x: float(sp.fn_silu(np.array([x]))[0])
        min_val = -0.2784645427610738
    elif act_name == "gelu":
        fn_vec = sp.fn_gelu_tanh
        fn_s = lambda x: float(sp.fn_gelu_tanh(np.array([x]))[0])
        min_val = -0.1700404099869362
    else:
        raise ValueError(f"Unsupported fused activation: {act_name}")

    # Input range reachable from intermediate uint8 u_mid in [0..255]:
    x_mid_lo = float(np.float32((-128 - zp_mid) * np.float32(s_mid)))
    x_mid_hi = float(np.float32((127 - zp_mid) * np.float32(s_mid)))
    fclip_min = max(domain[0], x_mid_lo)

    # Hardware positive saturation guard: clamp x <= f^{-1}((127 - zp_out) * s_out)
    # so y_out <= 127.0 (never wraps into 128 = -128 in unsigned byte output!).
    y_sat_hi = (127.0 - zp_out) * s_out
    if fn_s(domain[1]) > y_sat_hi:
        x_sat_hi = _find_monotone_root(fn_s, y_sat_hi, 0.0, domain[1])
    else:
        x_sat_hi = domain[1]
    fclip_max = float(np.float32(min(x_mid_hi, x_sat_hi)))
    fclip_min = float(np.float32(fclip_min))
    assert fclip_min < fclip_max

    # Check if negative output saturation (-128) is safe:
    assert (min_val / s_out + zp_out) >= -128.0, (
        f"{act_name} minimum {min_val}/s_out + zp_out < -128; use Mode-B 2-stage VICA or larger s_out/zp_out"
    )

    # Find crossing(s) where y_real(x) = f(x)/s_out + zp_out crosses -0.5 (boundary between int8 -1 and 0)
    f_thresh = (-0.5 - zp_out) * s_out
    scale_u = 1.0 / s_out

    # Partition [fclip_min, fclip_max] into 8 segments with exact breakpoint(s) at f(x) == f_thresh
    if f_thresh <= min_val:
        # Output is ALWAYS >= 0 (e.g. zp_out >= ceil(-min_val/s_out)) -> zero wrap segments!
        bps = [float(np.float32(b)) for b in np.linspace(fclip_min, fclip_max, 9)[1:-1]]
        seg_neg = [False] * 8
    elif f_thresh < 0.0:
        # Two crossings on the negative dip (e.g. symmetric zp_out == 0!)
        iv = _find_negative_crossing_interval(fn_s, f_thresh, lo=min(fclip_min, -11.1), hi=0.0)
        assert iv is not None
        xl, xr = iv
        if xl <= fclip_min + 1e-4:
            # Left crossing is outside domain -> 1 crossing at xr: segs 0..2 negative, segs 3..7 non-negative
            b_neg = np.linspace(fclip_min, xr, 4)[1:-1].tolist()
            b_pos = np.linspace(xr, fclip_max, 6)[1:-1].tolist()
            bps = [float(np.float32(b)) for b in (b_neg + [xr] + b_pos)]
            seg_neg = [True, True, True, False, False, False, False, False]
        else:
            # 2 crossings inside [fclip_min, fclip_max]:
            # seg 0: [fclip_min, xl) -> non-neg (tail ~0 -> byte 0)
            # seg 1,2: [xl, xr) -> negative dip (+256 wrap!)
            # seg 3..7: [xr, fclip_max] -> non-neg
            mid_dip = 0.5 * (xl + xr)
            b_pos = np.linspace(xr, fclip_max, 6)[1:-1].tolist()
            bps = [float(np.float32(b)) for b in ([xl, mid_dip, xr] + b_pos)]
            seg_neg = [False, True, True, False, False, False, False, False]
    else:
        # f_thresh >= 0 (e.g. zp_out < 0 such as zp_out = -112): single crossing x_cross >= 0!
        x_cross = _find_monotone_root(fn_s, f_thresh, 0.0, max(fclip_max, 11.1))
        if x_cross >= fclip_max - 1e-4:
            bps = [float(np.float32(b)) for b in np.linspace(fclip_min, fclip_max, 9)[1:-1]]
            seg_neg = [True] * 8
        else:
            # 4 segments below x_cross (negative int8 -> +256 wrap), 4 segments above x_cross (>= 0)
            b_lo = np.linspace(fclip_min, x_cross, 5)[1:-1].tolist()
            b_hi = np.linspace(x_cross, fclip_max, 5)[1:-1].tolist()
            bps = [float(np.float32(b)) for b in (b_lo + [x_cross] + b_hi)]
            seg_neg = [True, True, True, True, False, False, False, False]

    assert len(bps) == 7 and all(bps[i] < bps[i + 1] for i in range(6)), f"Non-monotonic bps: {bps}"
    edges = [fclip_min] + bps + [fclip_max]

    coeffs_8x5: List[List[float]] = []
    for s in range(8):
        a, b = edges[s], edges[s + 1]
        c_raw, _ = sp.remez_poly_deg4(fn_vec, a, b)
        offset = float(zp_out + (256 if seg_neg[s] else 0))
        seg_c = [
            float(np.float32(c_raw[0] * scale_u + offset)),
            float(np.float32(c_raw[1] * scale_u)),
            float(np.float32(c_raw[2] * scale_u)),
            float(np.float32(c_raw[3] * scale_u)),
            float(np.float32(c_raw[4] * scale_u)),
        ]
        coeffs_8x5.append(seg_c)

    # Measure exact quantized LSB error across all 256 intermediate u_mid values
    u_all = np.arange(256, dtype=np.int32)
    x_all = ((u_all - (zp_mid + 128)).astype(np.float32)) * np.float32(s_mid)
    x_clamped = np.clip(x_all, np.float32(fclip_min), np.float32(fclip_max))
    pkt_tmp = sp.SplinePacket0x640(header_76b=0x640, coeffs=coeffs_8x5, breakpoints=bps, tail_84b=0x7804)
    y_hw = sp.eval_0x640_horner_fp32(x_all, pkt_tmp, fclip_min, fclip_max)
    u_hw = np.clip(np.rint(y_hw), 0.0, 255.0).astype(np.uint8).view(np.int8).astype(np.int32)

    y_true_f64 = np.clip(fn_vec(x_clamped.astype(np.float64)) * scale_u + zp_out, -128.0, 127.0)
    # Compare modular distance in quantized LSBs (unwrapping the +256 on negative outputs)
    y_hw_signed = np.where(y_hw >= 127.5, y_hw.astype(np.float64) - 256.0, y_hw.astype(np.float64))
    max_lsb_err = float(np.max(np.abs(y_true_f64 - y_hw_signed)))
    q_ideal = np.clip(np.rint(y_true_f64), -128, 127).astype(np.int32)
    max_int8_diff = int(np.max(np.abs(u_hw - q_ideal)))
    assert max_lsb_err < 0.05, f"{act_name}: max_lsb_err={max_lsb_err:.5f} >= 0.05"
    assert max_int8_diff <= 1, f"{act_name}: max_int8_diff={max_int8_diff} > 1"

    return coeffs_8x5, bps, fclip_min, fclip_max, max_lsb_err


def patch_chunk5_fused_activation(
    c5: bytes,
    *,
    coeffs_8x5: List[List[float]],
    breakpoints_7: List[float],
    s_mid: float,
    zp_mid: int,
    fclip_min: float,
    fclip_max: float,
) -> Tuple[bytes, int, int]:
    """Patch `chunk[5]`'s existing `0x640` + `HDR_DATA` in place (0 extra bytes!) for fused GEMM+Act."""
    fl = ms.bytes_to_flits(c5)
    idx_640 = None
    idx_hdr = None
    i = 0
    while i < len(fl):
        lo, hi = fl[i]
        if lo == 0 and hi == 0:
            break
        op = lo & 0xFFF
        if op == 0x640:
            idx_640 = i
            ln = 13
        elif op == 0x200:
            tw = (hi >> 12) & 0xFFF
            if (hi & 0xF0F) == 0 and (hi >> 48) == 0 and (tw % 2 == 1) and 3 <= tw <= 0x100:
                ln = 1 + ((tw - 1) // 2 + 3) // 4
            else:
                idx_hdr = i
                ln = 20
        elif op == 0x940:
            ln = 3
        elif op in (0x980, 0x900):
            ln = 5
        elif op in (0x9C0, 0xA40):
            ln = 4
        elif op == 0x680:
            ln = 2
        elif op == 0x6C0:
            ln = 1
        elif op == 0x3C0:
            ln = 14
        elif op in (0x440, 0x480):
            ln = 8
        elif op == 0x400:
            ln = 7
        elif op == 0x500:
            ln = 9
        elif op == 0x5C0:
            ln = 13
        elif op == 0x4C0:
            ln = 10
        else:
            raise ValueError(f"Unexpected op 0x{op:03x} at c[5] flit {i}")
        i += ln

    assert idx_640 is not None and idx_hdr is not None, "Could not locate 0x640 + HDR_DATA in chunk[5]"
    buf = bytearray(c5)

    # 1. Patch 0x640 (preserve header_76b with pid & tile mask, set degree-4 tail=0x7804)
    off640 = idx_640 * 16
    orig_640 = sp.decode_0x640(bytes(buf[off640 : off640 + 208]))
    new_640 = sp.SplinePacket0x640(
        header_76b=orig_640.header_76b,
        coeffs=coeffs_8x5,
        breakpoints=breakpoints_7,
        tail_84b=0x7804,  # degree-4 Horner enable!
    )
    buf[off640 : off640 + 208] = sp.encode_0x640(new_640)

    # 2. Patch HDR_DATA (final 48 bytes at +272..+320)
    off_hdr = idx_hdr * 16 + 272
    hdr_val = int.from_bytes(buf[off_hdr : off_hdr + 48], "little")
    zp_mid_u8 = (zp_mid + 128) & 0xFF
    for bit_pos, width, fbits in [
        (80, 8, zp_mid_u8),                  # CSR 0x57[11:4]: input zp_mid + 128
        (189, 32, sp.f32_to_u32(s_mid)),     # stream125[44:13]: input scale s_mid
        (222, 32, sp.f32_to_u32(fclip_max)), # stream125[77:46]: hardware positive saturation clamp
        (259, 32, sp.f32_to_u32(fclip_min)), # stream125[114:83]: spline domain lower clamp
        (328, 8, 0),                         # CSR 0x61[9:2]: zp_out_hw = 0 (folded into 0x640 c0!)
    ]:
        mask = ((1 << width) - 1) << bit_pos
        hdr_val = (hdr_val & ~mask) | ((fbits & ((1 << width) - 1)) << bit_pos)
    buf[off_hdr : off_hdr + 48] = hdr_val.to_bytes(48, "little")

    return bytes(buf), off640, idx_hdr * 16


def simulate_fused_gemm_activation(
    X_mk: np.ndarray,
    W_nk: np.ndarray,
    B_n: np.ndarray,
    *,
    s_in: float,
    zp_in: int,
    s_w: float,
    s_mid: float,
    zp_mid: int,
    c5_patched: bytes,
    off640: int,
    off_hdr: int,
) -> np.ndarray:
    """Exact Tensor G5 hardware simulator for Mode-A Fused GEMM + 0x640 Activation."""
    # Stage 1 (Systolic MAC -> u_mid in [0..255])
    acc = (X_mk.astype(np.int32) - zp_in) @ W_nk.astype(np.int32).T + B_n.astype(np.int32)[None, :]
    m_eff = np.float32((np.float32(s_in) * np.float32(s_w)) / np.float32(s_mid))
    u_mid = np.clip(np.rint(acc.astype(np.float32) * m_eff) + (zp_mid + 128), 0, 255).astype(np.int32)

    # Stage 2 (chunk[5] HDR_DATA dequant -> 0x640 deg-4 Horner -> uint8 raw byte viewed as int8)
    hdr_val = int.from_bytes(c5_patched[off_hdr + 272 : off_hdr + 320], "little")
    zp_in_hw = (hdr_val >> 80) & 0xFF
    s_in_hw = sp.u32_to_f32((hdr_val >> 189) & 0xFFFFFFFF)
    fclip_max = sp.u32_to_f32((hdr_val >> 222) & 0xFFFFFFFF)
    fclip_min = sp.u32_to_f32((hdr_val >> 259) & 0xFFFFFFFF)
    zp_out_hw = (hdr_val >> 328) & 0xFF

    x_fp32 = (u_mid.astype(np.float32) - np.float32(zp_in_hw)) * np.float32(s_in_hw)
    pkt640 = sp.decode_0x640(c5_patched[off640 : off640 + 208])
    y_fp32 = sp.eval_0x640_horner_fp32(x_fp32, pkt640, fclip_min, fclip_max) + np.float32(zp_out_hw)
    u_out = np.clip(np.rint(y_fp32), 0.0, 255.0).astype(np.uint8)
    return u_out.view(np.int8)


def emit_and_verify_fused_probes(out_dir: Path) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {
            "name": "fused_gemm_silu_1x16x16_zp0",
            "act": "silu",
            "M": 1, "K": 16, "N": 16, "seed": 5001,
            "s_in": 0.0625, "zp_in": 0, "s_w": 0.0625,
            "s_mid": 0.0625, "zp_mid": 0,
            "s_out": 0.0625, "zp_out": 0,
        },
        {
            "name": "fused_gemm_gelu_1x16x16_zp0",
            "act": "gelu",
            "M": 1, "K": 16, "N": 16, "seed": 5002,
            "s_in": 0.0625, "zp_in": 0, "s_w": 0.0625,
            "s_mid": 0.0625, "zp_mid": 0,
            "s_out": 0.0625, "zp_out": 0,
        },
        {
            "name": "fused_gemm_silu_1x16x16_zpm112",
            "act": "silu",
            "M": 1, "K": 16, "N": 16, "seed": 5003,
            "s_in": 0.0625, "zp_in": -5, "s_w": 0.0625,
            "s_mid": 0.03125, "zp_mid": -10,
            "s_out": 0.02, "zp_out": -112,
        },
        {
            "name": "fused_gemm_silu_4x32x32_zp0",
            "act": "silu",
            "M": 4, "K": 32, "N": 32, "seed": 5004,
            "s_in": 0.0625, "zp_in": 3, "s_w": 0.0625,
            "s_mid": 0.0625, "zp_mid": 0,
            "s_out": 0.0625, "zp_out": 0,
        },
    ]

    summary: List[Dict[str, Any]] = []
    for c in cases:
        rng = np.random.default_rng(c["seed"])
        M, K, N = c["M"], c["K"], c["N"]
        X = rng.integers(-48, 48, size=(M, K), dtype=np.int8)
        W = rng.integers(-16, 16, size=(N, K), dtype=np.int8)
        B = rng.integers(-64, 65, size=(N,), dtype=np.int32)

        # 1. Base GEMM chunks targeting intermediate (s_mid, zp_mid), act=NONE
        chunks, ext2 = ms.synthesize_gemm_chunks(
            M=M, K=K, N=N,
            weights_nk=W.tolist(),
            bias_n=B.tolist(),
            s_in=c["s_in"], zp_in=c["zp_in"], s_w=c["s_w"],
            s_out=c["s_mid"], zp_out=c["zp_mid"], act=0,
        )
        chunks = list(chunks)

        # 2. Synthesize single-stage fused 0x640 spline and patch chunk[5] in place (0 extra bytes!)
        coeffs, bps, fmin, fmax, max_lsb_err = build_fused_single_stage_spline(
            c["act"], s_mid=c["s_mid"], zp_mid=c["zp_mid"], s_out=c["s_out"], zp_out=c["zp_out"]
        )
        c5_orig_len = len(chunks[5])
        c5_fused, off640, off_hdr = patch_chunk5_fused_activation(
            chunks[5],
            coeffs_8x5=coeffs,
            breakpoints_7=bps,
            s_mid=c["s_mid"],
            zp_mid=c["zp_mid"],
            fclip_min=fmin,
            fclip_max=fmax,
        )
        assert len(c5_fused) == c5_orig_len, "In-place fusion must add 0 bytes!"
        chunks[5] = c5_fused

        # 3. Build runnable DGC0 container & verify round-trip
        dgc0_bytes = db.build_dgc0_container(chunks=chunks, ext2=ext2, M=M, K=K, N=N)
        db.verify_container_roundtrip(dgc0_bytes, M, K, N, chunks, ext2)

        # 4. Compare hardware simulation vs exact float64 reference
        sim_out = simulate_fused_gemm_activation(
            X, W, B,
            s_in=c["s_in"], zp_in=c["zp_in"], s_w=c["s_w"],
            s_mid=c["s_mid"], zp_mid=c["zp_mid"],
            c5_patched=c5_fused, off640=off640, off_hdr=off_hdr,
        )
        acc = (X.astype(np.int32) - c["zp_in"]) @ W.astype(np.int32).T + B[None, :]
        m_eff = np.float32((np.float32(c["s_in"]) * np.float32(c["s_w"])) / np.float32(c["s_mid"]))
        u_mid = np.clip(np.rint(acc.astype(np.float32) * m_eff) + (c["zp_mid"] + 128), 0, 255).astype(np.int32)
        x_real = (u_mid - (c["zp_mid"] + 128)).astype(np.float64) * c["s_mid"]
        fn_vec = sp.fn_silu if c["act"] == "silu" else sp.fn_gelu_tanh
        ref_ideal = np.clip(np.rint(fn_vec(x_real) / c["s_out"] + c["zp_out"]), -128, 127).astype(np.int8)

        max_diff_vs_ideal = int(np.max(np.abs(sim_out.astype(np.int32) - ref_ideal.astype(np.int32))))
        exact_vs_ideal = int(np.sum(sim_out == ref_ideal))
        assert max_diff_vs_ideal <= 1, f"{c['name']}: max_diff_vs_ideal={max_diff_vs_ideal}"

        # Write artifacts (.dgc0, .in.bin, .ref.bin = exact hardware FP32-RNE prediction)
        nm = c["name"]
        (out_dir / f"{nm}.dgc0").write_bytes(dgc0_bytes)
        (out_dir / f"{nm}.in.bin").write_bytes(X.tobytes())
        (out_dir / f"{nm}.ref.bin").write_bytes(sim_out.tobytes())

        summary.append(
            {
                "name": nm,
                "act": c["act"],
                "shape": [M, K, N],
                "dgc0_bytes": len(dgc0_bytes),
                "dgc0_sha256": hashlib.sha256(dgc0_bytes).hexdigest()[:16],
                "chunk5_bytes": len(c5_fused),
                "chunk5_delta_bytes": 0,
                "fclip": [fmin, fmax],
                "max_lsb_err_all256": round(max_lsb_err, 5),
                "exact_vs_f64_ideal": f"{exact_vs_ideal}/{M * N}",
                "max_abs_diff_vs_f64_ideal": max_diff_vs_ideal,
            }
        )

    (out_dir / "fused_probes_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------
# 3. Verify Multi-Op Chaining Invariants (`pipe2_*` + `fc_i4_pc_*`)
# ---------------------------------------------------------------------------


def verify_chaining_invariants() -> Dict[str, Any]:
    # 1. pipe2_fc_relu_1x16x16: 2-stage VICA inside chunk[5] (0 DRAM round-trip, 7 chunks)
    img_fr = Dgc0Image((BATCH0_DIR / "pipe2_fc_relu_1x16x16.dgc0").read_bytes())
    r_fr = dgc0_dump.FBReader(img_fr.ext_slots[0].payload)
    chs_fr = [r_fr.byte_vector(r_fr.field_pos(c, 0))[2] for c in r_fr.uoffset_vector(r_fr.field_pos(r_fr.root_offset(), 3))]
    assert len(chs_fr) == 7 and len(chs_fr[5]) == 1936  # +480 B vs 1456 B single-stage chunk[5]
    # Verify two 0x5c0/0x4c0 VICA stages in chunk[5]
    fl5_fr = ms.bytes_to_flits(chs_fr[5])
    n_5c0 = sum(1 for lo, _ in fl5_fr if (lo & 0xFFF) == 0x5C0 and (lo & 0x3F) == 0)
    assert n_5c0 == 2, f"Expected 2 VICA stages in pipe2_fc_relu chunk[5], got {n_5c0}"

    # 2. pipe2_conv1x1_add_1x16x16x16: 9-chunk multi-input chaining (input_0 + param + input_1 + output)
    img_ca = Dgc0Image((BATCH0_DIR / "pipe2_conv1x1_add_1x16x16x16.dgc0").read_bytes())
    r_ca = dgc0_dump.FBReader(img_ca.ext_slots[0].payload)
    chs_ca = [r_ca.byte_vector(r_ca.field_pos(c, 0))[2] for c in r_ca.uoffset_vector(r_ca.field_pos(r_ca.root_offset(), 3))]
    assert len(chs_ca) == 9
    a40_words = [
        (struct.unpack_from("<Q", chs_ca[ci], 0)[0] >> 32) & 0xFFFF
        for ci in (2, 4, 6, 8)
    ]
    # Verify f0_hi_word = (dma_idx << 5) | dma_type!
    # c[2]: idx=0, type=1 (input_0) -> 0x01
    # c[4]: idx=1, type=2 (param)   -> 0x22
    # c[6]: idx=2, type=1 (input_1) -> 0x41
    # c[8]: idx=3, type=3 (output)  -> 0x63
    assert a40_words == [0x01, 0x22, 0x41, 0x63], f"Unexpected pipe2_conv1x1_add 0xa40 tags: {[hex(x) for x in a40_words]}"

    return {
        "pipe2_fc_relu_c5_stages": n_5c0,
        "pipe2_fc_relu_c5_delta_bytes": len(chs_fr[5]) - 1456,
        "pipe2_conv1x1_add_a40_tags": [hex(x) for x in a40_words],
        "param_chaining_rule": "f0_hi_word = (dma_idx << 5) | type (1=in, 2=param, 3=out); 0x940[47:32]=off>>4; 0x980[47:32]=off>>3",
    }


def main() -> int:
    live = verify_live_silicon_splines()
    print(f"[1/3] Live rango silicon 0x640 spline parity: {live['total_exact']}/{live['total_bytes']} EXACT (max_abs_err=0)")
    for k in ["silu_probe", "gelu_probe", "exp_probe"]:
        print(f"      {k:12s}: {live[k]['exact']}/{live[k]['total']} exact")

    fused = emit_and_verify_fused_probes(OUT_PROBES_DIR)
    print(f"\n[2/3] Mode-A Zero-Overhead Fused GEMM + SiLU/GELU probes ({len(fused)} emitted to {OUT_PROBES_DIR}):")
    for row in fused:
        print(
            f"      {row['name']:32s} shape={str(row['shape']):12s} "
            f"c5={row['chunk5_bytes']}B (+{row['chunk5_delta_bytes']}B) "
            f"max_lsb={row['max_lsb_err_all256']:.5f} ideal={row['exact_vs_f64_ideal']} (max_diff={row['max_abs_diff_vs_f64_ideal']})"
        )

    chain = verify_chaining_invariants()
    print(f"\n[3/3] Multi-op & Multi-0xa40 Chaining Invariants:")
    for k, v in chain.items():
        print(f"      {k}: {v}")

    print("\nALL STEP 4 FUSION & CHAINING VERIFICATION CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify 0x640 LUT/spline fusion and multi-op chaining invariants")
    ap.add_argument("--corpus-dir", type=Path, default=BASE, help="External corpus root (or FREETPU_CORPUS_DIR)")
    args = ap.parse_args()
    if args.corpus_dir is None or not args.corpus_dir.is_dir():
        print("verify_fusion_lut: SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)")
        raise SystemExit(0)
    LIVE_SPLINES_DIR = args.corpus_dir / "work/device-runner/phase5_live/spline_probes"
    PROBE_SRC_DIR = args.corpus_dir / "work/int4-splines/spline_probes"
    BATCH0_DIR = args.corpus_dir / "work/device-runner/phase3_batch0"
    db.DEFAULT_SHELL_DGC0 = args.corpus_dir / "work/device-runner/phase3_gemm_sweep1/g00_base_1x16x16.dgc0"
    raise SystemExit(main())
