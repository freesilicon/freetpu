#!/usr/bin/env python3
"""Clean-room Tensor G5 (buenos_a0) INT4 (W4A8) parameter packer, VLIW synthesizer, and probe emitter.

Supports all three ext[2] parameter layouts observed across the 19-model phase5_sweep_int4 corpus:
  1. W4A8 unpacked tile weights (`w_u8 = (w_i4 + 8) & 0xFF` in [0..15], 1 B/weight, identical tile
     layout to INT8 `[n_groups, pad4(K)//4, n_blk, 4]`), emitted by libedgetpu_tflite_compiler for
     per-tensor, per-channel, and blockwise (ignored) W4A8 FULLY_CONNECTED.
  2. W4A8 separate-DMA tail for per-channel FP32 scales (`8*N` B: 32B records `[4x fp32, 16B zero]`),
     separate-DMA int32 biases (`8*N` B: 32B records `[4x int32, 16B zero]`), or interleaved
     scales + biases (`16*N` B: 64B records `[4x fp32, 16B zero, 4x int32, 16B zero]`).
  3. True 4-bit nibble-packed weights (`dequant_i4_fc`, 0.5 B/weight -> 50% weight BW savings):
     `for kg in range(K // 2): for n in range(N): byte = ((W[n,2*kg]+8)&0xF) | (((W[n,2*kg+1]+8)&0xF)<<4)`
     followed by `N` contiguous LE float32 raw weight scales (`4*N` B).

Also wraps Phase 4 `multitile_synth` (read-only import, zero modifications to Phase 4 files) to
synthesize bit-exact 7-chunk W4A8 `chunk[0..6]` (`CSR 0x56` toggle + `(M in {4,8}, K=64, N=32)`
`post3` `0x3c0` rewind stride) and emit diagnostic silicon probes for `@rango-runner-5b`.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
sys.path.insert(0, str(_THIS_DIR))

import dgc0_builder as db  # noqa: E402
import dgc0_dump  # noqa: E402
import multitile_synth as ms  # noqa: E402

# Observed CSR 0x56 values (25-bit register in HDR_DATA / CSR_040)
CSR56_INT8_ZP128 = 0x0100000  # rhs_zero_point=128 (0x80<<13), rhs_data_type=0 (FIXED_POINT8)
CSR56_INT8_ZP8 = 0x0010000    # rhs_zero_point=8   (0x08<<13), rhs_data_type=0 (FIXED_POINT8)
CSR56_INT4_ZP8 = 0x0010070    # rhs_zero_point=8   (0x08<<13), rhs_data_type=7 (FIXED_POINT4)
CSR56_DEQUANT_BF16 = 0x0000026  # rhs_zero_point=0, rhs_data_type=2 (BFLOAT), lhs_data_type=6 (SINGLE)


def pad4(x: int) -> int:
    return (x + 3) & ~3


def ceil8(x: int) -> int:
    return (x + 7) & ~7


# ---------------------------------------------------------------------------
# 1. ext[2] Parameter Packing & Unpacking (all 3 observed layouts)
# ---------------------------------------------------------------------------


def pack_w4a8_unpacked_weights(
    weights_nk: Sequence[Sequence[int]],
    *,
    offset: int = 8,
) -> bytes:
    """Pack [N, K] signed weights into Tensor G5 unpacked tile layout (`[n_groups, k4//4, n_blk, 4]`).

    For W4A8 (`offset=8`), each weight `w in [-8, +7]` becomes `(w + 8) & 0xFF` in `[0..15]`.
    For W8A8 (`offset=128`), each weight `w in [-128, +127]` becomes `(w + 128) & 0xFF`.
    """
    N = len(weights_nk)
    K = len(weights_nk[0])
    k4 = pad4(K)
    n4 = pad4(N)
    n_blk = 32 if N > 32 else max(8, ceil8(N))
    n_groups = (N + 31) // 32
    active_pad = offset & 0xFF

    out = bytearray()
    for g in range(n_groups):
        n0 = 32 * g
        for kg in range(k4 // 4):
            for ni in range(n_blk):
                row = n0 + ni
                for kk in range(4):
                    col = 4 * kg + kk
                    if row < N and col < K:
                        out.append((int(weights_nk[row][col]) + offset) & 0xFF)
                    elif row < n4 and col < k4:
                        out.append(active_pad)
                    else:
                        out.append(0x00)
    return bytes(out)


def unpack_w4a8_unpacked_weights(
    raw: bytes,
    N: int,
    K: int,
    *,
    offset: int = 8,
) -> List[List[int]]:
    """Inverse of `pack_w4a8_unpacked_weights`."""
    k4 = pad4(K)
    n_blk = 32 if N > 32 else max(8, ceil8(N))
    n_groups = (N + 31) // 32
    expected = n_groups * (k4 // 4) * n_blk * 4
    if len(raw) < expected:
        raise ValueError(f"Weight buffer too short: {len(raw)} < {expected}")
    W = [[0] * K for _ in range(N)]
    pos = 0
    for g in range(n_groups):
        n0 = 32 * g
        for kg in range(k4 // 4):
            for ni in range(n_blk):
                row = n0 + ni
                for kk in range(4):
                    col = 4 * kg + kk
                    b = raw[pos]
                    pos += 1
                    if row < N and col < K:
                        W[row][col] = int(b) - offset
    return W


def pack_pc_scale_bias_tail(
    N: int,
    *,
    eff_scales_n: Optional[Sequence[float]] = None,
    bias_n: Optional[Sequence[int]] = None,
) -> bytes:
    """Pack separate-DMA `output.1::parameter` tail (`chunk[6]` 0xa40) in 32B/64B stride records.

    Per group of 4 output channels `4*j .. 4*j+3` (`j = 0 .. pad4(N)//4 - 1`):
      - If `eff_scales_n` present: 16 B (`4x LE float32`, `s_in * s_w[c] / s_out`) + 16 B zero pad.
      - If `bias_n` present:       16 B (`4x LE int32`, raw accumulator bias)      + 16 B zero pad.
      - If both present:           interleaved 64 B per 4 channels (`[scales32, bias32]`).
    """
    if eff_scales_n is None and bias_n is None:
        return b""
    n4 = pad4(N)
    out = bytearray()
    for j in range(n4 // 4):
        if eff_scales_n is not None:
            for q in range(4):
                c = 4 * j + q
                val = float(np.float32(eff_scales_n[c])) if c < N else 0.0
                out.extend(struct.pack("<f", val))
            out.extend(b"\x00" * 16)
        if bias_n is not None:
            for q in range(4):
                c = 4 * j + q
                val = int(bias_n[c]) if c < N else 0
                out.extend(struct.pack("<i", val))
            out.extend(b"\x00" * 16)
    return bytes(out)


def pack_int4_w4a8_ext2(
    weights_nk: Sequence[Sequence[int]],
    *,
    offset: int = 8,
    w_scales_n: Optional[Sequence[float]] = None,
    bias_n: Optional[Sequence[int]] = None,
    s_in: float = 1.0,
    s_out: float = 1.0,
    inline_bias_per_group: bool = False,
) -> Tuple[bytes, int, int]:
    """Pack complete W4A8 `ext[2]` buffer and return `(ext2_bytes, weights_bytes, tail_bytes)`.

    - For 7-chunk per-tensor FC with zero/no bias: `tail_bytes == 0`.
    - For 9-chunk per-channel or separate-DMA bias FC (`inline_bias_per_group=False`):
      `ext2 = unpacked_weights || scale_bias_tail`.
    - For Phase 4 inline-bias layout (`inline_bias_per_group=True`, `w_scales_n=None`):
      delegates to `[bias_group, weights_group]` layout with `+offset`.
    """
    N = len(weights_nk)
    if inline_bias_per_group and w_scales_n is None and bias_n is not None:
        # Phase 4 inline bias format with arbitrary offset (8 or 128)
        K = len(weights_nk[0])
        k4 = pad4(K)
        n4 = pad4(N)
        n_blk = 32 if N > 32 else max(8, ceil8(N))
        n_groups = (N + 31) // 32
        out = bytearray()
        for g in range(n_groups):
            n0 = 32 * g
            for ni in range(n_blk):
                out.extend(struct.pack("<i", int(bias_n[n0 + ni]) if n0 + ni < N else 0))
            for kg in range(k4 // 4):
                for ni in range(n_blk):
                    row = n0 + ni
                    for kk in range(4):
                        col = 4 * kg + kk
                        if row < N and col < K:
                            out.append((int(weights_nk[row][col]) + offset) & 0xFF)
                        elif row < n4 and col < k4:
                            out.append(offset & 0xFF)
                        else:
                            out.append(0x00)
        return bytes(out), len(out), 0

    w_bytes = pack_w4a8_unpacked_weights(weights_nk, offset=offset)
    eff_scales: Optional[List[float]] = None
    if w_scales_n is not None:
        s_in_f = np.float32(s_in)
        s_out_f = np.float32(s_out)
        eff_scales = [float((s_in_f * np.float32(sw)) / s_out_f) for sw in w_scales_n]
    nz_bias = bias_n if (bias_n is not None and any(int(b) != 0 for b in bias_n)) else None
    tail = pack_pc_scale_bias_tail(N, eff_scales_n=eff_scales, bias_n=nz_bias)
    return w_bytes + tail, len(w_bytes), len(tail)


def pack_int4_nibble_ext2(
    weights_nk: Sequence[Sequence[int]],
    raw_scales_n: Optional[Sequence[float]] = None,
    *,
    offset: int = 8,
) -> Tuple[bytes, int, int]:
    """Pack [N, K] INT4 weights into true 4-bit nibble-packed `ext[2]` (verified on `dequant_i4_fc`).

    Exact wire order:
      for kg in 0 .. K//2 - 1:
        for n in 0 .. N - 1:
          lo = (W[n, 2*kg]     + offset) & 0x0F   # even k -> bits [3:0]
          hi = (W[n, 2*kg + 1] + offset) & 0x0F   # odd  k -> bits [7:4]
          byte = lo | (hi << 4)
    Followed by `N` contiguous LE float32 raw weight scales (`4*N` B) when `raw_scales_n` is given.
    """
    N = len(weights_nk)
    K = len(weights_nk[0])
    if K % 2 != 0:
        raise ValueError(f"K must be even for 4-bit nibble packing, got K={K}")
    w_buf = bytearray((K // 2) * N)
    idx = 0
    for kg in range(K // 2):
        k0 = 2 * kg
        k1 = k0 + 1
        for n in range(N):
            w0 = int(weights_nk[n][k0])
            w1 = int(weights_nk[n][k1])
            if not (-8 <= w0 <= 7 and -8 <= w1 <= 7):
                raise ValueError(f"INT4 weight out of [-8, 7] at n={n}, k={k0}/{k1}: {w0}, {w1}")
            w_buf[idx] = ((w0 + offset) & 0x0F) | (((w1 + offset) & 0x0F) << 4)
            idx += 1
    s_buf = bytearray()
    if raw_scales_n is not None:
        if len(raw_scales_n) != N:
            raise ValueError(f"Expected {N} raw scales, got {len(raw_scales_n)}")
        for s in raw_scales_n:
            s_buf.extend(struct.pack("<f", float(np.float32(s))))
    return bytes(w_buf) + bytes(s_buf), len(w_buf), len(s_buf)


def unpack_int4_nibble_ext2(
    raw: bytes,
    N: int,
    K: int,
    *,
    has_scales: bool = True,
    offset: int = 8,
) -> Tuple[List[List[int]], Optional[List[float]]]:
    """Inverse of `pack_int4_nibble_ext2`."""
    w_len = (K // 2) * N
    s_len = 4 * N if has_scales else 0
    if len(raw) != w_len + s_len:
        raise ValueError(f"Expected {w_len + s_len} B, got {len(raw)} B")
    W = [[0] * K for _ in range(N)]
    idx = 0
    for kg in range(K // 2):
        for n in range(N):
            b = raw[idx]
            idx += 1
            W[n][2 * kg] = (b & 0x0F) - offset
            W[n][2 * kg + 1] = ((b >> 4) & 0x0F) - offset
    scales = list(struct.unpack_from(f"<{N}f", raw, w_len)) if has_scales else None
    return W, scales


# ---------------------------------------------------------------------------
# 2. 7-Chunk W4A8 VLIW Synthesizer (wraps Phase 4 multitile_synth read-only)
# ---------------------------------------------------------------------------


def _patch_csr040_reg56(c3: bytes, new_reg56: int) -> bytes:
    """Patch CSR 0x56 inside every CSR_040 packet of `chunk[3]` without touching Phase 4 files."""
    buf = bytearray(c3)
    n = len(buf) // 16
    i = 0
    patched = 0
    while i < n:
        lo, hi = struct.unpack_from("<QQ", buf, i * 16)
        if lo == 0 and hi == 0:
            break
        op = lo & 0xFFF
        if op == 0x940:
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
        elif op in (0x5C0, 0x640):
            ln = 13
        elif op == 0x4C0:
            ln = 10
        elif op == 0x200:
            tw = (hi >> 12) & 0xFFF
            if (hi & 0xF0F) == 0 and (hi >> 48) == 0 and (tw % 2 == 1) and 3 <= tw <= 0x100:
                ln = 1 + ((tw - 1) // 2 + 3) // 4
            else:
                ln = 20
        elif op == 0x040:
            tw = (hi >> 12) & 0xFFF
            n_csrs = (tw - 1) // 2
            ln = 1 + (max(0, n_csrs - 1) + 3) // 4
            # Patch u32 entry for reg 0x56 inside this CSR_040 packet
            # Slot 0 is hi0 >> 32; remaining slots are 4 u32s per flit in flits[1..ln-1]
            new_u32 = ((new_reg56 & 0x1FFFFFF) << 7) | 0x56
            if ((hi >> 32) & 0x7F) == 0x56:
                struct.pack_into("<I", buf, i * 16 + 12, new_u32)
                patched += 1
            for fi in range(i + 1, i + ln):
                for sub in range(4):
                    off = fi * 16 + sub * 4
                    u32 = struct.unpack_from("<I", buf, off)[0]
                    if u32 != 0 and (u32 & 0x7F) == 0x56:
                        struct.pack_into("<I", buf, off, new_u32)
                        patched += 1
        else:
            raise ValueError(f"Unexpected opcode 0x{op:03x} at flit {i}")
        i += ln
    if patched == 0:
        raise ValueError("CSR 0x56 not found in chunk[3] CSR_040 packets")
    return bytes(buf)


def _patch_post3_3c0_k64_n32_multim(c3: bytes, M: int, K: int, N: int) -> bytes:
    """Fix the 2-word `post3` `0x3c0` rewind stride for `(2 <= M < 48, K == 64, N <= 32)`.

    In Phase 4's 81-model corpus, `M in {4, 8}` with `K == 64` only occurred with `N == 64`
    (not `N == 32`). When `K == 64` and `N <= 32` (`n_groups == 1`) with `M in {4, 8}`,
    the second `0x3c0` in `post3` rewinds by `4*(M-1)` / `8*(M-1)` instead of `4` / `8`:
      flit[3].hi = (0x0001FFF4 if M==4 else 0x0001FFE4) << 32 | ((M-1) << 8)
      -> closed form: ((0x20000 - 4*(M-1)) << 32) | ((M - 1) << 8)
      flit[4].lo = 0x0000C000_00000000 | (((0x400000 - 8*(M-1)) & 0x3FFFFF) << 8)
    """
    if not (2 <= M < 48 and K == 64 and N <= 32):
        return c3
    buf = bytearray(c3)
    n = len(buf) // 16
    i = 0
    seen_3c0 = 0
    while i < n:
        lo, hi = struct.unpack_from("<QQ", buf, i * 16)
        if lo == 0 and hi == 0:
            break
        op = lo & 0xFFF
        if op == 0x3C0:
            seen_3c0 += 1
            f3_hi_cur = struct.unpack_from("<Q", buf, (i + 3) * 16 + 8)[0]
            if f3_hi_cur == (0x0001FFFC_00000000 | ((M - 1) << 8)):
                f3_hi = ((0x20000 - 4 * (M - 1)) << 32) | ((M - 1) << 8)
                f4_lo = 0x0000C000_00000000 | ((0x04000000 - (M - 1) * 0x800) & 0x03FFFFFF)
                struct.pack_into("<Q", buf, (i + 3) * 16 + 8, f3_hi)
                struct.pack_into("<Q", buf, (i + 4) * 16, f4_lo)
                return bytes(buf)
            ln = 14
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
        elif op in (0x440, 0x480):
            ln = 8
        elif op == 0x400:
            ln = 7
        elif op == 0x500:
            ln = 9
        elif op in (0x5C0, 0x640):
            ln = 13
        elif op == 0x4C0:
            ln = 10
        elif op == 0x200:
            tw = (hi >> 12) & 0xFFF
            ln = 1 + ((tw - 1) // 2 + 3) // 4 if ((hi & 0xF0F) == 0 and (hi >> 48) == 0 and (tw % 2 == 1) and 3 <= tw <= 0x100) else 20
        elif op == 0x040:
            tw = (hi >> 12) & 0xFFF
            ln = 1 + (max(0, (tw - 1) // 2 - 1) + 3) // 4
        else:
            break
        i += ln
    return bytes(buf)


def synthesize_int4_7chunk(
    *,
    M: int,
    K: int,
    N: int,
    weights_nk: Sequence[Sequence[int]],
    bias_n: Optional[Sequence[int]] = None,
    s_in: float = 0.0625,
    zp_in: int = 0,
    s_w: float = 0.0625,
    s_out: float = 0.25,
    zp_out: int = 0,
    act: int = 0,
    weight_is_int4: bool = True,
    csr56_override: Optional[int] = None,
) -> Tuple[List[bytes], bytes]:
    """Synthesize bit-exact 7-chunk W4A8/W8A8 GEMM `(chunks[0..6], ext2)`."""
    b_vec = list(bias_n) if bias_n is not None else [0] * N
    chunks, ext2_i8 = ms.synthesize_gemm_chunks(
        M=M,
        K=K,
        N=N,
        weights_nk=weights_nk,
        bias_n=b_vec,
        s_in=s_in,
        zp_in=zp_in,
        s_w=s_w,
        s_out=s_out,
        zp_out=zp_out,
        act=act,
    )
    chunks = list(chunks)
    if weight_is_int4:
        has_nz_bias = any(int(b) != 0 for b in b_vec)
        ext2, _, _ = pack_int4_w4a8_ext2(
            weights_nk,
            offset=8,
            bias_n=b_vec if has_nz_bias else None,
            inline_bias_per_group=has_nz_bias,
        )
        reg56 = CSR56_INT4_ZP8 if csr56_override is None else csr56_override
        chunks[3] = _patch_csr040_reg56(chunks[3], reg56)
        chunks[3] = _patch_post3_3c0_k64_n32_multim(chunks[3], M, K, N)
    else:
        ext2 = ext2_i8
        if csr56_override is not None:
            chunks[3] = _patch_csr040_reg56(chunks[3], csr56_override)
    return chunks, ext2


# ---------------------------------------------------------------------------
# 3. Bandwidth Savings Quantification
# ---------------------------------------------------------------------------


def bandwidth_comparison_table() -> List[Dict[str, Any]]:
    """Quantify ext[2] parameter bytes: INT8 vs W4A8 unpacked vs true 4-bit nibble-packed."""
    shapes = [
        ("sweep 1x16x16 pt", 16, 16, False, False),
        ("sweep 1x64x32 pc (dequant)", 64, 32, True, False),
        ("sweep 1x128x64 pc+bias", 128, 64, True, True),
        ("sweep 1x256x64 pt+bias", 256, 64, False, True),
        ("SmolLM2-135M q/o_proj 576x576", 576, 576, True, False),
        ("SmolLM2-135M gate/up 576x1536", 576, 1536, True, False),
        ("SmolLM2-135M down 1536x576", 1536, 576, True, False),
        ("Llama-1B/3B 2048x2048", 2048, 2048, True, False),
        ("Llama-1B/3B FFN 2048x5632", 2048, 5632, True, False),
    ]
    rows: List[Dict[str, Any]] = []
    for label, K, N, pc, bias in shapes:
        k4 = pad4(K)
        n_blk = 32 if N > 32 else max(8, ceil8(N))
        n_groups = (N + 31) // 32
        w_i8 = n_groups * (k4 // 4) * n_blk * 4
        w_i4_unpacked = w_i8  # compiler unpacks W4A8 FC to 1 byte/weight!
        w_i4_nib = (K // 2) * N  # true 4-bit nibble packing (dequant_i4_fc)
        pc_tail_strided = (8 * pad4(N)) if pc else 0
        pc_tail_packed = (4 * N) if pc else 0
        b_tail_strided = (8 * pad4(N)) if bias else 0
        tot_i8 = w_i8 + pc_tail_strided + b_tail_strided
        tot_i4_unpacked = w_i4_unpacked + pc_tail_strided + b_tail_strided
        tot_i4_nib_packed = w_i4_nib + pc_tail_packed + (4 * N if bias else 0)
        tot_i4_nib_strided = w_i4_nib + pc_tail_strided + b_tail_strided
        rows.append(
            {
                "shape": label,
                "K": K,
                "N": N,
                "int8_bytes": tot_i8,
                "w4a8_unpacked_bytes": tot_i4_unpacked,
                "w4a8_unpacked_savings_pct": round(100.0 * (1.0 - tot_i4_unpacked / tot_i8), 2),
                "int4_nibble_packed_bytes": tot_i4_nib_packed,
                "int4_nibble_packed_savings_pct": round(100.0 * (1.0 - tot_i4_nib_packed / tot_i8), 2),
                "int4_nibble_strided_bytes": tot_i4_nib_strided,
                "int4_nibble_strided_savings_pct": round(100.0 * (1.0 - tot_i4_nib_strided / tot_i8), 2),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 4. Diagnostic Silicon Probes for CSR 0x56 Ablation (@rango-runner-5b)
# ---------------------------------------------------------------------------


def emit_csr56_diagnostic_probes(out_dir: Path) -> List[Dict[str, Any]]:
    """Emit two tiny offline 1x16x16 probes isolating `CSR 0x56` (`0x0010000` vs `0x0010070`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sw = BASE / "work/int4-splines/sweep_int4"
    in_bytes = (sw / "fc_i4_pt_1x16x16_id.in.bin").read_bytes()
    ref_bytes = (sw / "fc_i4_pt_1x16x16_id.ref.bin").read_bytes()
    W_id = [[4 if o == i else 0 for i in range(16)] for o in range(16)]

    emitted: List[Dict[str, Any]] = []

    # Probe 1: ext[2] = unpacked w+8 (256 B), CSR 0x56 = 0x0010000 (rhs_data_type=0 [INT8], zp_w=8)
    # Prediction: max_abs_err <= 1 (exact match to fc_i8_pt_1x16x16_id_ref!)
    c1, e1 = synthesize_int4_7chunk(
        M=1, K=16, N=16, weights_nk=W_id, bias_n=[0] * 16,
        s_in=0.0625, zp_in=0, s_w=0.0625, s_out=0.25, zp_out=0, act=0,
        weight_is_int4=True, csr56_override=CSR56_INT8_ZP8,
    )
    dgc1 = db.build_dgc0_container(chunks=c1, ext2=e1, M=1, K=16, N=16)
    db.verify_container_roundtrip(dgc1, 1, 16, 16, c1, e1)
    p1 = out_dir / "probe_i4_1x16x16_csr56_0010000.dgc0"
    p1.write_bytes(dgc1)
    (out_dir / "probe_i4_1x16x16_csr56_0010000.in.bin").write_bytes(in_bytes)
    (out_dir / "probe_i4_1x16x16_csr56_0010000.ref.bin").write_bytes(ref_bytes)
    emitted.append(
        {
            "name": "probe_i4_1x16x16_csr56_0010000",
            "dgc0": str(p1),
            "bytes": len(dgc1),
            "ext2_bytes": len(e1),
            "csr56": hex(CSR56_INT8_ZP8),
            "hypothesis": "Clears 0x56[6:4] from 7 (FIXED_POINT4) to 0 (FIXED_POINT8) while keeping ext[2]=w+8 & zp_w=8; expected exact match to INT8 ref (max_abs_err<=1).",
        }
    )

    # Probe 2: ext[2] = 4-bit nibble-packed (128 B zero-padded to 256 B), CSR 0x56 = 0x0010070
    # Prediction: tests whether hardware 0x56[6:4]=7 expects nibble-packed data in WM SRAM.
    nib_raw, _, _ = pack_int4_nibble_ext2(W_id, raw_scales_n=None, offset=8)
    e2 = nib_raw + b"\x88" * (256 - len(nib_raw))
    c2, _ = synthesize_int4_7chunk(
        M=1, K=16, N=16, weights_nk=W_id, bias_n=[0] * 16,
        s_in=0.0625, zp_in=0, s_w=0.0625, s_out=0.25, zp_out=0, act=0,
        weight_is_int4=True, csr56_override=CSR56_INT4_ZP8,
    )
    dgc2 = db.build_dgc0_container(chunks=c2, ext2=e2, M=1, K=16, N=16)
    db.verify_container_roundtrip(dgc2, 1, 16, 16, c2, e2)
    p2 = out_dir / "probe_i4_1x16x16_csr56_0010070_nibpack.dgc0"
    p2.write_bytes(dgc2)
    (out_dir / "probe_i4_1x16x16_csr56_0010070_nibpack.in.bin").write_bytes(in_bytes)
    (out_dir / "probe_i4_1x16x16_csr56_0010070_nibpack.ref.bin").write_bytes(ref_bytes)
    emitted.append(
        {
            "name": "probe_i4_1x16x16_csr56_0010070_nibpack",
            "dgc0": str(p2),
            "bytes": len(dgc2),
            "ext2_bytes": len(e2),
            "csr56": hex(CSR56_INT4_ZP8),
            "hypothesis": "Keeps 0x56=0x0010070 (FIXED_POINT4) and feeds true 4-bit nibble-packed ext[2] (128B + 0x88 pad) to probe hardware 4-bit MAC decode.",
        }
    )

    (out_dir / "csr56_probes_manifest.json").write_text(json.dumps(emitted, indent=2) + "\n")
    return emitted


def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor G5 INT4 (W4A8) parameter packer & VLIW synthesizer")
    ap.add_argument("--shell", type=Path, default=None, help="Optional template DGC0 shell for emitting CSR 0x56 probes")
    ap.add_argument("--out-dir", type=Path, default=Path("out_csr56_probes"), help="Output directory when --shell is provided")
    args = ap.parse_args()

    # Self-test round trips
    rng = np.random.default_rng(42)
    W = rng.integers(-8, 8, size=(32, 64), dtype=np.int32).tolist()
    scales = [float(np.float32(0.0625 * (1.0 + 0.125 * (i % 8)))) for i in range(32)]
    raw_u = pack_w4a8_unpacked_weights(W, offset=8)
    assert unpack_w4a8_unpacked_weights(raw_u, 32, 64, offset=8) == W
    raw_n, wn_len, sn_len = pack_int4_nibble_ext2(W, raw_scales_n=scales, offset=8)
    assert wn_len == 1024 and sn_len == 128
    W_back, s_back = unpack_int4_nibble_ext2(raw_n, 32, 64, has_scales=True, offset=8)
    assert W_back == W and s_back == scales

    bw = bandwidth_comparison_table()
    print("pack_int4.py self-test PASS (unpacked + nibble round-trip exact)")
    shell = args.shell or (Path(os.environ["FREETPU_SHELL_DGC0"]) if os.environ.get("FREETPU_SHELL_DGC0") else None)
    if shell and shell.is_file():
        db.DEFAULT_SHELL_DGC0 = shell
        probes = emit_csr56_diagnostic_probes(args.out_dir)
        print(f"Emitted {len(probes)} CSR 0x56 diagnostic probes to {args.out_dir}")
    print("\nBandwidth Comparison Table:")
    for row in bw:
        print(
            f"  {row['shape']:30s} | INT8={row['int8_bytes']:8d}B | "
            f"W4A8_unp={row['w4a8_unpacked_bytes']:8d}B ({row['w4a8_unpacked_savings_pct']:5.2f}%) | "
            f"INT4_nib={row['int4_nibble_packed_bytes']:8d}B ({row['int4_nibble_packed_savings_pct']:5.2f}%)"
        )


if __name__ == "__main__":
    main()
