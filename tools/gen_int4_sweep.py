#!/usr/bin/env python3
"""Pure-Python TFLite v3 INT4 (W4A8 / W4A16 / Blockwise / Dequant) sweep generator for Tensor G5.

Deliverable Part B for tpu-int4-splines (100% offline):
Generates 19 validated `.tflite` models + deterministic `.in.bin` + exact CPU `.ref.bin`
and `manifest.json` (compatible with `work/device-runner/run_oracle_batch.py`) under:
    work/int4-splines/sweep_int4/

Coverage:
  - Per-tensor INT4 W4A8 FULLY_CONNECTED (K=16..256, N=16..64, M=1..8, zero/non-zero bias,
    no-bias, zp_in!=0 + RELU, plus an exact INT8 twin of 1x16x16 for 1-to-1 ISA bit-diffing)
  - Per-channel INT4 W4A8 FULLY_CONNECTED (N=16..64, K=32..128, M=1..4)
  - Blockwise sub-channel INT4 W4A8 FULLY_CONNECTED (TFL3 BlockwiseQuantization details_type=2,
    block_size=32, 64, 128; FLOAT16 [N, K//block_size] scale tensors)
  - Acceptance-boundary probes: W4A16 (INT16 act x per-channel INT4 weight) and
    DEQUANTIZE(INT4) -> FULLY_CONNECTED (FP32 dynamic/weight-only path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
import sweep_gen as sg  # noqa: E402
from sweep_gen import (  # noqa: E402
    ACT_NONE,
    BUILTIN_OP_FULLY_CONNECTED,
    BUILTIN_OPT_FULLY_CONNECTED,
    BUILTIN_OPT_NONE,
    FlatBufferBuilder,
    clamp_int8,
    clamp_int16,
    multiply_by_quantized_multiplier,
    prod,
    quantize_multiplier,
)

OUT_DIR = Path("out_sweep_int4")

# TFLite v3 schema constants
TENSOR_TYPE_FLOAT32 = 0
TENSOR_TYPE_FLOAT16 = 1
TENSOR_TYPE_INT32 = 2
TENSOR_TYPE_UINT8 = 3
TENSOR_TYPE_INT64 = 4
TENSOR_TYPE_INT16 = 7
TENSOR_TYPE_INT8 = 9
TENSOR_TYPE_INT4 = 17  # 0x11, signed two's complement [-8, +7], little-endian nibble packed

BUILTIN_OP_DEQUANTIZE = 6
ACT_RELU = 1

QUANT_DETAILS_NONE = 0
QUANT_DETAILS_CUSTOM = 1
QUANT_DETAILS_BLOCKWISE = 2

TENSOR_TYPE_BITS: Dict[int, int] = {
    TENSOR_TYPE_FLOAT32: 32,
    TENSOR_TYPE_FLOAT16: 16,
    TENSOR_TYPE_INT32: 32,
    TENSOR_TYPE_UINT8: 8,
    TENSOR_TYPE_INT64: 64,
    TENSOR_TYPE_INT16: 16,
    TENSOR_TYPE_INT8: 8,
    TENSOR_TYPE_INT4: 4,
}


# ---------------------------------------------------------------------------
# INT4 little-endian nibble packing (verified against TFLite & DarwiNN CopyElement)
# ---------------------------------------------------------------------------


def pack_int4_le(values: Sequence[int]) -> bytes:
    """Pack signed int4 values in [-8, +7] two's complement, low nibble = even index.

    Matches both TFLite TensorType_INT4 specification and DarwiNN CopyElement @ 0x2bd2400.
    """
    n = len(values)
    out = bytearray((n + 1) // 2)
    for idx, v in enumerate(values):
        if not (-8 <= v <= 7):
            raise ValueError(f"int4 value {v} out of range [-8, 7] at index {idx}")
        nib = v & 0x0F
        byte_i = idx >> 1
        if (idx & 1) == 0:
            out[byte_i] |= nib
        else:
            out[byte_i] |= nib << 4
    return bytes(out)


def unpack_int4_le(raw: bytes, numel: int) -> List[int]:
    """Unpack little-endian signed two's complement int4 buffer."""
    expected = (numel + 1) // 2
    if len(raw) != expected:
        raise ValueError(f"INT4 buffer length {len(raw)} != expected {expected} for numel={numel}")
    vals: List[int] = []
    for idx in range(numel):
        b = raw[idx >> 1]
        nib = (b & 0x0F) if (idx & 1) == 0 else ((b >> 4) & 0x0F)
        if nib >= 8:
            nib -= 16
        vals.append(nib)
    return vals


# ---------------------------------------------------------------------------
# Extended TFLite v3 FlatBuffer Builder (INT4 + BlockwiseQuantization + DEQUANTIZE)
# ---------------------------------------------------------------------------


def build_blockwise_quant_details(
    fb: FlatBufferBuilder,
    scales_tensor_idx: int,
    zero_points_tensor_idx: int,
    block_size: int,
) -> int:
    """TFLite schema.fbs table BlockwiseQuantization { scales:int; zero_points:int; block_size:int; }."""
    fb.start_table(3)
    fb.add_i32(0, scales_tensor_idx, 0)
    fb.add_i32(1, zero_points_tensor_idx, 0)
    fb.add_i32(2, block_size, 0)
    return fb.end_table()


def build_quant_params_ext(
    fb: FlatBufferBuilder,
    scales: Optional[Sequence[float]] = None,
    zero_points: Optional[Sequence[int]] = None,
    quantized_dimension: int = 0,
    blockwise: Optional[Dict[str, int]] = None,
) -> int:
    """Build TFLite QuantizationParameters table supporting per-tensor, per-channel, and BlockwiseQuantization."""
    scale_off = fb.create_f32_vector(scales) if scales is not None else 0
    zp_off = fb.create_i64_vector(zero_points) if zero_points is not None else 0
    details_type = QUANT_DETAILS_NONE
    details_off = 0
    if blockwise is not None:
        details_type = QUANT_DETAILS_BLOCKWISE
        details_off = build_blockwise_quant_details(
            fb,
            scales_tensor_idx=blockwise["scales"],
            zero_points_tensor_idx=blockwise.get("zero_points", -1),
            block_size=blockwise["block_size"],
        )

    fb.start_table(7)
    if scale_off:
        fb.add_uoffset(2, scale_off)
    if zp_off:
        fb.add_uoffset(3, zp_off)
    if details_type != QUANT_DETAILS_NONE:
        fb.add_u8(4, details_type, 0)
        fb.add_uoffset(5, details_off)
    fb.add_i32(6, quantized_dimension, 0)
    return fb.end_table()


def build_tensor_ext(fb: FlatBufferBuilder, spec: Dict[str, Any]) -> int:
    shape_off = fb.create_i32_vector(spec["shape"])
    name_off = fb.create_string(spec["name"])
    quant_off = 0
    if (
        spec.get("scales") is not None
        or spec.get("zero_points") is not None
        or spec.get("blockwise") is not None
    ):
        quant_off = build_quant_params_ext(
            fb,
            scales=spec.get("scales"),
            zero_points=spec.get("zero_points"),
            quantized_dimension=spec.get("quantized_dimension", 0),
            blockwise=spec.get("blockwise"),
        )
    fb.start_table(8)
    fb.add_uoffset(0, shape_off)
    fb.add_u8(1, spec["type"], 0)
    fb.add_u32(2, spec["buffer"], 0)
    fb.add_uoffset(3, name_off)
    if quant_off:
        fb.add_uoffset(4, quant_off)
    return fb.end_table()


def build_model_ext(
    opcodes: List[Tuple[int, int]],
    tensors_spec: List[Dict[str, Any]],
    inputs: List[int],
    outputs: List[int],
    operators_spec: List[Dict[str, Any]],
    buffers_data: List[Optional[bytes]],
    description: str,
    subgraph_name: str = "main",
) -> bytes:
    fb = FlatBufferBuilder()
    buf_offs = [sg.build_buffer(fb, b) for b in buffers_data]
    buffers_vec = fb.create_uoffset_vector(buf_offs)

    op_offs = []
    for op in operators_spec:
        opt_off = sg.build_options_table(fb, op)
        in_vec = fb.create_i32_vector(op["inputs"])
        out_vec = fb.create_i32_vector(op["outputs"])
        fb.start_table(6)
        fb.add_u32(0, op["opcode_index"], 0)
        fb.add_uoffset(1, in_vec)
        fb.add_uoffset(2, out_vec)
        fb.add_u8(3, op["builtin_options_type"], 0)
        if opt_off:
            fb.add_uoffset(4, opt_off)
        op_offs.append(fb.end_table())

    t_offs = [build_tensor_ext(fb, t) for t in tensors_spec]
    tensors_vec = fb.create_uoffset_vector(t_offs)
    in_vec = fb.create_i32_vector(inputs)
    out_vec = fb.create_i32_vector(outputs)
    ops_vec = fb.create_uoffset_vector(op_offs)
    sg_name = fb.create_string(subgraph_name)

    fb.start_table(5)
    fb.add_uoffset(0, tensors_vec)
    fb.add_uoffset(1, in_vec)
    fb.add_uoffset(2, out_vec)
    fb.add_uoffset(3, ops_vec)
    fb.add_uoffset(4, sg_name)
    sg_off = fb.end_table()
    sgs_vec = fb.create_uoffset_vector([sg_off])

    oc_offs = [sg.build_opcode(fb, code, ver) for code, ver in opcodes]
    ocs_vec = fb.create_uoffset_vector(oc_offs)
    desc_off = fb.create_string(description)

    fb.start_table(6)
    fb.add_u32(0, 3, 0)  # TFLite schema version 3
    fb.add_uoffset(1, ocs_vec)
    fb.add_uoffset(2, sgs_vec)
    fb.add_uoffset(3, desc_off)
    fb.add_uoffset(4, buffers_vec)
    root = fb.end_table()
    return fb.finish(root, b"TFL3")


# ---------------------------------------------------------------------------
# Strict FlatBuffer Validator (verifies INT4 byte packing & BlockwiseQuantization)
# ---------------------------------------------------------------------------


class _FBReader:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf

    def u8(self, pos: int) -> int:
        return self.buf[pos]

    def u16(self, pos: int) -> int:
        return struct.unpack_from("<H", self.buf, pos)[0]

    def i32(self, pos: int) -> int:
        return struct.unpack_from("<i", self.buf, pos)[0]

    def u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.buf, pos)[0]

    def field_pos(self, table_pos: int, field_id: int) -> Optional[int]:
        vtable_off = table_pos - self.i32(table_pos)
        vtable_len = self.u16(vtable_off)
        slot = 4 + field_id * 2
        if slot >= vtable_len:
            return None
        off = self.u16(vtable_off + slot)
        return None if off == 0 else table_pos + off

    def uoffset(self, pos: int) -> int:
        return pos + self.u32(pos)

    def uoffset_vec(self, field_pos: int) -> List[int]:
        v = self.uoffset(field_pos)
        n = self.u32(v)
        return [self.uoffset(v + 4 + 4 * i) for i in range(n)]

    def i32_vec(self, field_pos: int) -> List[int]:
        v = self.uoffset(field_pos)
        n = self.u32(v)
        return list(struct.unpack_from(f"<{n}i", self.buf, v + 4))

    def byte_vec(self, field_pos: int) -> bytes:
        v = self.uoffset(field_pos)
        n = self.u32(v)
        return self.buf[v + 4 : v + 4 + n]


def validate_int4_tflite(raw: bytes) -> Dict[str, Any]:
    """Strictly verify TFL3 magic, buffer sizes (ceil(numel/2) for INT4), ops, and BlockwiseQuantization."""
    if len(raw) < 16 or raw[4:8] != b"TFL3":
        raise ValueError("Not a TFL3 FlatBuffer")
    r = _FBReader(raw)
    root = r.u32(0)
    ver_f = r.field_pos(root, 0)
    assert ver_f is not None and r.u32(ver_f) == 3

    oc_tables = r.uoffset_vec(r.field_pos(root, 1))  # type: ignore[arg-type]
    sg_tables = r.uoffset_vec(r.field_pos(root, 2))  # type: ignore[arg-type]
    buf_tables = r.uoffset_vec(r.field_pos(root, 4))  # type: ignore[arg-type]
    assert len(sg_tables) == 1

    buf_lens: List[int] = []
    for bt in buf_tables:
        df = r.field_pos(bt, 0)
        buf_lens.append(len(r.byte_vec(df)) if df is not None else 0)

    opcodes: List[Tuple[int, int]] = []
    for ot in oc_tables:
        cf = r.field_pos(ot, 3)
        vf = r.field_pos(ot, 2)
        code = r.i32(cf) if cf is not None else 0
        ver = r.i32(vf) if vf is not None else 1
        opcodes.append((code, ver))

    sg0 = sg_tables[0]
    t_tables = r.uoffset_vec(r.field_pos(sg0, 0))  # type: ignore[arg-type]
    op_tables = r.uoffset_vec(r.field_pos(sg0, 3))  # type: ignore[arg-type]

    int4_tensors = 0
    blockwise_tensors = 0
    for idx, tt in enumerate(t_tables):
        shape = r.i32_vec(r.field_pos(tt, 0))  # type: ignore[arg-type]
        tf = r.field_pos(tt, 1)
        ttype = r.u8(tf) if tf is not None else 0
        bf = r.field_pos(tt, 2)
        bidx = r.u32(bf) if bf is not None else 0
        assert 0 <= bidx < len(buf_lens), f"tensor[{idx}] buffer {bidx} out of range"
        if ttype == TENSOR_TYPE_INT4:
            int4_tensors += 1
        if buf_lens[bidx] > 0:
            bits = TENSOR_TYPE_BITS[ttype]
            expected_bytes = (prod(shape) * bits + 7) // 8
            assert buf_lens[bidx] == expected_bytes, (
                f"tensor[{idx}] type={ttype} shape={shape} buf_len={buf_lens[bidx]} != expected {expected_bytes}"
            )
        qf = r.field_pos(tt, 4)
        if qf is not None:
            qt = r.uoffset(qf)
            dtf = r.field_pos(qt, 4)
            if dtf is not None and r.u8(dtf) == QUANT_DETAILS_BLOCKWISE:
                blockwise_tensors += 1
                df = r.field_pos(qt, 5)
                assert df is not None, "BlockwiseQuantization details_type=2 missing details table"
                bqt = r.uoffset(df)
                s_idx = r.i32(r.field_pos(bqt, 0))  # type: ignore[arg-type]
                bs = r.i32(r.field_pos(bqt, 2))  # type: ignore[arg-type]
                assert 0 <= s_idx < len(t_tables), f"Blockwise scales tensor idx {s_idx} invalid"
                assert bs in (32, 64, 128), f"Unexpected block_size={bs}"

    for opt in op_tables:
        oc_idx = r.u32(r.field_pos(opt, 0)) if r.field_pos(opt, 0) is not None else 0
        bcode, _ = opcodes[oc_idx]
        btf = r.field_pos(opt, 3)
        bopt = r.u8(btf) if btf is not None else 0
        if bcode == BUILTIN_OP_FULLY_CONNECTED:
            assert bopt == BUILTIN_OPT_FULLY_CONNECTED
        elif bcode == BUILTIN_OP_DEQUANTIZE:
            assert bopt == BUILTIN_OPT_NONE
        else:
            raise AssertionError(f"Unexpected builtin code {bcode}")

    return {
        "size_bytes": len(raw),
        "num_tensors": len(t_tables),
        "num_ops": len(op_tables),
        "int4_tensors": int4_tensors,
        "blockwise_tensors": blockwise_tensors,
    }


# ---------------------------------------------------------------------------
# Deterministic data & exact CPU reference evaluators
# ---------------------------------------------------------------------------


def det_int8(n: int, seed: int, lo: int = -48, hi: int = 47) -> List[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(lo, hi + 1, size=n, dtype=np.int32)]


def det_int16(n: int, seed: int, lo: int = -1024, hi: int = 1024) -> List[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(lo, hi + 1, size=n, dtype=np.int32)]


def det_int4_weights(N: int, K: int, seed: int, identity_val: Optional[int] = None) -> List[List[int]]:
    if identity_val is not None:
        return [[identity_val if o == i else 0 for i in range(K)] for o in range(N)]
    rng = np.random.default_rng(seed)
    # Full symmetric/asymmetric INT4 range [-8, +7] (includes -8 to probe +8 unsigned offset!)
    arr = rng.integers(-8, 8, size=(N, K), dtype=np.int32)
    return [[int(v) for v in row] for row in arr]


def eval_fc_quant_ref(
    x_2d: List[List[int]],
    w_nk: List[List[int]],
    bias_n: Optional[List[int]],
    *,
    s_in: float,
    zp_in: int,
    w_scales_n: List[float],
    s_out: float,
    zp_out: int,
    fused_act: int = ACT_NONE,
    is_int16: bool = False,
) -> List[int]:
    """Exact TFLite fixed-point reference for W4A8 / W8A8 / W4A16 FC (per-tensor & per-channel)."""
    M = len(x_2d)
    K = len(x_2d[0])
    N = len(w_nk)
    out: List[int] = []
    min_q = -32768 if is_int16 else -128
    max_q = 32767 if is_int16 else 127
    if fused_act == ACT_RELU:
        min_q = max(min_q, zp_out)

    mults = [quantize_multiplier((s_in * ws) / s_out) for ws in w_scales_n]
    for m in range(M):
        row = x_2d[m]
        for o in range(N):
            acc = bias_n[o] if bias_n is not None else 0
            wrow = w_nk[o]
            for k in range(K):
                acc += (row[k] - zp_in) * wrow[k]
            qm, sh = mults[o]
            q = multiply_by_quantized_multiplier(acc, qm, sh) + zp_out
            if q < min_q:
                q = min_q
            elif q > max_q:
                q = max_q
            out.append(q)
    return out


def eval_fc_blockwise_ref(
    x_2d: List[List[int]],
    w_nk: List[List[int]],
    scales_nb_f16: np.ndarray,
    bias_n: Optional[List[int]],
    *,
    s_in: float,
    zp_in: int,
    s_out: float,
    zp_out: int,
    block_size: int,
) -> List[int]:
    """Exact float32-accumulation reference for TFLite BlockwiseQuantization W4A8 FC."""
    M = len(x_2d)
    K = len(x_2d[0])
    N = len(w_nk)
    scales_f32 = scales_nb_f16.astype(np.float32)
    x_f32 = (np.array(x_2d, dtype=np.float32) - np.float32(zp_in)) * np.float32(s_in)
    w_arr = np.array(w_nk, dtype=np.float32)  # (N, K)
    # Dequantize weights per block
    w_deq = np.empty_like(w_arr, dtype=np.float32)
    num_blocks = K // block_size
    for b in range(num_blocks):
        k0, k1 = b * block_size, (b + 1) * block_size
        w_deq[:, k0:k1] = w_arr[:, k0:k1] * scales_f32[:, b : b + 1]
    y_f32 = x_f32 @ w_deq.T  # (M, N)
    if bias_n is not None:
        # Bias scale is s_in * w_scale (for blockwise tests we use zero bias or fp32 bias)
        y_f32 += np.array(bias_n, dtype=np.float32)[None, :]
    q_out = np.clip(np.rint(y_f32 / np.float32(s_out)) + zp_out, -128, 127).astype(np.int32)
    return [int(v) for v in q_out.reshape(-1)]


# ---------------------------------------------------------------------------
# Model builders for each sweep family
# ---------------------------------------------------------------------------


def make_fc_int4_or_int8(
    name: str,
    *,
    M: int,
    K: int,
    N: int,
    seed: int,
    weight_type: int = TENSOR_TYPE_INT4,
    per_channel: bool = False,
    identity_val: Optional[int] = None,
    use_bias: bool = True,
    zero_bias: bool = True,
    s_in: float = 0.0625,
    zp_in: int = 0,
    s_out: float = 0.25,
    zp_out: int = 0,
    fused_act: int = ACT_NONE,
    is_int16_act: bool = False,
) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    w_nk = det_int4_weights(N, K, seed=seed, identity_val=identity_val)
    w_flat = [v for row in w_nk for v in row]

    if per_channel:
        # Distinct power-of-2 & non-power-of-2 scales per output channel
        w_scales = [float(np.float32(0.0625 * (1.0 + 0.125 * (o % 8)))) for o in range(N)]
    else:
        w_scales = [0.0625]

    w_zps = [0] * len(w_scales)
    w_scales_full = w_scales if per_channel else [w_scales[0]] * N

    if is_int16_act:
        x_flat = det_int16(M * K, seed=seed + 101)
        x_bytes = struct.pack(f"<{M * K}h", *x_flat)
        act_ttype = TENSOR_TYPE_INT16
        zp_in = 0
        zp_out = 0
        s_in = 1.0 / 256.0
        s_out = 1.0 / 64.0
    else:
        x_flat = det_int8(M * K, seed=seed + 101)
        x_bytes = struct.pack(f"<{M * K}b", *x_flat)
        act_ttype = TENSOR_TYPE_INT8

    x_2d = [x_flat[m * K : (m + 1) * K] for m in range(M)]

    if weight_type == TENSOR_TYPE_INT4:
        w_bytes = pack_int4_le(w_flat)
        assert unpack_int4_le(w_bytes, N * K) == w_flat
        fc_ver = 10
    else:
        w_bytes = struct.pack(f"<{N * K}b", *w_flat)
        fc_ver = 9

    bias_vals: Optional[List[int]] = None
    buffers: List[Optional[bytes]] = [None, w_bytes]
    tensors: List[Dict[str, Any]] = [
        {
            "name": "input",
            "shape": [M, K],
            "type": act_ttype,
            "buffer": 0,
            "scales": [s_in],
            "zero_points": [zp_in],
        },
        {
            "name": "filter",
            "shape": [N, K],
            "type": weight_type,
            "buffer": 1,
            "scales": w_scales,
            "zero_points": w_zps,
            "quantized_dimension": 0,
        },
    ]

    op_inputs = [0, 1]
    if use_bias:
        if zero_bias:
            bias_vals = [0] * N
        else:
            rng = np.random.default_rng(seed + 202)
            bias_vals = [int(v) for v in rng.integers(-64, 65, size=N, dtype=np.int32)]
        bias_bytes = struct.pack(f"<{N}i", *bias_vals)
        buffers.append(bias_bytes)
        b_scales = [float(np.float32(s_in * ws)) for ws in w_scales]
        b_zps = [0] * len(b_scales)
        tensors.append(
            {
                "name": "bias",
                "shape": [N],
                "type": TENSOR_TYPE_INT32,
                "buffer": len(buffers) - 1,
                "scales": b_scales,
                "zero_points": b_zps,
                "quantized_dimension": 0,
            }
        )
        op_inputs.append(2)

    out_idx = len(tensors)
    tensors.append(
        {
            "name": "output",
            "shape": [M, N],
            "type": act_ttype,
            "buffer": 0,
            "scales": [s_out],
            "zero_points": [zp_out],
        }
    )

    ref_vals = eval_fc_quant_ref(
        x_2d,
        w_nk,
        bias_vals,
        s_in=s_in,
        zp_in=zp_in,
        w_scales_n=w_scales_full,
        s_out=s_out,
        zp_out=zp_out,
        fused_act=fused_act,
        is_int16=is_int16_act,
    )
    ref_bytes = (
        struct.pack(f"<{M * N}h", *ref_vals)
        if is_int16_act
        else struct.pack(f"<{M * N}b", *ref_vals)
    )

    model_bytes = build_model_ext(
        opcodes=[(BUILTIN_OP_FULLY_CONNECTED, fc_ver)],
        tensors_spec=tensors,
        inputs=[0],
        outputs=[out_idx],
        operators_spec=[
            {
                "opcode_index": 0,
                "inputs": op_inputs,
                "outputs": [out_idx],
                "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED,
                "fused_activation": fused_act,
                "weights_format": 0,
                "keep_num_dims": False,
                "asymmetric_quantize_inputs": False,
            }
        ],
        buffers_data=buffers,
        description=f"tpu-int4-splines {name}",
    )
    meta = {
        "suite": "sweep_int4",
        "op": "FULLY_CONNECTED",
        "weight_dtype": "int4" if weight_type == TENSOR_TYPE_INT4 else "int8",
        "act_dtype": "int16" if is_int16_act else "int8",
        "quant_mode": "per_channel" if per_channel else "per_tensor",
        "M": M,
        "K": K,
        "N": N,
        "use_bias": use_bias,
        "zero_bias": zero_bias if use_bias else None,
        "s_in": s_in,
        "zp_in": zp_in,
        "s_out": s_out,
        "zp_out": zp_out,
        "fused_act": "RELU" if fused_act == ACT_RELU else "NONE",
        "in_bytes": len(x_bytes),
        "out_bytes": len(ref_bytes),
    }
    return model_bytes, x_bytes, ref_bytes, meta


def make_fc_int4_blockwise(
    name: str,
    *,
    M: int,
    K: int,
    N: int,
    block_size: int,
    seed: int,
) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    """TFLite v3 BlockwiseQuantization (details_type=2) INT4 W4A8 FULLY_CONNECTED."""
    assert K % block_size == 0
    num_blocks = K // block_size
    s_in, zp_in = 0.0625, 0
    s_out, zp_out = 0.25, 0

    w_nk = det_int4_weights(N, K, seed=seed)
    w_flat = [v for row in w_nk for v in row]
    w_bytes = pack_int4_le(w_flat)

    rng = np.random.default_rng(seed + 303)
    # Positive FP16 per-block scales in [0.03125, 0.125]
    scales_f16 = (0.03125 + 0.015625 * rng.integers(0, 7, size=(N, num_blocks))).astype(np.float16)
    scales_bytes = scales_f16.tobytes()

    x_flat = det_int8(M * K, seed=seed + 101)
    x_bytes = struct.pack(f"<{M * K}b", *x_flat)
    x_2d = [x_flat[m * K : (m + 1) * K] for m in range(M)]

    ref_vals = eval_fc_blockwise_ref(
        x_2d,
        w_nk,
        scales_f16,
        bias_n=None,
        s_in=s_in,
        zp_in=zp_in,
        s_out=s_out,
        zp_out=zp_out,
        block_size=block_size,
    )
    ref_bytes = struct.pack(f"<{M * N}b", *ref_vals)

    # Tensor 0: input [M, K] int8
    # Tensor 1: scales [N, K//block_size] float16 (referenced by BlockwiseQuantization)
    # Tensor 2: filter [N, K] int4 with details_type=2 BlockwiseQuantization{scales=1, zp=-1, block_size}
    # Tensor 3: output [M, N] int8
    buffers: List[Optional[bytes]] = [None, scales_bytes, w_bytes]
    tensors: List[Dict[str, Any]] = [
        {
            "name": "input",
            "shape": [M, K],
            "type": TENSOR_TYPE_INT8,
            "buffer": 0,
            "scales": [s_in],
            "zero_points": [zp_in],
        },
        {
            "name": "filter_block_scales",
            "shape": [N, num_blocks],
            "type": TENSOR_TYPE_FLOAT16,
            "buffer": 1,
        },
        {
            "name": "filter",
            "shape": [N, K],
            "type": TENSOR_TYPE_INT4,
            "buffer": 2,
            "quantized_dimension": 0,
            "blockwise": {"scales": 1, "zero_points": -1, "block_size": block_size},
        },
        {
            "name": "output",
            "shape": [M, N],
            "type": TENSOR_TYPE_INT8,
            "buffer": 0,
            "scales": [s_out],
            "zero_points": [zp_out],
        },
    ]
    model_bytes = build_model_ext(
        opcodes=[(BUILTIN_OP_FULLY_CONNECTED, 12)],
        tensors_spec=tensors,
        inputs=[0],
        outputs=[3],
        operators_spec=[
            {
                "opcode_index": 0,
                "inputs": [0, 2],
                "outputs": [3],
                "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED,
                "fused_activation": ACT_NONE,
                "weights_format": 0,
                "keep_num_dims": False,
                "asymmetric_quantize_inputs": False,
            }
        ],
        buffers_data=buffers,
        description=f"tpu-int4-splines {name} (BlockwiseQuantization bs={block_size})",
    )
    meta = {
        "suite": "sweep_int4",
        "op": "FULLY_CONNECTED",
        "weight_dtype": "int4",
        "act_dtype": "int8",
        "quant_mode": f"blockwise_bs{block_size}",
        "block_size": block_size,
        "M": M,
        "K": K,
        "N": N,
        "s_in": s_in,
        "zp_in": zp_in,
        "s_out": s_out,
        "zp_out": zp_out,
        "in_bytes": len(x_bytes),
        "out_bytes": len(ref_bytes),
    }
    return model_bytes, x_bytes, ref_bytes, meta


def make_dequant_int4_fc(
    name: str,
    *,
    M: int,
    K: int,
    N: int,
    seed: int,
) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    """Weight-only DEQUANTIZE(INT4 per-channel) -> FLOAT32 FULLY_CONNECTED probe."""
    w_nk = det_int4_weights(N, K, seed=seed)
    w_flat = [v for row in w_nk for v in row]
    w_bytes = pack_int4_le(w_flat)
    w_scales = [float(np.float32(0.0625 * (1.0 + 0.125 * (o % 8)))) for o in range(N)]
    w_zps = [0] * N

    rng = np.random.default_rng(seed + 404)
    x_f32 = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    x_bytes = x_f32.tobytes()

    w_deq = np.array(w_nk, dtype=np.float32) * np.array(w_scales, dtype=np.float32)[:, None]
    y_f32 = (x_f32 @ w_deq.T).astype(np.float32)
    ref_bytes = y_f32.tobytes()

    buffers: List[Optional[bytes]] = [None, w_bytes]
    tensors: List[Dict[str, Any]] = [
        {"name": "input", "shape": [M, K], "type": TENSOR_TYPE_FLOAT32, "buffer": 0},
        {
            "name": "filter_i4",
            "shape": [N, K],
            "type": TENSOR_TYPE_INT4,
            "buffer": 1,
            "scales": w_scales,
            "zero_points": w_zps,
            "quantized_dimension": 0,
        },
        {"name": "filter_f32", "shape": [N, K], "type": TENSOR_TYPE_FLOAT32, "buffer": 0},
        {"name": "output", "shape": [M, N], "type": TENSOR_TYPE_FLOAT32, "buffer": 0},
    ]
    model_bytes = build_model_ext(
        opcodes=[
            (BUILTIN_OP_DEQUANTIZE, 5),
            (BUILTIN_OP_FULLY_CONNECTED, 9),
        ],
        tensors_spec=tensors,
        inputs=[0],
        outputs=[3],
        operators_spec=[
            {
                "opcode_index": 0,
                "inputs": [1],
                "outputs": [2],
                "builtin_options_type": BUILTIN_OPT_NONE,
            },
            {
                "opcode_index": 1,
                "inputs": [0, 2],
                "outputs": [3],
                "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED,
                "fused_activation": ACT_NONE,
                "weights_format": 0,
                "keep_num_dims": False,
                "asymmetric_quantize_inputs": False,
            },
        ],
        buffers_data=buffers,
        description=f"tpu-int4-splines {name} (DEQUANTIZE(INT4) -> FC)",
    )
    meta = {
        "suite": "sweep_int4",
        "op": "DEQUANTIZE+FULLY_CONNECTED",
        "weight_dtype": "int4",
        "act_dtype": "float32",
        "quant_mode": "dequant_per_channel",
        "M": M,
        "K": K,
        "N": N,
        "in_bytes": len(x_bytes),
        "out_bytes": len(ref_bytes),
    }
    return model_bytes, x_bytes, ref_bytes, meta


# ---------------------------------------------------------------------------
# Sweep assembly (19 models)
# ---------------------------------------------------------------------------


def generate_sweep(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: List[Tuple[str, Tuple[bytes, bytes, bytes, Dict[str, Any]]]] = [
        # Paired 1-to-1 ISA bit-diff baseline (INT8 vs INT4 identical scaled-identity)
        (
            "fc_i8_pt_1x16x16_id_ref",
            make_fc_int4_or_int8(
                "fc_i8_pt_1x16x16_id_ref",
                M=1, K=16, N=16, seed=1000,
                weight_type=TENSOR_TYPE_INT8, identity_val=4,
            ),
        ),
        (
            "fc_i4_pt_1x16x16_id",
            make_fc_int4_or_int8(
                "fc_i4_pt_1x16x16_id",
                M=1, K=16, N=16, seed=1000,
                weight_type=TENSOR_TYPE_INT4, identity_val=4,
            ),
        ),
        # Per-tensor INT4 W4A8 shape & feature sweep
        (
            "fc_i4_pt_1x32x16",
            make_fc_int4_or_int8("fc_i4_pt_1x32x16", M=1, K=32, N=16, seed=1001),
        ),
        (
            "fc_i4_pt_1x64x32",
            make_fc_int4_or_int8("fc_i4_pt_1x64x32", M=1, K=64, N=32, seed=1002),
        ),
        (
            "fc_i4_pt_1x128x32_bias",
            make_fc_int4_or_int8("fc_i4_pt_1x128x32_bias", M=1, K=128, N=32, seed=1003, zero_bias=False),
        ),
        (
            "fc_i4_pt_1x256x64_bias",
            make_fc_int4_or_int8("fc_i4_pt_1x256x64_bias", M=1, K=256, N=64, seed=1004, zero_bias=False),
        ),
        (
            "fc_i4_pt_4x64x32",
            make_fc_int4_or_int8("fc_i4_pt_4x64x32", M=4, K=64, N=32, seed=1005),
        ),
        (
            "fc_i4_pt_8x64x32",
            make_fc_int4_or_int8("fc_i4_pt_8x64x32", M=8, K=64, N=32, seed=1006),
        ),
        (
            "fc_i4_pt_1x32x16_zp_relu",
            make_fc_int4_or_int8(
                "fc_i4_pt_1x32x16_zp_relu",
                M=1, K=32, N=16, seed=1007,
                zp_in=-10, zp_out=-128, fused_act=ACT_RELU,
            ),
        ),
        (
            "fc_i4_pt_1x32x16_nobias",
            make_fc_int4_or_int8("fc_i4_pt_1x32x16_nobias", M=1, K=32, N=16, seed=1008, use_bias=False),
        ),
        # Per-channel INT4 W4A8 sweep
        (
            "fc_i4_pc_1x32x16",
            make_fc_int4_or_int8("fc_i4_pc_1x32x16", M=1, K=32, N=16, seed=2001, per_channel=True),
        ),
        (
            "fc_i4_pc_1x64x32_bias",
            make_fc_int4_or_int8("fc_i4_pc_1x64x32_bias", M=1, K=64, N=32, seed=2002, per_channel=True, zero_bias=False),
        ),
        (
            "fc_i4_pc_1x128x64_bias",
            make_fc_int4_or_int8("fc_i4_pc_1x128x64_bias", M=1, K=128, N=64, seed=2003, per_channel=True, zero_bias=False),
        ),
        (
            "fc_i4_pc_4x128x64",
            make_fc_int4_or_int8("fc_i4_pc_4x128x64", M=4, K=128, N=64, seed=2004, per_channel=True),
        ),
        # Blockwise sub-channel INT4 W4A8 (TFL3 BlockwiseQuantization details_type=2)
        (
            "fc_i4_blk32_1x128x32",
            make_fc_int4_blockwise("fc_i4_blk32_1x128x32", M=1, K=128, N=32, block_size=32, seed=3001),
        ),
        (
            "fc_i4_blk64_1x128x32",
            make_fc_int4_blockwise("fc_i4_blk64_1x128x32", M=1, K=128, N=32, block_size=64, seed=3002),
        ),
        (
            "fc_i4_blk128_1x256x64",
            make_fc_int4_blockwise("fc_i4_blk128_1x256x64", M=1, K=256, N=64, block_size=128, seed=3003),
        ),
        # Acceptance boundary probes: W4A16 & DEQUANTIZE(INT4)->FC
        (
            "fc_i4_w4a16_pc_1x64x32",
            make_fc_int4_or_int8(
                "fc_i4_w4a16_pc_1x64x32",
                M=1, K=64, N=32, seed=4001, per_channel=True, is_int16_act=True,
            ),
        ),
        (
            "dequant_i4_fc_1x64x32",
            make_dequant_int4_fc("dequant_i4_fc_1x64x32", M=1, K=64, N=32, seed=4002),
        ),
    ]

    manifest: Dict[str, Any] = {}
    for name, (model_bytes, in_bytes, ref_bytes, meta) in cases:
        vinfo = validate_int4_tflite(model_bytes)
        tfl_path = out_dir / f"{name}.tflite"
        in_path = out_dir / f"{name}.in.bin"
        ref_path = out_dir / f"{name}.ref.bin"
        tfl_path.write_bytes(model_bytes)
        in_path.write_bytes(in_bytes)
        ref_path.write_bytes(ref_bytes)

        meta["tflite_bytes"] = len(model_bytes)
        meta["tflite_sha256"] = hashlib.sha256(model_bytes).hexdigest()
        meta["in_sha256"] = hashlib.sha256(in_bytes).hexdigest()
        meta["ref_sha256"] = hashlib.sha256(ref_bytes).hexdigest()
        meta["validation"] = vinfo
        manifest[name] = {
            "tflite_path": str(tfl_path),
            "in_path": str(in_path),
            "ref_path": str(ref_path),
            "meta": meta,
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "out_dir": str(out_dir),
        "model_count": len(manifest),
        "manifest_path": str(manifest_path),
        "models": list(manifest.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate validated INT4 TFLite sweep pack for Tensor G5")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    summary = generate_sweep(args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
