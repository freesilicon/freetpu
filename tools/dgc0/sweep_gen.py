#!/usr/bin/env python3
"""Pure-Python Phase 3 TFLite v3 (TFL3) + INT8/INT16 CPU reference sweep generator.

Generates parameterized 1-op and 2-op TFLite models + exact reference vectors
(`.tflite`, `.in.bin`, `.ref.bin`, `manifest.json`) for Tensor G5 (`buenos`)
ISA compile-diff sweeps across:
  - DMA / SRAM tiling (`RESHAPE`, `SLICE`, `CONCATENATION`, dims 1..256)
  - INT8 & INT16x8 GEMM / systolic (`FULLY_CONNECTED`, `1x1 CONV_2D`, `3x3 CONV_2D`)
  - Vector / quant / activation (`ADD`, `MUL`, `RELU`, `LOGISTIC`, `QUANTIZE`)
  - 2-op chains (`FC->RELU`, `CONV1x1->ADD`, `ADD->LOGISTIC`)

Every generated `.tflite` passes an embedded strict TFLite v3 FlatBuffer
schema & BuiltinOperator/BuiltinOptions bijection validator before writing.

Zero external pip dependencies; Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Minimal Pure-Python FlatBuffer Builder (Little-Endian, Standard Spec)
# ---------------------------------------------------------------------------

class FlatBufferBuilder:
    """Standard backwards-growing FlatBuffer binary encoder."""

    def __init__(self, initial_size: int = 4096) -> None:
        self._buf = bytearray(initial_size)
        self._head = initial_size
        self._minalign = 1
        self._vtable: List[int] = []
        self._object_start = 0
        self._vtables: Dict[bytes, int] = {}
        self._in_table = False

    def offset(self) -> int:
        return len(self._buf) - self._head

    def _grow(self, needed: int) -> None:
        old_len = len(self._buf)
        used = old_len - self._head
        new_len = max(old_len * 2, used + needed + 1024)
        new_buf = bytearray(new_len)
        new_head = new_len - used
        new_buf[new_head:] = self._buf[self._head:]
        self._buf = new_buf
        self._head = new_head

    def prep(self, size: int, additional_bytes: int) -> None:
        if size > self._minalign:
            self._minalign = size
        used = len(self._buf) - self._head
        align_size = ((~(used + additional_bytes)) + 1) & (size - 1)
        needed = align_size + size + additional_bytes
        if self._head < needed:
            self._grow(needed)
        if align_size:
            self._head -= align_size
            self._buf[self._head : self._head + align_size] = b"\x00" * align_size

    def _place_bytes(self, raw: bytes) -> None:
        n = len(raw)
        if self._head < n:
            self._grow(n)
        self._head -= n
        self._buf[self._head : self._head + n] = raw

    def place_u8(self, x: int) -> None:
        self.prep(1, 0)
        self._place_bytes(struct.pack("<B", x & 0xFF))

    def place_i32(self, x: int) -> None:
        self.prep(4, 0)
        self._place_bytes(struct.pack("<i", x))

    def place_u32(self, x: int) -> None:
        self.prep(4, 0)
        self._place_bytes(struct.pack("<I", x & 0xFFFFFFFF))

    def place_i64(self, x: int) -> None:
        self.prep(8, 0)
        self._place_bytes(struct.pack("<q", x))

    def place_f32(self, x: float) -> None:
        self.prep(4, 0)
        self._place_bytes(struct.pack("<f", x))

    def place_uoffset(self, target_off: int) -> None:
        self.prep(4, 0)
        rel = self.offset() + 4 - target_off
        if rel <= 0:
            raise ValueError(f"Invalid forward offset: cur={self.offset()+4}, target={target_off}")
        self._place_bytes(struct.pack("<I", rel))

    def create_byte_vector(self, data: bytes) -> int:
        n = len(data)
        self.prep(4, n)
        self._place_bytes(data)
        self.place_u32(n)
        return self.offset()

    def create_string(self, s: str) -> int:
        raw = s.encode("utf-8") + b"\x00"
        self.prep(4, len(raw))
        self._place_bytes(raw)
        self.place_u32(len(raw) - 1)
        return self.offset()

    def create_i32_vector(self, vals: Sequence[int]) -> int:
        n = len(vals)
        self.prep(4, n * 4)
        for v in reversed(vals):
            self.place_i32(v)
        self.place_u32(n)
        return self.offset()

    def create_i64_vector(self, vals: Sequence[int]) -> int:
        n = len(vals)
        self.prep(4, n * 8)
        for v in reversed(vals):
            self.place_i64(v)
        self.place_u32(n)
        return self.offset()

    def create_f32_vector(self, vals: Sequence[float]) -> int:
        n = len(vals)
        self.prep(4, n * 4)
        for v in reversed(vals):
            self.place_f32(v)
        self.place_u32(n)
        return self.offset()

    def create_uoffset_vector(self, offsets: Sequence[int]) -> int:
        n = len(offsets)
        self.prep(4, n * 4)
        for off in reversed(offsets):
            self.place_uoffset(off)
        self.place_u32(n)
        return self.offset()

    def start_table(self, num_fields: int) -> None:
        if self._in_table:
            raise RuntimeError("Nested table start")
        self._in_table = True
        self._vtable = [0] * num_fields
        self._object_start = self.offset()

    def _slot(self, field_id: int) -> None:
        self._vtable[field_id] = self.offset()

    def add_bool(self, field_id: int, val: bool, default: bool = False) -> None:
        if val != default:
            self.place_u8(1 if val else 0)
            self._slot(field_id)

    def add_u8(self, field_id: int, val: int, default: int = 0) -> None:
        if val != default:
            self.place_u8(val)
            self._slot(field_id)

    def add_i32(self, field_id: int, val: int, default: int = 0) -> None:
        if val != default:
            self.place_i32(val)
            self._slot(field_id)

    def add_u32(self, field_id: int, val: int, default: int = 0) -> None:
        if val != default:
            self.place_u32(val)
            self._slot(field_id)

    def add_uoffset(self, field_id: int, off: Optional[int]) -> None:
        if off is not None and off != 0:
            self.place_uoffset(off)
            self._slot(field_id)

    def end_table(self) -> int:
        if not self._in_table:
            raise RuntimeError("end_table without start_table")
        self._in_table = False
        self.place_i32(0)
        obj_end = self.offset()
        obj_size = obj_end - self._object_start

        vlen = len(self._vtable)
        while vlen > 0 and self._vtable[vlen - 1] == 0:
            vlen -= 1

        slots = [0 if self._vtable[i] == 0 else (obj_end - self._vtable[i]) for i in range(vlen)]
        vt_size = (2 + vlen) * 2
        vt_bytes = struct.pack(f"<{2 + vlen}H", vt_size, obj_size, *slots)

        if vt_bytes in self._vtables:
            vt_off = self._vtables[vt_bytes]
        else:
            self.prep(2, len(vt_bytes) - 2 if len(vt_bytes) >= 2 else 0)
            self._place_bytes(vt_bytes)
            vt_off = self.offset()
            self._vtables[vt_bytes] = vt_off

        soff_idx = len(self._buf) - obj_end
        struct.pack_into("<i", self._buf, soff_idx, vt_off - obj_end)
        return obj_end

    def finish(self, root_table_off: int, file_identifier: bytes) -> bytes:
        if len(file_identifier) != 4:
            raise ValueError("file_identifier must be 4 bytes")
        self.prep(self._minalign, 8)
        self._place_bytes(file_identifier)
        self.place_uoffset(root_table_off)
        return bytes(self._buf[self._head:])


# ---------------------------------------------------------------------------
# Verified TFLite Schema v3 Constants (tensorflow/lite/schema/schema.fbs)
# ---------------------------------------------------------------------------

TENSOR_TYPE_FLOAT32 = 0
TENSOR_TYPE_INT32 = 2
TENSOR_TYPE_INT64 = 4
TENSOR_TYPE_INT16 = 7
TENSOR_TYPE_INT8 = 9

TENSOR_ELEM_BYTES = {
    TENSOR_TYPE_FLOAT32: 4,
    TENSOR_TYPE_INT32: 4,
    TENSOR_TYPE_INT64: 8,
    TENSOR_TYPE_INT16: 2,
    TENSOR_TYPE_INT8: 1,
}

BUILTIN_OP_ADD = 0
BUILTIN_OP_CONCATENATION = 2
BUILTIN_OP_CONV_2D = 3
BUILTIN_OP_FULLY_CONNECTED = 9
BUILTIN_OP_LOGISTIC = 14
BUILTIN_OP_MUL = 18
BUILTIN_OP_RELU = 19
BUILTIN_OP_RESHAPE = 22
BUILTIN_OP_SLICE = 65
BUILTIN_OP_QUANTIZE = 114

BUILTIN_OPT_NONE = 0
BUILTIN_OPT_CONV_2D = 1
BUILTIN_OPT_FULLY_CONNECTED = 8
BUILTIN_OPT_CONCATENATION = 10
BUILTIN_OPT_ADD = 11
BUILTIN_OPT_RESHAPE = 17
BUILTIN_OPT_MUL = 21
BUILTIN_OPT_SLICE = 48
BUILTIN_OPT_QUANTIZE = 89

# Exact legal BuiltinOptions discriminant(s) per BuiltinOperator for strict validation
EXPECTED_OPTIONS_BY_OP: Dict[int, Tuple[int, ...]] = {
    BUILTIN_OP_ADD: (BUILTIN_OPT_ADD,),
    BUILTIN_OP_CONCATENATION: (BUILTIN_OPT_CONCATENATION,),
    BUILTIN_OP_CONV_2D: (BUILTIN_OPT_CONV_2D,),
    BUILTIN_OP_FULLY_CONNECTED: (BUILTIN_OPT_FULLY_CONNECTED,),
    BUILTIN_OP_LOGISTIC: (BUILTIN_OPT_NONE,),
    BUILTIN_OP_MUL: (BUILTIN_OPT_MUL,),
    BUILTIN_OP_RELU: (BUILTIN_OPT_NONE,),
    BUILTIN_OP_RESHAPE: (BUILTIN_OPT_RESHAPE, BUILTIN_OPT_NONE),
    BUILTIN_OP_SLICE: (BUILTIN_OPT_SLICE, BUILTIN_OPT_NONE),
    BUILTIN_OP_QUANTIZE: (BUILTIN_OPT_QUANTIZE, BUILTIN_OPT_NONE),
}

PADDING_SAME = 0
PADDING_VALID = 1
ACT_NONE = 0


# ---------------------------------------------------------------------------
# Exact TFLite Reference Fixed-Point INT8 / INT16 Arithmetic
# ---------------------------------------------------------------------------

def quantize_multiplier(real_multiplier: float) -> Tuple[int, int]:
    if real_multiplier == 0.0:
        return 0, 0
    q, shift = math.frexp(real_multiplier)
    q_fixed = int(round(q * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    assert 0 <= q_fixed < (1 << 31)
    return q_fixed, shift


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    if a == -(1 << 31) and b == -(1 << 31):
        return (1 << 31) - 1
    ab_64 = a * b
    nudge = (1 << 30) if ab_64 >= 0 else (1 - (1 << 30))
    val = ab_64 + nudge
    return (val >> 31) if val >= 0 else -((-val) >> 31)


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def multiply_by_quantized_multiplier(x: int, quantized_multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    scaled = saturating_rounding_doubling_high_mul(x * (1 << left_shift), quantized_multiplier)
    return rounding_divide_by_pot(scaled, right_shift)


def clamp_int8(x: int) -> int:
    return -128 if x < -128 else (127 if x > 127 else x)


def clamp_int16(x: int) -> int:
    return -32768 if x < -32768 else (32767 if x > 32767 else x)


def prod(dims: Sequence[int]) -> int:
    p = 1
    for d in dims:
        p *= d
    return p


# ---------------------------------------------------------------------------
# TFLite FlatBuffer Serializer & Strict Schema Validator
# ---------------------------------------------------------------------------

def build_quant_params(
    fb: FlatBufferBuilder,
    scales: Sequence[float],
    zero_points: Sequence[int],
    quantized_dimension: int = 0,
) -> int:
    scale_off = fb.create_f32_vector(scales)
    zp_off = fb.create_i64_vector(zero_points)
    fb.start_table(7)
    fb.add_uoffset(2, scale_off)
    fb.add_uoffset(3, zp_off)
    fb.add_i32(6, quantized_dimension, 0)
    return fb.end_table()


def build_tensor(fb: FlatBufferBuilder, spec: Dict[str, Any]) -> int:
    shape_off = fb.create_i32_vector(spec["shape"])
    name_off = fb.create_string(spec["name"])
    quant_off = 0
    if spec.get("scales") is not None and spec.get("zero_points") is not None:
        quant_off = build_quant_params(
            fb, spec["scales"], spec["zero_points"], spec.get("quantized_dimension", 0)
        )
    fb.start_table(8)
    fb.add_uoffset(0, shape_off)
    fb.add_u8(1, spec["type"], 0)
    fb.add_u32(2, spec["buffer"], 0)
    fb.add_uoffset(3, name_off)
    fb.add_uoffset(4, quant_off)
    return fb.end_table()


def build_buffer(fb: FlatBufferBuilder, data: Optional[bytes] = None) -> int:
    data_off = fb.create_byte_vector(data) if data is not None else 0
    fb.start_table(1)
    fb.add_uoffset(0, data_off)
    return fb.end_table()


def build_opcode(fb: FlatBufferBuilder, builtin_code: int, version: int) -> int:
    fb.start_table(4)
    fb.add_u8(0, builtin_code if builtin_code < 127 else 127, 0)
    fb.add_i32(2, version, 1)
    fb.add_i32(3, builtin_code, 0)
    return fb.end_table()


def build_options_table(fb: FlatBufferBuilder, op: Dict[str, Any]) -> int:
    opt_type = op["builtin_options_type"]
    if opt_type == BUILTIN_OPT_NONE:
        return 0
    if opt_type == BUILTIN_OPT_ADD:
        fb.start_table(2)
        fb.add_u8(0, op.get("fused_activation", ACT_NONE), 0)
        fb.add_bool(1, op.get("pot_scale_int16", True), True)
        return fb.end_table()
    if opt_type == BUILTIN_OPT_MUL:
        fb.start_table(1)
        fb.add_u8(0, op.get("fused_activation", ACT_NONE), 0)
        return fb.end_table()
    if opt_type == BUILTIN_OPT_FULLY_CONNECTED:
        fb.start_table(4)
        fb.add_u8(0, op.get("fused_activation", ACT_NONE), 0)
        fb.add_u8(1, op.get("weights_format", 0), 0)
        fb.add_bool(2, op.get("keep_num_dims", False), False)
        fb.add_bool(3, op.get("asymmetric_quantize_inputs", False), False)
        return fb.end_table()
    if opt_type == BUILTIN_OPT_CONV_2D:
        fb.start_table(6)
        fb.add_u8(0, op.get("padding", PADDING_SAME), 0)
        fb.add_i32(1, op.get("stride_w", 1), 0)
        fb.add_i32(2, op.get("stride_h", 1), 0)
        fb.add_u8(3, op.get("fused_activation", ACT_NONE), 0)
        fb.add_i32(4, op.get("dilation_w", 1), 1)
        fb.add_i32(5, op.get("dilation_h", 1), 1)
        return fb.end_table()
    if opt_type == BUILTIN_OPT_RESHAPE:
        ns = fb.create_i32_vector(op["new_shape"])
        fb.start_table(1)
        fb.add_uoffset(0, ns)
        return fb.end_table()
    if opt_type == BUILTIN_OPT_CONCATENATION:
        fb.start_table(2)
        fb.add_i32(0, op["axis"], 0)
        fb.add_u8(1, op.get("fused_activation", ACT_NONE), 0)
        return fb.end_table()
    if opt_type in (BUILTIN_OPT_SLICE, BUILTIN_OPT_QUANTIZE):
        fb.start_table(0)
        return fb.end_table()
    raise ValueError(f"Unsupported builtin_options_type={opt_type}")


def build_model(
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
    buf_offs = [build_buffer(fb, b) for b in buffers_data]
    buffers_vec = fb.create_uoffset_vector(buf_offs)

    op_offs = []
    for op in operators_spec:
        opt_off = build_options_table(fb, op)
        in_vec = fb.create_i32_vector(op["inputs"])
        out_vec = fb.create_i32_vector(op["outputs"])
        fb.start_table(6)
        fb.add_u32(0, op["opcode_index"], 0)
        fb.add_uoffset(1, in_vec)
        fb.add_uoffset(2, out_vec)
        fb.add_u8(3, op["builtin_options_type"], 0)
        fb.add_uoffset(4, opt_off)
        op_offs.append(fb.end_table())

    t_offs = [build_tensor(fb, t) for t in tensors_spec]
    tensors_vec = fb.create_uoffset_vector(t_offs)
    sg_in_vec = fb.create_i32_vector(inputs)
    sg_out_vec = fb.create_i32_vector(outputs)
    ops_vec = fb.create_uoffset_vector(op_offs)
    sg_name_off = fb.create_string(subgraph_name)

    fb.start_table(5)
    fb.add_uoffset(0, tensors_vec)
    fb.add_uoffset(1, sg_in_vec)
    fb.add_uoffset(2, sg_out_vec)
    fb.add_uoffset(3, ops_vec)
    fb.add_uoffset(4, sg_name_off)
    sg_off = fb.end_table()
    subgraphs_vec = fb.create_uoffset_vector([sg_off])

    oc_offs = [build_opcode(fb, code, ver) for code, ver in opcodes]
    opcodes_vec = fb.create_uoffset_vector(oc_offs)
    desc_off = fb.create_string(description)

    fb.start_table(5)
    fb.add_u32(0, 3, 0)
    fb.add_uoffset(1, opcodes_vec)
    fb.add_uoffset(2, subgraphs_vec)
    fb.add_uoffset(3, desc_off)
    fb.add_uoffset(4, buffers_vec)
    root_off = fb.end_table()

    return fb.finish(root_off, b"TFL3")


def _fb_field(buf: bytes, table_pos: int, slot_idx: int) -> Optional[int]:
    soff = struct.unpack_from("<i", buf, table_pos)[0]
    vt_pos = table_pos - soff
    assert 0 <= vt_pos <= len(buf) - 4, f"vtable OOB: {vt_pos}"
    vt_size = struct.unpack_from("<H", buf, vt_pos)[0]
    vt_field_off = 4 + slot_idx * 2
    if vt_field_off >= vt_size:
        return None
    rel = struct.unpack_from("<H", buf, vt_pos + vt_field_off)[0]
    return None if rel == 0 else table_pos + rel


def _fb_uoff(buf: bytes, field_pos: int) -> int:
    tgt = field_pos + struct.unpack_from("<I", buf, field_pos)[0]
    assert 0 <= tgt < len(buf), f"uoffset OOB: {tgt}"
    return tgt


def validate_tflite_strict(blob: bytes) -> Dict[str, Any]:
    """Strict TFLite v3 FlatBuffer layout & BuiltinOptions schema verifier."""
    assert len(blob) >= 16 and blob[4:8] == b"TFL3", "Invalid TFL3 header"
    root = struct.unpack_from("<I", blob, 0)[0]
    ver_f = _fb_field(blob, root, 0)
    version = struct.unpack_from("<I", blob, ver_f)[0] if ver_f is not None else 0
    assert version == 3, f"Expected TFLite version 3, got {version}"

    # Parse operator codes
    oc_vec = _fb_uoff(blob, _fb_field(blob, root, 1))
    oc_cnt = struct.unpack_from("<I", blob, oc_vec)[0]
    opcodes: List[Tuple[int, int]] = []
    for i in range(oc_cnt):
        oc = _fb_uoff(blob, oc_vec + 4 + 4 * i)
        dep = blob[_fb_field(blob, oc, 0)] if _fb_field(blob, oc, 0) is not None else 0
        ver = struct.unpack_from("<i", blob, _fb_field(blob, oc, 2))[0] if _fb_field(blob, oc, 2) is not None else 1
        code = struct.unpack_from("<i", blob, _fb_field(blob, oc, 3))[0] if _fb_field(blob, oc, 3) is not None else dep
        if code < 127:
            assert dep == code, f"deprecated_builtin_code={dep} != builtin_code={code}"
        opcodes.append((code, ver))

    # Parse buffers
    bufs_vec = _fb_uoff(blob, _fb_field(blob, root, 4))
    buf_cnt = struct.unpack_from("<I", blob, bufs_vec)[0]
    assert buf_cnt >= 1
    buf_lens: List[int] = []
    for i in range(buf_cnt):
        bp = _fb_uoff(blob, bufs_vec + 4 + 4 * i)
        df = _fb_field(blob, bp, 0)
        if df is None:
            buf_lens.append(0)
        else:
            dv = _fb_uoff(blob, df)
            dlen = struct.unpack_from("<I", blob, dv)[0]
            assert dv + 4 + dlen <= len(blob)
            buf_lens.append(dlen)
    assert buf_lens[0] == 0, "TFLite buffer[0] must be empty sentinel"

    # Parse subgraph 0
    sgs_vec = _fb_uoff(blob, _fb_field(blob, root, 2))
    sg_cnt = struct.unpack_from("<I", blob, sgs_vec)[0]
    assert sg_cnt == 1
    sg = _fb_uoff(blob, sgs_vec + 4)

    t_vec = _fb_uoff(blob, _fb_field(blob, sg, 0))
    t_cnt = struct.unpack_from("<I", blob, t_vec)[0]
    for ti in range(t_cnt):
        tp = _fb_uoff(blob, t_vec + 4 + 4 * ti)
        sh_v = _fb_uoff(blob, _fb_field(blob, tp, 0))
        rank = struct.unpack_from("<I", blob, sh_v)[0]
        shape = list(struct.unpack_from(f"<{rank}i", blob, sh_v + 4))
        ttype = blob[_fb_field(blob, tp, 1)] if _fb_field(blob, tp, 1) is not None else 0
        bidx = struct.unpack_from("<I", blob, _fb_field(blob, tp, 2))[0] if _fb_field(blob, tp, 2) is not None else 0
        assert 0 <= bidx < buf_cnt
        if buf_lens[bidx] > 0:
            exp_bytes = prod(shape) * TENSOR_ELEM_BYTES[ttype]
            assert buf_lens[bidx] == exp_bytes, f"tensor[{ti}] buffer len {buf_lens[bidx]} != expected {exp_bytes}"

    in_vec = _fb_uoff(blob, _fb_field(blob, sg, 1))
    in_cnt = struct.unpack_from("<I", blob, in_vec)[0]
    in_indices = list(struct.unpack_from(f"<{in_cnt}i", blob, in_vec + 4))

    out_vec = _fb_uoff(blob, _fb_field(blob, sg, 2))
    out_cnt = struct.unpack_from("<I", blob, out_vec)[0]
    out_indices = list(struct.unpack_from(f"<{out_cnt}i", blob, out_vec + 4))

    # Strictly verify operators & BuiltinOptions discriminants
    ops_vec = _fb_uoff(blob, _fb_field(blob, sg, 3))
    op_cnt = struct.unpack_from("<I", blob, ops_vec)[0]
    op_summary = []
    for oi in range(op_cnt):
        op = _fb_uoff(blob, ops_vec + 4 + 4 * oi)
        oc_i = struct.unpack_from("<I", blob, _fb_field(blob, op, 0))[0] if _fb_field(blob, op, 0) is not None else 0
        assert 0 <= oc_i < len(opcodes)
        code, ver = opcodes[oc_i]
        opt_type = blob[_fb_field(blob, op, 3)] if _fb_field(blob, op, 3) is not None else 0
        opt_f = _fb_field(blob, op, 4)
        if code in EXPECTED_OPTIONS_BY_OP:
            assert opt_type in EXPECTED_OPTIONS_BY_OP[code], (
                f"Operator[{oi}] code={code} has invalid builtin_options_type={opt_type}; "
                f"expected one of {EXPECTED_OPTIONS_BY_OP[code]}"
            )
        if opt_type != BUILTIN_OPT_NONE:
            assert opt_f is not None, f"Operator[{oi}] has opt_type={opt_type} but missing options table"
            _fb_uoff(blob, opt_f)
        op_summary.append({"opcode": code, "version": ver, "opt_type": opt_type})

    return {
        "version": version,
        "opcodes": op_cnt,
        "subgraphs": sg_cnt,
        "tensors": t_cnt,
        "input_indices": in_indices,
        "output_indices": out_indices,
        "buffers": buf_cnt,
        "operators": op_summary,
    }


# ---------------------------------------------------------------------------
# Reference Op Implementations
# ---------------------------------------------------------------------------

def _det_int8(n: int, mul: int = 17, add: int = 23, span: int = 201, offset: int = -100) -> List[int]:
    return [((i * mul + add) % span) + offset for i in range(n)]


def _ref_add_int8(x0: List[int], s0: float, zp0: int, x1: List[int], s1: float, zp1: int, s_out: float, zp_out: int) -> List[int]:
    left_shift = 20
    twice_max = 2.0 * max(s0, s1)
    m0, sh0 = quantize_multiplier(s0 / twice_max)
    m1, sh1 = quantize_multiplier(s1 / twice_max)
    mo, sho = quantize_multiplier(twice_max / ((1 << left_shift) * s_out))
    out: List[int] = []
    for a, b in zip(x0, x1):
        sa = multiply_by_quantized_multiplier((a - zp0) * (1 << left_shift), m0, sh0)
        sb = multiply_by_quantized_multiplier((b - zp1) * (1 << left_shift), m1, sh1)
        out.append(clamp_int8(multiply_by_quantized_multiplier(sa + sb, mo, sho) + zp_out))
    return out


def _ref_mul_int8(x0: List[int], s0: float, zp0: int, x1: List[int], s1: float, zp1: int, s_out: float, zp_out: int) -> List[int]:
    mo, sho = quantize_multiplier((s0 * s1) / s_out)
    return [clamp_int8(multiply_by_quantized_multiplier((a - zp0) * (b - zp1), mo, sho) + zp_out) for a, b in zip(x0, x1)]


def _ref_fc_int8(x: List[int], M: int, K: int, N: int, w: List[int], bias: List[int], s_in: float, zp_in: int, s_w: float, s_out: float, zp_out: int) -> List[int]:
    mo, sho = quantize_multiplier((s_in * s_w) / s_out)
    out: List[int] = []
    for m in range(M):
        x_off = m * K
        for n in range(N):
            acc = bias[n]
            w_off = n * K
            for k in range(K):
                acc += (x[x_off + k] - zp_in) * w[w_off + k]
            out.append(clamp_int8(multiply_by_quantized_multiplier(acc, mo, sho) + zp_out))
    return out


def _ref_conv2d_int8(
    x: List[int], H: int, W: int, C_IN: int, C_OUT: int, KH: int, KW: int,
    w: List[int], bias: List[int], padding: int,
    s_in: float, zp_in: int, s_w: float, s_out: float, zp_out: int,
) -> Tuple[List[int], int, int]:
    mo, sho = quantize_multiplier((s_in * s_w) / s_out)
    if padding == PADDING_SAME:
        OH, OW = H, W
        pad_top = (KH - 1) // 2
        pad_left = (KW - 1) // 2
    else:
        OH, OW = H - KH + 1, W - KW + 1
        pad_top, pad_left = 0, 0
    out = [0] * (OH * OW * C_OUT)
    for oy in range(OH):
        for ox in range(OW):
            for oc in range(C_OUT):
                acc = bias[oc]
                w_oc = oc * KH * KW * C_IN
                for ky in range(KH):
                    iy = oy + ky - pad_top
                    if iy < 0 or iy >= H:
                        continue
                    for kx in range(KW):
                        ix = ox + kx - pad_left
                        if ix < 0 or ix >= W:
                            continue
                        x_p = (iy * W + ix) * C_IN
                        w_k = w_oc + (ky * KW + kx) * C_IN
                        for ic in range(C_IN):
                            acc += (x[x_p + ic] - zp_in) * w[w_k + ic]
                out[(oy * OW + ox) * C_OUT + oc] = clamp_int8(multiply_by_quantized_multiplier(acc, mo, sho) + zp_out)
    return out, OH, OW


def _ref_relu_int8(x: List[int], s_in: float, zp_in: int, s_out: float, zp_out: int) -> List[int]:
    mo, sho = quantize_multiplier(s_in / s_out)
    out: List[int] = []
    for v in x:
        clamped = v if v > zp_in else zp_in
        out.append(clamp_int8(multiply_by_quantized_multiplier(clamped - zp_in, mo, sho) + zp_out))
    return out


def _ref_logistic_int8(x: List[int], s_in: float, zp_in: int) -> List[int]:
    # TFLite INT8 logistic output quantization is fixed: scale = 1/256, zero_point = -128
    out: List[int] = []
    for v in x:
        real_x = (v - zp_in) * s_in
        y = 1.0 / (1.0 + math.exp(-real_x))
        q = int(round(y * 256.0)) - 128
        out.append(clamp_int8(q))
    return out


def _ref_requant_int8(x: List[int], s_in: float, zp_in: int, s_out: float, zp_out: int) -> List[int]:
    mo, sho = quantize_multiplier(s_in / s_out)
    return [clamp_int8(multiply_by_quantized_multiplier(v - zp_in, mo, sho) + zp_out) for v in x]


# ---------------------------------------------------------------------------
# Parameterized Model Builders
# ---------------------------------------------------------------------------

def make_reshape(in_shape: List[int], out_shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    assert prod(in_shape) == prod(out_shape)
    n = prod(in_shape)
    s, zp = 0.03125, -4
    x = _det_int8(n, 17, 11)
    shape_bytes = struct.pack(f"<{len(out_shape)}i", *out_shape)
    tensors = [
        {"name": "input_0", "shape": in_shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s], "zero_points": [zp]},
        {"name": "new_shape", "shape": [len(out_shape)], "type": TENSOR_TYPE_INT32, "buffer": 2},
        {"name": "output_0", "shape": out_shape, "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [s], "zero_points": [zp]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1], "outputs": [2],
        "builtin_options_type": BUILTIN_OPT_RESHAPE, "new_shape": out_shape,
    }]
    tflite = build_model([(BUILTIN_OP_RESHAPE, 1)], tensors, [0], [2], ops, [None, None, shape_bytes, None], name)
    raw = struct.pack(f"<{n}b", *x)
    return tflite, raw, raw, {"suite": "dma", "op": "RESHAPE", "in_shape": in_shape, "out_shape": out_shape}


def make_slice(in_shape: List[int], begin: List[int], size: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    assert len(in_shape) == len(begin) == len(size) == 4
    n_in = prod(in_shape)
    s, zp = 0.03125, -3
    x = _det_int8(n_in, 19, 7)
    N, H, W, C = in_shape
    b0, b1, b2, b3 = begin
    s0, s1, s2, s3 = size
    ref: List[int] = []
    for n in range(b0, b0 + s0):
        for h in range(b1, b1 + s1):
            for w in range(b2, b2 + s2):
                base = ((n * H + h) * W + w) * C
                ref.extend(x[base + b3 : base + b3 + s3])
    b_bytes = struct.pack("<4i", *begin)
    sz_bytes = struct.pack("<4i", *size)
    tensors = [
        {"name": "input_0", "shape": in_shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s], "zero_points": [zp]},
        {"name": "begin", "shape": [4], "type": TENSOR_TYPE_INT32, "buffer": 2},
        {"name": "size", "shape": [4], "type": TENSOR_TYPE_INT32, "buffer": 3},
        {"name": "output_0", "shape": size, "type": TENSOR_TYPE_INT8, "buffer": 4, "scales": [s], "zero_points": [zp]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3],
        "builtin_options_type": BUILTIN_OPT_SLICE,
    }]
    tflite = build_model([(BUILTIN_OP_SLICE, 2)], tensors, [0], [3], ops, [None, None, b_bytes, sz_bytes, None], name)
    return (
        tflite,
        struct.pack(f"<{n_in}b", *x),
        struct.pack(f"<{len(ref)}b", *ref),
        {"suite": "dma", "op": "SLICE", "in_shape": in_shape, "begin": begin, "size": size},
    )


def make_concat(shape_a: List[int], shape_b: List[int], axis: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    assert len(shape_a) == len(shape_b) == 4 and axis == 3
    out_shape = list(shape_a)
    out_shape[axis] += shape_b[axis]
    na, nb = prod(shape_a), prod(shape_b)
    s, zp = 0.03125, 0
    xa = _det_int8(na, 13, 5)
    xb = _det_int8(nb, 29, 11)
    outer = shape_a[0] * shape_a[1] * shape_a[2]
    ca, cb = shape_a[3], shape_b[3]
    ref: List[int] = []
    for o in range(outer):
        ref.extend(xa[o * ca : (o + 1) * ca])
        ref.extend(xb[o * cb : (o + 1) * cb])
    tensors = [
        {"name": "input_0", "shape": shape_a, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s], "zero_points": [zp]},
        {"name": "input_1", "shape": shape_b, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s], "zero_points": [zp]},
        {"name": "output_0", "shape": out_shape, "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [s], "zero_points": [zp]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1], "outputs": [2],
        "builtin_options_type": BUILTIN_OPT_CONCATENATION, "axis": axis, "fused_activation": ACT_NONE,
    }]
    tflite = build_model([(BUILTIN_OP_CONCATENATION, 2)], tensors, [0, 1], [2], ops, [None, None, None, None], name)
    in_bin = struct.pack(f"<{na}b", *xa) + struct.pack(f"<{nb}b", *xb)
    ref_bin = struct.pack(f"<{len(ref)}b", *ref)
    return tflite, in_bin, ref_bin, {"suite": "dma", "op": "CONCATENATION", "in_shapes": [shape_a, shape_b], "out_shape": out_shape, "axis": axis}


def make_fc_int8(M: int, K: int, N: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    s_in, zp_in = 0.03125, -5
    s_w, zp_w = 0.015625, 0
    s_bias = s_in * s_w
    s_out, zp_out = 0.0625, 3
    x = _det_int8(M * K, 19, 5, 121, -60)
    w = [(((o * 7 + i * 11 + 3) % 63) - 31) for o in range(N) for i in range(K)]
    bias = [((o * 29 - 100) % 250) - 125 for o in range(N)]
    ref = _ref_fc_int8(x, M, K, N, w, bias, s_in, zp_in, s_w, s_out, zp_out)
    w_bytes = struct.pack(f"<{N * K}b", *w)
    b_bytes = struct.pack(f"<{N}i", *bias)
    tensors = [
        {"name": "input_0", "shape": [M, K], "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s_in], "zero_points": [zp_in]},
        {"name": "weights", "shape": [N, K], "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s_w], "zero_points": [zp_w]},
        {"name": "bias", "shape": [N], "type": TENSOR_TYPE_INT32, "buffer": 3, "scales": [s_bias], "zero_points": [0]},
        {"name": "output_0", "shape": [M, N], "type": TENSOR_TYPE_INT8, "buffer": 4, "scales": [s_out], "zero_points": [zp_out]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3],
        "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED, "fused_activation": ACT_NONE,
    }]
    tflite = build_model([(BUILTIN_OP_FULLY_CONNECTED, 4)], tensors, [0], [3], ops, [None, None, w_bytes, b_bytes, None], name)
    return (
        tflite,
        struct.pack(f"<{M * K}b", *x),
        struct.pack(f"<{M * N}b", *ref),
        {"suite": "gemm", "op": "FULLY_CONNECTED", "dtype": "int8", "M": M, "K": K, "N": N},
    )


def make_fc_int16x8(M: int, K: int, N: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    """INT16 symmetric activations (zp=0) x INT8 symmetric weights + INT64 bias."""
    s_in, s_w, s_out = 0.0009765625, 0.015625, 0.001953125  # 2^-10, 2^-6, 2^-9
    s_bias = s_in * s_w
    x = [((i * 137 + 19) % 4001) - 2000 for i in range(M * K)]
    w = [(((o * 7 + i * 11 + 3) % 63) - 31) for o in range(N) for i in range(K)]
    bias = [((o * 53 - 200) % 500) - 250 for o in range(N)]
    mo, sho = quantize_multiplier((s_in * s_w) / s_out)
    ref: List[int] = []
    for m in range(M):
        for n in range(N):
            acc = bias[n]
            for k in range(K):
                acc += x[m * K + k] * w[n * K + k]
            ref.append(clamp_int16(multiply_by_quantized_multiplier(acc, mo, sho)))
    w_bytes = struct.pack(f"<{N * K}b", *w)
    b_bytes = struct.pack(f"<{N}q", *bias)
    tensors = [
        {"name": "input_0", "shape": [M, K], "type": TENSOR_TYPE_INT16, "buffer": 1, "scales": [s_in], "zero_points": [0]},
        {"name": "weights", "shape": [N, K], "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s_w], "zero_points": [0]},
        {"name": "bias", "shape": [N], "type": TENSOR_TYPE_INT64, "buffer": 3, "scales": [s_bias], "zero_points": [0]},
        {"name": "output_0", "shape": [M, N], "type": TENSOR_TYPE_INT16, "buffer": 4, "scales": [s_out], "zero_points": [0]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3],
        "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED, "fused_activation": ACT_NONE,
    }]
    tflite = build_model([(BUILTIN_OP_FULLY_CONNECTED, 9)], tensors, [0], [3], ops, [None, None, w_bytes, b_bytes, None], name)
    return (
        tflite,
        struct.pack(f"<{M * K}h", *x),
        struct.pack(f"<{M * N}h", *ref),
        {"suite": "gemm", "op": "FULLY_CONNECTED", "dtype": "int16x8", "M": M, "K": K, "N": N},
    )


def make_conv2d(H: int, W: int, C_IN: int, C_OUT: int, KH: int, KW: int, padding: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    s_in, zp_in = 0.03125, -4
    s_w, zp_w = 0.0078125, 0
    s_bias = s_in * s_w
    s_out, zp_out = 0.0625, 1
    x = _det_int8(H * W * C_IN, 13, 9, 81, -40)
    w = [
        (((oc * 5 + ky * 7 + kx * 11 + ic * 3 + 1) % 31) - 15)
        for oc in range(C_OUT) for ky in range(KH) for kx in range(KW) for ic in range(C_IN)
    ]
    bias = [((oc * 17 - 40) % 120) - 60 for oc in range(C_OUT)]
    ref, OH, OW = _ref_conv2d_int8(x, H, W, C_IN, C_OUT, KH, KW, w, bias, padding, s_in, zp_in, s_w, s_out, zp_out)
    w_bytes = struct.pack(f"<{len(w)}b", *w)
    b_bytes = struct.pack(f"<{C_OUT}i", *bias)
    tensors = [
        {"name": "input_0", "shape": [1, H, W, C_IN], "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s_in], "zero_points": [zp_in]},
        {"name": "filter", "shape": [C_OUT, KH, KW, C_IN], "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s_w], "zero_points": [zp_w]},
        {"name": "bias", "shape": [C_OUT], "type": TENSOR_TYPE_INT32, "buffer": 3, "scales": [s_bias], "zero_points": [0]},
        {"name": "output_0", "shape": [1, OH, OW, C_OUT], "type": TENSOR_TYPE_INT8, "buffer": 4, "scales": [s_out], "zero_points": [zp_out]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3],
        "builtin_options_type": BUILTIN_OPT_CONV_2D,
        "padding": padding, "stride_w": 1, "stride_h": 1, "fused_activation": ACT_NONE, "dilation_w": 1, "dilation_h": 1,
    }]
    tflite = build_model([(BUILTIN_OP_CONV_2D, 3)], tensors, [0], [3], ops, [None, None, w_bytes, b_bytes, None], name)
    return (
        tflite,
        struct.pack(f"<{len(x)}b", *x),
        struct.pack(f"<{len(ref)}b", *ref),
        {"suite": "gemm", "op": "CONV_2D", "in_shape": [1, H, W, C_IN], "kernel": [C_OUT, KH, KW, C_IN], "out_shape": [1, OH, OW, C_OUT], "padding": "SAME" if padding == PADDING_SAME else "VALID"},
    )


def make_add(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    s0, zp0, s1, zp1, so, zpo = 0.03125, -10, 0.03125, 12, 0.0625, -2
    x0 = _det_int8(n, 17, 23, 241, -120)
    x1 = _det_int8(n, 31, 7, 241, -120)
    ref = _ref_add_int8(x0, s0, zp0, x1, s1, zp1, so, zpo)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s0], "zero_points": [zp0]},
        {"name": "input_1", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s1], "zero_points": [zp1]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [so], "zero_points": [zpo]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1], "outputs": [2],
        "builtin_options_type": BUILTIN_OPT_ADD, "fused_activation": ACT_NONE, "pot_scale_int16": False,
    }]
    tflite = build_model([(BUILTIN_OP_ADD, 2)], tensors, [0, 1], [2], ops, [None, None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x0) + struct.pack(f"<{n}b", *x1), struct.pack(f"<{n}b", *ref), {"suite": "vector", "op": "ADD", "shape": shape}


def make_mul(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    s0, zp0, s1, zp1, so, zpo = 0.03125, -6, 0.03125, 4, 0.0625, -1
    x0 = _det_int8(n, 17, 13, 121, -60)
    x1 = _det_int8(n, 23, 5, 121, -60)
    ref = _ref_mul_int8(x0, s0, zp0, x1, s1, zp1, so, zpo)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s0], "zero_points": [zp0]},
        {"name": "input_1", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s1], "zero_points": [zp1]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [so], "zero_points": [zpo]},
    ]
    ops = [{
        "opcode_index": 0, "inputs": [0, 1], "outputs": [2],
        "builtin_options_type": BUILTIN_OPT_MUL, "fused_activation": ACT_NONE,
    }]
    tflite = build_model([(BUILTIN_OP_MUL, 2)], tensors, [0, 1], [2], ops, [None, None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x0) + struct.pack(f"<{n}b", *x1), struct.pack(f"<{n}b", *ref), {"suite": "vector", "op": "MUL", "shape": shape}


def make_relu(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    si, zpi, so, zpo = 0.03125, -8, 0.03125, -128
    x = _det_int8(n, 17, 3, 201, -100)
    ref = _ref_relu_int8(x, si, zpi, so, zpo)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [si], "zero_points": [zpi]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [so], "zero_points": [zpo]},
    ]
    ops = [{"opcode_index": 0, "inputs": [0], "outputs": [1], "builtin_options_type": BUILTIN_OPT_NONE}]
    tflite = build_model([(BUILTIN_OP_RELU, 2)], tensors, [0], [1], ops, [None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x), struct.pack(f"<{n}b", *ref), {"suite": "vector", "op": "RELU", "shape": shape}


def make_logistic(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    si, zpi = 0.0625, 0
    so, zpo = 1.0 / 256.0, -128
    x = _det_int8(n, 19, 11, 161, -80)
    ref = _ref_logistic_int8(x, si, zpi)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [si], "zero_points": [zpi]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [so], "zero_points": [zpo]},
    ]
    ops = [{"opcode_index": 0, "inputs": [0], "outputs": [1], "builtin_options_type": BUILTIN_OPT_NONE}]
    tflite = build_model([(BUILTIN_OP_LOGISTIC, 2)], tensors, [0], [1], ops, [None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x), struct.pack(f"<{n}b", *ref), {"suite": "vector", "op": "LOGISTIC", "shape": shape}


def make_quantize_requant(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    si, zpi, so, zpo = 0.03125, -10, 0.0625, 5
    x = _det_int8(n, 23, 9, 201, -100)
    ref = _ref_requant_int8(x, si, zpi, so, zpo)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [si], "zero_points": [zpi]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [so], "zero_points": [zpo]},
    ]
    ops = [{"opcode_index": 0, "inputs": [0], "outputs": [1], "builtin_options_type": BUILTIN_OPT_QUANTIZE}]
    tflite = build_model([(BUILTIN_OP_QUANTIZE, 2)], tensors, [0], [1], ops, [None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x), struct.pack(f"<{n}b", *ref), {"suite": "vector", "op": "QUANTIZE", "shape": shape}


def make_2op_fc_relu(M: int, K: int, N: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    s_in, zp_in = 0.03125, -5
    s_w = 0.015625
    s_mid, zp_mid = 0.0625, 3
    s_out, zp_out = 0.0625, -128
    x = _det_int8(M * K, 19, 5, 121, -60)
    w = [(((o * 7 + i * 11 + 3) % 63) - 31) for o in range(N) for i in range(K)]
    bias = [((o * 29 - 100) % 250) - 125 for o in range(N)]
    mid = _ref_fc_int8(x, M, K, N, w, bias, s_in, zp_in, s_w, s_mid, zp_mid)
    ref = _ref_relu_int8(mid, s_mid, zp_mid, s_out, zp_out)
    w_bytes = struct.pack(f"<{N * K}b", *w)
    b_bytes = struct.pack(f"<{N}i", *bias)
    tensors = [
        {"name": "input_0", "shape": [M, K], "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s_in], "zero_points": [zp_in]},
        {"name": "weights", "shape": [N, K], "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s_w], "zero_points": [0]},
        {"name": "bias", "shape": [N], "type": TENSOR_TYPE_INT32, "buffer": 3, "scales": [s_in * s_w], "zero_points": [0]},
        {"name": "fc_out", "shape": [M, N], "type": TENSOR_TYPE_INT8, "buffer": 4, "scales": [s_mid], "zero_points": [zp_mid]},
        {"name": "output_0", "shape": [M, N], "type": TENSOR_TYPE_INT8, "buffer": 5, "scales": [s_out], "zero_points": [zp_out]},
    ]
    ops = [
        {"opcode_index": 0, "inputs": [0, 1, 2], "outputs": [3], "builtin_options_type": BUILTIN_OPT_FULLY_CONNECTED, "fused_activation": ACT_NONE},
        {"opcode_index": 1, "inputs": [3], "outputs": [4], "builtin_options_type": BUILTIN_OPT_NONE},
    ]
    tflite = build_model(
        [(BUILTIN_OP_FULLY_CONNECTED, 4), (BUILTIN_OP_RELU, 2)],
        tensors, [0], [4], ops, [None, None, w_bytes, b_bytes, None, None], name,
    )
    return tflite, struct.pack(f"<{M * K}b", *x), struct.pack(f"<{M * N}b", *ref), {"suite": "two_op", "ops": ["FULLY_CONNECTED", "RELU"], "M": M, "K": K, "N": N}


def make_2op_conv1x1_add(H: int, W: int, C: int, name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    s_in, zp_in = 0.03125, -4
    s_w = 0.015625
    s_mid, zp_mid = 0.0625, 2
    s_out, zp_out = 0.0625, -1
    n = H * W * C
    x0 = _det_int8(n, 13, 9, 81, -40)
    x1 = _det_int8(n, 29, 3, 81, -40)
    w = [(((oc * 5 + ic * 3 + 1) % 31) - 15) for oc in range(C) for ic in range(C)]
    bias = [((oc * 17 - 40) % 120) - 60 for oc in range(C)]
    mid, _, _ = _ref_conv2d_int8(x0, H, W, C, C, 1, 1, w, bias, PADDING_SAME, s_in, zp_in, s_w, s_mid, zp_mid)
    ref = _ref_add_int8(mid, s_mid, zp_mid, x1, s_in, zp_in, s_out, zp_out)
    w_bytes = struct.pack(f"<{len(w)}b", *w)
    b_bytes = struct.pack(f"<{C}i", *bias)
    tensors = [
        {"name": "input_0", "shape": [1, H, W, C], "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s_in], "zero_points": [zp_in]},
        {"name": "input_1", "shape": [1, H, W, C], "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s_in], "zero_points": [zp_in]},
        {"name": "filter", "shape": [C, 1, 1, C], "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [s_w], "zero_points": [0]},
        {"name": "bias", "shape": [C], "type": TENSOR_TYPE_INT32, "buffer": 4, "scales": [s_in * s_w], "zero_points": [0]},
        {"name": "conv_out", "shape": [1, H, W, C], "type": TENSOR_TYPE_INT8, "buffer": 5, "scales": [s_mid], "zero_points": [zp_mid]},
        {"name": "output_0", "shape": [1, H, W, C], "type": TENSOR_TYPE_INT8, "buffer": 6, "scales": [s_out], "zero_points": [zp_out]},
    ]
    ops = [
        {"opcode_index": 0, "inputs": [0, 2, 3], "outputs": [4], "builtin_options_type": BUILTIN_OPT_CONV_2D, "padding": PADDING_SAME, "stride_w": 1, "stride_h": 1, "fused_activation": ACT_NONE},
        {"opcode_index": 1, "inputs": [4, 1], "outputs": [5], "builtin_options_type": BUILTIN_OPT_ADD, "fused_activation": ACT_NONE, "pot_scale_int16": False},
    ]
    tflite = build_model(
        [(BUILTIN_OP_CONV_2D, 3), (BUILTIN_OP_ADD, 2)],
        tensors, [0, 1], [5], ops, [None, None, None, w_bytes, b_bytes, None, None], name,
    )
    return tflite, struct.pack(f"<{n}b", *x0) + struct.pack(f"<{n}b", *x1), struct.pack(f"<{n}b", *ref), {"suite": "two_op", "ops": ["CONV_2D", "ADD"], "shape": [1, H, W, C]}


def make_2op_add_logistic(shape: List[int], name: str) -> Tuple[bytes, bytes, bytes, Dict[str, Any]]:
    n = prod(shape)
    s0, zp0, s1, zp1, sm, zpm = 0.03125, -4, 0.03125, 4, 0.0625, 0
    x0 = _det_int8(n, 17, 13, 121, -60)
    x1 = _det_int8(n, 29, 7, 121, -60)
    mid = _ref_add_int8(x0, s0, zp0, x1, s1, zp1, sm, zpm)
    ref = _ref_logistic_int8(mid, sm, zpm)
    tensors = [
        {"name": "input_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 1, "scales": [s0], "zero_points": [zp0]},
        {"name": "input_1", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 2, "scales": [s1], "zero_points": [zp1]},
        {"name": "add_out", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 3, "scales": [sm], "zero_points": [zpm]},
        {"name": "output_0", "shape": shape, "type": TENSOR_TYPE_INT8, "buffer": 4, "scales": [1.0 / 256.0], "zero_points": [-128]},
    ]
    ops = [
        {"opcode_index": 0, "inputs": [0, 1], "outputs": [2], "builtin_options_type": BUILTIN_OPT_ADD, "fused_activation": ACT_NONE, "pot_scale_int16": False},
        {"opcode_index": 1, "inputs": [2], "outputs": [3], "builtin_options_type": BUILTIN_OPT_NONE},
    ]
    tflite = build_model([(BUILTIN_OP_ADD, 2), (BUILTIN_OP_LOGISTIC, 2)], tensors, [0, 1], [3], ops, [None, None, None, None, None], name)
    return tflite, struct.pack(f"<{n}b", *x0) + struct.pack(f"<{n}b", *x1), struct.pack(f"<{n}b", *ref), {"suite": "two_op", "ops": ["ADD", "LOGISTIC"], "shape": shape}


# ---------------------------------------------------------------------------
# Sweep Registry
# ---------------------------------------------------------------------------

def get_sweep_specs(suite: str) -> List[Tuple[str, Callable[[], Tuple[bytes, bytes, bytes, Dict[str, Any]]]]]:
    dma: List[Tuple[str, Callable[[], Tuple[bytes, bytes, bytes, Dict[str, Any]]]]] = [
        ("dma_reshape_1x16x16x16_to_1x256x16", lambda: make_reshape([1, 16, 16, 16], [1, 256, 16], "dma_reshape_1x16x16x16_to_1x256x16")),
        ("dma_reshape_1x64x1x64_to_1x16x4x64", lambda: make_reshape([1, 64, 1, 64], [1, 16, 4, 64], "dma_reshape_1x64x1x64_to_1x16x4x64")),
        ("dma_reshape_1x256_to_1x16x16", lambda: make_reshape([1, 256], [1, 16, 16], "dma_reshape_1x256_to_1x16x16")),
        ("dma_slice_1x16x16x16_to_1x8x8x16", lambda: make_slice([1, 16, 16, 16], [0, 4, 4, 0], [1, 8, 8, 16], "dma_slice_1x16x16x16_to_1x8x8x16")),
        ("dma_slice_1x64x64x4_to_1x32x32x4", lambda: make_slice([1, 64, 64, 4], [0, 8, 8, 0], [1, 32, 32, 4], "dma_slice_1x64x64x4_to_1x32x32x4")),
        ("dma_slice_1x1x256x16_to_1x1x128x16", lambda: make_slice([1, 1, 256, 16], [0, 0, 64, 0], [1, 1, 128, 16], "dma_slice_1x1x256x16_to_1x1x128x16")),
        ("dma_concat_2x_1x16x16x8_c", lambda: make_concat([1, 16, 16, 8], [1, 16, 16, 8], 3, "dma_concat_2x_1x16x16x8_c")),
        ("dma_concat_2x_1x8x8x64_c", lambda: make_concat([1, 8, 8, 64], [1, 8, 8, 64], 3, "dma_concat_2x_1x8x8x64_c")),
        ("dma_concat_2x_1x1x1x128_c", lambda: make_concat([1, 1, 1, 128], [1, 1, 1, 128], 3, "dma_concat_2x_1x1x1x128_c")),
    ]
    gemm: List[Tuple[str, Callable[[], Tuple[bytes, bytes, bytes, Dict[str, Any]]]]] = [
        ("gemm_fc_1x16x16", lambda: make_fc_int8(1, 16, 16, "gemm_fc_1x16x16")),
        ("gemm_fc_1x16x32", lambda: make_fc_int8(1, 16, 32, "gemm_fc_1x16x32")),
        ("gemm_fc_1x64x64", lambda: make_fc_int8(1, 64, 64, "gemm_fc_1x64x64")),
        ("gemm_fc_4x32x32", lambda: make_fc_int8(4, 32, 32, "gemm_fc_4x32x32")),
        ("gemm_fc_1x128x64", lambda: make_fc_int8(1, 128, 64, "gemm_fc_1x128x64")),
        ("gemm_fc_int16x8_1x16x16", lambda: make_fc_int16x8(1, 16, 16, "gemm_fc_int16x8_1x16x16")),
        ("gemm_conv1x1_1x16x16x16", lambda: make_conv2d(16, 16, 16, 16, 1, 1, PADDING_SAME, "gemm_conv1x1_1x16x16x16")),
        ("gemm_conv1x1_1x8x8x64", lambda: make_conv2d(8, 8, 64, 64, 1, 1, PADDING_SAME, "gemm_conv1x1_1x8x8x64")),
        ("gemm_conv3x3_same_1x16x16x16", lambda: make_conv2d(16, 16, 16, 16, 3, 3, PADDING_SAME, "gemm_conv3x3_same_1x16x16x16")),
        ("gemm_conv3x3_valid_1x16x16x16", lambda: make_conv2d(16, 16, 16, 16, 3, 3, PADDING_VALID, "gemm_conv3x3_valid_1x16x16x16")),
    ]
    vector: List[Tuple[str, Callable[[], Tuple[bytes, bytes, bytes, Dict[str, Any]]]]] = [
        ("vec_add_1x16x16x16", lambda: make_add([1, 16, 16, 16], "vec_add_1x16x16x16")),
        ("vec_add_1x64x1x64", lambda: make_add([1, 64, 1, 64], "vec_add_1x64x1x64")),
        ("vec_mul_1x16x16x16", lambda: make_mul([1, 16, 16, 16], "vec_mul_1x16x16x16")),
        ("vec_relu_1x16x16x16", lambda: make_relu([1, 16, 16, 16], "vec_relu_1x16x16x16")),
        ("vec_logistic_1x16x16x16", lambda: make_logistic([1, 16, 16, 16], "vec_logistic_1x16x16x16")),
        ("vec_quantize_requant_1x16x16x16", lambda: make_quantize_requant([1, 16, 16, 16], "vec_quantize_requant_1x16x16x16")),
    ]
    two_op: List[Tuple[str, Callable[[], Tuple[bytes, bytes, bytes, Dict[str, Any]]]]] = [
        ("pipe2_fc_relu_1x16x16", lambda: make_2op_fc_relu(1, 16, 16, "pipe2_fc_relu_1x16x16")),
        ("pipe2_conv1x1_add_1x16x16x16", lambda: make_2op_conv1x1_add(16, 16, 16, "pipe2_conv1x1_add_1x16x16x16")),
        ("pipe2_add_logistic_1x16x16x16", lambda: make_2op_add_logistic([1, 16, 16, 16], "pipe2_add_logistic_1x16x16x16")),
    ]

    if suite == "dma":
        return dma
    if suite == "gemm":
        return gemm
    if suite == "vector":
        return vector
    if suite == "two_op":
        return two_op
    if suite == "smoke":
        return [dma[0], gemm[0], gemm[6], vector[0], vector[3], two_op[0]]
    if suite == "all":
        return dma + gemm + vector + two_op
    raise ValueError(f"Unknown suite: {suite}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 3 TFLite v3 ISA sweep models & CPU references")
    parser.add_argument("--suite", choices=["all", "dma", "gemm", "vector", "two_op", "smoke"], default="all")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for .tflite/.in.bin/.ref.bin/manifest.json")
    parser.add_argument("--validate-only", type=Path, default=None, help="Strictly validate an existing .tflite file and exit")
    args = parser.parse_args(argv)

    if args.validate_only is not None:
        parsed = validate_tflite_strict(args.validate_only.read_bytes())
        print(f"[VALID] {args.validate_only}: {json.dumps(parsed)}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = get_sweep_specs(args.suite)
    manifest: Dict[str, Any] = {}

    for stem, fn in specs:
        tflite, in_bin, ref_bin, meta = fn()
        parsed = validate_tflite_strict(tflite)

        t_path = args.out_dir / f"{stem}.tflite"
        i_path = args.out_dir / f"{stem}.in.bin"
        r_path = args.out_dir / f"{stem}.ref.bin"
        t_path.write_bytes(tflite)
        i_path.write_bytes(in_bin)
        r_path.write_bytes(ref_bin)

        manifest[stem] = {
            "tflite_path": str(t_path),
            "tflite_bytes": len(tflite),
            "tflite_sha256": hashlib.sha256(tflite).hexdigest(),
            "in_path": str(i_path),
            "in_bytes": len(in_bin),
            "in_sha256": hashlib.sha256(in_bin).hexdigest(),
            "ref_path": str(r_path),
            "ref_bytes": len(ref_bin),
            "ref_sha256": hashlib.sha256(ref_bin).hexdigest(),
            "parsed": parsed,
            "meta": meta,
        }
        print(f"[OK] {stem:36s} tflite={len(tflite):5d}B in={len(in_bin):5d}B ref={len(ref_bin):5d}B ops={parsed['operators']}")

    man_path = args.out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Generated & strictly validated {len(manifest)} models -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
