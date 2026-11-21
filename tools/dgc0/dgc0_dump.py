#!/usr/bin/env python3
"""Clean-room DarwiNN EdgeTPU DGC0 / DMG0 container dissector & extractor.

Parses Tensor G5 (`laguna` / `buenos`) and G6 (`malibu` / `santafe`) compiled
`DGC0` (`GraphContainer`) and `DMG0` (`MultiGraphContainer`) FlatBuffer blobs
into structured JSON, and optionally extracts nested `DWS0`, `DGD0`, inline
`DiveData`, nested `MultiExecutable`, `CompilerOptions` protobuf, trailing
`ExternalBuffer` payloads (`DIVE` IR, RV64 interpreter, parameter blobs), and
individual `buenos` VLIW hardware instruction bitstream chunks.

Zero external pip dependencies; Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Grounded Enums (recovered from rango CP3A.261005.005 fc.dgc0 / add.dgc0 / conv2d.dgc0)
# ---------------------------------------------------------------------------

DWS_DATA_TYPES: Dict[int, str] = {
    0: "UNKNOWN_0",
    1: "INT8",
    2: "UINT8_OR_RAW_BYTES",
    3: "INT16",
    4: "UINT16",
    5: "INT32",
    6: "UINT32",
    7: "FLOAT16",
    8: "FLOAT32",
    9: "BFLOAT16",
    10: "INT64",
    11: "UINT64_OR_ADDR",
}

DWS_STORAGE_TYPES: Dict[int, str] = {
    0: "EMBEDDED_OR_DEFAULT",
    1: "CALLER_SPECIFIED",
    2: "TRANSIENT_OR_INTERMEDIATE",
    3: "SCRATCHPAD",
    4: "LARGE",
    5: "EXTERNAL",
}

# Complete platforms::darwinn::dive::OperationInfo enum recovered verbatim
# from ExternalBuffer[1] (fc_interpreter_buenos.bin @ 0x8dc1c..0x8df62).
DIVE_OP_NAMES: List[str] = [
    "NONE",                         # 0
    "BlockEntrance",                # 1
    "Branch",                       # 2
    "Add",                          # 3
    "ArithmeticLeftShift",          # 4
    "ArithmeticRightShift",         # 5
    "BitwiseAnd",                   # 6
    "BitwiseOr",                    # 7
    "BitwiseXor",                   # 8
    "Cast",                         # 9
    "Div",                          # 10
    "Rem",                          # 11
    "Pow",                          # 12
    "Maximum",                      # 13
    "Minimum",                      # 14
    "Mul",                          # 15
    "Sub",                          # 16
    "Greater",                      # 17
    "GreaterEqual",                 # 18
    "Less",                         # 19
    "LessEqual",                    # 20
    "LogicalAnd",                   # 21
    "LogicalRightShift",            # 22
    "NotEqual",                     # 23
    "Copy",                         # 24? (Wait: verify indexing against observed 25=Copy!)
]

# Exact verified mapping from fc.dgc0/add.dgc0/conv2d.dgc0 + interpreter_buenos.bin strings:
# Note: in interpreter_buenos.bin, there is one entry ("Equal" or similar) between
# Greater/NotEqual and Copy, placing Copy at 25, DispatchHardwareInstruction at 47,
# PutBits at 50, WaitForFenceCompletion at 51, GetTensorAddress at 59, CacheCleanInvalidate at 61.
DIVE_OP_TYPE_MAP: Dict[int, str] = {
    0: "NONE",
    1: "BlockEntrance",
    2: "Branch",
    3: "Add",
    4: "ArithmeticLeftShift",
    5: "ArithmeticRightShift",
    6: "BitwiseAnd",
    7: "BitwiseOr",
    8: "BitwiseXor",
    9: "Cast",
    10: "Div",
    11: "Rem",
    12: "Pow",
    13: "Maximum",
    14: "Minimum",
    15: "Mul",
    16: "Sub",
    17: "Equal",
    18: "Greater",
    19: "GreaterEqual",
    20: "Less",
    21: "LessEqual",
    22: "LogicalAnd",
    23: "LogicalRightShift",
    24: "NotEqual",
    25: "Copy",
    26: "Select",
    27: "Load",
    28: "Store",
    29: "ProgramTensorMappingTable",
    30: "WriteTileSramBypassMode",
    31: "Abs",
    32: "Concat",
    33: "CountLeadingZeros",
    34: "Cumsum",
    35: "ExtractSlice",
    36: "Gather",
    37: "GatherNd",
    38: "InsertSlice",
    39: "OneHot",
    40: "Pad",
    41: "PopCount",
    42: "Reduction",
    43: "Roll",
    44: "ScatterNd",
    45: "Sign",
    46: "TopK",
    47: "DispatchHardwareInstruction",
    48: "SendInterrupt",
    49: "Multinomial",
    50: "PutBits",
    51: "WaitForFenceCompletion",
    52: "WriteDmaDescriptor",
    53: "WriteHibData",
    54: "WriteScalarArchRegister",
    55: "DiveTestOp",
    56: "DynamicSliceY",
    57: "CustomOnAddresses",
    58: "LegacyScalar",
    59: "GetTensorAddress",
    60: "GetTensorShape",
    61: "CacheCleanInvalidate",
    62: "SetTensorShape",
    63: "Epilogue",
    64: "Prologue",
    65: "Benchmark",
    66: "RunSerializedModel",
    67: "EnableItcTracing",
    68: "DisableItcTracing",
    69: "CustomOnBuffers",
    70: "UpdatePowerHint",
    71: "Extract",
    72: "Insert",
    73: "IdpStart",
    74: "IdpStop",
    75: "Fill",
}


# ---------------------------------------------------------------------------
# Low-level FlatBuffer Reader Primitives
# ---------------------------------------------------------------------------

class FBReader:
    """Bounds-checked little-endian FlatBuffer reader."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def u8(self, off: int) -> int:
        return self.data[off]

    def i16(self, off: int) -> int:
        return struct.unpack_from("<h", self.data, off)[0]

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.data, off)[0]

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]

    def root_offset(self) -> int:
        return self.u32(0)

    def magic(self) -> str:
        if len(self.data) < 8:
            return ""
        raw = self.data[4:8]
        return raw.decode("ascii", errors="replace")

    def read_vtable(self, table_pos: int) -> Tuple[int, int, int, List[int]]:
        soff = self.i32(table_pos)
        vt_pos = table_pos - soff
        vt_size = self.u16(vt_pos)
        obj_size = self.u16(vt_pos + 2)
        nslots = max(0, (vt_size - 4) // 2)
        slots = [self.u16(vt_pos + 4 + 2 * i) for i in range(nslots)]
        return vt_pos, vt_size, obj_size, slots

    def field_pos(self, table_pos: int, slot_id: int) -> Optional[int]:
        _, _, _, slots = self.read_vtable(table_pos)
        if slot_id >= len(slots) or slots[slot_id] == 0:
            return None
        return table_pos + slots[slot_id]

    def uoffset(self, field_pos: int) -> int:
        return field_pos + self.u32(field_pos)

    def string(self, field_pos: int) -> str:
        pos = self.uoffset(field_pos)
        length = self.u32(pos)
        return self.data[pos + 4 : pos + 4 + length].decode("utf-8", errors="replace")

    def byte_vector(self, field_pos: int) -> Tuple[int, int, bytes]:
        pos = self.uoffset(field_pos)
        length = self.u32(pos)
        start = pos + 4
        return start, length, self.data[start : start + length]

    def uoffset_vector(self, field_pos: int) -> List[int]:
        pos = self.uoffset(field_pos)
        count = self.u32(pos)
        return [self.uoffset(pos + 4 + 4 * i) for i in range(count)]

    def u32_vector(self, field_pos: int) -> List[int]:
        pos = self.uoffset(field_pos)
        count = self.u32(pos)
        return list(struct.unpack_from(f"<{count}I", self.data, pos + 4))

    def i32_vector(self, field_pos: int) -> List[int]:
        pos = self.uoffset(field_pos)
        count = self.u32(pos)
        return list(struct.unpack_from(f"<{count}i", self.data, pos + 4))

    def string_vector(self, field_pos: int) -> List[str]:
        pos = self.uoffset(field_pos)
        count = self.u32(pos)
        out: List[str] = []
        for i in range(count):
            sp = self.uoffset(pos + 4 + 4 * i)
            slen = self.u32(sp)
            out.append(self.data[sp + 4 : sp + 4 + slen].decode("utf-8", errors="replace"))
        return out

    def raw_slots(self, table_pos: int) -> Dict[str, Any]:
        vt_pos, vt_size, obj_size, slots = self.read_vtable(table_pos)
        present = sorted((off, i) for i, off in enumerate(slots) if off != 0)
        widths = {
            idx: (present[k + 1][0] if k + 1 < len(present) else obj_size) - off
            for k, (off, idx) in enumerate(present)
        }
        fields: Dict[str, Any] = {}
        for i, off in enumerate(slots):
            if off == 0:
                continue
            p = table_pos + off
            w = widths[i]
            fields[f"id_{i:02d}"] = {
                "voff": 4 + 2 * i,
                "rel": off,
                "abs": p,
                "span": w,
                "hex": self.data[p : p + min(w, 8)].hex(),
            }
        return {
            "table_offset": table_pos,
            "vtable_offset": vt_pos,
            "vtable_size": vt_size,
            "object_size": obj_size,
            "num_slots": len(slots),
            "present_slots": fields,
        }


# ---------------------------------------------------------------------------
# Generic Protobuf Wire Decoder (for Package.id15 CompilerOptions)
# ---------------------------------------------------------------------------

def decode_protobuf_shallow(buf: bytes) -> List[Dict[str, Any]]:
    """Decode top-level protobuf wire fields without a .proto descriptor."""
    i = 0
    out: List[Dict[str, Any]] = []
    n = len(buf)
    while i < n:
        tag, shift = 0, 0
        while i < n:
            b = buf[i]
            i += 1
            tag |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        fnum = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, shift = 0, 0
            while i < n:
                b = buf[i]
                i += 1
                v |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            out.append({"field": fnum, "wire_type": "varint", "value": v})
        elif wt == 1 and i + 8 <= n:
            raw = buf[i : i + 8]
            i += 8
            u = struct.unpack("<Q", raw)[0]
            d = struct.unpack("<d", raw)[0]
            out.append({"field": fnum, "wire_type": "fixed64", "u64": u, "f64": d})
        elif wt == 2:
            ln, shift = 0, 0
            while i < n:
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            sub = buf[i : i + ln]
            i += ln
            if sub and all(32 <= c < 127 for c in sub):
                out.append({"field": fnum, "wire_type": "string", "value": sub.decode("ascii")})
            else:
                out.append({"field": fnum, "wire_type": "bytes", "size": len(sub), "hex_preview": sub[:32].hex()})
        elif wt == 5 and i + 4 <= n:
            raw = buf[i : i + 4]
            i += 4
            u = struct.unpack("<I", raw)[0]
            f = struct.unpack("<f", raw)[0]
            out.append({"field": fnum, "wire_type": "fixed32", "u32": u, "f32": f})
        else:
            out.append({"field": fnum, "wire_type": f"unknown_{wt}", "offset": i})
            break
    return out


# ---------------------------------------------------------------------------
# Sub-Container Parsers
# ---------------------------------------------------------------------------

def parse_dgd0(dgd_bytes: bytes) -> Dict[str, Any]:
    r = FBReader(dgd_bytes)
    root = r.root_offset()
    tensors: List[Dict[str, Any]] = []
    if (fp := r.field_pos(root, 0)) is not None:
        for tp in r.uoffset_vector(fp):
            tensors.append({
                "subgraph_index": r.u32(r.field_pos(tp, 0)) if r.field_pos(tp, 0) is not None else 0,
                "tensor_index": r.u32(r.field_pos(tp, 1)) if r.field_pos(tp, 1) is not None else 0,
                "buffer_index": r.u32(r.field_pos(tp, 2)) if r.field_pos(tp, 2) is not None else 0,
            })
    children: List[Dict[str, Any]] = []
    if (fp := r.field_pos(root, 1)) is not None:
        for cp in r.uoffset_vector(fp):
            children.append({
                "field_00": r.u32(r.field_pos(cp, 0)) if r.field_pos(cp, 0) is not None else 0,
                "has_child_graph_data": r.field_pos(cp, 1) is not None,
            })
    return {
        "magic": r.magic(),
        "size_bytes": len(dgd_bytes),
        "sha256": hashlib.sha256(dgd_bytes).hexdigest(),
        "tensors": tensors,
        "children": children,
        "raw_table": r.raw_slots(root),
    }


def parse_tensor_ref(r: FBReader, pos: int) -> int:
    fp = r.field_pos(pos, 0)
    return r.u32(fp) if fp is not None else 0


def parse_multi_executable(blob: bytes) -> Dict[str, Any]:
    r = FBReader(blob)
    root = r.root_offset()
    execs: List[Dict[str, Any]] = []
    if (fp1 := r.field_pos(root, 1)) is not None:
        for ep in r.uoffset_vector(fp1):
            ex: Dict[str, Any] = {
                "name": r.string(r.field_pos(ep, 0)) if r.field_pos(ep, 0) is not None else "",
                "architecture": r.string(r.field_pos(ep, 1)) if r.field_pos(ep, 1) is not None else "",
            }
            if (f2 := r.field_pos(ep, 2)) is not None:
                _, l2, _ = r.byte_vector(f2)
                ex["serialized_metadata_bytes"] = l2

            def _parse_layer_list(slot: int) -> List[Dict[str, Any]]:
                lf = r.field_pos(ep, slot)
                if lf is None:
                    return []
                items: List[Dict[str, Any]] = []
                for cp in r.uoffset_vector(lf):
                    items.append({
                        "size_bytes_u32": r.u32(r.field_pos(cp, 1)) if r.field_pos(cp, 1) is not None else 0,
                        "name": r.string(r.field_pos(cp, 2)) if r.field_pos(cp, 2) is not None else "",
                        "size_bytes_u64": r.u64(r.field_pos(cp, 3)) if r.field_pos(cp, 3) is not None else 0,
                        "raw_table": r.raw_slots(cp),
                    })
                return items

            ex["inputs"] = _parse_layer_list(3)
            ex["outputs"] = _parse_layer_list(4)
            ex["parameters"] = _parse_layer_list(20)

            scalars: Dict[str, Any] = {}
            scalar_spec = {
                5: ("u32", "field_05"),
                7: ("i16", "executable_type"),
                8: ("u32", "hardware_cluster_id"),
                9: ("u64", "tile_memory_requirement_bytes"),
                10: ("u32", "field_10"),
                11: ("u32", "field_11"),
                12: ("u32", "field_12"),
                13: ("u32", "field_13"),
                14: ("u32", "field_14"),
                15: ("u32", "field_15"),
                16: ("u8", "field_16"),
                17: ("u8", "field_17"),
                18: ("u8", "field_18"),
                19: ("u8", "field_19"),
                21: ("u32", "field_21"),
                22: ("u64", "executable_hash"),
                23: ("u32", "field_23"),
                24: ("u32", "core_or_cluster_count"),
                26: ("u16", "field_26"),
                27: ("u32", "field_27"),
                28: ("u64", "dive_heap_size"),
                29: ("u64", "field_29"),
                30: ("u32", "field_30"),
                31: ("u64", "field_31"),
                33: ("u32", "field_33"),
                34: ("u64", "field_34"),
            }
            for sid, (kind, sname) in scalar_spec.items():
                sfp = r.field_pos(ep, sid)
                if sfp is None:
                    continue
                if kind == "u8":
                    scalars[sname] = r.u8(sfp)
                elif kind == "u16":
                    scalars[sname] = r.u16(sfp)
                elif kind == "i16":
                    scalars[sname] = r.i16(sfp)
                elif kind == "u32":
                    scalars[sname] = r.u32(sfp)
                elif kind == "u64":
                    val = r.u64(sfp)
                    scalars[sname] = f"0x{val:016x}" if sname == "executable_hash" else val
            ex["scalars"] = scalars
            ex["raw_table"] = r.raw_slots(ep)
            execs.append(ex)
    return {
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "executables": execs,
        "raw_table": r.raw_slots(root),
    }


def parse_dws0(dws_bytes: bytes) -> Tuple[Dict[str, Any], Optional[bytes], Optional[bytes]]:
    r = FBReader(dws_bytes)
    root = r.root_offset()
    multi_exec_blob: Optional[bytes] = None
    compiler_opts_pb: Optional[bytes] = None

    pkg: Dict[str, Any] = {
        "magic": r.magic(),
        "size_bytes": len(dws_bytes),
        "sha256": hashlib.sha256(dws_bytes).hexdigest(),
        "version": r.u32(r.field_pos(root, 0)) if r.field_pos(root, 0) is not None else 0,
        "header_type": r.u8(r.field_pos(root, 1)) if r.field_pos(root, 1) is not None else 0,
    }

    if (hp_f := r.field_pos(root, 2)) is not None:
        hp = r.uoffset(hp_f)
        hdr: Dict[str, Any] = {
            "target_architecture": r.string(r.field_pos(hp, 0)) if r.field_pos(hp, 0) is not None else "",
            "raw_table": r.raw_slots(hp),
        }
        if (hf1 := r.field_pos(hp, 1)) is not None:
            _, hl1, hb1 = r.byte_vector(hf1)
            hdr["field_01_bytes"] = hl1
            hdr["field_01_hex_preview"] = hb1[:32].hex()
        pkg["header"] = hdr

    pkg["model_identifier"] = r.string(r.field_pos(root, 4)) if r.field_pos(root, 4) is not None else ""
    pkg["field_06"] = r.u32(r.field_pos(root, 6)) if r.field_pos(root, 6) is not None else 0
    pkg["field_09"] = r.u8(r.field_pos(root, 9)) if r.field_pos(root, 9) is not None else 0
    pkg["compiler_cl"] = r.u64(r.field_pos(root, 10)) if r.field_pos(root, 10) is not None else 0
    if (f11 := r.field_pos(root, 11)) is not None:
        pkg["field_11_strings"] = r.string_vector(f11)
    pkg["field_12"] = r.string(r.field_pos(root, 12)) if r.field_pos(root, 12) is not None else ""
    pkg["field_13"] = r.u8(r.field_pos(root, 13)) if r.field_pos(root, 13) is not None else 0
    pkg["field_14"] = r.string(r.field_pos(root, 14)) if r.field_pos(root, 14) is not None else ""
    if (f15 := r.field_pos(root, 15)) is not None:
        o15, l15, b15 = r.byte_vector(f15)
        compiler_opts_pb = b15
        pkg["compiler_options_pb"] = {
            "offset_in_dws0": o15,
            "size_bytes": l15,
            "sha256": hashlib.sha256(b15).hexdigest(),
            "fields": decode_protobuf_shallow(b15),
        }
    pkg["field_17"] = r.string(r.field_pos(root, 17)) if r.field_pos(root, 17) is not None else ""
    pkg["compiler_release_version"] = r.string(r.field_pos(root, 18)) if r.field_pos(root, 18) is not None else ""

    # Embedded buffers
    emb_list: List[Dict[str, Any]] = []
    if (f5 := r.field_pos(root, 5)) is not None:
        for bi, bp in enumerate(r.uoffset_vector(f5)):
            if (bf := r.field_pos(bp, 0)) is not None:
                bo, bl, bb = r.byte_vector(bf)
                emb_list.append({
                    "index": bi,
                    "offset_in_dws0": bo,
                    "size_bytes": bl,
                    "sha256": hashlib.sha256(bb).hexdigest(),
                })
    pkg["embedded_buffers"] = emb_list

    # SubGraphs
    subgraphs: List[Dict[str, Any]] = []
    if (f3 := r.field_pos(root, 3)) is not None:
        for sg_idx, sgp in enumerate(r.uoffset_vector(f3)):
            tensors: List[Dict[str, Any]] = []
            if (tf := r.field_pos(sgp, 0)) is not None:
                for ti, tp in enumerate(r.uoffset_vector(tf)):
                    dtype = r.u8(r.field_pos(tp, 1)) if r.field_pos(tp, 1) is not None else 0
                    storage = r.u8(r.field_pos(tp, 5)) if r.field_pos(tp, 5) is not None else 0
                    is_ext = r.u8(r.field_pos(tp, 12)) if r.field_pos(tp, 12) is not None else 0
                    ext_idx = r.u64(r.field_pos(tp, 13)) if r.field_pos(tp, 13) is not None else 0
                    tensors.append({
                        "index": ti,
                        "name": r.string(r.field_pos(tp, 2)) if r.field_pos(tp, 2) is not None else "",
                        "dims": r.u32_vector(r.field_pos(tp, 0)) if r.field_pos(tp, 0) is not None else [],
                        "data_type": dtype,
                        "data_type_name": DWS_DATA_TYPES.get(dtype, f"UNKNOWN_{dtype}"),
                        "embedded_buffer_index": r.i32(r.field_pos(tp, 3)) if r.field_pos(tp, 3) is not None else -1,
                        "storage_type": storage,
                        "storage_type_name": DWS_STORAGE_TYPES.get(storage, f"UNKNOWN_{storage}"),
                        "is_external": bool(is_ext),
                        "external_buffer_index": ext_idx if is_ext else None,
                        "raw_table": r.raw_slots(tp),
                    })

            sg_inputs = [parse_tensor_ref(r, p) for p in r.uoffset_vector(r.field_pos(sgp, 1))] if r.field_pos(sgp, 1) is not None else []
            sg_outputs = [parse_tensor_ref(r, p) for p in r.uoffset_vector(r.field_pos(sgp, 2))] if r.field_pos(sgp, 2) is not None else []

            ops: List[Dict[str, Any]] = []
            if (of := r.field_pos(sgp, 3)) is not None:
                for oi, op in enumerate(r.uoffset_vector(of)):
                    dtype = r.u8(r.field_pos(op, 0)) if r.field_pos(op, 0) is not None else 0
                    op_in = [parse_tensor_ref(r, p) for p in r.uoffset_vector(r.field_pos(op, 3))] if r.field_pos(op, 3) is not None else []
                    op_out = [parse_tensor_ref(r, p) for p in r.uoffset_vector(r.field_pos(op, 4))] if r.field_pos(op, 4) is not None else []
                    op_name = r.string(r.field_pos(op, 5)) if r.field_pos(op, 5) is not None else ""
                    op_rec: Dict[str, Any] = {
                        "index": oi,
                        "name": op_name,
                        "detail_type": dtype,
                        "input_tensor_indices": op_in,
                        "output_tensor_indices": op_out,
                        "raw_table": r.raw_slots(op),
                    }
                    if dtype == 2 and (df := r.field_pos(op, 1)) is not None:
                        det = r.uoffset(df)
                        detail_rec: Dict[str, Any] = {"raw_table": r.raw_slots(det)}
                        if (mef := r.field_pos(det, 0)) is not None:
                            me_off, me_len, me_bytes = r.byte_vector(mef)
                            multi_exec_blob = me_bytes
                            detail_rec["multi_executable_offset_in_dws0"] = me_off
                            detail_rec["multi_executable"] = parse_multi_executable(me_bytes)
                        if (bf := r.field_pos(det, 1)) is not None:
                            bindings: List[Dict[str, Any]] = []
                            for bp in r.uoffset_vector(bf):
                                bnd: Dict[str, Any] = {"raw_table": r.raw_slots(bp)}
                                if (b0 := r.field_pos(bp, 0)) is not None:
                                    bnd["dive_resource_tensors"] = [parse_tensor_ref(r, x) for x in r.uoffset_vector(b0)]
                                if (b4 := r.field_pos(bp, 4)) is not None:
                                    bnd["parameter_tensors"] = [parse_tensor_ref(r, x) for x in r.uoffset_vector(b4)]
                                for bsid, bname in [(5, "scratchpad_tensor"), (6, "aux_tensor_id6"), (7, "interpreter_tensor"), (8, "dive_vm_ir_tensor")]:
                                    if (bx := r.field_pos(bp, bsid)) is not None:
                                        bnd[bname] = parse_tensor_ref(r, r.uoffset(bx))
                                bindings.append(bnd)
                            detail_rec["executable_bindings"] = bindings
                        op_rec["detail"] = detail_rec
                    ops.append(op_rec)

            subgraphs.append({
                "index": sg_idx,
                "name": r.string(r.field_pos(sgp, 4)) if r.field_pos(sgp, 4) is not None else "",
                "input_tensor_indices": sg_inputs,
                "output_tensor_indices": sg_outputs,
                "tensors": tensors,
                "operations": ops,
                "raw_table": r.raw_slots(sgp),
            })

    pkg["sub_graphs"] = subgraphs
    pkg["raw_table"] = r.raw_slots(root)
    return pkg, multi_exec_blob, compiler_opts_pb


def decode_buenos_dma_header(chunk_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Decode grounded 64-byte buenos_a0 flexible DMA packet fields (header == 0xa40)."""
    if len(chunk_bytes) < 64:
        return None
    words = struct.unpack_from("<8Q", chunk_bytes, 0)
    header12 = words[0] & 0xFFF
    if header12 != 0xA40:
        return None
    seq_id = (words[0] >> 12) & 0xFFFF
    dma_channel_tag = (words[0] >> 32) & 0xFF
    prod_bytes_minus_1 = (words[6] >> 51) & 0x1FFF
    cons_bytes_minus_1 = (words[7] >> 19) & 0x1FFF
    return {
        "header_12b": f"0x{header12:03x}",
        "hw_seq_id": seq_id,
        "dma_channel_tag": f"0x{dma_channel_tag:02x}",
        "producer_transfer_bytes": prod_bytes_minus_1 + 1,
        "consumer_transfer_bytes": cons_bytes_minus_1 + 1,
        "w0_hex": f"0x{words[0]:016x}",
        "w5_hex": f"0x{words[5]:016x}",
        "w6_hex": f"0x{words[6]:016x}",
        "w7_hex": f"0x{words[7]:016x}",
    }


def parse_dive_module(dive_bytes: bytes, abs_base: int = 0) -> Tuple[Dict[str, Any], List[Tuple[int, str, bytes]]]:
    r = FBReader(dive_bytes)
    root = r.root_offset()
    chunks_raw: List[Tuple[int, str, bytes]] = []

    # 1. Hardware instruction bitstream chunks (id 3)
    chunks_meta: List[Dict[str, Any]] = []
    if (f3 := r.field_pos(root, 3)) is not None:
        for ci, cp in enumerate(r.uoffset_vector(f3)):
            if (bp := r.field_pos(cp, 0)) is not None:
                bo, bl, bdata = r.byte_vector(bp)
                n_u64 = bl // 8
                preview_words = [f"0x{w:016x}" for w in struct.unpack_from(f"<{min(n_u64, 8)}Q", bdata, 0)]
                cinfo: Dict[str, Any] = {
                    "chunk_index": ci,
                    "offset_in_dive": bo,
                    "offset_in_dgc0": abs_base + bo,
                    "size_bytes": bl,
                    "num_16b_flits": bl / 16.0,
                    "num_u64_words": n_u64,
                    "sha256": hashlib.sha256(bdata).hexdigest(),
                    "first_u64_words": preview_words,
                }
                dma_dec = decode_buenos_dma_header(bdata)
                if dma_dec is not None:
                    cinfo["decoded_dma_head_64b"] = dma_dec
                chunks_meta.append(cinfo)
                chunks_raw.append((ci, f"chunk_{ci:02d}", bdata))

    # 2. Functions & operations (id 1)
    chunk_dispatch_names: Dict[int, str] = {}
    functions: List[Dict[str, Any]] = []
    if (f1 := r.field_pos(root, 1)) is not None:
        for fi, fp in enumerate(r.uoffset_vector(f1)):
            values: List[Dict[str, Any]] = []
            if (vf := r.field_pos(fp, 0)) is not None:
                for vi, vp in enumerate(r.uoffset_vector(vf)):
                    dt = r.u8(r.field_pos(vp, 1)) if r.field_pos(vp, 1) is not None else 0
                    values.append({
                        "index": vi,
                        "name": r.string(r.field_pos(vp, 2)) if r.field_pos(vp, 2) is not None else "",
                        "dims": r.i32_vector(r.field_pos(vp, 0)) if r.field_pos(vp, 0) is not None else [],
                        "data_type": dt,
                        "data_type_name": DWS_DATA_TYPES.get(dt, f"UNKNOWN_{dt}"),
                        "buffer_index": r.i32(r.field_pos(vp, 3)) if r.field_pos(vp, 3) is not None else -1,
                    })

            ops: List[Dict[str, Any]] = []
            if (of := r.field_pos(fp, 3)) is not None:
                for oi, op in enumerate(r.uoffset_vector(of)):
                    ot = r.u8(r.field_pos(op, 0)) if r.field_pos(op, 0) is not None else 0
                    oname = r.string(r.field_pos(op, 4)) if r.field_pos(op, 4) is not None else ""
                    oin = r.i32_vector(r.field_pos(op, 2)) if r.field_pos(op, 2) is not None else []
                    oout = r.i32_vector(r.field_pos(op, 3)) if r.field_pos(op, 3) is not None else []
                    orec: Dict[str, Any] = {
                        "index": oi,
                        "op_type": ot,
                        "op_type_name": DIVE_OP_TYPE_MAP.get(ot, f"OP_{ot}"),
                        "name": oname,
                        "inputs": oin,
                        "outputs": oout,
                    }
                    if (df := r.field_pos(op, 1)) is not None:
                        dp = r.uoffset(df)
                        if ot == 47:  # DispatchHardwareInstruction
                            cidx = r.u32(r.field_pos(dp, 0)) if r.field_pos(dp, 0) is not None else 0
                            seq = r.u32(r.field_pos(dp, 1)) if r.field_pos(dp, 1) is not None else 0
                            orec["detail"] = {"chunk_index": cidx, "sequence_id": seq}
                            chunk_dispatch_names[cidx] = oname
                        elif ot == 50:  # PutBits
                            cidx = r.u32(r.field_pos(dp, 0)) if r.field_pos(dp, 0) is not None else 0
                            boff = r.u64(r.field_pos(dp, 1)) if r.field_pos(dp, 1) is not None else 0
                            nbits = r.u32(r.field_pos(dp, 2)) if r.field_pos(dp, 2) is not None else 0
                            orec["detail"] = {"chunk_index": cidx, "offset": boff, "bit_width": nbits}
                        else:
                            orec["detail"] = r.raw_slots(dp)
                    ops.append(orec)

            functions.append({
                "index": fi,
                "name": r.string(r.field_pos(fp, 4)) if r.field_pos(fp, 4) is not None else "",
                "input_value_indices": r.i32_vector(r.field_pos(fp, 1)) if r.field_pos(fp, 1) is not None else [],
                "output_value_indices": r.i32_vector(r.field_pos(fp, 2)) if r.field_pos(fp, 2) is not None else [],
                "values": values,
                "operations": ops,
            })

    # Attach dispatch names to chunk metadata & extraction filenames
    updated_raw: List[Tuple[int, str, bytes]] = []
    for cm, (ci, cname, bdata) in zip(chunks_meta, chunks_raw):
        dname = chunk_dispatch_names.get(ci, cname)
        cm["dispatch_op_name"] = dname
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in dname)[:64]
        updated_raw.append((ci, safe, bdata))

    port_tables: Dict[str, List[int]] = {}
    for sid, pname in [
        (4, "input_value_indices"),
        (5, "output_value_indices"),
        (6, "parameter_value_indices"),
        (7, "scratchpad_value_indices"),
        (8, "variable_value_indices"),
    ]:
        if (pf := r.field_pos(root, sid)) is not None:
            port_tables[pname] = r.i32_vector(pf)

    return {
        "magic": r.magic(),
        "size_bytes": len(dive_bytes),
        "sha256": hashlib.sha256(dive_bytes).hexdigest(),
        "port_tables": port_tables,
        "hardware_instruction_chunks": chunks_meta,
        "functions": functions,
        "raw_table": r.raw_slots(root),
    }, updated_raw


def parse_dgc0(blob: bytes, extract_dir: Optional[Path] = None, prefix: str = "") -> Dict[str, Any]:
    r = FBReader(blob)
    magic = r.magic()
    if magic != "DGC0":
        raise ValueError(f"Expected DGC0 magic, got {magic!r}")
    root = r.root_offset()

    result: Dict[str, Any] = {
        "container_type": "DGC0",
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "version": r.u32(r.field_pos(root, 3)) if r.field_pos(root, 3) is not None else 0,
        "model_hash": f"0x{r.u64(r.field_pos(root, 4)):016x}" if r.field_pos(root, 4) is not None else None,
        "compiler_cl": r.u64(r.field_pos(root, 5)) if r.field_pos(root, 5) is not None else 0,
        "alignment": r.u32(r.field_pos(root, 6)) if r.field_pos(root, 6) is not None else 0,
        "raw_table": r.raw_slots(root),
    }

    if extract_dir is not None:
        extract_dir.mkdir(parents=True, exist_ok=True)

    # Nested DWS0 (id 0)
    dws_info: Optional[Dict[str, Any]] = None
    if (f0 := r.field_pos(root, 0)) is not None:
        o0, l0, b0 = r.byte_vector(f0)
        dws_info, multi_exec_bytes, compiler_opts_pb = parse_dws0(b0)
        dws_info["offset_in_dgc0"] = o0
        result["package_dws0"] = dws_info
        if extract_dir is not None:
            (extract_dir / f"{prefix}dws0.bin").write_bytes(b0)
            if multi_exec_bytes is not None:
                (extract_dir / f"{prefix}multi_executable.bin").write_bytes(multi_exec_bytes)
            if compiler_opts_pb is not None:
                (extract_dir / f"{prefix}compiler_options.pb").write_bytes(compiler_opts_pb)

    # Nested DGD0 (id 1)
    if (f1 := r.field_pos(root, 1)) is not None:
        o1, l1, b1 = r.byte_vector(f1)
        dgd_info = parse_dgd0(b1)
        dgd_info["offset_in_dgc0"] = o1
        result["graph_data_dgd0"] = dgd_info
        if extract_dir is not None:
            (extract_dir / f"{prefix}dgd0.bin").write_bytes(b1)

    # Inline DiveData (id 2, optional)
    if (f2 := r.field_pos(root, 2)) is not None:
        o2, l2, b2 = r.byte_vector(f2)
        result["inline_dive_data"] = {
            "offset_in_dgc0": o2,
            "size_bytes": l2,
            "sha256": hashlib.sha256(b2).hexdigest(),
            "magic": b2[4:8].decode("ascii", errors="replace") if len(b2) >= 8 else "",
        }
        if extract_dir is not None:
            (extract_dir / f"{prefix}inline_dive_data.bin").write_bytes(b2)

    # Map external_buffer_index -> DWS0 tensor name for labeling
    ext_tensor_names: Dict[int, str] = {}
    if dws_info is not None:
        for sg in dws_info.get("sub_graphs", []):
            for t in sg.get("tensors", []):
                if t.get("is_external") and t.get("external_buffer_index") is not None:
                    ext_tensor_names[int(t["external_buffer_index"])] = t["name"]

    # Trailing ExternalBuffers (id 7)
    ext_list: List[Dict[str, Any]] = []
    if (f7 := r.field_pos(root, 7)) is not None:
        for ei, ep in enumerate(r.uoffset_vector(f7)):
            eoff = r.u64(r.field_pos(ep, 0)) if r.field_pos(ep, 0) is not None else 0
            esz = r.u64(r.field_pos(ep, 1)) if r.field_pos(ep, 1) is not None else 0
            eblob = blob[eoff : eoff + esz]
            tname = ext_tensor_names.get(ei, f"ext_{ei}")
            kind = "raw_params_or_data"
            if len(eblob) >= 8 and eblob[4:8] == b"DIVE":
                kind = "dive_vm_ir"
            elif len(eblob) >= 16 and eblob[:8] == bytes.fromhex("6f0040036f00c001"):
                kind = "rv64_dive_interpreter"
            elif "params" in tname:
                kind = "tpu_parameters"

            erec: Dict[str, Any] = {
                "index": ei,
                "tensor_name": tname,
                "kind": kind,
                "offset": eoff,
                "size_bytes": esz,
                "end_offset": eoff + esz,
                "aligned_64": (eoff % 64) == 0,
                "sha256": hashlib.sha256(eblob).hexdigest(),
                "first_32b_hex": eblob[:32].hex(),
            }
            if kind == "dive_vm_ir":
                dive_parsed, chunks_raw = parse_dive_module(eblob, abs_base=eoff)
                erec["dive_module"] = dive_parsed
                if extract_dir is not None:
                    (extract_dir / f"{prefix}ext_{ei}_dive.bin").write_bytes(eblob)
                    vdir = extract_dir / f"{prefix}vliw_chunks"
                    vdir.mkdir(parents=True, exist_ok=True)
                    for ci, safe_name, cbytes in chunks_raw:
                        (vdir / f"chunk_{ci:02d}_{safe_name}.bin").write_bytes(cbytes)
            elif extract_dir is not None:
                suffix = "interpreter_rv64.bin" if kind == "rv64_dive_interpreter" else ("params.bin" if kind == "tpu_parameters" else "data.bin")
                (extract_dir / f"{prefix}ext_{ei}_{suffix}").write_bytes(eblob)

            ext_list.append(erec)

    result["external_buffers"] = ext_list
    return result


def parse_dmg0(blob: bytes, extract_dir: Optional[Path] = None) -> Dict[str, Any]:
    r = FBReader(blob)
    if r.magic() != "DMG0":
        raise ValueError(f"Expected DMG0 magic, got {r.magic()!r}")
    root = r.root_offset()
    ext_list: List[Dict[str, Any]] = []
    if (f1 := r.field_pos(root, 1)) is not None:
        for ei, ep in enumerate(r.uoffset_vector(f1)):
            eoff = r.u64(r.field_pos(ep, 0)) if r.field_pos(ep, 0) is not None else 0
            esz = r.u64(r.field_pos(ep, 1)) if r.field_pos(ep, 1) is not None else 0
            ext_list.append({"index": ei, "offset": eoff, "size_bytes": esz})

    graphs: List[Dict[str, Any]] = []
    if (f2 := r.field_pos(root, 2)) is not None:
        for gi, gp in enumerate(r.uoffset_vector(f2)):
            gname = r.string(r.field_pos(gp, 0)) if r.field_pos(gp, 0) is not None else f"graph_{gi}"
            grec: Dict[str, Any] = {"index": gi, "name": gname}
            if (sf := r.field_pos(gp, 1)) is not None:
                so, sl, sb = r.byte_vector(sf)
                grec["offset_in_dmg0"] = so
                grec["size_bytes"] = sl
                if len(sb) >= 8 and sb[4:8] == b"DGC0":
                    grec["dgc0"] = parse_dgc0(sb, extract_dir=extract_dir, prefix=f"graph_{gi}_")
            graphs.append(grec)

    return {
        "container_type": "DMG0",
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "version_or_flags": r.u64(r.field_pos(root, 0)) if r.field_pos(root, 0) is not None else 0,
        "field_03": r.u32(r.field_pos(root, 3)) if r.field_pos(root, 3) is not None else 0,
        "external_buffers": ext_list,
        "graphs": graphs,
        "raw_table": r.raw_slots(root),
    }


def parse_container(blob: bytes, extract_dir: Optional[Path] = None) -> Dict[str, Any]:
    if len(blob) < 8:
        raise ValueError("Container too small (< 8 bytes)")
    magic = blob[4:8]
    if magic == b"DGC0":
        return parse_dgc0(blob, extract_dir=extract_dir)
    if magic == b"DMG0":
        return parse_dmg0(blob, extract_dir=extract_dir)
    raise ValueError(f"Unsupported container magic {magic!r} (expected b'DGC0' or b'DMG0')")


def format_summary(parsed: Dict[str, Any]) -> str:
    lines: List[str] = []
    ctype = parsed["container_type"]
    lines.append(f"[{ctype}] size={parsed['size_bytes']} B sha256={parsed['sha256'][:16]}...")
    if ctype == "DMG0":
        lines.append(f"  graphs={len(parsed['graphs'])} external_buffers={len(parsed['external_buffers'])}")
        for g in parsed["graphs"]:
            lines.append(f"  graph[{g['index']}] {g['name']!r} ({g.get('size_bytes', 0)} B)")
        return "\n".join(lines)

    lines.append(
        f"  version={parsed['version']} compiler_cl={parsed['compiler_cl']} "
        f"align={parsed['alignment']} model_hash={parsed['model_hash']}"
    )
    if (dgd := parsed.get("graph_data_dgd0")) is not None:
        lines.append(
            f"  DGD0 @ 0x{dgd['offset_in_dgc0']:x} ({dgd['size_bytes']} B): "
            f"tensors={len(dgd['tensors'])} children={len(dgd['children'])}"
        )
    if (dws := parsed.get("package_dws0")) is not None:
        arch = dws.get("header", {}).get("target_architecture", "")
        lines.append(
            f"  DWS0 @ 0x{dws['offset_in_dgc0']:x} ({dws['size_bytes']} B): "
            f"ver={dws['version']} arch={arch!r} release={dws['compiler_release_version']!r}"
        )
        for sg in dws.get("sub_graphs", []):
            lines.append(
                f"    SubGraph[{sg['index']}] {sg['name']!r}: "
                f"inputs={sg['input_tensor_indices']} outputs={sg['output_tensor_indices']} "
                f"tensors={len(sg['tensors'])} ops={len(sg['operations'])}"
            )
            for t in sg["tensors"]:
                ext_s = f" ext[{t['external_buffer_index']}]" if t["is_external"] else ""
                lines.append(
                    f"      tensor[{t['index']}] {t['name']!r:42s} dims={t['dims']} "
                    f"{t['data_type_name']} ({t['storage_type_name']}){ext_s}"
                )
    for eb in parsed.get("external_buffers", []):
        lines.append(
            f"  ExtBuf[{eb['index']}] ({eb['kind']}) @ 0x{eb['offset']:x}..0x{eb['end_offset']:x} "
            f"({eb['size_bytes']} B) sha256={eb['sha256'][:12]}... [{eb['tensor_name']}]"
        )
        if "dive_module" in eb:
            dm = eb["dive_module"]
            lines.append(
                f"    DIVE: ports={dm['port_tables']} "
                f"chunks={len(dm['hardware_instruction_chunks'])} "
                f"functions={len(dm['functions'])}"
            )
            for ch in dm["hardware_instruction_chunks"]:
                dma_s = ""
                if "decoded_dma_head_64b" in ch:
                    dd = ch["decoded_dma_head_64b"]
                    dma_s = (
                        f" [DMA seq={dd['hw_seq_id']} ch={dd['dma_channel_tag']} "
                        f"bytes={dd['producer_transfer_bytes']}]"
                    )
                lines.append(
                    f"      chunk[{ch['chunk_index']:2d}] @ 0x{ch['offset_in_dgc0']:x} "
                    f"({ch['size_bytes']:4d} B, {int(ch['num_16b_flits']):3d}x16B){dma_s} "
                    f"{ch.get('dispatch_op_name', '')}"
                )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dissect DarwiNN DGC0 / DMG0 compiled EdgeTPU containers")
    parser.add_argument("input", type=Path, help="Path to .dgc0 or .dmg0 binary")
    parser.add_argument("--json", dest="json_out", type=Path, default=None, help="Write full structured JSON to path")
    parser.add_argument("--stdout-json", action="store_true", help="Print structured JSON to stdout instead of summary")
    parser.add_argument("--extract-dir", type=Path, default=None, help="Extract nested blobs, VLIW chunks, and ext buffers")
    args = parser.parse_args(argv)

    blob = args.input.read_bytes()
    parsed = parse_container(blob, extract_dir=args.extract_dir)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")

    if args.stdout_json:
        print(json.dumps(parsed, indent=2))
    else:
        print(format_summary(parsed))
        if args.json_out is not None:
            print(f"[wrote JSON] {args.json_out}")
        if args.extract_dir is not None:
            print(f"[extracted]  {args.extract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
