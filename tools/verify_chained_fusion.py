#!/usr/bin/env python3
"""Verify Phase 6 Chained Multi-Op Fusion invariants on a local .dgc0 corpus.

Verifies all 3 Phase 6 fusion probes (`work/device-runner/phase6_perf/fusion_probes/`):
  1. `fuse_fc64_silu_fc64` : 2-GEMM chained MLP (`FC1 -> 0x640 SiLU -> FC2`, 9 chunks, 8704 B ext[2])
  2. `fuse_swiglu_64`      : 3-GEMM chained SwiGLU (`gate -> 0x640 SiLU * up (0x100 MUL) -> down`, 11 chunks, 13056 B ext[2])
  3. `fuse_qkv_64`         : 3-branch Q/K/V + in-SRAM 6-tile concat (`0x600`/`0x540`/`0x6c0`, 11 chunks, 13056 B ext[2])

Checks performed:
  - 8/8 (34,816 / 34,816 B) bit-exact `ext[2]` reconstruction via concatenated `multitile_synth.pack_ext2`
  - Exact DIVE VM IR `Add(param_base, i * 4352)` + `PutBits(chunk=2*i+4, offset=37, width=64)` chaining
  - Exact `0xa40` DMA channel numbering: `0x01` (input), `(i+1)<<5 | 0x02` (params), `(S+1)<<5 | 0x03` (output)
  - Exact `0x940`/`0x980` parameter SRAM ring offset stepping (`+0x110` flits = `4352 B` per stage)
  - Exact `0x640` degree-4 `tail=0x7804` SiLU spline presence in `fc64_silu_fc64 c[5]` & `swiglu_64 c[5]`
  - Exact `0x100` 20-flit binary elementwise MUL (`silu(gate) * up`) in `swiglu_64 c[7]`
  - Exact 6-tile `0x6c0` + `mask=0x3f` `0x640`/`HDR_DATA` concat in `qkv_64 c[9]`
  - Exact live bare-metal silicon output parity (`bm.out.bin == ref.bin`, `max_abs_err == 0`)
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
sys.path.insert(0, str(_THIS_DIR))

import dgc0_builder as db  # noqa: E402
import verify_fusion_lut as vfl  # noqa: E402
import verify_int4_19 as vi4  # noqa: E402

dd = db.dgc0_dump
ms = db.ms
sp = vfl.sp

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
BASE: Optional[Path] = Path(_CORPUS_ENV) if _CORPUS_ENV else None
FUSION_DIR = (BASE / "work" / "device-runner" / "phase6_perf" / "fusion_probes") if BASE else Path("phase6_perf/fusion_probes")

MODELS = [
    ("fuse_fc64_silu_fc64", 2, 9, [0x01, 0x22, 0x42, 0x63], 64),
    ("fuse_swiglu_64", 3, 11, [0x01, 0x22, 0x42, 0x62, 0x83], 64),
    ("fuse_qkv_64", 3, 11, [0x01, 0x22, 0x42, 0x62, 0x83], 192),
]


def load_probe(name: str):
    blob = (FUSION_DIR / f"{name}.dgc0").read_bytes()
    info = dd.parse_dgc0(blob)
    eb0 = info["external_buffers"][0]
    eb2 = info["external_buffers"][2]
    ext0 = blob[eb0["offset"] : eb0["offset"] + eb0["size_bytes"]]
    ext2 = blob[eb2["offset"] : eb2["offset"] + eb2["size_bytes"]]
    meta, chunks = dd.parse_dive_module(ext0)
    return info, meta, chunks, ext2


def verify_ext2_concat(name: str, ext2: bytes, num_stages: int) -> Tuple[int, int]:
    vi4.SW_DIR = FUSION_DIR
    tensors = vi4.load_tflite_tensors(name)
    weights: List[Tuple[str, np.ndarray]] = []
    biases: List[Tuple[str, np.ndarray]] = []
    for tname, t in tensors.items():
        buf = t["buf"]
        if not buf:
            continue
        if len(t["shape"]) == 2:
            weights.append((tname, np.frombuffer(buf, dtype=np.int8).reshape(t["shape"])))
        elif len(t["shape"]) == 1:
            biases.append((tname, np.frombuffer(buf, dtype=np.int32)))

    assert len(weights) == num_stages and len(biases) == num_stages
    rebuilt = b""
    for (_, W), (_, B) in zip(weights, biases):
        N, K = W.shape
        rebuilt += ms.pack_ext2(N, K, W.tolist(), B.tolist(), include_bias=True)
    assert len(rebuilt) == len(ext2), f"{name}: ext2 size mismatch {len(rebuilt)} vs {len(ext2)}"
    assert rebuilt == ext2, f"{name}: ext2 bytes mismatch!"
    return num_stages, len(ext2)


def verify_dive_chaining(name: str, meta: Dict, num_stages: int) -> None:
    ops = meta["functions"][0]["operations"]
    putbits = [op for op in ops if op["op_type_name"] == "PutBits"]
    adds = [op for op in ops if op["op_type_name"] == "Add"]

    # PutBits targets: chunk 2 (input), chunks 4,6,.. (params), chunk 2*num_stages+4 (output)
    expected_chunks = [2] + [4 + 2 * i for i in range(num_stages)] + [4 + 2 * num_stages]
    actual_chunks = [op["detail"]["chunk_index"] for op in putbits]
    assert actual_chunks == expected_chunks, f"{name}: PutBits chunks {actual_chunks} != {expected_chunks}"
    for op in putbits:
        assert op["detail"]["offset"] == 37 and op["detail"]["bit_width"] == 64

    # Add immediates for stages 1..num_stages-1: i * 0x1100 (4352 B)
    expected_imms = [i * 4352 for i in range(1, num_stages)]
    actual_imms = [
        struct.unpack("<I", bytes.fromhex(op["detail"]["present_slots"]["id_00"]["hex"]))[0]
        for op in adds
    ]
    assert actual_imms == expected_imms, f"{name}: Add offsets {actual_imms} != {expected_imms}"


def verify_chunks_and_ring(name: str, chunks: List, expected_chs: List[int], num_stages: int) -> None:
    # 1. Verify 0xa40 DMA channels
    actual_chs = []
    for _, _, raw in chunks:
        fl = ms.bytes_to_flits(raw)
        if fl and (fl[0][0] & 0xFFF) == 0xA40:
            _, w1, _, _ = struct.unpack_from("<4I", raw, 0)
            actual_chs.append(w1 & 0xFF)
    assert actual_chs == expected_chs, f"{name}: 0xa40 channels {actual_chs} != {expected_chs}"

    # 2. Verify 0x940 / 0x980 parameter SRAM ring stepping across stages (chunks 5, 7, [9])
    for s in range(num_stages):
        ci = 5 + 2 * s
        raw = chunks[ci][2]
        w0_940, w1_940 = struct.unpack_from("<2I", raw, 0)
        f2_940 = struct.unpack_from("<I", raw, 2 * 16)[0]
        assert (w0_940 & 0xFFF) == 0x940, f"{name} c[{ci}] missing leading 0x940"
        assert w1_940 == 0x10000 + s * 0x110, f"{name} c[{ci}] 0x940 w1=0x{w1_940:x}"
        assert f2_940 == 0xE0040311 + s * 0x880000, f"{name} c[{ci}] 0x940 f2=0x{f2_940:x}"

        w0_980a, w1_980a = struct.unpack_from("<2I", raw, 3 * 16)
        f5_w3 = struct.unpack_from("<I", raw, 5 * 16 + 12)[0]
        assert (w0_980a & 0xFFF) == 0x980
        assert w1_980a == 0x20000 + s * 0x220
        assert f5_w3 == 0x00202 + s * 0x11000

        w0_980b, w1_980b = struct.unpack_from("<2I", raw, 8 * 16)
        f10_w3 = struct.unpack_from("<I", raw, 10 * 16 + 12)[0]
        assert (w0_980b & 0xFFF) == 0x980
        assert w1_980b == 0x20110 + s * 0x220
        assert f10_w3 == 0x08A02 + s * 0x11000


def verify_stage_ops(name: str, chunks: List) -> None:
    if name in ("fuse_fc64_silu_fc64", "fuse_swiglu_64"):
        # c[5] flit 54 must be degree-4 0x640 SiLU (tail_84b == 0x7804)
        raw5 = chunks[5][2]
        pkt = sp.decode_0x640(raw5[54 * 16 : 67 * 16])
        assert pkt.tail_84b == 0x7804, f"{name} c[5] expected deg-4 0x7804 spline, got 0x{pkt.tail_84b:x}"

    if name == "fuse_swiglu_64":
        # c[7] flit 29 must be 20-flit 0x100 binary elementwise MUL (tile mask 0x3)
        raw7 = chunks[7][2]
        w0 = struct.unpack_from("<I", raw7, 29 * 16)[0]
        assert w0 == 0x00003100, f"fuse_swiglu_64 c[7] expected 0x3100 binary MUL, got 0x{w0:08x}"

    if name == "fuse_qkv_64":
        # c[9] flit 246: 0x640 with 6-tile mask 0x3f; flits 259..270: 12x 0x6c0 across 6 tiles
        raw9 = chunks[9][2]
        w0_640 = struct.unpack_from("<I", raw9, 246 * 16)[0]
        assert w0_640 == 0x0003F640, f"fuse_qkv_64 c[9] expected 0x3f640, got 0x{w0_640:08x}"
        for k in range(12):
            w0_6c0 = struct.unpack_from("<I", raw9, (259 + k) * 16)[0]
            assert (w0_6c0 & 0xFFF) == 0x6C0, f"fuse_qkv_64 c[9] flit {259+k} expected 0x6c0"
        w0_hdr = struct.unpack_from("<I", raw9, 271 * 16)[0]
        assert w0_hdr == 0x0003F200, f"fuse_qkv_64 c[9] expected 0x3f200 HDR_DATA, got 0x{w0_hdr:08x}"


def verify_silicon_parity(name: str, expected_bytes: int) -> None:
    ref = np.frombuffer((FUSION_DIR / f"{name}.ref.bin").read_bytes(), dtype=np.int8)
    bm = np.frombuffer((FUSION_DIR / f"{name}.bm.out.bin").read_bytes(), dtype=np.int8)
    assert len(ref) == expected_bytes and len(bm) == expected_bytes
    max_err = int(np.max(np.abs(ref.astype(np.int32) - bm.astype(np.int32))))
    assert max_err == 0, f"{name}: silicon max_abs_err={max_err} != 0"


def main() -> int:
    total_slices = 0
    total_ext2_bytes = 0
    for name, num_stages, num_chunks, chs, out_bytes in MODELS:
        _, meta, chunks, ext2 = load_probe(name)
        assert len(chunks) == num_chunks, f"{name}: chunks {len(chunks)} != {num_chunks}"
        ns, nb = verify_ext2_concat(name, ext2, num_stages)
        total_slices += ns
        total_ext2_bytes += nb
        verify_dive_chaining(name, meta, num_stages)
        verify_chunks_and_ring(name, chunks, chs, num_stages)
        verify_stage_ops(name, chunks)
        verify_silicon_parity(name, out_bytes)
        print(
            f"[PASS] {name:20s}: {num_chunks:2d} chunks, {ns}x4352B ({nb:5d}B) ext[2] BIT-EXACT, "
            f"0xa40={[hex(c) for c in chs]}, bm==ref ({out_bytes}B, err=0)"
        )

    # Cross-model structural identity checks:
    _, _, fc_ch, _ = load_probe("fuse_fc64_silu_fc64")
    _, _, sw_ch, _ = load_probe("fuse_swiglu_64")
    _, _, qkv_ch, _ = load_probe("fuse_qkv_64")
    for ci in range(5):
        assert fc_ch[ci][2] == sw_ch[ci][2], f"fc.c[{ci}] != sw.c[{ci}]"
    for ci in range(3):
        assert fc_ch[ci][2] == qkv_ch[ci][2], f"fc.c[{ci}] != qkv.c[{ci}]"
    print("[PASS] Cross-model LEGO invariants: fc.c[0..4]==sw.c[0..4] (0 byte diffs), fc.c[0..2]==qkv.c[0..2]")
    print(f"ALL CHECKS PASSED: 3/3 fusion probes, {total_slices}/{total_slices} ext[2] slices ({total_ext2_bytes}/{total_ext2_bytes} B BIT-EXACT).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify Phase 6 chained multi-op fusion invariants (MLP, SwiGLU, QKV).")
    ap.add_argument("--corpus-dir", type=Path, default=BASE, help="External corpus root (or FREETPU_CORPUS_DIR)")
    args = ap.parse_args()
    if args.corpus_dir is None or not (args.corpus_dir / "work" / "device-runner" / "phase6_perf" / "fusion_probes").is_dir():
        print("verify_chained_fusion: SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)")
        raise SystemExit(0)
    FUSION_DIR = args.corpus_dir / "work" / "device-runner" / "phase6_perf" / "fusion_probes"
    raise SystemExit(main())
