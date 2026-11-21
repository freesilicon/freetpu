#!/usr/bin/env python3
"""Clean-room DarwiNN DGC0 weight / VLIW / parameter patcher & repacker.

Parses a compiled Tensor G5 (`buenos`) `DGC0` container, supports bit-exact
round-trip repacking (preserving FlatBuffer vtables, 4096 B `DGD0` alignment,
64 B `DWS0`/`ExternalBuffer` alignment, and `0x0a` external-gap fill), and
cleanly patches:
  - FC INT8 systolic weights + INT32 biases in `ExternalBuffer[2]` using the
    silicon-verified `[K//4, N, 4] + 128` (`^ 0x80`) layout
  - Arbitrary `ExternalBuffer[idx]` blobs (automatically recalculating all
    trailing offsets, 64 B alignments, `ExternalBuffer` descriptors, and
    matching `DWS0` tensor dimensions when sizes change)
  - Individual `buenos` VLIW hardware instruction chunks inside `"DIVE"`
    (`ExternalBuffer[0]`)

Zero external pip dependencies; Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dgc0_dump  # noqa: E402


def _align_up(val: int, align: int) -> int:
    if align <= 1:
        return val
    return (val + align - 1) & ~(align - 1)


def pack_buenos_fc_params(weights_nk: Sequence[Sequence[int]], bias_n: Sequence[int]) -> bytes:
    """Pack FC biases (int32[N]) + weights (int8[N][K]) into buenos ExternalBuffer format.

    Verified bit-for-bit (320/320 B) against live rango silicon `fc.dgc0`:
      - Bytes [0 : 4*N]:       little-endian int32 bias[0..N-1]
      - Bytes [4*N : 4*N+N*K]: weights tiled as [K//4, N, 4] with +128 (^ 0x80) offset.
    """
    N = len(weights_nk)
    if N == 0:
        raise ValueError("Empty weights matrix")
    K = len(weights_nk[0])
    if len(bias_n) != N:
        raise ValueError(f"bias length {len(bias_n)} != N={N}")
    if K % 4 != 0:
        raise ValueError(f"K={K} must be a multiple of 4 for [K//4, N, 4] buenos packing")

    out = bytearray()
    out.extend(struct.pack(f"<{N}i", *[int(b) for b in bias_n]))
    for kb in range(K // 4):
        for o in range(N):
            row = weights_nk[o]
            if len(row) != K:
                raise ValueError(f"Row {o} length {len(row)} != K={K}")
            for k in range(4):
                w = int(row[kb * 4 + k])
                if not (-128 <= w <= 127):
                    raise ValueError(f"Weight {w} out of int8 range [-128, 127]")
                out.append((w + 128) & 0xFF)
    return bytes(out)


def unpack_buenos_fc_params(blob: bytes, N: int, K: int) -> Tuple[List[List[int]], List[int]]:
    """Inverse of pack_buenos_fc_params for verification/inspection."""
    expected = 4 * N + N * K
    if len(blob) != expected:
        raise ValueError(f"Expected {expected} bytes for ({N}x{K}), got {len(blob)}")
    bias = list(struct.unpack_from(f"<{N}i", blob, 0))
    weights = [[0] * K for _ in range(N)]
    idx = 4 * N
    for kb in range(K // 4):
        for o in range(N):
            for k in range(4):
                weights[o][kb * 4 + k] = blob[idx] - 128
                idx += 1
    return weights, bias


@dataclass
class ExtBufSlot:
    index: int
    offset_field_pos: int  # Absolute offset of uint64 ExternalBuffer.offset inside DGC0 header
    size_field_pos: int    # Absolute offset of uint64 ExternalBuffer.size inside DGC0 header
    orig_offset: int
    orig_size: int
    payload: bytes


class Dgc0Image:
    """Mutable structured view over a DGC0 binary for exact round-trip & clean patching."""

    def __init__(self, raw: bytes) -> None:
        if len(raw) < 16 or raw[4:8] != b"DGC0":
            raise ValueError("Input is not a valid DGC0 container")
        self.orig_sha256 = hashlib.sha256(raw).hexdigest()
        r = dgc0_dump.FBReader(raw)
        self.root = r.root_offset()
        self.alignment = r.u32(r.field_pos(self.root, 6)) if r.field_pos(self.root, 6) is not None else 64
        if self.alignment == 0:
            self.alignment = 64

        # Verify DGD0 (4096 B) and DWS0 (64 B) alignments
        f0 = r.field_pos(self.root, 0)
        f1 = r.field_pos(self.root, 1)
        if f0 is None or f1 is None:
            raise ValueError("DGC0 missing required DWS0 (id 0) or DGD0 (id 1)")
        self.dws_off, self.dws_len, _ = r.byte_vector(f0)
        self.dgd_off, self.dgd_len, _ = r.byte_vector(f1)
        if self.dgd_off % 4096 != 0:
            raise ValueError(f"DGD0 offset 0x{self.dgd_off:x} is not 4096 B aligned")
        if self.dws_off % 64 != 0:
            raise ValueError(f"DWS0 offset 0x{self.dws_off:x} is not 64 B aligned")

        # Parse ExternalBuffer slots
        f7 = r.field_pos(self.root, 7)
        if f7 is None:
            raise ValueError("DGC0 has no external_buffers (id 7)")
        eps = r.uoffset_vector(f7)
        if not eps:
            raise ValueError("DGC0 external_buffers vector is empty")

        self.ext_slots: List[ExtBufSlot] = []
        for idx, ep in enumerate(eps):
            off_fp = r.field_pos(ep, 0)
            sz_fp = r.field_pos(ep, 1)
            if off_fp is None or sz_fp is None:
                raise ValueError(f"ExternalBuffer[{idx}] missing offset or size slot")
            eo = r.u64(off_fp)
            es = r.u64(sz_fp)
            self.ext_slots.append(
                ExtBufSlot(
                    index=idx,
                    offset_field_pos=off_fp,
                    size_field_pos=sz_fp,
                    orig_offset=eo,
                    orig_size=es,
                    payload=raw[eo : eo + es],
                )
            )

        # Header slab covers [0 .. first_ext_offset) preserving all FlatBuffer vtables & offsets
        self.header_end = self.ext_slots[0].orig_offset
        if self.header_end % self.alignment != 0:
            raise ValueError(f"First ExternalBuffer offset 0x{self.header_end:x} not aligned to {self.alignment}")
        self.header = bytearray(raw[: self.header_end])

        # Detect alignment gap fill byte observed in the source binary (DarwiNN uses 0x0a)
        self.ext_pad_byte = 0x0A
        for i in range(len(self.ext_slots)):
            end_i = self.ext_slots[i].orig_offset + self.ext_slots[i].orig_size
            next_start = self.ext_slots[i + 1].orig_offset if i + 1 < len(self.ext_slots) else len(raw)
            if next_start > end_i:
                self.ext_pad_byte = raw[end_i]
                break

    def _update_dws0_ext_tensor_size(self, ext_idx: int, old_size: int, new_size: int) -> None:
        """Update matching DWS0 Tensor.dims[0] and Executable parameter sizes when an ext buffer resizes."""
        if old_size == new_size:
            return
        dws_view = bytes(self.header[self.dws_off : self.dws_off + self.dws_len])
        r = dgc0_dump.FBReader(dws_view)
        dr = r.root_offset()
        f3 = r.field_pos(dr, 3)
        if f3 is None:
            return
        for sgp in r.uoffset_vector(f3):
            tf = r.field_pos(sgp, 0)
            if tf is None:
                continue
            for tp in r.uoffset_vector(tf):
                is_ext_f = r.field_pos(tp, 12)
                if is_ext_f is None or r.u8(is_ext_f) == 0:
                    continue
                eidx_f = r.field_pos(tp, 13)
                eidx = r.u64(eidx_f) if eidx_f is not None else 0
                if eidx == ext_idx:
                    df = r.field_pos(tp, 0)
                    if df is not None:
                        dvec = r.uoffset(df)
                        if r.u32(dvec) == 1 and r.u32(dvec + 4) == old_size:
                            struct.pack_into("<I", self.header, self.dws_off + dvec + 4, new_size)

            # Also update nested MultiExecutable parameter layer size if ext_idx is a parameter tensor
            of = r.field_pos(sgp, 3)
            if of is not None:
                for op in r.uoffset_vector(of):
                    dt_f = r.field_pos(op, 0)
                    if dt_f is None or r.u8(dt_f) != 2:
                        continue
                    det = r.uoffset(r.field_pos(op, 1))
                    mef = r.field_pos(det, 0)
                    if mef is None:
                        continue
                    me_off, me_len, me_bytes = r.byte_vector(mef)
                    mr = dgc0_dump.FBReader(me_bytes)
                    mroot = mr.root_offset()
                    ef = mr.field_pos(mroot, 1)
                    if ef is None:
                        continue
                    for ep in mr.uoffset_vector(ef):
                        pf = mr.field_pos(ep, 20)
                        if pf is None:
                            continue
                        for cp in mr.uoffset_vector(pf):
                            s32_f = mr.field_pos(cp, 1)
                            s64_f = mr.field_pos(cp, 3)
                            if s32_f is not None and mr.u32(s32_f) == old_size:
                                struct.pack_into("<I", self.header, self.dws_off + me_off + s32_f, new_size)
                            if s64_f is not None and mr.u64(s64_f) == old_size:
                                struct.pack_into("<Q", self.header, self.dws_off + me_off + s64_f, new_size)

    def patch_external_buffer(self, ext_idx: int, new_payload: bytes) -> None:
        if not (0 <= ext_idx < len(self.ext_slots)):
            raise IndexError(f"ext_idx {ext_idx} out of range [0, {len(self.ext_slots)})")
        slot = self.ext_slots[ext_idx]
        old_size = len(slot.payload)
        slot.payload = new_payload
        self._update_dws0_ext_tensor_size(ext_idx, old_size, len(new_payload))

    def patch_vliw_chunk(self, chunk_idx: int, new_chunk: bytes) -> None:
        """Patch a VLIW hardware instruction chunk inside ExternalBuffer[0] ('DIVE')."""
        dive_slot = self.ext_slots[0]
        dive = bytearray(dive_slot.payload)
        if len(dive) < 8 or dive[4:8] != b"DIVE":
            raise ValueError("ExternalBuffer[0] is not a DIVE FlatBuffer")
        r = dgc0_dump.FBReader(bytes(dive))
        root = r.root_offset()
        f3 = r.field_pos(root, 3)
        if f3 is None:
            raise ValueError("DIVE module has no hardware_instruction_chunks (id 3)")
        chunks = r.uoffset_vector(f3)
        if not (0 <= chunk_idx < len(chunks)):
            raise IndexError(f"chunk_idx {chunk_idx} out of range [0, {len(chunks)})")
        bp = r.field_pos(chunks[chunk_idx], 0)
        if bp is None:
            raise ValueError(f"DIVE chunk[{chunk_idx}] has no byte vector")
        boff, blen, _ = r.byte_vector(bp)
        if len(new_chunk) != blen:
            raise ValueError(
                f"In-place VLIW chunk patch requires matching length ({len(new_chunk)} != {blen} B) "
                "to preserve FlatBuffer relative offsets inside DIVE"
            )
        dive[boff : boff + blen] = new_chunk
        dive_slot.payload = bytes(dive)

    def repack(self) -> bytes:
        """Repack header + all ExternalBuffers with exact 64 B alignment and updated descriptors."""
        hdr = bytearray(self.header)
        out = bytearray()
        cur = len(hdr)

        # Compute new offsets and patch ExternalBuffer[i].{offset, size} inside header
        new_offsets: List[int] = []
        for slot in self.ext_slots:
            cur = _align_up(cur, self.alignment)
            new_offsets.append(cur)
            struct.pack_into("<Q", hdr, slot.offset_field_pos, cur)
            struct.pack_into("<Q", hdr, slot.size_field_pos, len(slot.payload))
            cur += len(slot.payload)

        # Emit header + aligned ExternalBuffer payloads
        out.extend(hdr)
        for slot, off in zip(self.ext_slots, new_offsets):
            if off > len(out):
                out.extend(bytes([self.ext_pad_byte]) * (off - len(out)))
            out.extend(slot.payload)

        # Final file size aligned to self.alignment (64 B), matching DarwiNN compiler behavior
        final_len = _align_up(len(out), self.alignment)
        if final_len > len(out):
            out.extend(bytes([self.ext_pad_byte]) * (final_len - len(out)))
        return bytes(out)


def _compute_fc_identity_patch(scale_val: int = 16, N: int = 16, K: int = 16) -> Tuple[List[List[int]], List[int]]:
    """Build an N x K scaled-identity int8 weight matrix and zero bias vector."""
    weights = [[scale_val if o == i else 0 for i in range(K)] for o in range(N)]
    bias = [0] * N
    return weights, bias


def run_self_test(fc_path: Path, out_dir: Path) -> Dict[str, object]:
    raw = fc_path.read_bytes()
    orig_sha = hashlib.sha256(raw).hexdigest()

    # 1. Exact bit-for-bit round-trip check
    img = Dgc0Image(raw)
    repacked = img.repack()
    rt_sha = hashlib.sha256(repacked).hexdigest()
    if repacked != raw or rt_sha != orig_sha:
        raise AssertionError(f"Round-trip mismatch: orig={orig_sha} vs repacked={rt_sha}")

    # 2. Verify pack/unpack bijection on live fc.dgc0 ExternalBuffer[2] (16x16 FC)
    orig_params = img.ext_slots[2].payload
    unpacked_w, unpacked_b = unpack_buenos_fc_params(orig_params, 16, 16)
    repacked_params = pack_buenos_fc_params(unpacked_w, unpacked_b)
    assert repacked_params == orig_params, "FC weight pack/unpack bijection failed"

    # 3. Generate a patched scaled-identity FC variant (w = 16 * I, bias = 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_w, id_b = _compute_fc_identity_patch(scale_val=16, N=16, K=16)
    patched_params = pack_buenos_fc_params(id_w, id_b)
    img.patch_external_buffer(2, patched_params)
    patched_blob = img.repack()
    patched_path = out_dir / "fc_patched_identity.dgc0"
    patched_path.write_bytes(patched_blob)

    # Re-parse patched blob with dgc0_dump to prove structural validity
    parsed = dgc0_dump.parse_dgc0(patched_blob)
    assert len(patched_blob) == len(raw)
    assert parsed["external_buffers"][2]["size_bytes"] == 320
    assert parsed["external_buffers"][1]["sha256"] == dgc0_dump.parse_dgc0(raw)["external_buffers"][1]["sha256"]

    # 4. Also test a harmless VLIW round-trip patch on chunk[0] (re-writing identical bytes -> exact SHA)
    img2 = Dgc0Image(raw)
    c0_meta = dgc0_dump.parse_dgc0(raw)["external_buffers"][0]["dive_module"]["hardware_instruction_chunks"][0]
    c0_bytes = raw[c0_meta["offset_in_dgc0"] : c0_meta["offset_in_dgc0"] + c0_meta["size_bytes"]]
    img2.patch_vliw_chunk(0, c0_bytes)
    assert hashlib.sha256(img2.repack()).hexdigest() == orig_sha, "VLIW identity patch broke SHA-256"

    return {
        "input": str(fc_path),
        "input_bytes": len(raw),
        "orig_sha256": orig_sha,
        "roundtrip_sha256": rt_sha,
        "roundtrip_bit_exact": True,
        "fc_weight_bijection_ok": True,
        "vliw_identity_patch_ok": True,
        "patched_variant_path": str(patched_path),
        "patched_variant_bytes": len(patched_blob),
        "patched_variant_sha256": hashlib.sha256(patched_blob).hexdigest(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean-room DGC0 weight / VLIW / external-buffer patcher & repacker")
    parser.add_argument("input", type=Path, help="Input .dgc0 file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .dgc0 file")
    parser.add_argument("--roundtrip", action="store_true", help="Repack without modification and assert bit-exact SHA-256")
    parser.add_argument("--self-test", action="store_true", help="Run round-trip SHA-256 + FC weight bijection + patched variant test")
    parser.add_argument("--self-test-dir", type=Path, default=Path("."), help="Directory for --self-test patched artifact")
    parser.add_argument("--patch-fc-identity", type=int, default=None, metavar="SCALE", help="Patch ExternalBuffer[2] with SCALE*I int8 weights & zero bias")
    parser.add_argument("--patch-fc-json", type=Path, default=None, help="Patch ExternalBuffer[2] from JSON {'weights': [[...]], 'bias': [...]}")
    parser.add_argument("--patch-ext", action="append", default=[], metavar="IDX:PATH", help="Replace ExternalBuffer[IDX] with raw file at PATH")
    parser.add_argument("--patch-vliw", action="append", default=[], metavar="CHUNK_IDX:PATH", help="Patch DIVE VLIW chunk[CHUNK_IDX] in-place from PATH")
    args = parser.parse_args(argv)

    if args.self_test:
        res = run_self_test(args.input, args.self_test_dir)
        print(json.dumps(res, indent=2))
        return 0

    raw = args.input.read_bytes()
    img = Dgc0Image(raw)

    if args.patch_fc_identity is not None:
        w, b = _compute_fc_identity_patch(scale_val=args.patch_fc_identity, N=16, K=16)
        img.patch_external_buffer(2, pack_buenos_fc_params(w, b))

    if args.patch_fc_json is not None:
        spec = json.loads(args.patch_fc_json.read_text(encoding="utf-8"))
        img.patch_external_buffer(2, pack_buenos_fc_params(spec["weights"], spec["bias"]))

    for item in args.patch_ext:
        idx_s, path_s = item.split(":", 1)
        img.patch_external_buffer(int(idx_s), Path(path_s).read_bytes())

    for item in args.patch_vliw:
        idx_s, path_s = item.split(":", 1)
        img.patch_vliw_chunk(int(idx_s), Path(path_s).read_bytes())

    out_blob = img.repack()
    out_sha = hashlib.sha256(out_blob).hexdigest()

    if args.roundtrip:
        if out_sha != img.orig_sha256 or out_blob != raw:
            print(f"[FAIL] Round-trip mismatch: orig={img.orig_sha256} vs out={out_sha}", file=sys.stderr)
            return 1
        print(f"[PASS] Bit-exact round-trip verified: {len(out_blob)} B sha256={out_sha}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(out_blob)
        print(f"[WROTE] {args.output} ({len(out_blob)} B, sha256={out_sha})")
    elif not args.roundtrip:
        print(f"[OK] Repacked in memory ({len(out_blob)} B, sha256={out_sha}); pass -o PATH to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
