#!/usr/bin/env python3
"""Offline cross-batch DMA & TTU decoder and invariant verifier for buenos_a0 .dgc0 artifacts.

Verifies exact bitfield formulas across:
- Baseline models: work/device-runner/{fc,add,conv2d}.dgc0
- Batch 0 models:  work/device-runner/phase3_batch0/*.dgc0
- Batch 1 models:  work/device-runner/phase3_isa_dma_b1/*.dgc0
- Batch 2 models:  work/device-runner/phase3_isa_dma_b2/*.dgc0
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
from dgc0_dump import parse_dgc0  # type: ignore

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
ROOT: Optional[Path] = Path(_CORPUS_ENV) if _CORPUS_ENV else None


def read_u64_words(raw_bytes: bytes) -> List[int]:
    if len(raw_bytes) % 8 != 0:
        raise ValueError(f"Chunk byte length {len(raw_bytes)} is not a multiple of 8")
    return list(struct.unpack(f"<{len(raw_bytes) // 8}Q", raw_bytes))


def decode_0xa40(words: List[int]) -> Dict[str, Any]:
    assert len(words) >= 8
    w0, w1, w2, w3, w4, w5, w6, w7 = words[:8]
    assert (w0 & 0xFFF) == 0xA40
    hw_seq_id = (w0 >> 12) & 0xFFFF
    w0_b28_31 = (w0 >> 28) & 0xF
    tag = (w0 >> 32) & 0xFF
    w0_hi = (w0 >> 40) & 0xFFFFFF
    addr_slot_64 = ((w4 >> 40) & 0xFFFFFF) | ((w5 & 0xFFFFFFFFFF) << 24)
    prod_bytes = (((w6 >> 51) | ((w7 & 0x7FFFF) << 13)) & 0xFFFFFFFF) + 1
    cons_bytes = ((w7 >> 19) & 0xFFFFFFFF) + 1
    return {
        "opcode": 0xA40,
        "hw_seq_id": hw_seq_id,
        "w0_b28_31": w0_b28_31,
        "tag": tag,
        "tag_slot": tag >> 5,
        "tag_kind": tag & 0xF,
        "w0_hi24": w0_hi,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4_lo40": w4 & 0xFFFFFFFFFF,
        "addr_slot_64": addr_slot_64,
        "w5_b40_63": (w5 >> 40) & 0xFFFFFF,
        "w6_lo51": w6 & ((1 << 51) - 1),
        "prod_bytes": prod_bytes,
        "cons_bytes": cons_bytes,
        "w7_hi13": (w7 >> 51) & 0x1FFF,
        "raw_words": words[:8],
    }


def decode_0x9c0(words: List[int]) -> Dict[str, Any]:
    assert len(words) >= 8
    w8, w9, w10, w11, w12, w13, w14, w15 = words[:8]
    assert (w8 & 0xFFF) == 0x9C0
    return {
        "opcode": 0x9C0,
        "hw_seq_id": (w8 >> 12) & 0xFFFF,
        "w8_b28_49": (w8 >> 28) & ((1 << 22) - 1),
        "le32_flag": (w8 >> 50) & 1,
        "w8_hi13": (w8 >> 51) & 0x1FFF,
        "w9": w9,
        "w10": w10,
        "w11": w11,
        "w12": w12,
        "gt32_flag": (w12 >> 45) & 1,
        "w12_b48_51": (w12 >> 48) & 0xF,
        "tail_rem_m1": (w12 >> 50) & 0x1F,
        "head_cap_m1": (w12 >> 55) & 0x1F,
        "w12_hi4": (w12 >> 60) & 0xF,
        "w13": w13,
        "w14": w14,
        "w14_lo15": w14 & 0x7FFF,
        "w14_bytes": w14 >> 15,
        "w15": w15,
        "raw_words": words[:8],
    }


def decode_0x940(words: List[int]) -> Dict[str, Any]:
    assert len(words) >= 6
    w0, w1, w2, w3, w4, w5 = words[:6]
    assert (w0 & 0xFFF) == 0x940
    # Bits [187:64] inside w1..w2 form 4 x 31-bit dims at global bit offsets 64+31*i (local 0..123 in w1|w2<<64).
    combined_128 = w1 | (w2 << 64)
    dims = [(combined_128 >> (31 * i)) & ((1 << 31) - 1) for i in range(4)]
    w2_hi4 = (combined_128 >> 124) & 0xF
    return {
        "opcode": 0x940,
        "hw_seq_id": (w0 >> 12) & 0xFFFF,
        "w0_b28_31": (w0 >> 28) & 0xF,
        "is_input_bit32": (w0 >> 32) & 1,
        "sram_off_32b": (w0 >> 33) & 0x7FFF,
        "w0_b48_63": (w0 >> 48) & 0xFFFF,
        "dims": dims,
        "w2_hi4": w2_hi4,
        "w3_lo32": w3 & 0xFFFFFFFF,
        "w3_hi32": (w3 >> 32) & 0xFFFFFFFF,
        "w3_b32_34": (w3 >> 32) & 0x7,
        "w3_slots_m1": (w3 >> 35) & 0xFFF,
        "w3_b47_49": (w3 >> 47) & 0x7,
        "w3_b50": (w3 >> 50) & 1,
        "w3_b51": (w3 >> 51) & 1,
        "w3_slack": (w3 >> 52) & 0xFFF,
        "w4": w4,
        "w4_lo16": w4 & 0xFFFF,
        "w4_b16_25": (w4 >> 16) & 0x3FF,
        "w4_b26_35": (w4 >> 26) & 0x3FF,
        "w4_hi28": (w4 >> 36) & 0xFFFFFFF,
        "w5": w5,
        "raw_words": words[:6],
    }


def decode_0x980(words: List[int]) -> Dict[str, Any]:
    assert len(words) >= 10
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = words[:10]
    assert (w0 & 0xFFF) == 0x980
    # Payload after bit 64 is shifted +1 bit vs 0x940 (w1[0] is 0, dims start at bit 65)
    comb_dims = (w1 | (w2 << 64)) >> 1
    dims = [(comb_dims >> (31 * i)) & ((1 << 31) - 1) for i in range(4)]
    w3_shifted = (w3 | (w4 << 64)) >> 1
    tile_mask_16b = ((w7 >> 56) & 0xFF) | ((w8 & 0xFF) << 8)
    return {
        "opcode": 0x980,
        "hw_seq_id": (w0 >> 12) & 0xFFFF,
        "is_input_bit32": (w0 >> 32) & 1,
        "sram_off_mod32k_16b": (w0 >> 33) & 0xFFFF,
        "w0_hi15": (w0 >> 49) & 0x7FFF,
        "dims": dims,
        "w3_shifted_lo32": w3_shifted & 0xFFFFFFFF,
        "w3_shifted_hi32": (w3_shifted >> 32) & 0xFFFFFFFF,
        "w5_abs_off_64b": (w5 >> 42) & 0x3FFFFF,
        "w5_lo42": w5 & ((1 << 42) - 1),
        "w6": w6,
        "w7_lo56": w7 & ((1 << 56) - 1),
        "tile_mask_16b": tile_mask_16b,
        "w8_hi56": w8 >> 8,
        "w9": w9,
        "raw_words": words[:10],
    }


def collect_dgc0_paths() -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
def collect_dgc0_paths(root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for name in ("fc", "add", "conv2d"):
        p = root / "work" / "device-runner" / f"{name}.dgc0"
        if p.exists():
            out.append(("baseline", p))
    for batch, folder in (
        ("batch0", root / "work" / "device-runner" / "phase3_batch0"),
        ("batch1", root / "work" / "device-runner" / "phase3_isa_dma_b1"),
        ("batch2", root / "work" / "device-runner" / "phase3_isa_dma_b2"),
    ):
        for p in sorted(folder.glob("*.dgc0")):
            out.append((batch, p))
    return out


def load_manifests(root: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for mf in (
        root / "work" / "dgc0-dissector" / "sweep_models" / "manifest.json",
        root / "work" / "isa-dma" / "sweep_batch1" / "manifest.json",
        root / "work" / "isa-dma" / "sweep_batch2" / "manifest.json",
    ):
        if mf.exists():
            data = json.loads(mf.read_text())
            for k, v in data.items():
                meta[k] = v
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline cross-batch DMA & TTU decoder and invariant verifier")
    ap.add_argument("--corpus-dir", type=Path, default=ROOT, help="External DGC0 corpus directory (or FREETPU_CORPUS_DIR)")
    ap.add_argument("--out-report", type=Path, default=None, help="Optional JSON report output path")
    args = ap.parse_args()
    if args.corpus_dir is None or not args.corpus_dir.is_dir():
        print("verify_dma_decoder: SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)")
        return 0
    dgc0_list = collect_dgc0_paths(args.corpus_dir)
    manifests = load_manifests(args.corpus_dir)
    print(f"Total .dgc0 files discovered: {len(dgc0_list)}")

    failures: List[str] = []
    zero_chunk_models: List[str] = []

    stats = {
        "models_total": 0,
        "models_tpu": 0,
        "models_rv64_only": 0,
        "cnt_chunk1": 0,
        "cnt_0xa40": 0,
        "cnt_0xa40_in": 0,
        "cnt_0xa40_par": 0,
        "cnt_0xa40_out": 0,
        "cnt_0x9c0": 0,
        "cnt_0x940": 0,
        "cnt_0x980": 0,
    }

    # Collect special audit tables for inspection
    odd_0x9c0: List[str] = []
    odd_0x940: List[str] = []
    odd_0xa40: List[str] = []
    param_0x940_rows: List[str] = []
    input_0x940_samples: List[str] = []
    fc_rows: List[str] = []
    special_models: Dict[str, Any] = {}

    for batch, path in dgc0_list:
        stats["models_total"] += 1
        model_name = path.stem
        blob = path.read_bytes()
        parsed = parse_dgc0(blob)
        sg = parsed["package_dws0"]["sub_graphs"][0]
        exes: List[Dict[str, Any]] = []
        for op in sg.get("operations", []):
            me = op.get("detail", {}).get("multi_executable")
            if me and me.get("executables"):
                exes.extend(me["executables"])
        tensors = sg.get("tensors", [])
        if not exes:
            stats["models_rv64_only"] += 1
            zero_chunk_models.append(f"{batch}/{model_name}")
            continue

        vm_bufs = [b for b in parsed.get("external_buffers", []) if b.get("kind") == "dive_vm_ir"]
        if not vm_bufs:
            failures.append(f"{model_name}: executable present but no dive_vm_ir external_buffer")
            continue
        dm = vm_bufs[0]["dive_module"]
        chunks_meta = dm["hardware_instruction_chunks"]
        if not chunks_meta:
            stats["models_rv64_only"] += 1
            zero_chunk_models.append(f"{batch}/{model_name}")
            continue

        stats["models_tpu"] += 1
        exe = exes[0]
        in_ports = [x["size_bytes_u64"] for x in exe.get("inputs", [])]
        out_ports = [x["size_bytes_u64"] for x in exe.get("outputs", [])]
        par_ports = [x["size_bytes_u64"] for x in exe.get("parameters", [])]
        par_tensors = [
            math.prod(t["dims"])
            for t in tensors
            if "params" in t.get("name", "") and t.get("dims")
        ]
        scr_tensors = [
            math.prod(t["dims"])
            for t in tensors
            if t.get("name", "").startswith("dive_scratch_")
        ]
        scr_bytes = scr_tensors[0] if scr_tensors else None
        f14 = exe.get("scalars", {}).get("field_14")
        ops = dm["functions"][0]["operations"]
        putbits_by_chunk: Dict[int, List[Dict[str, Any]]] = {}
        for op in ops:
            if op.get("op_type_name") == "PutBits":
                d = op.get("detail", {})
                ci = d.get("chunk_index")
                putbits_by_chunk.setdefault(ci, []).append({"name": op.get("name"), **d})

        chunk_words: List[Tuple[int, str, List[int]]] = []
        for ch in chunks_meta:
            ci = ch["chunk_index"]
            off = ch["offset_in_dgc0"]
            sz = ch["size_bytes"]
            words = read_u64_words(blob[off : off + sz])
            chunk_words.append((ci, ch.get("dispatch_op_name", ""), words))

        # 1. Verify chunk[1] structure (except conv2d/special? Check all!)
        if len(chunk_words) >= 2:
            ci, dname, w = chunk_words[1]
            if len(w) == 10 and (w[4] & 0xFFF) == 0x940:
                stats["cnt_chunk1"] += 1
                exp_hdr = (0x000000007FFFF680, 0x000FFFF8000003C0, 0, 0)
                if tuple(w[:4]) != exp_hdr:
                    failures.append(f"{model_name} chunk[1] header mismatch: {[hex(x) for x in w[:4]]}")
            else:
                failures.append(f"{model_name} chunk[1] unexpected len={len(w)} op4={hex(w[4]&0xfff) if len(w)>4 else None}")

        # 2. Scan and verify all 0xa40 descriptors + 0x9c0 tail descriptors
        in_seen = 0
        out_seen = 0
        par_seen = 0
        output_seq_by_chunk: Dict[int, int] = {}

        model_a40s = []
        model_9c0s = []
        model_940s = []
        model_980s = []

        for ci, dname, w in chunk_words:
            # Check chunk head for 0xa40
            if len(w) >= 8 and (w[0] & 0xFFF) == 0xA40:
                stats["cnt_0xa40"] += 1
                d = decode_0xa40(w[:8])
                model_a40s.append((ci, dname, d))
                # Constant bit checks
                if d["w0_b28_31"] != 0 or d["w0_hi24"] != 0:
                    odd_0xa40.append(f"{model_name} ch[{ci}] w0_b28_31={d['w0_b28_31']} w0_hi={hex(d['w0_hi24'])}")
                if d["w1"] != 0x0000002000000000:
                    failures.append(f"{model_name} ch[{ci}] 0xa40 w1={hex(d['w1'])}")
                if d["w2"] != 0 or d["w3"] != 0 or d["w4_lo40"] != 0 or d["addr_slot_64"] != 0:
                    failures.append(f"{model_name} ch[{ci}] 0xa40 nonzero w2/w3/w4/addr")
                if d["w5_b40_63"] != 2:
                    failures.append(f"{model_name} ch[{ci}] 0xa40 w5_hi={hex(d['w5_b40_63'])}")
                if d["w6_lo51"] != 0x0000800000000000:
                    failures.append(f"{model_name} ch[{ci}] 0xa40 w6_lo51={hex(d['w6_lo51'])}")
                if d["w7_hi13"] != 0:
                    failures.append(f"{model_name} ch[{ci}] 0xa40 w7_hi13={hex(d['w7_hi13'])}")

                # PutBits check
                pbs = putbits_by_chunk.get(ci, [])
                if not any(pb.get("offset") == 37 and pb.get("bit_width") == 64 for pb in pbs):
                    failures.append(f"{model_name} ch[{ci}] 0xa40 missing PutBits(off=37,w=64): {pbs}")

                # Check prod_bytes == cons_bytes and match expected tensor size
                if d["prod_bytes"] != d["cons_bytes"]:
                    failures.append(
                        f"{model_name} ch[{ci}] 0xa40 prod={d['prod_bytes']} != cons={d['cons_bytes']}"
                    )

                kind = d["tag_kind"]
                if kind == 1:
                    stats["cnt_0xa40_in"] += 1
                    exp = in_ports[in_seen]
                    in_seen += 1
                    if d["prod_bytes"] != exp:
                        failures.append(
                            f"{model_name} ch[{ci}] in[{in_seen-1}] 0xa40 bytes={d['prod_bytes']} != exp={exp}"
                        )
                elif kind == 2:
                    stats["cnt_0xa40_par"] += 1
                    par_list = par_ports if par_ports else par_tensors
                    exp = par_list[par_seen]
                    par_seen += 1
                    if d["prod_bytes"] != exp:
                        failures.append(
                            f"{model_name} ch[{ci}] par[{par_seen-1}] 0xa40 bytes={d['prod_bytes']} != exp={exp}"
                        )
                elif kind == 3:
                    stats["cnt_0xa40_out"] += 1
                    raw_out = out_ports[out_seen]
                    out_seen += 1
                    exp_dma = ((raw_out + 3) // 4) * 4
                    output_seq_by_chunk[ci] = d["hw_seq_id"]
                    if d["prod_bytes"] != exp_dma:
                        failures.append(
                            f"{model_name} ch[{ci}] out[{out_seen-1}] 0xa40 bytes={d['prod_bytes']} != exp_dma={exp_dma} (raw={raw_out})"
                        )
                    # Verify 0x9c0 tail immediately at words[8:16]
                    if len(w) >= 16 and (w[8] & 0xFFF) == 0x9C0:
                        stats["cnt_0x9c0"] += 1
                        t = decode_0x9c0(w[8:16])
                        model_9c0s.append((ci, t))
                        if t["hw_seq_id"] != d["hw_seq_id"] + 1:
                            failures.append(
                                f"{model_name} ch[{ci}] 0x9c0 seq={t['hw_seq_id']} != a40_seq+1={d['hw_seq_id']+1}"
                            )
                        if t["w10"] != 0 or t["w11"] != 0 or t["w13"] != 0 or t["w15"] != 0:
                            failures.append(
                                f"{model_name} ch[{ci}] 0x9c0 nonzero w10/w11/w13/w15: "
                                f"{hex(t['w10'])},{hex(t['w11'])},{hex(t['w13'])},{hex(t['w15'])}"
                            )
                        exp_le32 = 1 if exp_dma <= 32 else 0
                        exp_gt32 = 1 if exp_dma > 32 else 0
                        exp_w9 = (((exp_dma + 31) // 32) - 1) << 5
                        exp_tail_rem = (exp_dma - 1) % 32
                        exp_head_cap = min(exp_dma, 32) - 1
                        exp_w14 = (exp_dma << 15) | 0x4040
                        if t["le32_flag"] != exp_le32:
                            failures.append(f"{model_name} ch[{ci}] 0x9c0 le32={t['le32_flag']} != {exp_le32}")
                        if t["w9"] != exp_w9:
                            failures.append(f"{model_name} ch[{ci}] 0x9c0 w9={hex(t['w9'])} != {hex(exp_w9)} (dma={exp_dma})")
                        if t["gt32_flag"] != exp_gt32 or t["w12_b48_51"] != 0xC:
                            odd_0x9c0.append(
                                f"{model_name} ch[{ci}] dma={exp_dma} w12={hex(t['w12'])} gt32={t['gt32_flag']} nib={hex(t['w12_b48_51'])}"
                            )
                        if t["tail_rem_m1"] != exp_tail_rem or t["head_cap_m1"] != exp_head_cap or t["w12_hi4"] != 0:
                            failures.append(
                                f"{model_name} ch[{ci}] 0x9c0 w12={hex(t['w12'])} dma={exp_dma}: "
                                f"rem={t['tail_rem_m1']}!={exp_tail_rem} cap={t['head_cap_m1']}!={exp_head_cap}"
                            )
                        if t["w14"] != exp_w14:
                            failures.append(f"{model_name} ch[{ci}] 0x9c0 w14={hex(t['w14'])} != {hex(exp_w14)}")
                        # Check remaining w8 bits and low w12 bits
                        if t["w8_b28_49"] != 0 or t["w8_hi13"] != 0 or (t["w12"] & ((1 << 45) - 1)) != 0 or ((t["w12"] >> 46) & 3) != 0:
                            odd_0x9c0.append(
                                f"{model_name} ch[{ci}] extra w8={hex(w[8])} w12={hex(t['w12'])}"
                            )
                    else:
                        failures.append(f"{model_name} ch[{ci}] output 0xa40 missing 0x9c0 at words[8:16]!")
                else:
                    failures.append(f"{model_name} ch[{ci}] unknown 0xa40 tag={hex(d['tag'])}")

            # Scan for all 0x940 (48 B / 6 u64) and 0x980 (80 B / 10 u64) opcodes inside chunk.
            # Notice TPU instructions start on 16-byte boundaries (2 x u64 words) except wait-can 0x940 appear at even u64 word idx?
            # Let's scan at every even u64 index (16-byte flit aligned)! Let's check odd u64 just in case too.
            for idx in range(0, len(w), 2):
                op = w[idx] & 0xFFF
                if op == 0x940 and idx + 6 <= len(w):
                    # Verify non-zero low pattern and w5==4 to be sure it's a real 0x940 packet
                    if w[idx + 5] == 4:
                        stats["cnt_0x940"] += 1
                        r = decode_0x940(w[idx : idx + 6])
                        model_940s.append((ci, idx, dname, r))
                elif op == 0x980 and idx + 10 <= len(w) and ((w[idx] >> 49) & 0x7FFF) == 1:
                    stats["cnt_0x980"] += 1
                    d980 = decode_0x980(w[idx : idx + 10])
                    if d980["w9"] != 0 or d980["w8_hi56"] != 0:
                        failures.append(f"{model_name} ch[{ci}]@w{idx} 0x980 w8_hi/w9 nonzero: {hex(d980['w8_hi56'])},{hex(d980['w9'])}")
                    if d980["w3_shifted_lo32"] not in (0x10010800, 0x80010800):
                        failures.append(f"{model_name} ch[{ci}]@w{idx} 0x980 unexpected w3_shifted_lo32={hex(d980['w3_shifted_lo32'])}")
                    model_980s.append((ci, idx, dname, w[idx : idx + 10]))

        if in_seen != len(in_ports):
            failures.append(f"{model_name}: matched {in_seen} input 0xa40 vs {len(in_ports)} ports")
        if out_seen != len(out_ports):
            failures.append(f"{model_name}: matched {out_seen} output 0xa40 vs {len(out_ports)} ports")

        # 3. Verify 0x940 rules per role!
        # Let's track which 0x940 corresponds to input_0 (chunk[1]), input_1/2, param (before output chunk).
        input_940_idx = 0
        for ci, idx, dname, r in model_940s:
            # Verify fixed constants
            if r["w5"] != 4 or r["w0_b28_31"] != 0 or r["w0_b48_63"] != 1:
                odd_0x940.append(f"{model_name} ch[{ci}]@w{idx} w0={hex(r['raw_words'][0])} w5={hex(r['w5'])}")
            if r["w2_hi4"] != 0:
                odd_0x940.append(f"{model_name} ch[{ci}]@w{idx} w2_hi4={hex(r['w2_hi4'])}")

            if r["w3_lo32"] == 0x10010800:
                # Input 0x940
                if input_940_idx < len(in_ports):
                    in_b = in_ports[input_940_idx]
                    input_940_idx += 1
                    in32 = ((in_b + 31) // 32) * 32
                    if in_b <= 32:
                        exp_dims = [0, 0, 0, 0]
                        exp_w3_hi = 0
                    else:
                        d1 = max(0, (in_b + 16383) // 16384 - 1)
                        exp_dims = [0x7F0081FF, 0x7F008000 | d1, 0x7F008000, 0x7F008000]
                        if in_b < 16384:
                            slots_m1 = (in32 // 32) - 1
                            slack = 0x400 - (in32 // 16)
                            exp_w3_hi = (slack << 20) | (1 << 18) | (slots_m1 << 3)
                        else:
                            exp_w3_hi = 0
                    if r["dims"] != exp_dims:
                        failures.append(
                            f"{model_name} ch[{ci}]@w{idx} in_b={in_b} 0x940 dims={[hex(x) for x in r['dims']]} != {[hex(x) for x in exp_dims]}"
                        )
                    if r["w3_hi32"] != exp_w3_hi:
                        failures.append(
                            f"{model_name} ch[{ci}]@w{idx} in_b={in_b} 0x940 w3_hi={hex(r['w3_hi32'])} != {hex(exp_w3_hi)}"
                        )
                    # Check w4 bitfield hypothesis:
                    # bits [15:0] = 0x0210; bits [27:16] = row_or_M_tag (0x004 default, FC adds (M-1)<<2, add/concat adds sram_off_32b?);
                    # Wait: check exact bits [41:28] vs [27:16]! Let's collect every non-standard w4 too!
                    if (r["w4"] & 0xFFFF) != 0x0210:
                        odd_0x940.append(f"{model_name} ch[{ci}]@w{idx} w4_lo16={hex(r['w4']&0xffff)}")
                    if r["w4"] != 0x3FFFC040210 or (in_b in (1, 17, 32, 48, 49, 65, 256, 4128, 16384, 32768, 65536, 98304, 131072, 262144) or "add" in model_name or "concat" in model_name or "fc" in model_name):
                        input_0x940_samples.append(
                            f"{model_name:32s} ch[{ci}]@w{idx:3d} in_b={in_b:6d} sram_off32={r['sram_off_32b']:#06x} w4={r['w4']:#018x} w4_hi={r['w4']>>28:#x} w4_mid={(r['w4']>>16)&0xfff:#05x}"
                        )
            elif r["w3_lo32"] == 0x80010800:
                # Param (FC/Conv/unaligned slice) 0x940
                par_list = par_ports if par_ports else par_tensors
                pbytes = par_list[0] if par_list else None
                if pbytes is not None:
                    p32 = ((pbytes + 31) // 32) * 32
                    exp_dims = [0x78008FFF, 0x78008000, 0x78008000, 0x78008000]
                    slots_m1 = (p32 // 32) - 1
                    slack = 0x1000 - (p32 // 16)
                    exp_w3_hi = (slack << 20) | (1 << 18) | (slots_m1 << 3)
                    if r["dims"] != exp_dims:
                        odd_0x940.append(
                            f"{model_name} ch[{ci}]@w{idx} par_b={pbytes} dims={[hex(x) for x in r['dims']]} != {[hex(x) for x in exp_dims]}"
                        )
                    if r["w3_hi32"] != exp_w3_hi:
                        odd_0x940.append(
                            f"{model_name} ch[{ci}]@w{idx} par_b={pbytes} w3_hi={hex(r['w3_hi32'])} != {hex(exp_w3_hi)}"
                        )
                param_0x940_rows.append(
                    f"{model_name:32s} ch[{ci}]@w{idx:3d} par_b={str(pbytes):>6s} w0={r['raw_words'][0]:#018x} w3={r['raw_words'][3]:#018x} w4={r['w4']:#018x}"
                )
            else:
                odd_0x940.append(
                    f"{model_name} ch[{ci}]@w{idx} UNKNOWN w3_lo={hex(r['w3_lo32'])} w0={hex(r['raw_words'][0])} w4={hex(r['w4'])}"
                )

        # Record FC details
        mf_entry = manifests.get(model_name, {})
        meta = mf_entry.get("meta", {})
        if meta.get("op") == "FULLY_CONNECTED" or model_name == "fc":
            dims = meta.get("dims") or []
            fc_w4_ch1 = None
            for ci, idx, dname, r in model_940s:
                if ci == 1:
                    fc_w4_ch1 = r["w4"]
            fc_rows.append(
                f"{model_name:24s} dims={str(dims):14s} in={in_ports} par={par_ports or par_tensors} out={out_ports} scr={scr_bytes} f14={f14} ch1_w4={hex(fc_w4_ch1) if fc_w4_ch1 is not None else None} chunks={len(chunk_words)}"
            )

        # Save detailed inspect structures for key models
        if model_name in (
            "b2_slice_c_off1",
            "b2_slice_w_off3",
            "b2_concat_c_4_8",
            "b2_q_1x1536x1x64",
            "b1_q_1x256x1x256",
            "b1_q_1x512x1x256",
            "b1_q_1x1024x1x256",
            "b2_fc_3x16x16",
            "b2_fc_1x256x64",
            "fc",
            "add",
            "conv2d",
        ):
            special_models[model_name] = {
                "chunks": [(ci, dname, len(w) * 8) for ci, dname, w in chunk_words],
                "scr": scr_bytes,
                "f14": f14,
                "a40": [(ci, d["hw_seq_id"], hex(d["tag"]), d["prod_bytes"]) for ci, _, d in model_a40s],
                "9c0": [(ci, t["hw_seq_id"], t["le32_flag"], hex(t["w9"]), hex(t["w12"]), hex(t["w14"])) for ci, t in model_9c0s],
                "940": [
                    (
                        ci,
                        idx,
                        hex(r["raw_words"][0]),
                        [hex(x) for x in r["dims"]],
                        hex(r["raw_words"][3]),
                        hex(r["w4"]),
                    )
                    for ci, idx, _, r in model_940s
                ],
                "980": [(ci, idx, [hex(x) for x in w980]) for ci, idx, _, w980 in model_980s],
            }

    print("Summary stats:", json.dumps(stats, indent=2))
    print(f"RV64-only (0 TPU chunks, {len(zero_chunk_models)}): {zero_chunk_models}")
    print(f"Failures ({len(failures)}):")
    for f in failures[:50]:
        print("  FAIL:", f)

    print(f"\nOdd 0xa40 ({len(odd_0xa40)}):")
    for x in odd_0xa40[:20]:
        print(" ", x)

    print(f"\nOdd 0x9c0 ({len(odd_0x9c0)}):")
    for x in odd_0x9c0[:20]:
        print(" ", x)

    print(f"\nOdd 0x940 ({len(odd_0x940)}):")
    for x in odd_0x940[:30]:
        print(" ", x)

    print(f"\nParam 0x940 rows ({len(param_0x940_rows)}):")
    for x in param_0x940_rows:
        print(" ", x)

    print(f"\nSample Input 0x940 rows:")
    for x in input_0x940_samples:
        print(" ", x)

    print(f"\nFC Models ({len(fc_rows)}):")
    for x in fc_rows:
        print(" ", x)

    if args.out_report is not None:
        args.out_report.write_text(
            json.dumps(
                {
                    "stats": stats,
                    "zero_chunk_models": zero_chunk_models,
                    "failures": failures,
                    "odd_0xa40": odd_0xa40,
                    "odd_0x9c0": odd_0x9c0,
                    "odd_0x940": odd_0x940,
                    "special_models": special_models,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.out_report}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
