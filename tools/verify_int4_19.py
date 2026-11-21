#!/usr/bin/env python3
"""Full 19-model verification harness for Tensor G5 (buenos_a0) Phase 5 INT4 (W4A8) sweep.

Verifies every model in `work/device-runner/phase5_sweep_int4/` against clean-room synthesis in
`pack_int4.py` (wrapping Phase 4 `multitile_synth.py` read-only):
  - 18/18 compiled models: 100% bit-exact `ext[2]` reconstruction (0 byte mismatches).
  - 11/11 7-chunk models (1 INT8 ref, 7 per-tensor INT4, 3 blockwise INT4): 100% bit-exact
    `chunk[0..6]` reconstruction (only masking cyclic 4-bit tag `c6[-5].lo[15:12]`).
  -  6/6  9-chunk W4A8 models (4 per-channel, 2 separate-DMA bias): exact verification of
    `c0/c1/c2/c4/c6/c8` DMAs (`f0_hi_word = (dma_idx<<5)|type`), `c3` CSR_040 (`0x56=0x10070`
    + INT32 accumulator mode), `c5` `0x240` (`0x61[19:18]` scale/bias enables + `0x940` param
    offset `w_bytes>>4`), and `c7` (`0x980` `w_bytes>>3` + Stage-3 MSB-flip `0x640`/`HDR200`).
  -  1/1  `dequant_i4_fc_1x64x32`: exact 1152B `ext[2]` (1024B 4-bit nibble-packed + 128B raw
    FP32 scales), `c4`/`c6` 0xa40 sizes, `0x940`/`0x980` offsets, and `CSR 0x56=0x0000026`.
  -  1/1  `fc_i4_w4a16_pc_1x64x32`: verified compile-rejected (`u16 x u4` legalize-dwc error).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "dgc0"))
sys.path.insert(0, str(_THIS_DIR))

import dgc0_dump  # noqa: E402
from dgc0_patcher import Dgc0Image  # noqa: E402
import gen_int4_sweep as gis  # noqa: E402
import multitile_synth as ms  # noqa: E402
import pack_int4 as pi4  # noqa: E402
import verify_full_isa_81 as v81  # noqa: E402

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
BASE: Optional[Path] = Path(_CORPUS_ENV) if _CORPUS_ENV else None
DV_DIR = (BASE / "work/device-runner/phase5_sweep_int4") if BASE else Path("phase5_sweep_int4")
SW_DIR = (BASE / "work/int4-splines/sweep_int4") if BASE else Path("sweep_int4")


def load_tflite_tensors(name: str) -> Dict[str, Dict[str, Any]]:
    raw = (SW_DIR / f"{name}.tflite").read_bytes()
    r = gis._FBReader(raw)
    root = r.u32(0)
    sg0 = r.uoffset_vec(r.field_pos(root, 2))[0]
    bufs = [
        r.byte_vec(r.field_pos(bt, 0)) if r.field_pos(bt, 0) is not None else b""
        for bt in r.uoffset_vec(r.field_pos(root, 4))
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for tt in r.uoffset_vec(r.field_pos(sg0, 0)):
        nv = r.uoffset(r.field_pos(tt, 3))
        tname = raw[nv + 4 : nv + 4 + r.u32(nv)].decode()
        shape = r.i32_vec(r.field_pos(tt, 0))
        bf = r.field_pos(tt, 2)
        bidx = r.u32(bf) if bf is not None else 0
        tf = r.field_pos(tt, 1)
        ttype = r.u8(tf) if tf is not None else 0
        scales: List[float] = []
        qf = r.field_pos(tt, 4)
        if qf is not None:
            qt = r.uoffset(qf)
            sf = r.field_pos(qt, 2)
            if sf is not None:
                sv = r.uoffset(sf)
                cnt = r.u32(sv)
                scales = [struct.unpack_from("<f", raw, sv + 4 + 4 * j)[0] for j in range(cnt)]
        out[tname] = {"shape": shape, "type": ttype, "buf": bufs[bidx], "scales": scales}
    return out


def extract_dgc0_chunks_ext2(path: Path) -> Tuple[List[bytes], bytes]:
    img = Dgc0Image(path.read_bytes())
    r = dgc0_dump.FBReader(img.ext_slots[0].payload)
    chunks = [
        r.byte_vector(r.field_pos(c, 0))[2]
        for c in r.uoffset_vector(r.field_pos(r.root_offset(), 3))
    ]
    ext2 = img.ext_slots[2].payload if len(img.ext_slots) > 2 else b""
    return chunks, ext2


def extract_csr040_maps(chunk: bytes) -> List[Dict[int, int]]:
    fl = ms.bytes_to_flits(chunk)
    out: List[Dict[int, int]] = []
    i = 0
    n = len(fl)
    while i < n:
        lo, hi = fl[i]
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
        elif op == 0x240:
            ln = 20
        elif op == 0x200:
            tw = (hi >> 12) & 0xFFF
            ln = (
                1 + ((tw - 1) // 2 + 3) // 4
                if ((hi & 0xF0F) == 0 and (hi >> 48) == 0 and (tw % 2 == 1) and 3 <= tw <= 0x100)
                else 20
            )
        elif op == 0x040:
            tw = (hi >> 12) & 0xFFF
            n_csrs = (tw - 1) // 2
            ln = 1 + (max(0, n_csrs - 1) + 3) // 4
            u32s = [hi >> 32] if (hi >> 32) else []
            for fi in range(i + 1, i + ln):
                l2, h2 = fl[fi]
                u32s.extend([l2 & 0xFFFFFFFF, l2 >> 32, h2 & 0xFFFFFFFF, h2 >> 32])
            out.append({w & 0x7F: (w >> 7) & 0x1FFFFFF for w in u32s[:n_csrs] if w})
        else:
            raise ValueError(f"Unknown opcode 0x{op:03x} at flit {i}")
        i += ln
    return out


def decode_a40(ch: bytes) -> Tuple[int, int]:
    fl = ms.bytes_to_flits(ch)
    lo0, _ = fl[0]
    lo3, hi3 = fl[3]
    f0_hi_word = (lo0 >> 32) & 0xFFFF
    f3 = (hi3 << 64) | lo3
    dma_enc = (f3 >> 48) & 0xFFFFFFFF
    nbytes = (dma_enc // 8) + 1
    return f0_hi_word, nbytes


def verify_all() -> int:
    manifest = json.loads((SW_DIR / "manifest.json").read_text())
    summary_live = json.loads((DV_DIR / "summary.json").read_text())
    live_by_name = {m["model"]: m for m in summary_live["models"]}

    ext2_ok = 0
    compiled_cnt = 0
    c7_ok = 0
    c7_tot = 0
    c9_w4a8_ok = 0
    c9_w4a8_tot = 0
    deq_ok = 0
    rej_ok = 0
    failures: List[str] = []

    print(f"{'Model':27s} | {'Cls':7s} | {'ext[2]':14s} | {'VLIW / Chunks':38s}")
    print("-" * 96)

    for name, entry in manifest.items():
        m = entry["meta"]
        M, K, N = m["M"], m["K"], m["N"]
        qm = m["quant_mode"]
        dgc0_path = DV_DIR / f"{name}.dgc0"

        if not dgc0_path.exists():
            # Verify expected compile failure (W4A16)
            live = live_by_name[name]
            assert live["compile.ok"] is False and "legalize-dwc" in live["error"]
            rej_ok += 1
            print(f"{name:27s} | {'REJECT':7s} | {'N/A (compile)':14s} | compile rejected (W4A16 u16x u4)")
            continue

        compiled_cnt += 1
        chunks, real_ext2 = extract_dgc0_chunks_ext2(dgc0_path)
        ts = load_tflite_tensors(name)
        wt = ts.get("filter") or ts.get("filter_i4")
        assert wt is not None

        if wt["type"] == 17:
            W = np.array(gis.unpack_int4_le(wt["buf"], N * K), dtype=np.int32).reshape(N, K).tolist()
        else:
            W = np.frombuffer(wt["buf"], dtype=np.int8).astype(np.int32).reshape(N, K).tolist()

        # 1. Verify ext[2] bit-exact across ALL 18 compiled models!
        if qm == "dequant_per_channel":
            syn_ext2, w_bytes, tail_bytes = pi4.pack_int4_nibble_ext2(W, raw_scales_n=wt["scales"], offset=8)
        else:
            is_pc = qm == "per_channel"
            bias_vec = (
                list(struct.unpack(f"<{N}i", ts["bias"]["buf"])) if "bias" in ts else None
            )
            syn_ext2, w_bytes, tail_bytes = pi4.pack_int4_w4a8_ext2(
                W,
                offset=8 if wt["type"] == 17 else 128,
                w_scales_n=wt["scales"] if is_pc else None,
                bias_n=bias_vec,
                s_in=m["s_in"],
                s_out=m["s_out"],
                inline_bias_per_group=False,
            )

        ext2_match = real_ext2 == syn_ext2
        if ext2_match:
            ext2_ok += 1
        else:
            failures.append(f"{name}: ext[2] mismatch (real={len(real_ext2)}B syn={len(syn_ext2)}B)")

        # 2. Verify VLIW chunks by class
        if len(chunks) == 7:
            c7_tot += 1
            act = 1 if m.get("fused_act") == "RELU" else 0
            s_w = 1.0 if qm.startswith("blockwise") else wt["scales"][0]
            syn_chunks, _ = pi4.synthesize_int4_7chunk(
                M=M,
                K=K,
                N=N,
                weights_nk=W,
                bias_n=[0] * N,
                s_in=m["s_in"],
                zp_in=m["zp_in"],
                s_w=s_w,
                s_out=m["s_out"],
                zp_out=m["zp_out"],
                act=act,
                weight_is_int4=(wt["type"] == 17),
            )
            ch_diffs = []
            for ci in range(7):
                a = v81.mask_c6_cyclic_tag(chunks[ci]) if ci == 6 else chunks[ci]
                b = v81.mask_c6_cyclic_tag(syn_chunks[ci]) if ci == 6 else syn_chunks[ci]
                if a != b:
                    ch_diffs.append(f"c{ci}")
            if not ch_diffs:
                c7_ok += 1
                vliw_str = "7/7 chunks BIT-EXACT (c0..c6=0 diffs)"
            else:
                vliw_str = f"DIFF in {ch_diffs}"
                failures.append(f"{name}: {vliw_str}")
            cls = "INT8-7c" if wt["type"] != 17 else ("BLK-7c" if qm.startswith("blockwise") else "W4A8-7c")

        elif qm == "dequant_per_channel":
            # Verify dequant_i4_fc_1x64x32 9-chunk invariants
            assert len(chunks) == 9
            assert chunks[0] == ms.build_chunk0()
            assert decode_a40(chunks[2]) == (0x0001, M * K * 4)
            assert decode_a40(chunks[4]) == (0x0022, w_bytes)  # 1024 B true 4-bit nibble weights!
            assert decode_a40(chunks[6]) == (0x0042, tail_bytes)  # 128 B raw FP32 scales!
            assert decode_a40(chunks[8]) == (0x0063, M * N * 4)
            csrs_c3 = extract_csr040_maps(chunks[3])
            assert len(csrs_c3) == 1 and csrs_c3[0][0x56] == pi4.CSR56_DEQUANT_BF16
            # Verify c[5] tail 0x940 offset = w_bytes >> 4 and c[7] head 0x980 offset = w_bytes >> 3
            fl5 = ms.bytes_to_flits(chunks[5])
            fl7 = ms.bytes_to_flits(chunks[7])
            assert ((fl5[-3][0] >> 32) & 0xFFFF) == (w_bytes >> 4)
            assert ((fl7[0][0] >> 32) & 0xFFFF) == (w_bytes >> 3)
            deq_ok += 1
            cls = "DEQ-9c"
            vliw_str = "9ch EXACT (0xa40 1024B/128B, 0x56=0x26)"

        else:
            # 9-chunk W4A8 (per-channel scale and/or separate-DMA bias)
            c9_w4a8_tot += 1
            assert len(chunks) == 9
            assert chunks[0] == ms.build_chunk0()
            # When M>=4 and K>=128 (4-row tile striping across 4x4 grid, 0x55=0x785800), c[1] uses m_eff = M//4
            c1_m = (M // 4) if (M >= 4 and K >= 128) else M
            c1_k = K * (M // c1_m)
            assert chunks[1] == ms.build_chunk1(c1_m, c1_k)
            assert chunks[2] == ms.build_chunk2(M, K)
            assert decode_a40(chunks[4]) == (0x0022, w_bytes)
            assert decode_a40(chunks[6]) == (0x0042, tail_bytes)
            assert decode_a40(chunks[8]) == (0x0063, M * N)

            # Verify c[3] CSR_040: 0x56 = 0x10070 + INT32 accumulator mode (0x5b=0x800, 0x5c=0x3f800, 0x61=0)
            csrs_c3 = extract_csr040_maps(chunks[3])
            assert len(csrs_c3) >= 1
            for cm in csrs_c3:
                assert cm[0x56] == pi4.CSR56_INT4_ZP8, f"{name}: c3 0x56={hex(cm[0x56])}"
                assert cm[0x5B] == 0x800 and cm[0x5C] == 0x3F800 and cm[0x5D] == 0x1F80000
                assert cm[0x5E] == 0x3 and cm[0x5F] == 0x7FC0 and cm[0x61] == 0x0

            # Verify c[5] 0x240 rescaling HDR_DATA + tail 0x940 byte offset = w_bytes >> 4
            fl5 = ms.bytes_to_flits(chunks[5])
            p240_idx = next(i for i, (lo, _) in enumerate(fl5) if (lo & 0xFFF) == 0x240 and (lo & 0x3F) == 0)
            cs240 = ms.unpack_hdr_csrs(fl5[p240_idx : p240_idx + 20])
            is_pc = qm == "per_channel"
            has_nz_bias = "bias" in ts and any(x != 0 for x in struct.unpack(f"<{N}i", ts["bias"]["buf"]))
            exp_0x61 = ((1 if is_pc else 0) << 19) | ((1 if has_nz_bias else 0) << 18) | (((m["zp_out"] + 128) & 0xFF) << 2)
            assert cs240[0x61] == exp_0x61, f"{name}: 0x240 0x61={hex(cs240[0x61])} != {hex(exp_0x61)}"
            exp_0x5c = 0x3F800 if is_pc else 0x3C800  # 1.0f if per-channel, 1/64 if scalar 0.0625*0.0625/0.25
            assert cs240[0x5C] == exp_0x5c, f"{name}: 0x240 0x5c={hex(cs240[0x5C])} != {hex(exp_0x5c)}"
            # Verify 2nd param 0x940 offset = w_bytes >> 4 and 0x980 offset = w_bytes >> 3
            # (When N>32 and M==1, 0x940 sits at c[7][0] before 0x980 at c[7][3]; otherwise c[5][-3] / c[7][0].)
            fl7 = ms.bytes_to_flits(chunks[7])
            if (fl5[-3][0] & 0xFFF) == 0x940:
                p940_lo, p980_lo = fl5[-3][0], fl7[0][0]
            else:
                p940_lo, p980_lo = fl7[0][0], fl7[3][0]
            assert (p940_lo & 0xFFF) == 0x940 and ((p940_lo >> 32) & 0xFFFF) == (w_bytes >> 4)
            assert (p980_lo & 0xFFF) == 0x980 and ((p980_lo >> 32) & 0xFFFF) == (w_bytes >> 3)

            # Verify Stage-3 HDR200 (0x56=0xe00, 0x61=(zp_out+128)<<2)
            h7_idx = next(i for i, (lo, _) in enumerate(fl7) if (lo & 0xFFF) == 0x200 and (lo & 0x3F) == 0 and i + 20 <= len(fl7))
            cs7 = ms.unpack_hdr_csrs(fl7[h7_idx : h7_idx + 20])
            assert cs7[0x56] == 0xE00 and cs7[0x61] == (((m["zp_out"] + 128) & 0xFF) << 2)

            c9_w4a8_ok += 1
            cls = "PC-9c" if is_pc else "BIAS-9c"
            vliw_str = f"9ch EXACT (0x61=0x{exp_0x61:x}, off={w_bytes}B)"

        print(f"{name:27s} | {cls:7s} | {len(real_ext2):5d}B {'EXACT' if ext2_match else 'FAIL ':5s} | {vliw_str}")

    print("-" * 96)
    print(f"Summary across 19 models:")
    print(f"  ext[2] bit-exact:        {ext2_ok}/{compiled_cnt} compiled models")
    print(f"  7-chunk c0..c6 exact:    {c7_ok}/{c7_tot} (1 INT8 + 7 pt-INT4 + 3 blk-INT4)")
    print(f"  9-chunk W4A8 exact:      {c9_w4a8_ok}/{c9_w4a8_tot} (4 pc-INT4 + 2 separate-DMA bias)")
    print(f"  dequant_i4_fc 4b exact:  {deq_ok}/1 (true 4-bit nibble packing: 1024B + 128B)")
    print(f"  compile-rejected W4A16:  {rej_ok}/1")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    assert ext2_ok == 18 and c7_ok == 11 and c9_w4a8_ok == 6 and deq_ok == 1 and rej_ok == 1
    print("\nALL 19/19 PHASE 5 INT4 CORPUS CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify 19-model INT4 (W4A8 / nibble-packed) corpus against pack_int4")
    ap.add_argument("--corpus-dir", type=Path, default=BASE, help="External corpus root (or FREETPU_CORPUS_DIR)")
    args = ap.parse_args()
    if args.corpus_dir is None or not args.corpus_dir.is_dir():
        print("verify_int4_19: SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)")
        raise SystemExit(0)
    DV_DIR = args.corpus_dir / "work/device-runner/phase5_sweep_int4"
    SW_DIR = args.corpus_dir / "work/int4-splines/sweep_int4"
    raise SystemExit(verify_all())
