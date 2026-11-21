#!/usr/bin/env python3
"""Verify exact 4096/4096 math on all 5 live causal patch probes in phase3_isa_vec_patch_proof."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

_CORPUS_ENV = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
BASE: Optional[Path] = Path(_CORPUS_ENV) if _CORPUS_ENV else None
PATCH_OUT = (BASE / "work/device-runner/phase3_isa_vec_patch_proof") if BASE else Path("phase3_isa_vec_patch_proof")
B1B_IN = (BASE / "work/isa-vec/batch1b") if BASE else Path("batch1b")


def u32_f32(u: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0])


def main() -> None:
    manifest = json.loads((B1B_IN / "manifest.json").read_text())

    # 1. p1_q_s1_mul_05_to_15 (q_s_0_5 with Stage 1 mul = u32_f32(0x3fc00001))
    inp1 = np.fromfile(manifest["q_s_0_5"]["in_path"], dtype=np.int8).astype(np.float32)
    hw1 = np.fromfile(PATCH_OUT / "p1_q_s1_mul_05_to_15.out.bin", dtype=np.int8)
    m1 = u32_f32(0x3FC00001)
    pred1 = np.clip(np.rint(inp1 * m1), -128, 127).astype(np.int8)
    print(f"p1_q_s1_mul_05_to_15 exact: {int((hw1 == pred1).sum())}/4096 (m1={float(m1):.8f})")
    if (hw1 != pred1).sum():
        bad = np.where(hw1 != pred1)[0][:5]
        print("  diffs:", [(int(i), int(inp1[i]), int(hw1[i]), int(pred1[i])) for i in bad])

    # 2. p2_q_s2_mul_half_x2 (Stage 1 = 0.5 -> int8 q_mid, then Stage 2 * 0.125 / 0.0625 = * 2.0, clipped at Stage 2 [min=-8.0, max=7.9375] / 0.0625)
    hw2 = np.fromfile(PATCH_OUT / "p2_q_s2_mul_half_x2.out.bin", dtype=np.int8)
    q_mid2 = np.clip(np.rint(inp1 * np.float32(0.5)), -128, 127)
    y_real2 = np.clip(q_mid2 * np.float32(0.125), np.float32(-8.0), np.float32(7.9375))
    pred2 = np.clip(np.rint(y_real2 * np.float32(16.0)), -128, 127).astype(np.int8)
    print(f"p2_q_s2_mul_half_x2 exact: {int((hw2 == pred2).sum())}/4096")
    if (hw2 != pred2).sum():
        bad = np.where(hw2 != pred2)[0][:5]
        print("  diffs:", [(int(i), int(inp1[i]), float(q_mid2[i]), int(hw2[i]), int(pred2[i])) for i in bad])

    # 3. p3_mul_nonpot_s1_to_05 (mul_nonpot with Stage 1 0x100@100 mul = 0.5, Stage 2 untouched s_out=0.03 -> 1/0.03=33.333336)
    # Wait! Note Stage 1 in mul_nonpot had max=129 (127 - (-2)) and min=-126 (-128 - (-2)) AND zp_out=-2 (`0x61 = 0x1f8`)!
    # Also remember mul_nonpot has 0x640@36 = 1/0.05 = 20, @76 = 1/0.04 = 25, @107 = 1/0.03 = 33.333336!
    inp3 = np.fromfile(manifest["mul_nonpot"]["in_path"], dtype=np.int8)
    hw3 = np.fromfile(PATCH_OUT / "p3_mul_nonpot_s1_to_05.out.bin", dtype=np.int8)
    x0_3 = inp3[:4096].astype(np.float32) - np.float32(-3)
    x1_3 = inp3[4096:].astype(np.float32) - np.float32(7)
    # Test: does Stage 1 compute clip(rint(x0_3 * x1_3 * 0.5) - 2, -128, 127)? OR rint(x0_3*x1_3*0.5) clamped to [-126, 129] then Stage 2 * 0.03 * 33.3333 + zp?
    # Note Stage 2 in mul_nonpot had min=-3.78, max=3.87 (which corresponds to q_mid in [-126, 129] * 0.03!)
    q_mid3 = np.clip(np.rint(x0_3 * x1_3 * np.float32(0.5)), -126, 129)
    pred3_s1 = np.clip(q_mid3 + (-2), -128, 127).astype(np.int8)
    print(f"p3_mul_nonpot_s1_to_05 (Stage1=0.5 + Stage2 clamp/s2) exact: {int((hw3 == pred3_s1).sum())}/4096")
    if (hw3 != pred3_s1).sum():
        bad = np.where(hw3 != pred3_s1)[0][:5]
        print("  diffs:", [(int(i), int(x0_3[i]), int(x1_3[i]), int(hw3[i]), int(pred3_s1[i])) for i in bad])

    # 4. p4_add_relu_to_clip_m128_96 (add_act_relu with 0x5d=0x160000 (+96) & 0x5f=0x0030c0 (-128.0f))
    inp4 = np.fromfile(manifest["add_act_relu"]["in_path"], dtype=np.int8)
    hw4 = np.fromfile(PATCH_OUT / "p4_add_relu_to_clip_m128_96.out.bin", dtype=np.int8)
    pred4 = np.clip(inp4[:4096].astype(np.int32) + inp4[4096:].astype(np.int32), -128, 96).astype(np.int8)
    print(f"p4_add_relu_to_clip_m128_96 exact: {int((hw4 == pred4).sum())}/4096")
    if (hw4 != pred4).sum():
        bad = np.where(hw4 != pred4)[0][:5]
        print("  diffs:", [(int(i), int(inp4[i]), int(inp4[4096+i]), int(hw4[i]), int(pred4[i])) for i in bad])

    # 5. p5_q_zp_out_1_to_10 (q_zp_in0_out1 with 0x61=0x114 (zp_out=10), 0x5d=0x075000 (117*0.0625=7.3125f))
    inp5 = np.fromfile(manifest["q_zp_in0_out1"]["in_path"], dtype=np.int8).astype(np.int32)
    hw5 = np.fromfile(PATCH_OUT / "p5_q_zp_out_1_to_10.out.bin", dtype=np.int8)
    # Note 0x5e=0x800001 kept min=-129*0.0625 -> no lower clip changes unless < -129; max=117 -> +127!
    pred5 = np.clip(inp5 + 10, -128, 127).astype(np.int8)
    print(f"p5_q_zp_out_1_to_10 exact: {int((hw5 == pred5).sum())}/4096")
    if (hw5 != pred5).sum():
        bad = np.where(hw5 != pred5)[0][:5]
        print("  diffs:", [(int(i), int(inp5[i]), int(hw5[i]), int(pred5[i])) for i in bad])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify FP32-RNE math on causal vector patch probes")
    ap.add_argument("--corpus-dir", type=Path, default=BASE, help="External corpus root (or FREETPU_CORPUS_DIR)")
    args = ap.parse_args()
    if args.corpus_dir is None or not args.corpus_dir.is_dir():
        print("verify_patch_proof: SKIPPED (pass --corpus-dir or set FREETPU_CORPUS_DIR)")
        raise SystemExit(0)
    PATCH_OUT = args.corpus_dir / "work/device-runner/phase3_isa_vec_patch_proof"
    B1B_IN = args.corpus_dir / "work/isa-vec/batch1b"
    main()
