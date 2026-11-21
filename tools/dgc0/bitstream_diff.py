#!/usr/bin/env python3
"""Structural + 64-bit LSB-first VLIW XOR differ for DarwiNN DGC0 binaries.

Compares two or more Tensor G5/G6 compiled `DGC0` files against the first
(baseline) file, highlighting:
  - Container-level & vtable field diffs (`DGC0`, `DWS0`, `Executable`, `DIVE`)
  - RV64 interpreter parity & parameter/weight byte diffs
  - Chunk-by-chunk 64-bit LSB-first VLIW word XOR diffs with exact bit spans
    (both word-local bit indices and chunk-global bit positions)

Zero external pip dependencies; Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure sibling dgc0_dump is importable whether invoked as script or module
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dgc0_dump  # noqa: E402


def _bit_ranges(mask: int) -> List[Tuple[int, int]]:
    """Return contiguous inclusive [lo, hi] bit-index ranges where mask has 1s."""
    ranges: List[Tuple[int, int]] = []
    bit = 0
    while mask:
        if mask & 1:
            lo = bit
            while mask & 1:
                mask >>= 1
                bit += 1
            ranges.append((lo, bit - 1))
        else:
            mask >>= 1
            bit += 1
    return ranges


def _fmt_ranges(ranges: List[Tuple[int, int]], base: int = 0) -> str:
    parts = []
    for lo, hi in ranges:
        glo, ghi = lo + base, hi + base
        parts.append(f"{glo}" if glo == ghi else f"{ghi}:{glo}")
    return "[" + ", ".join(parts) + "]"


def _extract_dive_chunks(blob: bytes, parsed: Dict[str, Any]) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    chunks_bytes: List[bytes] = []
    chunks_meta: List[Dict[str, Any]] = []
    for eb in parsed.get("external_buffers", []):
        if eb.get("kind") == "dive_vm_ir" and "dive_module" in eb:
            dm = eb["dive_module"]
            for ch in dm.get("hardware_instruction_chunks", []):
                off = ch["offset_in_dgc0"]
                sz = ch["size_bytes"]
                chunks_bytes.append(blob[off : off + sz])
                chunks_meta.append(ch)
            break
    return chunks_bytes, chunks_meta


def _extract_params_blob(blob: bytes, parsed: Dict[str, Any]) -> Optional[bytes]:
    for eb in parsed.get("external_buffers", []):
        if eb.get("kind") == "tpu_parameters":
            return blob[eb["offset"] : eb["end_offset"]]
    return None


def _extract_exec_scalars(parsed: Dict[str, Any]) -> Dict[str, Any]:
    dws = parsed.get("package_dws0", {})
    for sg in dws.get("sub_graphs", []):
        for op in sg.get("operations", []):
            me = op.get("detail", {}).get("multi_executable", {})
            execs = me.get("executables", [])
            if execs:
                return execs[0].get("scalars", {})
    return {}


def _diff_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k in sorted(set(a.keys()) | set(b.keys())):
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            out[k] = {"baseline": va, "target": vb}
    return out


def diff_vliw_chunks(
    base_bytes: bytes,
    targ_bytes: bytes,
    max_words: int = 64,
) -> Dict[str, Any]:
    """Perform 64-bit LSB-first word XOR diff between two VLIW bitstream chunks."""
    common_bytes = min(len(base_bytes), len(targ_bytes))
    common_u64 = common_bytes // 8
    total_changed_words = 0
    total_changed_bits = 0
    word_diffs: List[Dict[str, Any]] = []

    for wi in range(common_u64):
        wa = struct.unpack_from("<Q", base_bytes, wi * 8)[0]
        wb = struct.unpack_from("<Q", targ_bytes, wi * 8)[0]
        x = wa ^ wb
        if x != 0:
            total_changed_words += 1
            pop = x.bit_count()
            total_changed_bits += pop
            if len(word_diffs) < max_words:
                ranges = _bit_ranges(x)
                word_diffs.append({
                    "word_index": wi,
                    "byte_offset": wi * 8,
                    "baseline_hex": f"0x{wa:016x}",
                    "target_hex": f"0x{wb:016x}",
                    "xor_hex": f"0x{x:016x}",
                    "changed_bits": pop,
                    "word_bit_ranges": [[lo, hi] for lo, hi in ranges],
                    "global_bit_ranges": [[lo + wi * 64, hi + wi * 64] for lo, hi in ranges],
                })

    return {
        "baseline_bytes": len(base_bytes),
        "target_bytes": len(targ_bytes),
        "identical": base_bytes == targ_bytes,
        "common_u64_words": common_u64,
        "changed_u64_words": total_changed_words,
        "changed_bits_in_common": total_changed_bits,
        "truncated_word_diffs": total_changed_words > max_words,
        "word_diffs": word_diffs,
    }


def compare_pair(
    base_path: Path,
    base_blob: bytes,
    base_parsed: Dict[str, Any],
    targ_path: Path,
    targ_blob: bytes,
    targ_parsed: Dict[str, Any],
    max_words: int = 64,
) -> Dict[str, Any]:
    # 1. Top-level DGC0 & vtable diffs
    top_a = {k: base_parsed.get(k) for k in ("size_bytes", "version", "model_hash", "compiler_cl", "alignment")}
    top_b = {k: targ_parsed.get(k) for k in ("size_bytes", "version", "model_hash", "compiler_cl", "alignment")}
    dgc0_diff = _diff_dicts(top_a, top_b)

    # 2. DWS0 Package & Executable scalar diffs
    dws_a = base_parsed.get("package_dws0", {})
    dws_b = targ_parsed.get("package_dws0", {})
    pkg_fields = ("size_bytes", "version", "model_identifier", "compiler_cl", "compiler_release_version")
    pkg_diff = _diff_dicts({k: dws_a.get(k) for k in pkg_fields}, {k: dws_b.get(k) for k in pkg_fields})
    exec_diff = _diff_dicts(_extract_exec_scalars(base_parsed), _extract_exec_scalars(targ_parsed))

    # 3. SubGraph tensor summary comparison
    def _tensor_summary(p: Dict[str, Any]) -> List[Dict[str, Any]]:
        sgs = p.get("package_dws0", {}).get("sub_graphs", [])
        if not sgs:
            return []
        return [
            {"name": t["name"], "dims": t["dims"], "dtype": t["data_type_name"], "storage": t["storage_type_name"]}
            for t in sgs[0].get("tensors", [])
        ]

    # 4. External buffers (RV64 interpreter parity & params byte diff)
    interp_a = next((e for e in base_parsed.get("external_buffers", []) if e["kind"] == "rv64_dive_interpreter"), None)
    interp_b = next((e for e in targ_parsed.get("external_buffers", []) if e["kind"] == "rv64_dive_interpreter"), None)
    interp_parity = {
        "baseline_sha256": interp_a["sha256"] if interp_a else None,
        "target_sha256": interp_b["sha256"] if interp_b else None,
        "identical": (interp_a is not None and interp_b is not None and interp_a["sha256"] == interp_b["sha256"]),
    }

    pa = _extract_params_blob(base_blob, base_parsed)
    pb = _extract_params_blob(targ_blob, targ_parsed)
    params_diff: Dict[str, Any] = {
        "baseline_bytes": len(pa) if pa is not None else 0,
        "target_bytes": len(pb) if pb is not None else 0,
        "identical": pa == pb,
    }
    if pa is not None and pb is not None and pa != pb:
        common = min(len(pa), len(pb))
        changed = [i for i in range(common) if pa[i] != pb[i]]
        params_diff["changed_bytes_in_common"] = len(changed)
        params_diff["first_changed_offsets"] = changed[:32]

    # 5. VLIW hardware instruction chunks XOR diff
    ca_bytes, ca_meta = _extract_dive_chunks(base_blob, base_parsed)
    cb_bytes, cb_meta = _extract_dive_chunks(targ_blob, targ_parsed)
    n_chunks = max(len(ca_bytes), len(cb_bytes))
    chunk_diffs: List[Dict[str, Any]] = []
    for ci in range(n_chunks):
        ma = ca_meta[ci] if ci < len(ca_meta) else {}
        mb = cb_meta[ci] if ci < len(cb_meta) else {}
        ba = ca_bytes[ci] if ci < len(ca_bytes) else b""
        bb = cb_bytes[ci] if ci < len(cb_bytes) else b""
        rec: Dict[str, Any] = {
            "chunk_index": ci,
            "baseline_dispatch_op": ma.get("dispatch_op_name"),
            "target_dispatch_op": mb.get("dispatch_op_name"),
            "baseline_dma_head": ma.get("decoded_dma_head_64b"),
            "target_dma_head": mb.get("decoded_dma_head_64b"),
            "vliw_xor": diff_vliw_chunks(ba, bb, max_words=max_words),
        }
        chunk_diffs.append(rec)

    return {
        "baseline": str(base_path),
        "target": str(targ_path),
        "dgc0_diff": dgc0_diff,
        "package_dws0_diff": pkg_diff,
        "executable_scalars_diff": exec_diff,
        "baseline_tensors": _tensor_summary(base_parsed),
        "target_tensors": _tensor_summary(targ_parsed),
        "rv64_interpreter_parity": interp_parity,
        "parameters_diff": params_diff,
        "vliw_chunk_diffs": chunk_diffs,
    }


def format_report(reports: List[Dict[str, Any]], max_words_print: int = 12) -> str:
    lines: List[str] = []
    for rep in reports:
        lines.append("=" * 78)
        lines.append(f"DIFF: {Path(rep['baseline']).name}  vs  {Path(rep['target']).name}")
        lines.append("=" * 78)
        if rep["dgc0_diff"]:
            lines.append("  [DGC0 Container Diff]")
            for k, v in rep["dgc0_diff"].items():
                lines.append(f"    {k}: {v['baseline']} -> {v['target']}")
        if rep["executable_scalars_diff"]:
            lines.append("  [Executable Scalars Diff]")
            for k, v in rep["executable_scalars_diff"].items():
                lines.append(f"    {k}: {v['baseline']} -> {v['target']}")
        ip = rep["rv64_interpreter_parity"]
        lines.append(f"  [RV64 Interpreter] identical={ip['identical']}")
        pd = rep["parameters_diff"]
        lines.append(
            f"  [Parameters Blob]  {pd['baseline_bytes']} B vs {pd['target_bytes']} B "
            f"(identical={pd['identical']}, changed_common={pd.get('changed_bytes_in_common', 0)})"
        )
        lines.append("  [VLIW Hardware Instruction Chunks (64-bit LSB-first XOR)]")
        for ch in rep["vliw_chunk_diffs"]:
            vx = ch["vliw_xor"]
            ci = ch["chunk_index"]
            op_a = ch["baseline_dispatch_op"] or ""
            status = "IDENTICAL" if vx["identical"] else f"DIFF ({vx['changed_u64_words']}/{vx['common_u64_words']} u64 words, {vx['changed_bits_in_common']} bits)"
            lines.append(
                f"    chunk[{ci:2d}] {vx['baseline_bytes']:4d}B vs {vx['target_bytes']:4d}B | {status} | {op_a}"
            )
            if ch["baseline_dma_head"] or ch["target_dma_head"]:
                da = ch["baseline_dma_head"] or {}
                db = ch["target_dma_head"] or {}
                if da != db:
                    lines.append(
                        f"      DMA64: seq={da.get('hw_seq_id')}->{db.get('hw_seq_id')} "
                        f"ch={da.get('dma_channel_tag')}->{db.get('dma_channel_tag')} "
                        f"bytes={da.get('producer_transfer_bytes')}->{db.get('producer_transfer_bytes')}"
                    )
            for wd in vx["word_diffs"][:max_words_print]:
                wi = wd["word_index"]
                wr = _fmt_ranges([(a, b) for a, b in wd["word_bit_ranges"]])
                gr = _fmt_ranges([(a, b) for a, b in wd["global_bit_ranges"]])
                lines.append(
                    f"      w{wi:03d} (+0x{wd['byte_offset']:03x}): "
                    f"{wd['baseline_hex']} ^ {wd['target_hex']} = {wd['xor_hex']} "
                    f"wbits={wr} gbits={gr}"
                )
            if len(vx["word_diffs"]) > max_words_print or vx["truncated_word_diffs"]:
                rem = vx["changed_u64_words"] - min(len(vx["word_diffs"]), max_words_print)
                if rem > 0:
                    lines.append(f"      ... ({rem} more changed u64 words)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Structural + 64-bit LSB-first VLIW XOR differ for DGC0 files")
    parser.add_argument("files", nargs="+", type=Path, help="Two or more .dgc0 files (first is baseline)")
    parser.add_argument("--max-words", type=int, default=64, help="Max changed u64 words recorded per chunk")
    parser.add_argument("--json", dest="json_out", type=Path, default=None, help="Write full diff JSON to path")
    args = parser.parse_args(argv)

    if len(args.files) < 2:
        parser.error("Provide at least two .dgc0 files to compare")

    base_path = args.files[0]
    base_blob = base_path.read_bytes()
    base_parsed = dgc0_dump.parse_dgc0(base_blob)

    reports: List[Dict[str, Any]] = []
    for targ_path in args.files[1:]:
        targ_blob = targ_path.read_bytes()
        targ_parsed = dgc0_dump.parse_dgc0(targ_blob)
        reports.append(
            compare_pair(base_path, base_blob, base_parsed, targ_path, targ_blob, targ_parsed, max_words=args.max_words)
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")

    print(format_report(reports))
    if args.json_out is not None:
        print(f"[wrote JSON] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
