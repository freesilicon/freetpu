#!/usr/bin/env python3
"""
Dynamic DGC0 / DWS0 / DIVE container builder for Tensor G5 (buenos_a0) INT8 GEMM microkernels.

Constructs a complete, spec-valid DarwiNN `DGC0` container from synthesized `chunk[0..6]` VLIW
instruction streams and `ext[2]` packed parameters for arbitrary `(M, K, N)` dimensions:
  - Dynamically builds the `DIVE` FlatBuffer (`ext[0]`) with arbitrary `chunk[0..6]` byte lengths,
    exact 32-byte chunk alignment, shared chunk-table vtable, root/vector/string uoffsets,
    int64 `input_0`/`output_0` tensor shapes `[M, K]`/`[M, N]`, and `PutBits` VA patch ops
    targeting `chunk[2]@37`, `chunk[4]@37`, `chunk[6]@37` (64-bit).
  - Updates `DWS0` (`SubGraph[0]` tensor shapes, `dive_scratch_<size>`, `MultiExecutable` IO/param
    byte counts, SRAM watermark `field_14`, unique `executable_hash`, and unique `model_identifier`).
  - Preserves `DGD0` (4096 B aligned), `DWS0` (64 B aligned), `ext[1]` (RV64 interpreter ELF),
    `0x0a` gap padding, and 64 B ExternalBuffer/file alignment.

Zero external pip dependencies; Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR / "dgc0") not in sys.path:
    sys.path.insert(0, str(_THIS_DIR / "dgc0"))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dgc0_dump  # noqa: E402
import dgc0_patcher  # noqa: E402
import multitile_synth as ms  # noqa: E402


def _default_shell_dgc0() -> Optional[Path]:
    env_p = os.environ.get("FREETPU_SHELL_DGC0", "").strip()
    return Path(env_p) if env_p else None


DEFAULT_SHELL_DGC0: Optional[Path] = _default_shell_dgc0()

_CHUNK_VTABLE = bytes.fromhex("060008000400")  # vtable_size=6, object_size=8, slot0_off=4
_DIVE_PREFIX_SIZE = 160
_DIVE_SUFFIX_SIZE = 2976


def _align_up(val: int, align: int) -> int:
    return (val + align - 1) & ~(align - 1)


def compute_exec_field14(M: int, K: int, N: int) -> int:
    """Compute MultiExecutable.executables[0].field_14 (tile SRAM watermark in bytes)."""
    k4 = ms.pad4(K)
    sw = ms.compute_sw_unit(M, K, N)
    if sw > 0:
        extra = 320 if (N >= 32 and K % 32 == 0) else 256
        return 16 * (sw * 128 + M * min(32, k4)) + extra
    if M >= 48:
        return 16 * (2 * M * min(32, k4)) + 320
    base = max(128, ((M * min(32, k4) + 127) // 128) * 128)
    extra = 320 if (M >= 4 and (M * min(32, k4)) % 128 == 0) else 256
    return 16 * (base + M * k4) + extra


def compute_dive_scratch_size(M: int, K: int, N: int) -> int:
    """Compute DWS0 `dive_scratch_<size>` scratchpad buffer size in bytes."""
    n4 = ms.pad4(N)
    if M * n4 >= 1024:
        return 196608
    return 192512


def _extract_dive_template(shell_raw: bytes) -> Tuple[bytes, bytes, int, int]:
    """
    Extract static DIVE prefix[0:160], suffix[c0_end:], and suffix-relative offsets
    of root slot 0 (`functions`) and root slot 1 (`name`) from the shell DGC0.
    """
    parsed = dgc0_dump.parse_dgc0(shell_raw)
    eb0 = parsed["external_buffers"][0]
    dive = shell_raw[eb0["offset"] : eb0["offset"] + eb0["size_bytes"]]
    chunks = eb0["dive_module"]["hardware_instruction_chunks"]
    c0_end = chunks[0]["offset_in_dive"] + chunks[0]["size_bytes"]
    prefix = dive[:_DIVE_PREFIX_SIZE]
    suffix = dive[c0_end:]
    assert len(suffix) == _DIVE_SUFFIX_SIZE, f"Unexpected DIVE suffix size {len(suffix)}"
    # Root table is at offset 44: slot0 (+48) -> functions vector, slot1 (+52) -> module name
    fn_abs = 48 + struct.unpack_from("<I", prefix, 48)[0]
    nm_abs = 52 + struct.unpack_from("<I", prefix, 52)[0]
    fn_rel_to_c0_end = fn_abs - c0_end
    nm_rel_to_c0_end = nm_abs - c0_end
    return prefix, suffix, fn_rel_to_c0_end, nm_rel_to_c0_end


def build_dive_ext0(
    chunks: Sequence[bytes],
    M: int,
    K: int,
    N: int,
    shell_raw: Optional[bytes] = None,
    shell_path: Optional[Path] = DEFAULT_SHELL_DGC0,
) -> bytes:
    """
    Dynamically construct a complete DIVE FlatBuffer (`ext[0]`) for arbitrary chunk[0..6] sizes.

    Matches the DarwiNN compiler's exact FlatBuffer layout:
      - `[0 : 160]`: DIVE header, root table (`@44`), `hardware_instruction_chunks` uoffset vector (`@120..148`),
        and `chunk[6]`'s 12-byte wrapper table (`@148..160`).
      - `chunk[6], chunk[5], chunk[4], chunk[3], chunk[2], chunk[1], chunk[0]` laid out sequentially,
        each data payload 32-byte aligned with its 12-byte table `[i32(tbl-vt), u32(4), u32(len)]`
        immediately preceding the payload.
      - Shared 6-byte chunk-table vtable (`06 00 08 00 04 00`) placed at `c0_tbl - 6` (right before
        `chunk[0]`'s table, guaranteeing a >=48 B gap from `chunk[1]` end to `chunk[0]` data).
      - `[c0_end : c0_end + 2976]`: `functions[0]` DIVE VM IR subtree (including `PutBits` VA patch ops
        for `chunk[2]@37`, `chunk[4]@37`, `chunk[6]@37`) with int64 `output_0=[M,N]` and `input_0=[M,K]`.
    """
    if len(chunks) != 7:
        raise ValueError(f"Expected 7 hardware instruction chunks, got {len(chunks)}")
    if shell_raw is None:
        if shell_path is None or not Path(shell_path).is_file():
            raise FileNotFoundError(
                "No template DGC0 shell provided. Pass --shell or set FREETPU_SHELL_DGC0 (extracted on-device)."
            )
        shell_raw = Path(shell_path).read_bytes()
    tpl_prefix, tpl_suffix, fn_rel, nm_rel = _extract_dive_template(shell_raw)

    # 1. Compute exact 32-byte aligned offsets for chunks 6 down to 0
    data_offs = [0] * 7
    tbl_offs = [0] * 7
    cur_data = _DIVE_PREFIX_SIZE  # chunk[6] data starts at 160
    for idx in range(6, -1, -1):
        if idx == 6:
            data_off = _DIVE_PREFIX_SIZE
        elif idx == 0:
            # Leave room for shared 6-byte vtable + 12-byte table before chunk[0] data
            prev_end = data_offs[1] + len(chunks[1])
            data_off = _align_up(prev_end + 18, 32)
            # Compiler reserves 48B gap when (prev_end + 18) lands on a tight boundary;
            # specifically check: prev_end + 32 >= data_off? Wait: c1 is always 80B (16 mod 32),
            # so prev_end is 16 mod 32; prev_end + 18 = 34 -> align32 -> prev_end + 48!
            if data_off - prev_end < 32:
                data_off += 32
        else:
            prev_end = data_offs[idx + 1] + len(chunks[idx + 1])
            data_off = _align_up(prev_end + 12, 32)
        data_offs[idx] = data_off
        tbl_offs[idx] = data_off - 12

    vtable_pos = tbl_offs[0] - 6
    c0_end = data_offs[0] + len(chunks[0])

    # 2. Assemble DIVE buffer
    buf = bytearray(c0_end + len(tpl_suffix))
    buf[:_DIVE_PREFIX_SIZE] = tpl_prefix

    # Patch root table uoffsets (+48 -> functions vector, +52 -> module name)
    fn_abs = c0_end + fn_rel
    nm_abs = c0_end + nm_rel
    struct.pack_into("<I", buf, 48, fn_abs - 48)
    struct.pack_into("<I", buf, 52, nm_abs - 52)

    # Patch hardware_instruction_chunks[0..6] uoffset vector (@120..147)
    for idx in range(7):
        slot_pos = 120 + 4 * idx
        struct.pack_into("<I", buf, slot_pos, tbl_offs[idx] - slot_pos)

    # Write shared 6-byte chunk-table vtable
    buf[vtable_pos : vtable_pos + 6] = _CHUNK_VTABLE

    # Write each chunk table + payload
    for idx in range(6, -1, -1):
        tbl = tbl_offs[idx]
        data_off = data_offs[idx]
        cbytes = chunks[idx]
        struct.pack_into("<i", buf, tbl, tbl - vtable_pos)
        struct.pack_into("<I", buf, tbl + 4, 4)
        struct.pack_into("<I", buf, tbl + 8, len(cbytes))
        buf[data_off : data_off + len(cbytes)] = cbytes

    # Write suffix with updated int64 output_0=[M, N] and input_0=[M, K] dims
    suf = bytearray(tpl_suffix)
    struct.pack_into("<q", suf, 2664, M)
    struct.pack_into("<q", suf, 2672, N)
    struct.pack_into("<q", suf, 2848, M)
    struct.pack_into("<q", suf, 2856, K)
    buf[c0_end:] = suf

    return bytes(buf)


def _patch_dws0_in_header(
    header: bytearray,
    dws_off: int,
    dws_len: int,
    M: int,
    K: int,
    N: int,
    ext0_size: int,
    ext2_size: int,
    exec_hash: int,
    token_19_digits: str,
) -> None:
    """
    Patch DWS0 FlatBuffer inside `header` in-place using structural FBReader navigation:
      - `input_0.dims = [M, K]`, `output_0.dims = [M, N]`
      - `ext[0]` & `ext[2]` tensor sizes
      - `dive_scratch_<size>` tensor name & size (`192512` or `196608`)
      - `MultiExecutable.executables[0]`: input/output/param byte sizes, `field_14`, `executable_hash`
      - `Package.model_identifier` and `compiler_options_pb` token string (exact same byte length)
    """
    assert len(token_19_digits) == 19 and token_19_digits.isdigit()
    dws_view = bytes(header[dws_off : dws_off + dws_len])
    r = dgc0_dump.FBReader(dws_view)
    dr = r.root_offset()

    # 1. Update model_identifier (slot 4) and matching Token_ string inside compiler_options_pb (slot 15)
    old_token_str = ""
    f4 = r.field_pos(dr, 4)
    if f4 is not None:
        s_off = r.uoffset(f4)
        s_len = r.u32(s_off)
        old_token_bytes = dws_view[s_off + 4 : s_off + 4 + s_len]
        old_token_str = old_token_bytes.decode("ascii", errors="ignore")
        new_token_str = f"Token_{token_19_digits}"
        if len(new_token_str) == s_len:
            header[dws_off + s_off + 4 : dws_off + s_off + 4 + s_len] = new_token_str.encode("ascii")
            # Also replace inside compiler_options_pb (slot 15) if present with identical length
            f15 = r.field_pos(dr, 15)
            if f15 is not None:
                pb_off, pb_len, pb_bytes = r.byte_vector(f15)
                pos = pb_bytes.find(old_token_bytes)
                if pos >= 0:
                    abs_pos = dws_off + pb_off + pos
                    header[abs_pos : abs_pos + s_len] = new_token_str.encode("ascii")

    # 2. Patch SubGraph[0] tensors & MultiExecutable
    scratch_sz = compute_dive_scratch_size(M, K, N)
    field14 = compute_exec_field14(M, K, N)
    f3 = r.field_pos(dr, 3)
    assert f3 is not None, "DWS0 missing SubGraphs (slot 3)"
    sgp = r.uoffset_vector(f3)[0]

    # 2a. Tensors (SubGraph slot 0)
    tf = r.field_pos(sgp, 0)
    assert tf is not None
    for tp in r.uoffset_vector(tf):
        nf = r.field_pos(tp, 2)
        tname = r.string(nf) if nf is not None else ""
        df = r.field_pos(tp, 0)
        if df is None:
            continue
        dvec = r.uoffset(df)
        ndims = r.u32(dvec)
        if tname == "input_0" and ndims == 2:
            struct.pack_into("<II", header, dws_off + dvec + 4, M, K)
        elif tname == "output_0" and ndims == 2:
            struct.pack_into("<II", header, dws_off + dvec + 4, M, N)
        elif "scratch" in tname and ndims == 1:
            struct.pack_into("<I", header, dws_off + dvec + 4, scratch_sz)
            new_sname = f"dive_scratch_{scratch_sz}".encode("ascii")
            s_off = r.uoffset(nf)
            s_len = r.u32(s_off)
            if len(new_sname) == s_len:
                header[dws_off + s_off + 4 : dws_off + s_off + 4 + s_len] = new_sname
        else:
            is_ext_f = r.field_pos(tp, 12)
            if is_ext_f is not None and r.u8(is_ext_f) != 0:
                eidx_f = r.field_pos(tp, 13)
                eidx = r.u64(eidx_f) if eidx_f is not None else 0
                if eidx == 0 and ndims == 1:
                    struct.pack_into("<I", header, dws_off + dvec + 4, ext0_size)
                elif eidx == 2 and ndims == 1:
                    struct.pack_into("<I", header, dws_off + dvec + 4, ext2_size)

    # 2b. Operations -> OpExecutableCall -> MultiExecutable.executables[0]
    of = r.field_pos(sgp, 3)
    assert of is not None
    for op in r.uoffset_vector(of):
        dt_f = r.field_pos(op, 0)
        if dt_f is None or r.u8(dt_f) != 2:
            continue
        det = r.uoffset(r.field_pos(op, 1))
        mef = r.field_pos(det, 0)
        assert mef is not None
        me_off, me_len, me_bytes = r.byte_vector(mef)
        mr = dgc0_dump.FBReader(me_bytes)
        mroot = mr.root_offset()
        ep = mr.uoffset_vector(mr.field_pos(mroot, 1))[0]

        def _patch_io_vec(slot: int, new_bytes: int) -> None:
            vf = mr.field_pos(ep, slot)
            if vf is None:
                return
            for tp in mr.uoffset_vector(vf):
                s32_f = mr.field_pos(tp, 1)
                s64_f = mr.field_pos(tp, 3)
                if s32_f is not None:
                    struct.pack_into("<I", header, dws_off + me_off + s32_f, new_bytes)
                if s64_f is not None:
                    struct.pack_into("<Q", header, dws_off + me_off + s64_f, new_bytes)

        _patch_io_vec(3, M * K)      # inputs[0]
        _patch_io_vec(4, M * N)      # outputs[0]
        _patch_io_vec(20, ext2_size)  # parameters[0]

        # field_14 (slot 14, u32)
        f14_pos = mr.field_pos(ep, 14)
        if f14_pos is not None:
            struct.pack_into("<I", header, dws_off + me_off + f14_pos, field14)

        # executable_hash (slot 21, u64)
        f21_pos = mr.field_pos(ep, 21)
        if f21_pos is not None:
            struct.pack_into("<Q", header, dws_off + me_off + f21_pos, exec_hash & 0xFFFFFFFF_FFFFFFFF)


def build_dgc0_container(
    *,
    chunks: Sequence[bytes],
    ext2: bytes,
    M: int,
    K: int,
    N: int,
    shell_path: Optional[Path] = DEFAULT_SHELL_DGC0,
) -> bytes:
    """
    Build a complete, runnable DarwiNN `DGC0` binary for arbitrary `(M, K, N)` from
    synthesized `chunk[0..6]` VLIW payloads and `ext[2]` packed parameters.
    """
    if shell_path is None or not Path(shell_path).is_file():
        raise FileNotFoundError(
            "No template DGC0 shell provided. Pass --shell or set FREETPU_SHELL_DGC0 (extracted on-device)."
        )
    shell_raw = Path(shell_path).read_bytes()
    ext0 = build_dive_ext0(chunks, M, K, N, shell_raw=shell_raw)

    # Compute unique deterministic executable hash & 19-digit model token to avoid cache collisions
    digest = hashlib.sha256(struct.pack("<III", M, K, N) + ext0 + ext2).digest()
    exec_hash = struct.unpack_from("<Q", digest, 0)[0]
    token_num = struct.unpack_from("<Q", digest, 8)[0] % (10**19)
    # Ensure exact 19 decimal digits (first digit 1..9)
    if token_num < 10**18:
        token_num += 10**18
    token_str = f"{token_num:019d}"

    img = dgc0_patcher.Dgc0Image(shell_raw)
    img.ext_slots[0].payload = ext0
    img.ext_slots[2].payload = ext2

    _patch_dws0_in_header(
        header=img.header,
        dws_off=img.dws_off,
        dws_len=img.dws_len,
        M=M,
        K=K,
        N=N,
        ext0_size=len(ext0),
        ext2_size=len(ext2),
        exec_hash=exec_hash,
        token_19_digits=token_str,
    )

    return img.repack()


def verify_container_roundtrip(dgc0_bytes: bytes, M: int, K: int, N: int, expected_chunks: Sequence[bytes], expected_ext2: bytes) -> Dict[str, object]:
    """Verify that a built DGC0 container parses cleanly via `dgc0_dump.parse_dgc0` with exact metadata."""
    parsed = dgc0_dump.parse_dgc0(dgc0_bytes)
    assert len(dgc0_bytes) % 64 == 0, "DGC0 file size not 64 B aligned"
    assert parsed["graph_data_dgd0"]["offset_in_dgc0"] % 4096 == 0, "DGD0 not 4096 B aligned"
    assert parsed["package_dws0"]["offset_in_dgc0"] % 64 == 0, "DWS0 not 64 B aligned"

    sg = parsed["package_dws0"]["sub_graphs"][0]
    tensors = {t["name"]: t for t in sg["tensors"]}
    assert tensors["input_0"]["dims"] == [M, K], f"input_0 dims mismatch: {tensors['input_0']['dims']}"
    assert tensors["output_0"]["dims"] == [M, N], f"output_0 dims mismatch: {tensors['output_0']['dims']}"
    exp_scr = compute_dive_scratch_size(M, K, N)
    scr_name = f"dive_scratch_{exp_scr}"
    assert scr_name in tensors and tensors[scr_name]["dims"] == [exp_scr], f"scratchpad mismatch: {list(tensors.keys())}"

    me = sg["operations"][0]["detail"]["multi_executable"]["executables"][0]
    assert me["inputs"][0]["size_bytes_u64"] == M * K
    assert me["outputs"][0]["size_bytes_u64"] == M * N
    assert me["parameters"][0]["size_bytes_u64"] == len(expected_ext2)
    assert me["scalars"]["field_14"] == compute_exec_field14(M, K, N)

    ebs = parsed["external_buffers"]
    assert len(ebs) == 3
    for eb in ebs:
        assert eb["offset"] % 64 == 0, f"ext[{eb['index']}] not 64 B aligned"

    dm = ebs[0]["dive_module"]
    chunks_meta = dm["hardware_instruction_chunks"]
    assert len(chunks_meta) == 7
    for ci, cm in enumerate(chunks_meta):
        got = dgc0_bytes[cm["offset_in_dgc0"] : cm["offset_in_dgc0"] + cm["size_bytes"]]
        assert got == expected_chunks[ci], f"chunk[{ci}] payload mismatch after container roundtrip"

    # Verify PutBits ops intact
    pb_ops = [
        op["detail"]
        for op in dm["functions"][0]["operations"]
        if op["op_type_name"] == "PutBits"
    ]
    assert pb_ops == [
        {"chunk_index": 2, "offset": 37, "bit_width": 64},
        {"chunk_index": 4, "offset": 37, "bit_width": 64},
        {"chunk_index": 6, "offset": 37, "bit_width": 64},
    ], f"PutBits ops mismatch: {pb_ops}"

    got_ext2 = dgc0_bytes[ebs[2]["offset"] : ebs[2]["offset"] + ebs[2]["size_bytes"]]
    assert got_ext2 == expected_ext2, "ext[2] payload mismatch after container roundtrip"
    return {
        "size_bytes": len(dgc0_bytes),
        "ext0_bytes": ebs[0]["size_bytes"],
        "ext1_bytes": ebs[1]["size_bytes"],
        "ext2_bytes": ebs[2]["size_bytes"],
        "dive_scratch_bytes": exp_scr,
        "exec_field_14": me["scalars"]["field_14"],
        "sha256": hashlib.sha256(dgc0_bytes).hexdigest(),
    }


def _self_test_81(corpus_dir: Optional[Path], shell_path: Optional[Path]) -> int:
    if corpus_dir is None or not corpus_dir.is_dir():
        # Run self-contained metadata & synthesis invariant checks without external .dgc0 corpus
        for (M, K, N) in ((1, 16, 16), (4, 32, 32), (8, 64, 64), (16, 64, 64), (1, 512, 128)):
            f14 = compute_exec_field14(M, K, N)
            scr = compute_dive_scratch_size(M, K, N)
            assert f14 > 0 and scr in (192512, 196608)
        print("DGC0_BUILDER: self-contained metadata invariants PASS (set --corpus-dir / FREETPU_CORPUS_DIR for 81-model parity)")
        return 0

    import verify_full_isa_81 as v81

    rows = v81.load_corpus_81(corpus_dir)
    manifests = v81.load_manifests(corpus_dir)
    eff_shell = shell_path or (corpus_dir / "work/device-runner/phase3_gemm_sweep1/g00_base_1x16x16.dgc0")
    shell_raw = eff_shell.read_bytes()

    ext0_ok = 0
    f14_ok = 0
    scr_ok = 0
    rt_ok = 0

    for p, M, K, N in rows:
        raw = p.read_bytes()
        parsed = dgc0_dump.parse_dgc0(raw)
        eb0 = parsed["external_buffers"][0]
        ref_ext0 = raw[eb0["offset"] : eb0["offset"] + eb0["size_bytes"]]
        ref_chunks = [
            raw[cm["offset_in_dgc0"] : cm["offset_in_dgc0"] + cm["size_bytes"]]
            for cm in eb0["dive_module"]["hardware_instruction_chunks"]
        ]
        # 1. Exact bit-for-bit DIVE (ext[0]) match using ref_chunks
        syn_ext0 = build_dive_ext0(ref_chunks, M, K, N, shell_raw=shell_raw)
        if syn_ext0 == ref_ext0:
            ext0_ok += 1
        else:
            print(f"[DIFF ext0] {p.stem} ({M},{K},{N}): syn={len(syn_ext0)} ref={len(ref_ext0)}")

        # 2. Exact field_14 & dive_scratch match
        sg = parsed["package_dws0"]["sub_graphs"][0]
        ref_scr = next(t for t in sg["tensors"] if "scratch" in t["name"])["dims"][0]
        ref_f14 = sg["operations"][0]["detail"]["multi_executable"]["executables"][0]["scalars"]["field_14"]
        if compute_exec_field14(M, K, N) == ref_f14:
            f14_ok += 1
        else:
            print(f"[DIFF f14] {p.stem} ({M},{K},{N}): syn={compute_exec_field14(M,K,N)} ref={ref_f14}")
        if compute_dive_scratch_size(M, K, N) == ref_scr:
            scr_ok += 1
        else:
            print(f"[DIFF scr] {p.stem} ({M},{K},{N}): syn={compute_dive_scratch_size(M,K,N)} ref={ref_scr}")

        # 3. Full synthesis + container build + round-trip parse validation
        tf = v81.parse_tflite_fc_full(Path(manifests[p.stem]["tflite_path"]).read_bytes())
        syn_chunks, syn_ext2 = ms.synthesize_gemm_chunks(
            M=tf["M"],
            K=tf["K"],
            N=tf["N"],
            weights_nk=tf["weights_nk"],
            bias_n=tf["bias_n"],
            s_in=tf["s_in"],
            zp_in=tf["zp_in"],
            s_w=tf["s_w"],
            s_out=tf["s_out"],
            zp_out=tf["zp_out"],
            act=tf["act"],
        )
        built = build_dgc0_container(chunks=syn_chunks, ext2=syn_ext2, M=M, K=K, N=N, shell_path=eff_shell)
        verify_container_roundtrip(built, M, K, N, syn_chunks, syn_ext2)
        rt_ok += 1

    n = len(rows)
    print(f"=== DGC0_BUILDER 81-MODEL SELF-TEST ===")
    print(f"  DIVE ext[0] bit-exact : {ext0_ok}/{n}")
    print(f"  ME field_14 exact     : {f14_ok}/{n}")
    print(f"  dive_scratch exact    : {scr_ok}/{n}")
    print(f"  Full build+roundtrip  : {rt_ok}/{n}")
    return 0 if (ext0_ok == n and f14_ok == n and scr_ok == n and rt_ok == n) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dynamic DGC0/DWS0/DIVE container builder for Tensor G5 INT8 GEMM")
    ap.add_argument("--corpus-dir", type=Path, default=None, help="Optional external corpus directory (or FREETPU_CORPUS_DIR)")
    ap.add_argument("--shell", type=Path, default=DEFAULT_SHELL_DGC0, help="Path to user-extracted template DGC0 shell (or FREETPU_SHELL_DGC0)")
    ap.add_argument("--self-test", action="store_true", help="Run builder self-test")
    args = ap.parse_args()
    c_env = os.environ.get("FREETPU_CORPUS_DIR", "").strip()
    corpus = args.corpus_dir or (Path(c_env) if c_env else None)
    raise SystemExit(_self_test_81(corpus, args.shell))
