"""Backend 1: Edge TPU Tensor G5 (buenos_a0) lowering & execution.

Lowers portable TileKernelIR directly into:
  - FreeTPU's VLIW GEMM emitter (`tools/emit_custom_gemm.py` + `tools/multitile_synth.py`)
  - Mode-A `0x640` minimax spline fusion (`tools/spline_0x640.py` + `tools/verify_fusion_lut.py`)
  - `libopenbuenos` ctypes C-API bridge (`OPENBUENOS_MOCK=1` offline or live `/dev/edgetpu-soc`)
  - Optional ADB dispatch to rooted Pixel 10 Pro Fold (`rango`) via `freetpu_tile_runner`
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir import (
    DType,
    ExecutionResult,
    LoweredArtifact,
    OpKind,
    QuantParams,
    Target,
    TileKernelIR,
)
from ..libopenbuenos_bridge import (
    OB_ACT_GELU,
    OB_ACT_NONE,
    OB_ACT_SILU,
    OB_STAT_FLAG_MOCK,
    OpenBuenosSession,
)
from .cpu_ref import quantize_w8a16_weights


def _tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "tools"


def _ensure_tools_on_path() -> None:
    tools = _tools_dir()
    dgc0_dir = tools / "dgc0"
    for p in (str(dgc0_dir), str(tools)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _op_to_ob_act(op_kind: OpKind) -> int:
    if op_kind == OpKind.FUSED_MATMUL_SILU:
        return OB_ACT_SILU
    if op_kind == OpKind.FUSED_MATMUL_GELU:
        return OB_ACT_GELU
    return OB_ACT_NONE


def _is_lt_1x(m: int, k: int, n: int) -> bool:
    dims = {512, 576, 1024, 1536, 2048}
    return m == 1 and k in dims and n in dims


def _is_lt_64x(m: int, k: int, n: int) -> bool:
    return (
        m == 64
        and k in {512, 576, 1024, 1536}
        and 128 <= n <= 1536
        and (n % 32 == 0)
        and (k + n <= 3072)
    )


def _select_buenos_tile_schedule(m: int, k: int, n: int, ir: TileKernelIR) -> Dict[str, Any]:
    if _is_lt_1x(m, k, n):
        return {"regime": "lt_1x_single_tile", "tile_m": 1, "tile_k": k, "tile_n": n, "tiles": 1}
    if _is_lt_64x(m, k, n):
        return {"regime": "lt_64x_single_tile", "tile_m": 64, "tile_k": k, "tile_n": n, "tiles": 1}
    max_m = min(16, ir.block_spec.block_m)
    max_k = min(64, ir.block_spec.block_k)
    max_n = min(64, ir.block_spec.block_n)
    tm_cnt = (m + max_m - 1) // max_m
    tk_cnt = (k + max_k - 1) // max_k
    tn_cnt = (n + max_n - 1) // max_n
    return {
        "regime": "multitile_4x4",
        "tile_m": min(m, max_m),
        "tile_k": min(k, max_k),
        "tile_n": min(n, max_n),
        "tiles": tm_cnt * tk_cnt * tn_cnt,
    }


def _synthesize_mode_a_spline(
    act_name: str,
    *,
    s_mid: float,
    zp_mid: int,
    s_out: float,
    zp_out: int,
) -> Dict[str, Any]:
    _ensure_tools_on_path()
    import spline_0x640 as sp  # type: ignore
    import verify_fusion_lut as vfl  # type: ignore

    coeffs_8x5, bps, fclip_min, fclip_max, max_lsb_err = vfl.build_fused_single_stage_spline(
        act_name,
        s_mid=s_mid,
        zp_mid=zp_mid,
        s_out=s_out,
        zp_out=zp_out,
    )
    pkt = sp.SplinePacket0x640(
        header_76b=(0xFFFF << 12) | 0x640,
        coeffs=coeffs_8x5,
        breakpoints=bps,
        tail_84b=0x7804,
    )
    pkt_bytes = sp.encode_0x640(pkt)
    return {
        "activation": act_name,
        "coeffs_8x5": coeffs_8x5,
        "breakpoints": bps,
        "fclip_min": fclip_min,
        "fclip_max": fclip_max,
        "max_lsb_err": max_lsb_err,
        "pkt_0x640": pkt,
        "pkt_0x640_bytes": pkt_bytes,
    }


def _eval_spline_on_acc_i32(
    acc_i32: np.ndarray,
    *,
    s_in: float,
    s_w: float,
    s_out: float,
    zp_out: int,
    spline_info: Dict[str, Any],
) -> np.ndarray:
    _ensure_tools_on_path()
    import spline_0x640 as sp  # type: ignore

    # Replicate Mode-A chunk[5] hardware dataflow:
    #   1) MAC quantizes acc to intermediate uint8 u_mid at (s_mid = s_in*s_w, zp_mid = zp_out)
    #      or directly when s_mid = max(s_in*s_w, s_out)
    s_mid = float(spline_info["s_mid"])
    zp_mid = int(spline_info["zp_mid"])
    real_mul = np.float32(np.float32(s_in * s_w) / np.float32(s_mid))
    u_mid = np.clip(
        np.rint(acc_i32.astype(np.float32) * real_mul + float(zp_mid + 128)),
        0.0,
        255.0,
    ).astype(np.uint8)
    x_fp32 = ((u_mid.astype(np.int32) - (zp_mid + 128)) * np.float32(s_mid)).astype(np.float32)
    y_fp32 = sp.eval_0x640_horner_fp32(
        x_fp32.reshape(-1),
        spline_info["pkt_0x640"],
        float(spline_info["fclip_min"]),
        float(spline_info["fclip_max"]),
    )
    u_out = np.clip(np.rint(y_fp32), 0.0, 255.0).astype(np.uint8)
    return u_out.view(np.int8).reshape(acc_i32.shape)


def lower_buenos_a0(
    ir: TileKernelIR,
    shapes: Dict[str, Tuple[int, ...]],
    *,
    quant: Optional[QuantParams] = None,
) -> LoweredArtifact:
    """Lower TileKernelIR into buenos_a0 VLIW chunks, 0x640 spline, and libopenbuenos plan."""
    _ensure_tools_on_path()
    import emit_custom_gemm as ecg  # type: ignore
    import multitile_synth as ms  # type: ignore

    qparams = quant or ir.quant
    first_in = shapes[ir.input_names[0]]
    m_dim, k_dim = int(first_in[0]), int(first_in[1])
    if ir.op_kind == OpKind.SWIGLU:
        wg_shape = shapes[ir.swiglu_roles["w_gate"]]
        n_inter = int(wg_shape[0] if ir.rhs_transposed else wg_shape[1])
        if ir.has_down_proj:
            wd_shape = shapes[ir.swiglu_roles["w_down"]]
            n_dim = int(wd_shape[0] if ir.rhs_transposed else wd_shape[1])
        else:
            n_dim = n_inter
    else:
        b_shape = shapes[ir.input_names[1]]
        n_dim = int(b_shape[0] if ir.rhs_transposed else b_shape[1])
        n_inter = n_dim

    schedule = _select_buenos_tile_schedule(m_dim, k_dim, n_inter, ir)
    synth_m = min(m_dim, 16)
    synth_k = min(k_dim, 64)
    synth_n = min(n_inter, 64)
    # Synthesize representative VLIW chunks + ext[2] via multitile_synth / emit_custom_gemm
    w_zero = [[0] * synth_k for _ in range(synth_n)]
    b_zero = [0] * synth_n
    chunks, ext2 = ms.synthesize_gemm_chunks(
        M=synth_m,
        K=synth_k,
        N=synth_n,
        weights_nk=w_zero,
        bias_n=b_zero,
        s_in=qparams.s_in,
        zp_in=qparams.zp_in,
        s_w=qparams.s_w,
        s_out=qparams.s_out,
        zp_out=qparams.zp_out,
        act=ecg.ACT_NONE,
    )

    spline_meta: Optional[Dict[str, Any]] = None
    act_name: Optional[str] = None
    if ir.op_kind in (OpKind.FUSED_MATMUL_SILU, OpKind.SWIGLU):
        act_name = "silu"
    elif ir.op_kind == OpKind.FUSED_MATMUL_GELU:
        act_name = "gelu"

    if act_name is not None:
        s_mid = max(float(qparams.s_in * qparams.s_w), float(qparams.s_out))
        zp_mid = int(qparams.zp_out)
        # Ensure s_mid covers activation dynamic range without negative saturation
        min_val = -0.2784645427610738 if act_name == "silu" else -0.1700404099869362
        s_out_eff = float(qparams.s_out)
        if (min_val / s_out_eff + qparams.zp_out) < -128.0:
            s_out_eff = max(s_out_eff, abs(min_val) / max(1.0, 127.0 + qparams.zp_out))
        spline_meta = _synthesize_mode_a_spline(
            act_name,
            s_mid=s_mid,
            zp_mid=zp_mid,
            s_out=s_out_eff,
            zp_out=qparams.zp_out,
        )
        spline_meta["s_mid"] = s_mid
        spline_meta["zp_mid"] = zp_mid
        spline_meta["s_out_eff"] = s_out_eff

    dgc0_bytes: Optional[bytes] = None
    shell_env = os.environ.get("FREETPU_SHELL_DGC0", "").strip()
    if shell_env and Path(shell_env).is_file():
        dgc0_bytes, _, _, _ = ecg.emit_custom_gemm_dgc0(
            weights_nk=w_zero,
            bias_n=b_zero,
            x_mk=[[0] * synth_k for _ in range(synth_m)],
            s_in=qparams.s_in,
            zp_in=qparams.zp_in,
            s_w=qparams.s_w,
            s_out=qparams.s_out,
            zp_out=qparams.zp_out,
            act=ecg.ACT_NONE,
            shell_path=Path(shell_env),
        )

    summary: Dict[str, Any] = {
        "backend": "buenos_a0",
        "op_kind": ir.op_kind.value,
        "dtype": ir.dtype.value,
        "schedule": schedule,
        "vliw_emitter": "tools/emit_custom_gemm.py + tools/multitile_synth.py",
        "vliw_synth_shape": [synth_m, synth_k, synth_n],
        "chunk_sizes": [len(c) for c in chunks],
        "ext2_bytes": len(ext2),
        "libopenbuenos_op": (
            "ob_gemm_synthesize_dgc0 + ob_graph_invoke"
            if ir.dtype == DType.INT8
            else "ob_gemm_matmul_f32_quant_batch"
        ),
        "mode_a_0x640_spline": (
            {
                "activation": spline_meta["activation"],
                "packet_bytes": len(spline_meta["pkt_0x640_bytes"]),
                "tail_84b": "0x7804",
                "max_lsb_err": round(float(spline_meta["max_lsb_err"]), 6),
                "fclip": [spline_meta["fclip_min"], spline_meta["fclip_max"]],
                "breakpoints": [round(float(b), 6) for b in spline_meta["breakpoints"]],
            }
            if spline_meta is not None
            else None
        ),
        "dgc0_emitted_bytes": len(dgc0_bytes) if dgc0_bytes is not None else 0,
    }
    if spline_meta is not None:
        summary["_spline_runtime"] = spline_meta

    return LoweredArtifact(
        target=Target.BUENOS_A0,
        ir=ir,
        shapes=shapes,
        summary=summary,
        dgc0_bytes=dgc0_bytes,
        vliw_chunks=chunks,
        ext2_bytes=ext2,
    )


def _run_adb_rango(
    ir: TileKernelIR,
    inputs: Dict[str, Any],
    *,
    adb_serial: str,
    quant: QuantParams,
    bias: Optional[np.ndarray],
    scales: Optional[Dict[str, np.ndarray]],
    warmup: int,
    iters: int,
    perf_mode: int,
) -> ExecutionResult:
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_runner = repo_root / "runtime" / "build" / "freetpu_tile_runner"
    if not local_runner.is_file():
        res = subprocess.run(
            ["make", "-C", str(repo_root / "runtime"), "tile-runner"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not local_runner.is_file():
            raise RuntimeError(f"Failed building Android ARM64 freetpu_tile_runner:\n{res.stderr}")

    remote_dir = "/data/local/tmp/freetpu_tile_run"
    remote_runner = "/data/local/tmp/tpu_open_stack/freetpu_tile_runner"

    def _adb(*cmd_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["adb", "-s", adb_serial, *cmd_args],
            capture_output=True,
            text=True,
            check=False,
        )

    # Ensure remote workspace and runner binary
    _adb("shell", f"mkdir -p {remote_dir}")
    push_r = _adb("push", str(local_runner), remote_runner)
    if push_r.returncode != 0:
        raise RuntimeError(f"adb push runner failed: {push_r.stderr}")
    _adb("shell", f"chmod 0755 {remote_runner}")

    with tempfile.TemporaryDirectory(prefix="freetpu-adb-") as tmp:
        tmp_p = Path(tmp)
        cmd_parts: List[str] = [
            remote_runner,
            "--op",
            ir.op_kind.value,
            "--dtype",
            ir.dtype.value,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--perf-mode",
            str(perf_mode),
        ]

        if ir.op_kind == OpKind.SWIGLU:
            x = np.asarray(inputs[ir.swiglu_roles["x"]], dtype=np.float32)
            wg_name = ir.swiglu_roles["w_gate"]
            wu_name = ir.swiglu_roles["w_up"]
            wd_name = ir.swiglu_roles.get("w_down")

            def _to_f32_nk(arr: np.ndarray, name: str) -> np.ndarray:
                a = np.asarray(arr)
                if a.dtype == np.int8 and scales and name in scales:
                    s = np.asarray(scales[name], dtype=np.float32)
                    a = a.astype(np.float32) * (s[:, None] if ir.rhs_transposed else s[None, :])
                else:
                    a = a.astype(np.float32)
                return np.ascontiguousarray(a if ir.rhs_transposed else a.T, dtype=np.float32)

            wg_nk = _to_f32_nk(inputs[wg_name], wg_name)
            wu_nk = _to_f32_nk(inputs[wu_name], wu_name)
            wd_nk = _to_f32_nk(inputs[wd_name], wd_name) if wd_name else None
            m_dim, k_dim = x.shape
            n_inter = wg_nk.shape[0]
            n_out = wd_nk.shape[0] if wd_nk is not None else n_inter
            n_dim = n_out

            (tmp_p / "in.bin").write_bytes(np.ascontiguousarray(x, dtype=np.float32).tobytes())
            (tmp_p / "wg.bin").write_bytes(wg_nk.tobytes())
            (tmp_p / "wu.bin").write_bytes(wu_nk.tobytes())
            for fn in ("in.bin", "wg.bin", "wu.bin"):
                _adb("push", str(tmp_p / fn), f"{remote_dir}/{fn}")
            if wd_nk is not None:
                (tmp_p / "wd.bin").write_bytes(wd_nk.tobytes())
                _adb("push", str(tmp_p / "wd.bin"), f"{remote_dir}/wd.bin")
            cmd_parts.extend([
                "--m", str(m_dim),
                "--k", str(k_dim),
                "--n", str(n_out),
                "--n-inter", str(n_inter),
                "--in", f"{remote_dir}/in.bin",
                "--w", f"{remote_dir}/wg.bin",
                "--w-up", f"{remote_dir}/wu.bin",
                "--w-down", f"{remote_dir}/wd.bin",
                "--out", f"{remote_dir}/out.bin",
            ])
        else:
            a = np.asarray(inputs[ir.input_names[0]])
            b = np.asarray(inputs[ir.input_names[1]])
            m_dim, k_dim = a.shape
            b_nk = b if ir.rhs_transposed else b.T
            n_dim = b_nk.shape[0]
            n_out = n_dim

            if ir.dtype == DType.INT8:
                (tmp_p / "in.bin").write_bytes(np.ascontiguousarray(a, dtype=np.int8).tobytes())
                (tmp_p / "w.bin").write_bytes(np.ascontiguousarray(b_nk, dtype=np.int8).tobytes())
                _adb("push", str(tmp_p / "in.bin"), f"{remote_dir}/in.bin")
                _adb("push", str(tmp_p / "w.bin"), f"{remote_dir}/w.bin")
                if bias is not None:
                    (tmp_p / "bias.bin").write_bytes(np.ascontiguousarray(bias, dtype=np.int32).tobytes())
                    _adb("push", str(tmp_p / "bias.bin"), f"{remote_dir}/bias.bin")
                    cmd_parts.extend(["--bias", f"{remote_dir}/bias.bin"])
                cmd_parts.extend([
                    "--s-in", str(quant.s_in),
                    "--zp-in", str(quant.zp_in),
                    "--s-w", str(quant.s_w),
                    "--s-out", str(quant.s_out),
                    "--zp-out", str(quant.zp_out),
                ])
            else:
                if b_nk.dtype == np.int8 and scales and ir.input_names[1] in scales:
                    s = np.asarray(scales[ir.input_names[1]], dtype=np.float32)
                    b_f = b_nk.astype(np.float32) * s[:, None]
                else:
                    b_f = b_nk.astype(np.float32)
                (tmp_p / "in.bin").write_bytes(np.ascontiguousarray(a, dtype=np.float32).tobytes())
                (tmp_p / "w.bin").write_bytes(np.ascontiguousarray(b_f, dtype=np.float32).tobytes())
                _adb("push", str(tmp_p / "in.bin"), f"{remote_dir}/in.bin")
                _adb("push", str(tmp_p / "w.bin"), f"{remote_dir}/w.bin")

            cmd_parts.extend([
                "--m", str(m_dim),
                "--k", str(k_dim),
                "--n", str(n_dim),
                "--in", f"{remote_dir}/in.bin",
                "--w", f"{remote_dir}/w.bin",
                "--out", f"{remote_dir}/out.bin",
            ])

        shell_cmd = " ".join(cmd_parts)
        exec_res = _adb("shell", f"su -c '{shell_cmd}'")
        if exec_res.returncode != 0:
            # Fallback without su if shell already has /dev/edgetpu-soc access
            exec_res = _adb("shell", shell_cmd)
        if exec_res.returncode != 0:
            raise RuntimeError(
                f"rango execution failed (rc={exec_res.returncode}):\nstdout={exec_res.stdout}\nstderr={exec_res.stderr}"
            )
        report = json.loads(exec_res.stdout)
        pull_r = _adb("pull", f"{remote_dir}/out.bin", str(tmp_p / "out.bin"))
        if pull_r.returncode != 0:
            raise RuntimeError(f"adb pull out.bin failed: {pull_r.stderr}")
        out_raw = (tmp_p / "out.bin").read_bytes()
        if ir.dtype == DType.INT8:
            out_arr = np.frombuffer(out_raw, dtype=np.int8).reshape(m_dim, n_dim).copy()
        else:
            out_arr = np.frombuffer(out_raw, dtype=np.float32).reshape(m_dim, n_dim).copy()

    hw_us = float(report.get("hw_us", 0.0))
    min_hw_us = float(report.get("min_hw_us", hw_us))
    wall_us = float(report.get("wall_us", hw_us))
    if ir.op_kind == OpKind.SWIGLU:
        flops = 4.0 * m_dim * k_dim * n_inter + 2.0 * m_dim * n_inter * n_out
    else:
        flops = 2.0 * m_dim * k_dim * n_dim
    denom_s = max(hw_us * 1e-6, 1e-12)
    gflops = round((flops / denom_s) / 1e9, 3)

    return ExecutionResult(
        target=Target.BUENOS_A0,
        op_kind=ir.op_kind,
        dtype=ir.dtype,
        output=out_arr,
        hw_us=round(hw_us, 3),
        min_hw_us=round(min_hw_us, 3),
        wall_us=round(wall_us, 3),
        gflops=gflops,
        tflops=round(gflops / 1000.0, 5),
        silicon="Pixel Tensor G5 (buenos_a0)",
        stack="libopenbuenos (zero Google .so) + /dev/edgetpu-soc",
        mock=bool(report.get("mock", False)),
        metadata=report,
    )


def run_buenos_a0(
    ir: TileKernelIR,
    inputs: Dict[str, Any],
    *,
    quant: Optional[QuantParams] = None,
    bias: Optional[np.ndarray] = None,
    scales: Optional[Dict[str, np.ndarray]] = None,
    adb_serial: Optional[str] = None,
    mock: Optional[bool] = None,
    warmup: int = 2,
    iters: int = 10,
    perf_mode: int = 5,
    dry_run: bool = False,
) -> ExecutionResult:
    """Lower and execute a TileKernelIR on Edge TPU buenos_a0 (mock or live rango)."""
    qparams = quant or ir.quant
    shapes = {k: tuple(np.asarray(v).shape) for k, v in inputs.items()}
    lowered = lower_buenos_a0(ir, shapes, quant=qparams)

    serial = adb_serial or os.environ.get("FREETPU_ADB_SERIAL", "").strip()
    if dry_run:
        clean_meta = {k: v for k, v in lowered.summary.items() if k != "_spline_runtime"}
        return ExecutionResult(
            target=Target.BUENOS_A0,
            op_kind=ir.op_kind,
            dtype=ir.dtype,
            output=None,
            hw_us=None,
            wall_us=0.0,
            silicon="Pixel Tensor G5 (buenos_a0)",
            stack="dry_run (emit_custom_gemm + spline_0x640 + libopenbuenos)",
            mock=True,
            lowered=lowered,
            metadata=clean_meta,
        )

    if serial and mock is not True:
        res = _run_adb_rango(
            ir,
            inputs,
            adb_serial=serial,
            quant=qparams,
            bias=bias,
            scales=scales,
            warmup=warmup,
            iters=iters,
            perf_mode=perf_mode,
        )
        res.lowered = lowered
        return res

    # Host ctypes libopenbuenos path (OPENBUENOS_MOCK=1 or local Linux /dev/edgetpu-soc)
    t0 = time.perf_counter()
    with OpenBuenosSession(mock=mock, perf_mode=perf_mode) as sess:
        hw_samples: List[float] = []
        last_stats = None

        if ir.op_kind == OpKind.SWIGLU:
            x = np.asarray(inputs[ir.swiglu_roles["x"]], dtype=np.float32)
            wg_name = ir.swiglu_roles["w_gate"]
            wu_name = ir.swiglu_roles["w_up"]
            wd_name = ir.swiglu_roles.get("w_down")

            def _to_f32_nk(arr: np.ndarray, name: str) -> np.ndarray:
                a = np.asarray(arr)
                if a.dtype == np.int8 and scales and name in scales:
                    s = np.asarray(scales[name], dtype=np.float32)
                    a = a.astype(np.float32) * (s[:, None] if ir.rhs_transposed else s[None, :])
                else:
                    a = a.astype(np.float32)
                return np.ascontiguousarray(a if ir.rhs_transposed else a.T, dtype=np.float32)

            wg_nk = _to_f32_nk(inputs[wg_name], wg_name)
            wu_nk = _to_f32_nk(inputs[wu_name], wu_name)
            wd_nk = _to_f32_nk(inputs[wd_name], wd_name) if wd_name else None
            m_dim, k_dim = x.shape
            n_inter = wg_nk.shape[0]
            n_out = wd_nk.shape[0] if wd_nk is not None else n_inter
            n_dim = n_out

            for i in range(max(1, iters)):
                (c_gate, c_up), st1 = sess.matmul_f32_quant_batch(
                    x, [wg_nk, wu_nk], [OB_ACT_SILU, OB_ACT_NONE]
                )
                hidden = (c_gate * c_up).astype(np.float32)
                if wd_nk is not None:
                    (out_arr,), st2 = sess.matmul_f32_quant_batch(hidden, [wd_nk], [OB_ACT_NONE])
                    hw_samples.append(float(st1.hw_execution_us + st2.hw_execution_us))
                    last_stats = st2
                else:
                    out_arr = hidden
                    hw_samples.append(float(st1.hw_execution_us))
                    last_stats = st1
            flops = (4.0 * m_dim * k_dim * n_inter) + (
                2.0 * m_dim * n_inter * n_out if wd_nk is not None else 0.0
            )
        else:
            a = np.asarray(inputs[ir.input_names[0]])
            b = np.asarray(inputs[ir.input_names[1]])
            m_dim, k_dim = a.shape
            b_nk = b if ir.rhs_transposed else b.T
            n_dim = b_nk.shape[0]
            n_out = n_dim
            flops = 2.0 * m_dim * k_dim * n_dim

            if ir.dtype == DType.INT8:
                acc_i32, last_stats = sess.matmul_int8(
                    a, b_nk, bias_n=bias, zp_a=qparams.zp_in, zp_b=qparams.zp_w
                )
                if ir.op_kind == OpKind.MATMUL:
                    real_mul = np.float32(np.float32(qparams.s_in * qparams.s_w) / np.float32(qparams.s_out))
                    out_arr = np.clip(
                        np.rint(acc_i32.astype(np.float32) * real_mul) + qparams.zp_out,
                        -128,
                        127,
                    ).astype(np.int8)
                else:
                    spline_rt = lowered.summary["_spline_runtime"]
                    out_arr = _eval_spline_on_acc_i32(
                        acc_i32,
                        s_in=qparams.s_in,
                        s_w=qparams.s_w,
                        s_out=spline_rt["s_out_eff"],
                        zp_out=qparams.zp_out,
                        spline_info=spline_rt,
                    )
                hw_samples.append(float(last_stats.hw_execution_us))
            else:
                if b_nk.dtype == np.int8 and scales and ir.input_names[1] in scales:
                    s = np.asarray(scales[ir.input_names[1]], dtype=np.float32)
                    b_f = b_nk.astype(np.float32) * s[:, None]
                else:
                    b_f = b_nk.astype(np.float32)
                ob_act = _op_to_ob_act(ir.op_kind)
                a_f = np.ascontiguousarray(a, dtype=np.float32)
                for _ in range(max(1, iters)):
                    (out_arr,), last_stats = sess.matmul_f32_quant_batch(a_f, [b_f], [ob_act])
                    hw_samples.append(float(last_stats.hw_execution_us))
                if bias is not None and ir.op_kind == OpKind.MATMUL:
                    out_arr = out_arr + np.asarray(bias, dtype=np.float32)[None, :]

        wall_us = ((time.perf_counter() - t0) * 1e6) / max(1, iters)
        med_hw = float(np.median(hw_samples)) if hw_samples else 0.0
        min_hw = float(np.min(hw_samples)) if hw_samples else 0.0
        clean_meta = {k: v for k, v in lowered.summary.items() if k != "_spline_runtime"}
        clean_meta["libopenbuenos_path"] = str(sess.bridge.lib_path)
        clean_meta["tiles_dispatched"] = int(last_stats.tiles_dispatched) if last_stats else 0
        clean_meta["cache_hits"] = int(last_stats.cache_hits) if last_stats else 0
        clean_meta["mock_flag"] = bool(last_stats.flags & OB_STAT_FLAG_MOCK) if last_stats else sess.is_mock

        gflops = None
        if med_hw > 0.0:
            gflops = round((flops / (med_hw * 1e-6)) / 1e9, 3)

        return ExecutionResult(
            target=Target.BUENOS_A0,
            op_kind=ir.op_kind,
            dtype=ir.dtype,
            output=out_arr,
            hw_us=round(med_hw, 3) if not sess.is_mock else None,
            min_hw_us=round(min_hw, 3) if not sess.is_mock else None,
            wall_us=round(wall_us, 3),
            gflops=gflops,
            tflops=round(gflops / 1000.0, 5) if gflops is not None else None,
            silicon="Tensor G5 (buenos_a0 mock)" if sess.is_mock else "Pixel Tensor G5 (buenos_a0)",
            stack="libopenbuenos (ctypes)" + (" [OPENBUENOS_MOCK=1]" if sess.is_mock else ""),
            mock=sess.is_mock,
            lowered=lowered,
            metadata=clean_meta,
        )
