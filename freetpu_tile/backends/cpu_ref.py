"""Backend 0: Pure NumPy deterministic reference executor for FreeTPU Tile DSL."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

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


def to_bf16_f32(arr: np.ndarray) -> np.ndarray:
    """Round float32 values to IEEE-754 bfloat16 precision (round-to-nearest-even) and back."""
    x = np.asarray(arr, dtype=np.float32)
    u32 = x.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = np.where(np.isnan(x), u32 | np.uint32(0x00400000), u32 + rounding_bias)
    bf16_bits = rounded & np.uint32(0xFFFF0000)
    return bf16_bits.view(np.float32)


def silu_f32(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    sig = np.where(x64 >= 0.0, 1.0 / (1.0 + np.exp(-x64)), np.exp(x64) / (1.0 + np.exp(x64)))
    return (x64 * sig).astype(np.float32)


def gelu_tanh_f32(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    k = math.sqrt(2.0 / math.pi)
    return (0.5 * x64 * (1.0 + np.tanh(k * (x64 + 0.044715 * (x64 ** 3))))).astype(np.float32)


def quantize_w8a16_weights(
    w_kn: np.ndarray, *, rhs_transposed: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float weights to symmetric per-channel int8 + float32 scales."""
    w = np.asarray(w_kn, dtype=np.float32)
    if rhs_transposed:
        # w is [N, K]
        amax = np.max(np.abs(w), axis=1, keepdims=True)
        scales = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(w / scales), -127, 127).astype(np.int8)
        return q, scales.reshape(-1)
    # w is [K, N]
    amax = np.max(np.abs(w), axis=0, keepdims=True)
    scales = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scales), -127, 127).astype(np.int8)
    return q, scales.reshape(-1)


def _apply_act(val: np.ndarray, op_kind: OpKind) -> np.ndarray:
    if op_kind == OpKind.FUSED_MATMUL_SILU:
        return silu_f32(val)
    if op_kind == OpKind.FUSED_MATMUL_GELU:
        return gelu_tanh_f32(val)
    return np.asarray(val, dtype=np.float32)


def _int8_gemm_acc(
    a_mk: np.ndarray,
    b: np.ndarray,
    *,
    rhs_transposed: bool,
    zp_in: int,
    zp_w: int,
    bias_n: Optional[np.ndarray],
) -> np.ndarray:
    a_i32 = np.asarray(a_mk, dtype=np.int32) - int(zp_in)
    b_i32 = np.asarray(b, dtype=np.int32) - int(zp_w)
    if rhs_transposed:
        acc = a_i32 @ b_i32.T
    else:
        acc = a_i32 @ b_i32
    if bias_n is not None:
        acc = acc + np.asarray(bias_n, dtype=np.int32)[None, :]
    return acc


def _int8_requantize(
    acc: np.ndarray,
    op_kind: OpKind,
    quant: QuantParams,
) -> np.ndarray:
    s_in = np.float32(quant.s_in)
    s_w = np.float32(quant.s_w)
    s_out = np.float32(quant.s_out)
    if op_kind == OpKind.MATMUL:
        real_mul = np.float32(np.float32(s_in * s_w) / s_out)
        scaled = acc.astype(np.float32) * real_mul
        q = np.rint(scaled).astype(np.int32) + int(quant.zp_out)
        return np.clip(q, -128, 127).astype(np.int8)

    # Fused SiLU / GELU: dequantize accumulator using s_in * s_w, apply activation, requantize to s_out, zp_out
    real_in_mul = np.float32(s_in * s_w)
    x_real = acc.astype(np.float32) * real_in_mul
    act_real = _apply_act(x_real, op_kind)
    q = np.rint(act_real / float(s_out)).astype(np.int32) + int(quant.zp_out)
    return np.clip(q, -128, 127).astype(np.int8)


def lower_cpu_ref(
    ir: TileKernelIR,
    shapes: Dict[str, tuple[int, ...]],
) -> LoweredArtifact:
    return LoweredArtifact(
        target=Target.CPU_REF,
        ir=ir,
        shapes=shapes,
        summary={
            "backend": "numpy_reference",
            "op_kind": ir.op_kind.value,
            "dtype": ir.dtype.value,
            "block_spec": {
                "block_m": ir.block_spec.block_m,
                "block_n": ir.block_spec.block_n,
                "block_k": ir.block_spec.block_k,
            },
        },
    )


def run_cpu_ref(
    ir: TileKernelIR,
    inputs: Dict[str, Any],
    *,
    quant: Optional[QuantParams] = None,
    bias: Optional[np.ndarray] = None,
    scales: Optional[Dict[str, np.ndarray]] = None,
) -> ExecutionResult:
    qparams = quant or ir.quant
    scales = scales or {}
    t0 = time.perf_counter()

    if ir.op_kind == OpKind.SWIGLU:
        if ir.dtype == DType.INT8:
            raise ValueError("INT8 dtype is not supported for swiglu; use dtype='w8a16' or dtype='bf16'")
        if bias is not None:
            raise ValueError("bias is not supported for op_kind='swiglu'")
        x_name = ir.swiglu_roles["x"]
        wg_name = ir.swiglu_roles["w_gate"]
        wu_name = ir.swiglu_roles["w_up"]
        x = np.asarray(inputs[x_name])
        wg = np.asarray(inputs[wg_name])
        wu = np.asarray(inputs[wu_name])

        def _proj(lhs: np.ndarray, rhs: np.ndarray, role_key: str) -> np.ndarray:
            if ir.dtype == DType.BF16:
                a_f = to_bf16_f32(lhs)
                b_f = to_bf16_f32(rhs)
                return a_f @ (b_f.T if ir.rhs_transposed else b_f)
            if ir.dtype == DType.W8A16:
                a_f = to_bf16_f32(lhs)
                if rhs.dtype == np.int8:
                    s = np.asarray(scales[role_key], dtype=np.float32)
                    b_f = rhs.astype(np.float32) * (s[:, None] if ir.rhs_transposed else s[None, :])
                else:
                    b_f = rhs.astype(np.float32)
                return a_f @ (b_f.T if ir.rhs_transposed else b_f)
            # INT8 dequantized SwiGLU reference
            acc = _int8_gemm_acc(
                lhs, rhs, rhs_transposed=ir.rhs_transposed, zp_in=qparams.zp_in, zp_w=qparams.zp_w, bias_n=None
            )
            return acc.astype(np.float32) * np.float32(qparams.s_in * qparams.s_w)

        gate = silu_f32(_proj(x, wg, wg_name))
        up = _proj(x, wu, wu_name)
        hidden = (gate * up).astype(np.float32)
        if ir.has_down_proj:
            wd_name = ir.swiglu_roles["w_down"]
            wd = np.asarray(inputs[wd_name])
            if ir.dtype == DType.INT8 and wd.dtype == np.int8:
                b_f = wd.astype(np.float32) * np.float32(qparams.s_w)
                out = to_bf16_f32(hidden) @ (b_f.T if ir.rhs_transposed else b_f)
            elif ir.dtype == DType.W8A16 and wd.dtype == np.int8:
                s_d = np.asarray(scales[wd_name], dtype=np.float32)
                b_f = wd.astype(np.float32) * (s_d[:, None] if ir.rhs_transposed else s_d[None, :])
                out = to_bf16_f32(hidden) @ (b_f.T if ir.rhs_transposed else b_f)
            else:
                out = to_bf16_f32(hidden) @ (to_bf16_f32(wd).T if ir.rhs_transposed else to_bf16_f32(wd))
        else:
            out = hidden
        if ir.dtype == DType.BF16:
            out = to_bf16_f32(out)
        m_dim, k_dim = x.shape
        n_inter = wg.shape[0] if ir.rhs_transposed else wg.shape[1]
        n_out = out.shape[1]
        flops = (4.0 * m_dim * k_dim * n_inter) + (2.0 * m_dim * n_inter * n_out if ir.has_down_proj else 0.0)
    else:
        a_name, b_name = ir.gemm_roles if len(ir.gemm_roles) == 2 else (ir.input_names[0], ir.input_names[1])
        a = np.asarray(inputs[a_name])
        b = np.asarray(inputs[b_name])
        m_dim, k_dim = a.shape
        n_dim = b.shape[0] if ir.rhs_transposed else b.shape[1]
        flops = 2.0 * m_dim * k_dim * n_dim

        if bias is not None:
            bias_arr = np.asarray(bias)
            if bias_arr.ndim != 1 or bias_arr.shape[0] != n_dim:
                raise ValueError(f"bias must have shape ({n_dim},), got {bias_arr.shape}")

        if ir.dtype == DType.INT8:
            acc = _int8_gemm_acc(
                a,
                b,
                rhs_transposed=ir.rhs_transposed,
                zp_in=qparams.zp_in,
                zp_w=qparams.zp_w,
                bias_n=bias,
            )
            out = _int8_requantize(acc, ir.op_kind, qparams)
        elif ir.dtype == DType.W8A16:
            a_f = to_bf16_f32(a)
            if b.dtype == np.int8:
                s_b = np.asarray(scales[b_name], dtype=np.float32)
                b_f = b.astype(np.float32) * (s_b[:, None] if ir.rhs_transposed else s_b[None, :])
            else:
                b_f = b.astype(np.float32)
            mm = a_f @ (b_f.T if ir.rhs_transposed else b_f)
            if bias is not None:
                mm = mm + np.asarray(bias, dtype=np.float32)[None, :]
            out = _apply_act(mm, ir.op_kind)
        else:  # BF16
            a_f = to_bf16_f32(a)
            b_f = to_bf16_f32(b)
            mm = a_f @ (b_f.T if ir.rhs_transposed else b_f)
            if bias is not None:
                mm = mm + np.asarray(bias, dtype=np.float32)[None, :]
            out = to_bf16_f32(_apply_act(mm, ir.op_kind))

    dt_us = (time.perf_counter() - t0) * 1e6
    gflops = round((flops / max(dt_us * 1e-6, 1e-12)) / 1e9, 3)
    return ExecutionResult(
        target=Target.CPU_REF,
        op_kind=ir.op_kind,
        dtype=ir.dtype,
        output=out,
        hw_us=None,
        wall_us=dt_us,
        gflops=gflops,
        tflops=round(gflops / 1000.0, 5),
        silicon="host-cpu",
        stack=f"numpy-{np.__version__}",
        mock=False,
        lowered=lower_cpu_ref(ir, {k: tuple(np.asarray(v).shape) for k, v in inputs.items()}),
    )
