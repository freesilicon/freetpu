"""Backend 2: Cloud TPU (v5e-1 / v6e-1) JAX Pallas/Mosaic lowering & execution.

Evolves Track B's proven `@tpu_tiled_kernel` / `pallas_opt_bench.py` /
`llama8b_decode_bench.py` Pallas/Mosaic patterns:
  - 3D `(M // bm, N // bn, K // bk)` `pl.pallas_call` + `pl.BlockSpec`
  - `pltpu.VMEM((bm, bn), jnp.float32)` accumulator scratch
  - `CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))`
  - Fused `SiLU` / `GELU` / `W8A16` dequantization in VMEM -> `tpu_custom_call`
  - Full offline `--dry-run` syntax + BlockSpec + IR lowering verification
"""

from __future__ import annotations

import ast
import math
import textwrap
import time
from typing import Any, Dict, Optional, Tuple

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
from .cpu_ref import quantize_w8a16_weights


def _align_m_pallas(m_dim: int, block_m: int) -> Tuple[int, int]:
    """Return (m_pad, bm) satisfying Cloud TPU Mosaic MXU 8-row minimum alignment."""
    bm = max(8, block_m)
    if bm < 128 and bm % 8 != 0:
        bm = ((bm + 7) // 8) * 8
    m_pad = max(bm, ((m_dim + bm - 1) // bm) * bm)
    return m_pad, bm


def _validate_pallas_blocks(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    bm: int,
    bn: int,
    bk: int,
) -> Tuple[int, int, int, int]:
    """Validate and resolve Cloud TPU Mosaic block sizes."""
    _, eff_bm = _align_m_pallas(m_dim, bm)
    eff_bk = bk
    eff_bn = bn
    if k_dim % eff_bk != 0:
        raise ValueError(f"Cloud TPU Pallas requires K ({k_dim}) divisible by block_k ({eff_bk})")
    if n_dim % eff_bn != 0:
        raise ValueError(f"Cloud TPU Pallas requires N ({n_dim}) divisible by block_n ({eff_bn})")
    if eff_bk % 128 != 0 or eff_bn % 128 != 0:
        raise ValueError(
            f"Cloud TPU Mosaic requires block_k ({eff_bk}) and block_n ({eff_bn}) divisible by 128"
        )
    m_pad = ((m_dim + eff_bm - 1) // eff_bm) * eff_bm
    return m_pad, eff_bm, eff_bn, eff_bk


def _epilogue_expr(op_kind: OpKind, dtype: DType) -> str:
    if op_kind == OpKind.FUSED_MATMUL_SILU:
        act = "v * jax.nn.sigmoid(v)"
    elif op_kind == OpKind.FUSED_MATMUL_GELU:
        act = "0.5 * v * (1.0 + jnp.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))"
    else:
        act = "v"

    if dtype == DType.BF16 or dtype == DType.W8A16:
        return f"({act}).astype(jnp.bfloat16)"
    return f"({act}).astype(jnp.float32)"


def generate_pallas_source(
    ir: TileKernelIR,
    shapes: Dict[str, Tuple[int, ...]],
    *,
    quant: Optional[QuantParams] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Emit self-contained JAX Pallas/Mosaic Python source for TileKernelIR."""
    qparams = quant or ir.quant
    a_shape = shapes[ir.input_names[0]]
    m_dim, k_dim = int(a_shape[0]), int(a_shape[1])

    if ir.op_kind == OpKind.SWIGLU:
        wg_shape = shapes[ir.swiglu_roles["w_gate"]]
        n_inter = int(wg_shape[0] if ir.rhs_transposed else wg_shape[1])
        if ir.has_down_proj:
            wd_shape = shapes[ir.swiglu_roles["w_down"]]
            n_out = int(wd_shape[0] if ir.rhs_transposed else wd_shape[1])
        else:
            n_out = n_inter
        m_pad, bm, bn_inter, bk = _validate_pallas_blocks(
            m_dim, k_dim, n_inter, ir.block_spec.block_m, ir.block_spec.block_n, ir.block_spec.block_k
        )
        grid = (m_pad // bm, n_inter // bn_inter, k_dim // bk)
        if ir.has_down_proj:
            _, _, bn_down, bk_down = _validate_pallas_blocks(
                m_dim, n_inter, n_out, ir.block_spec.block_m, ir.block_spec.block_n, ir.block_spec.block_k
            )
            grid_down = (m_pad // bm, n_out // bn_down, n_inter // bk_down)
        else:
            bn_down, bk_down, grid_down = bn_inter, bk, None
        n_dim = n_out
    else:
        b_shape = shapes[ir.input_names[1]]
        n_dim = int(b_shape[0] if ir.rhs_transposed else b_shape[1])
        m_pad, bm, bn, bk = _validate_pallas_blocks(
            m_dim, k_dim, n_dim, ir.block_spec.block_m, ir.block_spec.block_n, ir.block_spec.block_k
        )
        grid = (m_pad // bm, n_dim // bn, k_dim // bk)
        n_inter, bn_inter, bn_down, bk_down, grid_down = n_dim, bn, bn, bk, None

    rhs_t = ir.rhs_transposed
    out_dtype_str = "jnp.bfloat16" if ir.dtype in (DType.BF16, DType.W8A16) else "jnp.float32"

    def _gemm_kernel_block(fn_name: str, act_kind: OpKind, cur_k: int, cur_n: int, cur_bk: int, cur_bn: int) -> str:
        num_k = cur_k // cur_bk
        epi = _epilogue_expr(act_kind, ir.dtype)
        if ir.dtype == DType.W8A16:
            sig = "def kernel(a_ref, w_ref, s_ref, o_ref, acc_ref):"
            deq = (
                "w_f = w_ref[...].astype(jnp.bfloat16)\n"
                "        s_tile = s_ref[...]\n"
                f"        b_tile = w_f * jnp.broadcast_to(s_tile, ({cur_bn if rhs_t else cur_bk}, {cur_bk if rhs_t else cur_bn}))"
            )
            scale_in_spec = (
                f"pl.BlockSpec(({cur_bn}, 1), lambda i, j, k: (j, 0))"
                if rhs_t
                else f"pl.BlockSpec((1, {cur_bn}), lambda i, j, k: (0, j))"
            )
            extra_spec = f",\n            {scale_in_spec}"
            call_args = "(a, w, s)"
            in_params = "a, w, s"
        elif ir.dtype == DType.INT8:
            sig = "def kernel(a_ref, b_ref, o_ref, acc_ref):"
            deq = "b_tile = b_ref[...]"
            extra_spec = ""
            call_args = "(a, b)"
            in_params = "a, b"
        else:
            sig = "def kernel(a_ref, b_ref, o_ref, acc_ref):"
            deq = "b_tile = b_ref[...]"
            extra_spec = ""
            call_args = "(a, b)"
            in_params = "a, b"

        if ir.dtype == DType.INT8:
            dot_expr = (
                "jax.lax.dot_general(a_tile, b_tile, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32).astype(jnp.float32)"
                if rhs_t
                else "jnp.dot(a_tile, b_tile, preferred_element_type=jnp.int32).astype(jnp.float32)"
            )
            scale_scale = float(qparams.s_in * qparams.s_w)
            epilogue_pre = f"v = acc_ref[...] * {scale_scale!r}\n            o_ref[...] = {epi}"
        else:
            dot_expr = (
                "jax.lax.dot_general(a_tile, b_tile, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)"
                if rhs_t
                else "jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)"
            )
            epilogue_pre = f"v = acc_ref[...]\n            o_ref[...] = {epi}"

        b_spec_str = (
            f"pl.BlockSpec(({cur_bn}, {cur_bk}), lambda i, j, k: (j, k))"
            if rhs_t
            else f"pl.BlockSpec(({cur_bk}, {cur_bn}), lambda i, j, k: (k, j))"
        )
        lines = [
            f"def {fn_name}({in_params}):",
            f"    {sig}",
            "        k_idx = pl.program_id(2)",
            "        a_tile = a_ref[...]",
            f"        {deq}",
            f"        prod = {dot_expr}",
            "",
            "        @pl.when(k_idx == 0)",
            "        def _init():",
            "            acc_ref[...] = prod",
            "",
            "        @pl.when(k_idx > 0)",
            "        def _acc():",
            "            acc_ref[...] += prod",
            "",
            f"        @pl.when(k_idx == {num_k} - 1)",
            "        def _epilogue():",
            f"            {epilogue_pre}",
            "",
            "    return pl.pallas_call(",
            "        kernel,",
            f"        out_shape=jax.ShapeDtypeStruct(({m_pad}, {cur_n}), {out_dtype_str}),",
            f"        grid=({m_pad // bm}, {cur_n // cur_bn}, {num_k}),",
            "        in_specs=[",
            f"            pl.BlockSpec(({bm}, {cur_bk}), lambda i, j, k: (i, k)),",
            f"            {b_spec_str}{extra_spec}",
            "        ],",
            f"        out_specs=pl.BlockSpec(({bm}, {cur_bn}), lambda i, j, k: (i, j)),",
            f"        scratch_shapes=[pltpu.VMEM(({bm}, {cur_bn}), jnp.float32)],",
            "        compiler_params=CompilerParams(",
            '            dimension_semantics=("parallel", "parallel", "arbitrary"),',
            "            disable_bounds_checks=True,",
            "        ),",
            f"    ){call_args}",
        ]
        return "\n".join(lines) + "\n"

    header = textwrap.dedent(
        """\
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(
            pltpu, "TPUCompilerParams"
        )
        """
    )

    if ir.op_kind == OpKind.SWIGLU:
        k_gate = _gemm_kernel_block("_pallas_gate_silu", OpKind.FUSED_MATMUL_SILU, k_dim, n_inter, bk, bn_inter)
        k_up = _gemm_kernel_block("_pallas_up", OpKind.MATMUL, k_dim, n_inter, bk, bn_inter)
        if ir.has_down_proj:
            k_down = _gemm_kernel_block("_pallas_down", OpKind.MATMUL, n_inter, n_dim, bk_down, bn_down)
            if ir.dtype == DType.W8A16:
                runner = textwrap.dedent(
                    f"""\
                    @jax.jit
                    def run_kernel(x, w_gate, s_gate, w_up, s_up, w_down, s_down):
                        g = _pallas_gate_silu(x, w_gate, s_gate)
                        u = _pallas_up(x, w_up, s_up)
                        hidden = (g * u).astype(jnp.bfloat16)
                        out = _pallas_down(hidden, w_down, s_down)
                        return out[:{m_dim}, :{n_dim}]
                    """
                )
            else:
                runner = textwrap.dedent(
                    f"""\
                    @jax.jit
                    def run_kernel(x, w_gate, w_up, w_down):
                        g = _pallas_gate_silu(x, w_gate)
                        u = _pallas_up(x, w_up)
                        hidden = (g * u).astype({out_dtype_str})
                        out = _pallas_down(hidden, w_down)
                        return out[:{m_dim}, :{n_dim}]
                    """
                )
            source = f"{header}\n{k_gate}\n{k_up}\n{k_down}\n{runner}"
        else:
            if ir.dtype == DType.W8A16:
                runner = textwrap.dedent(
                    f"""\
                    @jax.jit
                    def run_kernel(x, w_gate, s_gate, w_up, s_up):
                        g = _pallas_gate_silu(x, w_gate, s_gate)
                        u = _pallas_up(x, w_up, s_up)
                        return (g * u)[:{m_dim}, :{n_inter}]
                    """
                )
            else:
                runner = textwrap.dedent(
                    f"""\
                    @jax.jit
                    def run_kernel(x, w_gate, w_up):
                        g = _pallas_gate_silu(x, w_gate)
                        u = _pallas_up(x, w_up)
                        return (g * u)[:{m_dim}, :{n_inter}]
                    """
                )
            source = f"{header}\n{k_gate}\n{k_up}\n{runner}"
    else:
        k_main = _gemm_kernel_block("_pallas_main", ir.op_kind, k_dim, n_dim, bk, bn)
        if ir.dtype == DType.W8A16:
            runner = textwrap.dedent(
                f"""\
                @jax.jit
                def run_kernel(a, w, s):
                    return _pallas_main(a, w, s)[:{m_dim}, :{n_dim}]
                """
            )
        else:
            runner = textwrap.dedent(
                f"""\
                @jax.jit
                def run_kernel(a, b):
                    return _pallas_main(a, b)[:{m_dim}, :{n_dim}]
                """
            )
        source = f"{header}\n{k_main}\n{runner}"

    meta = {
        "backend": "cloud_tpu_pallas",
        "op_kind": ir.op_kind.value,
        "dtype": ir.dtype.value,
        "m_orig": m_dim,
        "m_pad": m_pad,
        "k": k_dim,
        "n": n_dim,
        "block_m": bm,
        "block_n": bn if ir.op_kind != OpKind.SWIGLU else bn_inter,
        "block_k": bk,
        "grid": list(grid),
        "grid_down": list(grid_down) if grid_down is not None else None,
        "rhs_transposed": rhs_t,
        "tpu_custom_call_expected": True,
    }
    return source, meta


def lower_cloud_tpu_pallas(
    ir: TileKernelIR,
    shapes: Dict[str, Tuple[int, ...]],
    *,
    quant: Optional[QuantParams] = None,
) -> LoweredArtifact:
    """Lower TileKernelIR to Cloud TPU Pallas/Mosaic code and verify syntax + BlockSpec invariants."""
    source, meta = generate_pallas_source(ir, shapes, quant=quant)
    ast.parse(source)
    compile(source, f"<freetpu_tile:{ir.name}:pallas>", "exec")
    meta["ast_verified"] = True
    return LoweredArtifact(
        target=Target.CLOUD_TPU_PALLAS,
        ir=ir,
        shapes=shapes,
        summary=meta,
        source_code=source,
    )


def run_cloud_tpu_pallas(
    ir: TileKernelIR,
    inputs: Dict[str, Any],
    *,
    quant: Optional[QuantParams] = None,
    scales: Optional[Dict[str, np.ndarray]] = None,
    warmup: int = 4,
    iters: int = 20,
    dry_run: bool = False,
) -> ExecutionResult:
    """Lower and (when JAX+TPU is present and dry_run=False) execute on Cloud TPU Pallas/Mosaic."""
    shapes = {k: tuple(np.asarray(v).shape) for k, v in inputs.items()}
    lowered = lower_cloud_tpu_pallas(ir, shapes, quant=quant)
    meta = dict(lowered.summary)

    if dry_run:
        return ExecutionResult(
            target=Target.CLOUD_TPU_PALLAS,
            op_kind=ir.op_kind,
            dtype=ir.dtype,
            output=None,
            hw_us=None,
            wall_us=0.0,
            silicon="cloud-tpu (dry_run)",
            stack="pallas/mosaic (dry_run AST+BlockSpec verified)",
            mock=True,
            lowered=lowered,
            metadata=meta,
        )

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed in this environment; pass dry_run=True locally or execute via Colab TPU (`freetpu_tile.colab_runner`)."
        ) from exc

    devices = jax.devices()
    backend = jax.default_backend()
    if backend != "tpu":
        raise RuntimeError(
            f"Cloud TPU Pallas execution requires a TPU backend, found {backend!r} ({devices}). "
            "Use --dry-run offline or run on Colab TPU via `freetpu_tile.colab_runner`."
        )

    ns: Dict[str, Any] = {}
    exec(compile(lowered.source_code or "", f"<freetpu_tile:{ir.name}>", "exec"), ns, ns)
    run_kernel = ns["run_kernel"]

    m_pad = int(meta["m_pad"])
    m_orig = int(meta["m_orig"])
    scales = scales or {}

    def _pad_a(arr: np.ndarray) -> Any:
        arr_np = np.asarray(arr)
        if m_pad != m_orig:
            pad = np.zeros((m_pad, arr_np.shape[1]), dtype=arr_np.dtype)
            pad[:m_orig, :] = arr_np
            arr_np = pad
        if ir.dtype == DType.INT8:
            return jnp.asarray(arr_np.astype(np.int8))
        return jnp.asarray(arr_np.astype(np.float32)).astype(jnp.bfloat16)

    def _prep_w(name: str) -> tuple[Any, ...]:
        w_np = np.asarray(inputs[name])
        if ir.dtype == DType.W8A16:
            if w_np.dtype == np.int8 and name in scales:
                q_w = w_np
                s_w = np.asarray(scales[name], dtype=np.float32)
            else:
                q_w, s_w = quantize_w8a16_weights(w_np, rhs_transposed=ir.rhs_transposed)
            s_2d = s_w.reshape(-1, 1) if ir.rhs_transposed else s_w.reshape(1, -1)
            return (jnp.asarray(q_w, dtype=jnp.int8), jnp.asarray(s_2d, dtype=jnp.float32).astype(jnp.bfloat16))
        if ir.dtype == DType.INT8:
            return (jnp.asarray(w_np.astype(np.int8)),)
        return (jnp.asarray(w_np.astype(np.float32)).astype(jnp.bfloat16),)

    if ir.op_kind == OpKind.SWIGLU:
        x_j = _pad_a(inputs[ir.swiglu_roles["x"]])
        args_j: list[Any] = [x_j]
        args_j.extend(_prep_w(ir.swiglu_roles["w_gate"]))
        args_j.extend(_prep_w(ir.swiglu_roles["w_up"]))
        if ir.has_down_proj:
            args_j.extend(_prep_w(ir.swiglu_roles["w_down"]))
    else:
        a_j = _pad_a(inputs[ir.input_names[0]])
        args_j = [a_j, *_prep_w(ir.input_names[1])]

    jax.block_until_ready(args_j)
    lowered_hlo = run_kernel.lower(*args_j).as_text()
    has_tpu_cc = "tpu_custom_call" in lowered_hlo
    meta["tpu_custom_call"] = has_tpu_cc

    for _ in range(max(1, warmup)):
        out_j = run_kernel(*args_j)
        jax.block_until_ready(out_j)

    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        out_j = run_kernel(*args_j)
        jax.block_until_ready(out_j)
        ts.append(time.perf_counter() - t0)

    med_us = float(np.median(ts)) * 1e6
    min_us = float(np.min(ts)) * 1e6
    out_np = np.asarray(out_j.astype(jnp.float32))
    if ir.dtype == DType.INT8:
        qparams = quant or ir.quant
        out_np = np.clip(
            np.rint(out_np / float(qparams.s_out)) + int(qparams.zp_out),
            -128,
            127,
        ).astype(np.int8)

    if ir.op_kind == OpKind.SWIGLU:
        n_inter = int(inputs[ir.swiglu_roles["w_gate"]].shape[0 if ir.rhs_transposed else 1])
        flops = 4.0 * m_orig * int(meta["k"]) * n_inter + (
            2.0 * m_orig * n_inter * int(meta["n"]) if ir.has_down_proj else 0.0
        )
    else:
        flops = 2.0 * m_orig * int(meta["k"]) * int(meta["n"])
    gflops = round((flops / max(med_us * 1e-6, 1e-12)) / 1e9, 3)

    return ExecutionResult(
        target=Target.CLOUD_TPU_PALLAS,
        op_kind=ir.op_kind,
        dtype=ir.dtype,
        output=out_np,
        hw_us=round(med_us, 3),
        min_hw_us=round(min_us, 3),
        wall_us=round(med_us, 3),
        gflops=gflops,
        tflops=round(gflops / 1000.0, 5),
        silicon=str(getattr(devices[0], "device_kind", "Cloud TPU")),
        stack=f"jax-{getattr(jax, '__version__', 'unknown')} (pallas/tpu_custom_call={has_tpu_cc})",
        mock=False,
        lowered=lowered,
        metadata=meta,
    )
