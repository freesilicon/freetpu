"""Symbolic Tile DSL frontend (@tile_kernel) and multi-backend dispatcher."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .backends.buenos_a0 import lower_buenos_a0, run_buenos_a0
from .backends.cloud_tpu_pallas import lower_cloud_tpu_pallas, run_cloud_tpu_pallas
from .backends.cpu_ref import lower_cpu_ref, run_cpu_ref
from .ir import (
    DType,
    ExecutionResult,
    LoweredArtifact,
    OpKind,
    ParityReport,
    QuantParams,
    Target,
    TileBlockSpec,
    TileKernelIR,
    TileOp,
)


def cdiv(a: int, b: int) -> int:
    """Ceiling division helper matching Triton / cuTile tile-grid idioms."""
    return (int(a) + int(b) - 1) // int(b)


@dataclass(frozen=True)
class PidExpr:
    axis: int


class TileValue:
    """Symbolic SSA value produced during @tile_kernel tracing."""

    def __init__(self, vid: str, op: str, inputs: Tuple["TileValue | TileTensorProxy", ...] = (), attrs: Optional[Dict[str, Any]] = None) -> None:
        self.vid = vid
        self.op = op
        self.inputs = inputs
        self.attrs = attrs or {}

    def __matmul__(self, other: "TileValue | TileTensorProxy") -> "TileValue":
        return dot(self, other)

    def __mul__(self, other: "TileValue | TileTensorProxy") -> "TileValue":
        return mul(self, other)

    def __add__(self, other: "TileValue | TileTensorProxy") -> "TileValue":
        return add(self, other)


class TileTensorProxy:
    """Symbolic parameter proxy bound to a @tile_kernel argument."""

    def __init__(self, name: str, tracer: "_TileTracer") -> None:
        self.name = name
        self._tracer = tracer

    def __matmul__(self, other: "TileValue | TileTensorProxy") -> TileValue:
        return dot(self, other)


class _TileTracer:
    def __init__(self) -> None:
        self._counter = 0
        self.ops: List[TileOp] = []
        self.values: Dict[str, TileValue] = {}
        self.stored_output: Optional[Tuple[str, TileValue]] = None

    def next_id(self, prefix: str = "v") -> str:
        vid = f"{prefix}{self._counter}"
        self._counter += 1
        return vid

    def emit(
        self,
        op: str,
        inputs: Sequence[TileValue | TileTensorProxy],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> TileValue:
        norm_inputs: List[TileValue] = []
        for item in inputs:
            if isinstance(item, TileTensorProxy):
                norm_inputs.append(self.ensure_loaded(item))
            elif isinstance(item, TileValue):
                norm_inputs.append(item)
            else:
                raise TypeError(f"Unsupported operand in @tile_kernel op {op}: {type(item).__name__}")
        vid = self.next_id()
        tv = TileValue(vid=vid, op=op, inputs=tuple(norm_inputs), attrs=attrs)
        self.values[vid] = tv
        self.ops.append(
            TileOp(
                op=op,
                inputs=tuple(v.vid for v in norm_inputs),
                output=vid,
                attrs=dict(attrs or {}),
            )
        )
        return tv

    def ensure_loaded(self, proxy: TileTensorProxy) -> TileValue:
        vid = f"arg_{proxy.name}"
        if vid not in self.values:
            tv = TileValue(vid=vid, op="load", attrs={"param": proxy.name})
            self.values[vid] = tv
            self.ops.append(TileOp(op="load", inputs=(proxy.name,), output=vid, attrs={"param": proxy.name}))
        return self.values[vid]


_TRACER_TLS = threading.local()


def _current_tracer() -> _TileTracer:
    tracer = getattr(_TRACER_TLS, "tracer", None)
    if tracer is None:
        raise RuntimeError("Tile DSL primitives (load/dot/silu/gelu/mul/store) may only be called inside @tile_kernel tracing.")
    return tracer


def program_id(axis: int) -> PidExpr:
    if axis not in (0, 1, 2):
        raise ValueError(f"program_id axis must be 0, 1, or 2, got {axis}")
    return PidExpr(axis=axis)


def load(
    tensor: TileTensorProxy,
    index: Any = None,
    *,
    shape: Optional[Tuple[int, ...]] = None,
) -> TileValue:
    tracer = _current_tracer()
    if not isinstance(tensor, TileTensorProxy):
        raise TypeError("load() first argument must be a kernel tensor parameter")
    vid = tracer.next_id("ld_")
    attrs: Dict[str, Any] = {"param": tensor.name}
    if shape is not None:
        attrs["tile_shape"] = tuple(int(s) for s in shape)
    if index is not None:
        attrs["index"] = repr(index)
    tv = TileValue(vid=vid, op="load", attrs=attrs)
    tracer.values[vid] = tv
    tracer.ops.append(TileOp(op="load", inputs=(tensor.name,), output=vid, attrs=attrs))
    return tv


def dot(
    lhs: TileValue | TileTensorProxy,
    rhs: TileValue | TileTensorProxy,
    *,
    rhs_transposed: Optional[bool] = None,
) -> TileValue:
    attrs: Dict[str, Any] = {}
    if rhs_transposed is not None:
        attrs["rhs_transposed"] = bool(rhs_transposed)
    return _current_tracer().emit("dot", (lhs, rhs), attrs=attrs)


def silu(x: TileValue | TileTensorProxy) -> TileValue:
    return _current_tracer().emit("silu", (x,))


def gelu(x: TileValue | TileTensorProxy) -> TileValue:
    return _current_tracer().emit("gelu", (x,))


def mul(lhs: TileValue | TileTensorProxy, rhs: TileValue | TileTensorProxy) -> TileValue:
    return _current_tracer().emit("mul", (lhs, rhs))


def add(lhs: TileValue | TileTensorProxy, rhs: TileValue | TileTensorProxy) -> TileValue:
    return _current_tracer().emit("add", (lhs, rhs))


def store(out: TileTensorProxy, val: TileValue, index: Any = None) -> None:
    tracer = _current_tracer()
    if not isinstance(out, TileTensorProxy):
        raise TypeError("store() first argument must be the output tensor parameter")
    if not isinstance(val, TileValue):
        raise TypeError("store() value must be a TileValue expression")
    attrs: Dict[str, Any] = {"param": out.name}
    if index is not None:
        attrs["index"] = repr(index)
    tracer.ops.append(TileOp(op="store", inputs=(val.vid,), output=out.name, attrs=attrs))
    tracer.stored_output = (out.name, val)


def _trace_param_source(val: TileValue) -> str:
    if val.op == "load":
        return str(val.attrs["param"])
    raise ValueError(f"Expected loaded tensor parameter, got op={val.op}")


def _canonicalize_ir(
    fn: Callable[..., Any],
    *,
    block_spec: TileBlockSpec,
    dtype: DType,
    rhs_transposed: bool,
    quant: QuantParams,
) -> TileKernelIR:
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    if len(param_names) < 3:
        raise ValueError(f"@tile_kernel {fn.__name__} requires at least 3 tensor parameters (inputs..., out)")

    tracer = _TileTracer()
    proxies = {name: TileTensorProxy(name, tracer) for name in param_names}
    prev = getattr(_TRACER_TLS, "tracer", None)
    _TRACER_TLS.tracer = tracer
    try:
        ret = fn(**proxies)
    finally:
        _TRACER_TLS.tracer = prev

    if tracer.stored_output is None:
        if isinstance(ret, TileValue):
            out_name = param_names[-1]
            store_val = ret
            tracer.ops.append(TileOp(op="store", inputs=(ret.vid,), output=out_name, attrs={"param": out_name}))
        else:
            raise ValueError(f"@tile_kernel {fn.__name__} must call freetpu.store(out, val) or return a TileValue")
    else:
        out_name, store_val = tracer.stored_output

    input_names = tuple(p for p in param_names if p != out_name)

    # Pattern match & fuse canonical Tile IR
    explicit_rhs_t = rhs_transposed
    swiglu_roles: Dict[str, str] = {}
    has_down_proj = False

    def _check_dot_pair(node: TileValue) -> Tuple[str, str]:
        nonlocal explicit_rhs_t
        if node.op != "dot" or len(node.inputs) != 2:
            raise ValueError(f"Expected dot(lhs, rhs), got {node.op}")
        if "rhs_transposed" in node.attrs:
            explicit_rhs_t = bool(node.attrs["rhs_transposed"])
        return _trace_param_source(node.inputs[0]), _trace_param_source(node.inputs[1])

    def _match_gated_mul(node: TileValue) -> Optional[Tuple[str, str, str]]:
        if node.op != "mul" or len(node.inputs) != 2:
            return None
        lhs, rhs = node.inputs[0], node.inputs[1]
        if lhs.op == "silu" and lhs.inputs[0].op == "dot" and rhs.op == "dot":
            x1, wg = _check_dot_pair(lhs.inputs[0])
            x2, wu = _check_dot_pair(rhs)
            if x1 != x2:
                raise ValueError(f"SwiGLU gate and up projections must share input tensor: {x1} vs {x2}")
            return x1, wg, wu
        if rhs.op == "silu" and rhs.inputs[0].op == "dot" and lhs.op == "dot":
            x1, wg = _check_dot_pair(rhs.inputs[0])
            x2, wu = _check_dot_pair(lhs)
            if x1 != x2:
                raise ValueError(f"SwiGLU gate and up projections must share input tensor: {x1} vs {x2}")
            return x1, wg, wu
        return None

    if store_val.op == "dot":
        # Either plain matmul OR 3-proj SwiGLU: dot(mul(silu(dot(x, wg)), dot(x, wu)), w_down)
        lhs_node, rhs_node = store_val.inputs[0], store_val.inputs[1]
        matched_mul = _match_gated_mul(lhs_node)
        if matched_mul is not None:
            x_name, wg_name, wu_name = matched_mul
            wd_name = _trace_param_source(rhs_node)
            op_kind = OpKind.SWIGLU
            has_down_proj = True
            swiglu_roles = {"x": x_name, "w_gate": wg_name, "w_up": wu_name, "w_down": wd_name}
        else:
            _check_dot_pair(store_val)
            op_kind = OpKind.MATMUL
    elif store_val.op == "silu" and store_val.inputs[0].op == "dot":
        _check_dot_pair(store_val.inputs[0])
        op_kind = OpKind.FUSED_MATMUL_SILU
    elif store_val.op == "gelu" and store_val.inputs[0].op == "dot":
        _check_dot_pair(store_val.inputs[0])
        op_kind = OpKind.FUSED_MATMUL_GELU
    else:
        matched_mul = _match_gated_mul(store_val)
        if matched_mul is not None:
            x_name, wg_name, wu_name = matched_mul
            op_kind = OpKind.SWIGLU
            has_down_proj = False
            swiglu_roles = {"x": x_name, "w_gate": wg_name, "w_up": wu_name}
        else:
            raise ValueError(
                f"Unsupported @tile_kernel expression graph ending at {store_val.op!r}. "
                "Supported core patterns: matmul, fused matmul+silu, fused matmul+gelu, swiglu."
            )

    return TileKernelIR(
        name=fn.__name__,
        op_kind=op_kind,
        dtype=dtype,
        block_spec=block_spec,
        rhs_transposed=explicit_rhs_t,
        input_names=input_names,
        output_name=out_name,
        ops=tuple(tracer.ops),
        quant=quant,
        has_down_proj=has_down_proj,
        swiglu_roles=swiglu_roles,
    )


class TileKernel:
    """Compiled portable tile kernel supporting cpu_ref, buenos_a0, and cloud_tpu_pallas."""

    def __init__(self, fn: Callable[..., Any], ir: TileKernelIR) -> None:
        self.fn = fn
        self.ir = ir
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.last_result: Optional[ExecutionResult] = None

    def _bind_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected = self.ir.input_names
        if len(args) > len(expected):
            raise TypeError(
                f"{self.ir.name}() expects {len(expected)} input tensors {expected}, got {len(args)} positional args"
            )
        bound: Dict[str, Any] = {}
        for name, val in zip(expected, args):
            bound[name] = val
        for name in expected[len(args) :]:
            if name not in kwargs:
                raise TypeError(f"{self.ir.name}() missing required input tensor {name!r}")
            bound[name] = kwargs[name]
        return bound

    def lower(
        self,
        *args: Any,
        target: str | Target = Target.BUENOS_A0,
        shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        quant: Optional[QuantParams] = None,
        **kwargs: Any,
    ) -> LoweredArtifact:
        tgt = Target.parse(target)
        if shapes is None:
            bound = self._bind_inputs(args, kwargs)
            shapes = {
                k: tuple(v) if isinstance(v, tuple) else tuple(np.asarray(v).shape)
                for k, v in bound.items()
            }
        if tgt == Target.CPU_REF:
            return lower_cpu_ref(self.ir, shapes)
        if tgt == Target.BUENOS_A0:
            return lower_buenos_a0(self.ir, shapes, quant=quant)
        return lower_cloud_tpu_pallas(self.ir, shapes, quant=quant)

    def run(
        self,
        *args: Any,
        target: str | Target = Target.CPU_REF,
        dry_run: bool = False,
        quant: Optional[QuantParams] = None,
        bias: Optional[np.ndarray] = None,
        scales: Optional[Dict[str, np.ndarray]] = None,
        adb_serial: Optional[str] = None,
        mock: Optional[bool] = None,
        warmup: int = 3,
        iters: int = 10,
        perf_mode: int = 5,
        **kwargs: Any,
    ) -> ExecutionResult:
        tgt = Target.parse(target)
        inputs = self._bind_inputs(args, kwargs)
        if tgt == Target.CPU_REF:
            res = run_cpu_ref(self.ir, inputs, quant=quant, bias=bias, scales=scales)
        elif tgt == Target.BUENOS_A0:
            res = run_buenos_a0(
                self.ir,
                inputs,
                quant=quant,
                bias=bias,
                scales=scales,
                adb_serial=adb_serial,
                mock=mock,
                warmup=warmup,
                iters=iters,
                perf_mode=perf_mode,
                dry_run=dry_run,
            )
        else:
            res = run_cloud_tpu_pallas(
                self.ir,
                inputs,
                quant=quant,
                scales=scales,
                warmup=warmup,
                iters=iters,
                dry_run=dry_run,
            )
        self.last_result = res
        return res

    def __call__(
        self,
        *args: Any,
        target: str | Target = Target.CPU_REF,
        **kwargs: Any,
    ) -> Any:
        res = self.run(*args, target=target, **kwargs)
        return res.output

    def parity_check(
        self,
        *args: Any,
        targets: Sequence[str | Target] = (Target.CPU_REF, Target.BUENOS_A0, Target.CLOUD_TPU_PALLAS),
        dry_run_targets: Sequence[str | Target] = (),
        quant: Optional[QuantParams] = None,
        bias: Optional[np.ndarray] = None,
        scales: Optional[Dict[str, np.ndarray]] = None,
        adb_serial: Optional[str] = None,
        mock: Optional[bool] = None,
        warmup: int = 3,
        iters: int = 10,
        max_abs_tol: Optional[float] = None,
        nmse_tol: float = 5e-3,
        **kwargs: Any,
    ) -> ParityReport:
        inputs = self._bind_inputs(args, kwargs)
        shapes = {k: tuple(np.asarray(v).shape) for k, v in inputs.items()}
        dry_set = {Target.parse(t) for t in dry_run_targets}

        ref_res = self.run(
            *args,
            target=Target.CPU_REF,
            quant=quant,
            bias=bias,
            scales=scales,
            **kwargs,
        )
        results: Dict[str, ExecutionResult] = {Target.CPU_REF.value: ref_res}
        parity: Dict[str, Dict[str, Any]] = {}
        all_ok = True

        ref_f = np.asarray(ref_res.output, dtype=np.float64)
        ref_energy = float(np.sum(ref_f * ref_f))

        for raw_t in targets:
            tgt = Target.parse(raw_t)
            if tgt == Target.CPU_REF:
                continue
            is_dry = tgt in dry_set
            res = self.run(
                *args,
                target=tgt,
                dry_run=is_dry,
                quant=quant,
                bias=bias,
                scales=scales,
                adb_serial=adb_serial,
                mock=mock,
                warmup=warmup,
                iters=iters,
                **kwargs,
            )
            results[tgt.value] = res
            if is_dry or res.output is None:
                parity[tgt.value] = {"status": "dry_run_lowered", "ok": True}
                continue

            got_f = np.asarray(res.output, dtype=np.float64)
            diff = np.abs(got_f - ref_f)
            max_err = float(np.max(diff))
            mean_err = float(np.mean(diff))
            mse = float(np.sum(diff * diff))
            nmse = mse / max(ref_energy, 1e-30)

            if self.ir.dtype == DType.INT8:
                tol = 1.0 if max_abs_tol is None else max_abs_tol
                ok = max_err <= tol
            else:
                if max_abs_tol is not None:
                    ok = max_err <= max_abs_tol
                else:
                    ok = nmse <= nmse_tol

            if not ok:
                all_ok = False
            parity[tgt.value] = {
                "status": "checked",
                "ok": ok,
                "max_abs_err": max_err,
                "mean_abs_err": mean_err,
                "nmse": nmse,
            }

        return ParityReport(
            op_kind=self.ir.op_kind,
            dtype=self.ir.dtype,
            shapes=shapes,
            reference_target=Target.CPU_REF,
            results=results,
            parity=parity,
            passed=all_ok,
        )


def tile_kernel(
    *,
    block_m: int = 64,
    block_n: int = 512,
    block_k: int = 512,
    dtype: str | DType = DType.BF16,
    rhs_transposed: bool = False,
    quant: Optional[QuantParams] = None,
) -> Callable[[Callable[..., Any]], TileKernel]:
    """Decorator defining a portable single-source tile kernel across Edge & Cloud TPUs."""
    dt = DType.parse(dtype)
    bspec = TileBlockSpec(block_m=block_m, block_n=block_n, block_k=block_k)
    qparams = quant or QuantParams()

    def _decorator(fn: Callable[..., Any]) -> TileKernel:
        ir = _canonicalize_ir(
            fn,
            block_spec=bspec,
            dtype=dt,
            rhs_transposed=rhs_transposed,
            quant=qparams,
        )
        return TileKernel(fn=fn, ir=ir)

    return _decorator
