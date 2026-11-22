"""Portable declarative Tile IR & result types for the FreeTPU Tile DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OpKind(str, Enum):
    MATMUL = "matmul"
    FUSED_MATMUL_SILU = "fused_matmul_silu"
    FUSED_MATMUL_GELU = "fused_matmul_gelu"
    SWIGLU = "swiglu"


class DType(str, Enum):
    BF16 = "bf16"
    INT8 = "int8"
    W8A16 = "w8a16"

    @classmethod
    def parse(cls, value: str | "DType") -> "DType":
        if isinstance(value, DType):
            return value
        norm = str(value).strip().lower()
        for member in cls:
            if member.value == norm:
                return member
        raise ValueError(f"Unsupported dtype {value!r}; expected one of {[d.value for d in cls]}")


class Target(str, Enum):
    CPU_REF = "cpu_ref"
    BUENOS_A0 = "buenos_a0"
    CLOUD_TPU_PALLAS = "cloud_tpu_pallas"

    @classmethod
    def parse(cls, value: str | "Target") -> "Target":
        if isinstance(value, Target):
            return value
        norm = str(value).strip().lower()
        aliases = {
            "cpu": cls.CPU_REF,
            "numpy": cls.CPU_REF,
            "buenos": cls.BUENOS_A0,
            "edge": cls.BUENOS_A0,
            "edgetpu": cls.BUENOS_A0,
            "rango": cls.BUENOS_A0,
            "pallas": cls.CLOUD_TPU_PALLAS,
            "cloud_tpu": cls.CLOUD_TPU_PALLAS,
            "tpu": cls.CLOUD_TPU_PALLAS,
        }
        if norm in aliases:
            return aliases[norm]
        for member in cls:
            if member.value == norm:
                return member
        raise ValueError(f"Unsupported target {value!r}; expected one of {[t.value for t in cls]}")


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for INT8 / Mode-A 0x640 spline kernels."""

    s_in: float = 0.03125
    zp_in: int = 0
    s_w: float = 0.015625
    zp_w: int = 0
    s_out: float = 0.0625
    zp_out: int = 0


@dataclass(frozen=True)
class TileBlockSpec:
    """Explicit tile block dimensions (BM, BN, BK) shared across Edge and Cloud TPUs."""

    block_m: int = 64
    block_n: int = 512
    block_k: int = 512

    def __post_init__(self) -> None:
        if self.block_m <= 0 or self.block_n <= 0 or self.block_k <= 0:
            raise ValueError(
                f"Tile block dimensions must be positive, got ({self.block_m}, {self.block_n}, {self.block_k})"
            )


@dataclass(frozen=True)
class TileOp:
    """Single IR node inside a traced @tile_kernel graph."""

    op: str
    inputs: Tuple[str, ...]
    output: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TileKernelIR:
    """Backend-neutral declarative Tile IR emitted by @freetpu.tile_kernel."""

    name: str
    op_kind: OpKind
    dtype: DType
    block_spec: TileBlockSpec
    rhs_transposed: bool
    input_names: Tuple[str, ...]
    output_name: str
    ops: Tuple[TileOp, ...]
    quant: QuantParams = field(default_factory=QuantParams)
    has_down_proj: bool = False
    gemm_roles: Tuple[str, str] = ()  # (lhs_param, rhs_param) for MATMUL / FUSED_MATMUL_*
    swiglu_roles: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_kind": self.op_kind.value,
            "dtype": self.dtype.value,
            "block_spec": {
                "block_m": self.block_spec.block_m,
                "block_n": self.block_spec.block_n,
                "block_k": self.block_spec.block_k,
            },
            "rhs_transposed": self.rhs_transposed,
            "input_names": list(self.input_names),
            "output_name": self.output_name,
            "has_down_proj": self.has_down_proj,
            "gemm_roles": list(self.gemm_roles),
            "swiglu_roles": dict(self.swiglu_roles),
            "ops": [
                {"op": o.op, "inputs": list(o.inputs), "output": o.output, "attrs": dict(o.attrs)}
                for o in self.ops
            ],
        }


@dataclass
class LoweredArtifact:
    """Backend-specific lowered artifact produced without executing on hardware."""

    target: Target
    ir: TileKernelIR
    shapes: Dict[str, Tuple[int, ...]]
    summary: Dict[str, Any]
    source_code: Optional[str] = None
    dgc0_bytes: Optional[bytes] = None
    vliw_chunks: Optional[List[bytes]] = None
    ext2_bytes: Optional[bytes] = None


@dataclass
class ExecutionResult:
    """Output tensor + hardware telemetry from executing a @tile_kernel."""

    target: Target
    op_kind: OpKind
    dtype: DType
    output: Any
    hw_us: Optional[float]
    wall_us: float
    min_hw_us: Optional[float] = None
    gflops: Optional[float] = None
    tflops: Optional[float] = None
    silicon: str = ""
    stack: str = ""
    mock: bool = False
    lowered: Optional[LoweredArtifact] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.value,
            "op_kind": self.op_kind.value,
            "dtype": self.dtype.value,
            "silicon": self.silicon,
            "stack": self.stack,
            "mock": self.mock,
            "hw_us": self.hw_us,
            "min_hw_us": self.min_hw_us,
            "wall_us": round(self.wall_us, 3),
            "gflops": self.gflops,
            "tflops": self.tflops,
            "metadata": self.metadata,
        }


@dataclass
class ParityReport:
    """Cross-target numerical parity & latency comparison against cpu_ref."""

    op_kind: OpKind
    dtype: DType
    shapes: Dict[str, Tuple[int, ...]]
    reference_target: Target
    results: Dict[str, ExecutionResult]
    parity: Dict[str, Dict[str, Any]]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_kind": self.op_kind.value,
            "dtype": self.dtype.value,
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "reference_target": self.reference_target.value,
            "passed": self.passed,
            "targets": {k: v.summary_dict() for k, v in self.results.items()},
            "parity": self.parity,
        }
