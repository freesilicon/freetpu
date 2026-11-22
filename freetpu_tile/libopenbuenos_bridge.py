"""Clean-room ctypes C-API bridge to libopenbuenos (mock & live /dev/edgetpu-soc)."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

OB_OK = 0
OB_ERR_OVERFLOW = -11

OB_ACT_NONE = 0
OB_ACT_RELU = 1
OB_ACT_RELU6 = 3
OB_ACT_SILU = 4
OB_ACT_GELU = 5

OB_BUF_BIDIRECTIONAL = 0
OB_BUF_TO_DEVICE = 1 << 0
OB_BUF_FROM_DEVICE = 1 << 1
OB_BUF_ZERO_INIT = 1 << 3

OB_STAT_FLAG_MOCK = 1 << 0
OB_STAT_FLAG_CACHE_HIT = 1 << 1
OB_STAT_FLAG_CPU_FALLBACK = 1 << 2
OB_STAT_FLAG_EMITTER_USED = 1 << 3

OB_QK8_0 = 32
OB_MAX_GRAPH_IO = 8


class ObConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("mock_mode", ctypes.c_int),
        ("device_path", ctypes.c_char_p),
        ("dma_heap_path", ctypes.c_char_p),
        ("max_inflight", ctypes.c_uint32),
        ("graph_cache_capacity", ctypes.c_uint32),
        ("buffer_pool_max_bytes", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
        ("core_power_state", ctypes.c_uint8),
        ("mem_power_state", ctypes.c_uint8),
        ("qos_class", ctypes.c_uint8),
        ("cluster_id", ctypes.c_uint8),
        ("priority", ctypes.c_uint8),
        ("perf_mode", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
    ]


class ObInvokeStats(ctypes.Structure):
    _fields_ = [
        ("hw_execution_us", ctypes.c_uint64),
        ("wall_time_us", ctypes.c_uint64),
        ("seq", ctypes.c_uint64),
        ("model_handle", ctypes.c_uint32),
        ("vii_resp_code", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("tiles_dispatched", ctypes.c_uint32),
        ("cache_hits", ctypes.c_uint32),
    ]


class ObDgc0Info(ctypes.Structure):
    _fields_ = [
        ("num_inputs", ctypes.c_uint32),
        ("num_outputs", ctypes.c_uint32),
        ("input_sizes", ctypes.c_uint64 * OB_MAX_GRAPH_IO),
        ("output_sizes", ctypes.c_uint64 * OB_MAX_GRAPH_IO),
        ("dive_scratch_size", ctypes.c_uint64),
        ("ext_vm_ir_size", ctypes.c_uint64),
        ("ext_image_size", ctypes.c_uint64),
        ("ext_params_size", ctypes.c_uint64),
        ("compiler_cl", ctypes.c_uint64),
        ("model_handle", ctypes.c_uint32),
        ("cluster_id", ctypes.c_uint8),
        ("has_params", ctypes.c_uint8),
        ("cache_hit", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class ObGemmDesc(ctypes.Structure):
    _fields_ = [
        ("m_dim", ctypes.c_uint32),
        ("k_dim", ctypes.c_uint32),
        ("n_dim", ctypes.c_uint32),
        ("s_in", ctypes.c_float),
        ("zp_in", ctypes.c_int32),
        ("s_w", ctypes.c_float),
        ("zp_w", ctypes.c_int32),
        ("s_out", ctypes.c_float),
        ("zp_out", ctypes.c_int32),
        ("activation", ctypes.c_uint32),
        ("weights_nk", ctypes.POINTER(ctypes.c_int8)),
        ("bias_n", ctypes.POINTER(ctypes.c_int32)),
        ("packed_params", ctypes.POINTER(ctypes.c_uint8)),
        ("packed_params_bytes", ctypes.c_size_t),
        ("cache_key", ctypes.c_uint64),
    ]


class ObBlockQ8_0(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("d", ctypes.c_uint16),
        ("qs", ctypes.c_int8 * OB_QK8_0),
    ]


class ObGemmF32QuantJob(ctypes.Structure):
    _fields_ = [
        ("b_nk", ctypes.c_void_p),
        ("c_mn", ctypes.POINTER(ctypes.c_float)),
        ("n_dim", ctypes.c_uint32),
        ("activation", ctypes.c_uint32),
        ("is_q4_0", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _copy_build_tree(src_dir: Path, docs_dir: Optional[Path], dst_root: Path) -> Path:
    import shutil

    dst_lib = dst_root / "libopenbuenos"
    dst_lib.mkdir(parents=True, exist_ok=True)
    for rel in ("Makefile", "CMakeLists.txt"):
        f = src_dir / rel
        if f.is_file():
            shutil.copy2(f, dst_lib / rel)
    for sub in ("include", "src"):
        s = src_dir / sub
        d = dst_lib / sub
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
    if docs_dir is not None and docs_dir.is_dir():
        dst_docs = dst_root / "docs"
        for sub in ("baremetal-vii", "kernel-abi"):
            s = docs_dir / sub
            d = dst_docs / sub
            if s.is_dir():
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(s, d)
    return dst_lib


def find_libopenbuenos_path(build_if_missing: bool = True) -> Optional[Path]:
    """Locate (or optionally build via make) the libopenbuenos shared library."""
    env_p = os.environ.get("OPENBUENOS_LIB_PATH", "").strip()
    if env_p:
        p = Path(env_p)
        if p.is_file():
            return p

    root = _repo_root()
    ext = "dylib" if platform.system() == "Darwin" else "so"
    cache_root = Path.home() / ".cache" / "freetpu" / "build_tree"
    cache_src = cache_root / "libopenbuenos"
    candidates = [
        root / "libopenbuenos" / "build" / "host" / f"libopenbuenos.{ext}",
        root / "build-cmake" / f"libopenbuenos.{ext}",
        root / "libopenbuenos" / "build" / "cmake-host" / f"libopenbuenos.{ext}",
        cache_src / "build" / "host" / f"libopenbuenos.{ext}",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    src_dir = root / "libopenbuenos"
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        pkg_docs = Path(__file__).resolve().parent / "_packaged_docs"
        if pkg_docs.is_dir():
            docs_dir = pkg_docs
    if build_if_missing and (src_dir / "Makefile").is_file():
        in_site_packages = "site-packages" in src_dir.parts or "dist-packages" in src_dir.parts
        build_dir = src_dir if (os.access(src_dir, os.W_OK) and not in_site_packages) else cache_src
        if build_dir is cache_src:
            build_dir = _copy_build_tree(src_dir, docs_dir, cache_root)
        res = subprocess.run(
            ["make", "-C", str(build_dir), "host-shared"],
            capture_output=True,
            text=True,
            check=False,
        )
        built = build_dir / "build" / "host" / f"libopenbuenos.{ext}"
        if res.returncode == 0 and built.is_file():
            return built
    return None


def pack_f32_to_q8_0_bytes(w_nk: np.ndarray) -> Tuple[bytes, int]:
    """Pack float32 weights [N, K] into contiguous ob_block_q8_0_t bytes (K padded to 32)."""
    w = np.asarray(w_nk, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D [N, K] weight array, got shape {w.shape}")
    n_dim, k_dim = w.shape
    k_pad = ((k_dim + OB_QK8_0 - 1) // OB_QK8_0) * OB_QK8_0
    if k_pad != k_dim:
        pad = np.zeros((n_dim, k_pad), dtype=np.float32)
        pad[:, :k_dim] = w
        w = pad
    nb_k = k_pad // OB_QK8_0
    blocks = w.reshape(n_dim, nb_k, OB_QK8_0)
    amax = np.max(np.abs(blocks), axis=-1)
    scales_f32 = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
    scales_f16 = scales_f32.astype(np.float16)
    scales_q = scales_f16.astype(np.float32)[..., None]
    quants = np.clip(np.rint(blocks / scales_q), -127, 127).astype(np.int8)

    dtype_q8 = np.dtype([("d", "<u2"), ("qs", "i1", (OB_QK8_0,))])
    out = np.empty((n_dim, nb_k), dtype=dtype_q8)
    out["d"] = scales_f16.view(np.uint16)
    out["qs"] = quants
    return out.tobytes(), k_pad


class OpenBuenosLibrary:
    """Thin ctypes wrapper binding libopenbuenos C-API symbols."""

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        resolved = lib_path or find_libopenbuenos_path(build_if_missing=True)
        if resolved is None or not resolved.is_file():
            raise FileNotFoundError(
                "libopenbuenos shared library not found; run `make -C libopenbuenos host-shared`"
            )
        self.lib_path = resolved
        self.lib = ctypes.CDLL(str(resolved))
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.ob_config_default.argtypes = []
        lib.ob_config_default.restype = ObConfig

        lib.ob_context_create.argtypes = [ctypes.POINTER(ObConfig), ctypes.POINTER(ctypes.c_void_p)]
        lib.ob_context_create.restype = ctypes.c_int

        lib.ob_context_destroy.argtypes = [ctypes.c_void_p]
        lib.ob_context_destroy.restype = None

        lib.ob_status_string.argtypes = [ctypes.c_int]
        lib.ob_status_string.restype = ctypes.c_char_p

        lib.ob_context_last_error.argtypes = [ctypes.c_void_p]
        lib.ob_context_last_error.restype = ctypes.c_char_p

        lib.ob_context_is_mock.argtypes = [ctypes.c_void_p]
        lib.ob_context_is_mock.restype = ctypes.c_int

        lib.ob_gemm_register_builtin_emitter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.ob_gemm_register_builtin_emitter.restype = ctypes.c_int

        lib.ob_gemm_synthesize_dgc0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ObGemmDesc),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.ob_gemm_synthesize_dgc0.restype = ctypes.c_int

        lib.ob_gemm_packed_weight_bytes.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.ob_gemm_packed_weight_bytes.restype = ctypes.c_size_t

        lib.ob_gemm_pack_weights_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        lib.ob_gemm_pack_weights_int8.restype = ctypes.c_int

        lib.ob_gemm_matmul_int8.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ObInvokeStats),
        ]
        lib.ob_gemm_matmul_int8.restype = ctypes.c_int

        lib.ob_gemm_matmul_f32_quant_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ObGemmF32QuantJob),
            ctypes.c_uint32,
            ctypes.POINTER(ObInvokeStats),
        ]
        lib.ob_gemm_matmul_f32_quant_batch.restype = ctypes.c_int

        lib.ob_buffer_alloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.ob_buffer_alloc.restype = ctypes.c_int

        lib.ob_buffer_free.argtypes = [ctypes.c_void_p]
        lib.ob_buffer_free.restype = None

        lib.ob_buffer_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.ob_buffer_write.restype = ctypes.c_int

        lib.ob_buffer_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.ob_buffer_read.restype = ctypes.c_int

        lib.ob_graph_register_dgc0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.ob_graph_register_dgc0.restype = ctypes.c_int

        lib.ob_graph_release.argtypes = [ctypes.c_void_p]
        lib.ob_graph_release.restype = None

        lib.ob_graph_invoke.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ObInvokeStats),
        ]
        lib.ob_graph_invoke.restype = ctypes.c_int

    def status_str(self, code: int) -> str:
        raw = self.lib.ob_status_string(int(code))
        return raw.decode("utf-8", errors="replace") if raw else f"OB_STATUS({code})"


class OpenBuenosSession:
    """Managed libopenbuenos context supporting OPENBUENOS_MOCK=1 and live /dev/edgetpu-soc."""

    def __init__(
        self,
        *,
        mock: Optional[bool] = None,
        perf_mode: int = 5,
        dive_interp_path: Optional[str] = None,
        lib_path: Optional[Path] = None,
    ) -> None:
        self.bridge = OpenBuenosLibrary(lib_path=lib_path)
        if mock is None:
            env_mock = os.environ.get("OPENBUENOS_MOCK", "").strip()
            if env_mock == "1":
                mock = True
            elif env_mock == "0":
                mock = False
            else:
                mock = not Path("/dev/edgetpu-soc").exists()

        cfg = self.bridge.lib.ob_config_default()
        cfg.mock_mode = 1 if mock else 0
        cfg.max_inflight = 16
        cfg.graph_cache_capacity = 256
        cfg.perf_mode = perf_mode

        self.ctx = ctypes.c_void_p()
        st = self.bridge.lib.ob_context_create(ctypes.byref(cfg), ctypes.byref(self.ctx))
        if st != OB_OK or not self.ctx.value:
            raise RuntimeError(f"ob_context_create failed: {self.bridge.status_str(st)}")

        self.is_mock = bool(self.bridge.lib.ob_context_is_mock(self.ctx))
        self.emitter_registered = False
        interp = dive_interp_path or os.environ.get("OPENBUENOS_DIVE_INTERP_PATH", "").strip()
        if interp and Path(interp).is_file():
            est = self.bridge.lib.ob_gemm_register_builtin_emitter(
                self.ctx, interp.encode("utf-8")
            )
            self.emitter_registered = est == OB_OK

    def close(self) -> None:
        if getattr(self, "ctx", None) and self.ctx.value:
            self.bridge.lib.ob_context_destroy(self.ctx)
            self.ctx = ctypes.c_void_p()

    def __enter__(self) -> "OpenBuenosSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def matmul_int8(
        self,
        a_mk: np.ndarray,
        b_nk: np.ndarray,
        bias_n: Optional[np.ndarray] = None,
        *,
        zp_a: int = 0,
        zp_b: int = 0,
    ) -> Tuple[np.ndarray, ObInvokeStats]:
        a = np.ascontiguousarray(a_mk, dtype=np.int8)
        b = np.ascontiguousarray(b_nk, dtype=np.int8)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got A={a.shape}, B_nk={b.shape}")
        m_dim, k_dim = a.shape
        n_dim, b_k = b.shape
        if b_k != k_dim:
            raise ValueError(f"Shape mismatch: A={a.shape}, B_nk={b.shape}")
        bias_arr: Optional[np.ndarray] = None
        if bias_n is not None:
            bias_arr = np.ascontiguousarray(bias_n, dtype=np.int32)
            if bias_arr.ndim != 1 or bias_arr.shape[0] != n_dim:
                raise ValueError(
                    f"INT8 bias_n must have shape ({n_dim},), got {bias_arr.shape}"
                )
        c_out = np.zeros((m_dim, n_dim), dtype=np.int32)
        stats = ObInvokeStats()
        st = self.bridge.lib.ob_gemm_matmul_int8(
            self.ctx,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            bias_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if bias_arr is not None
            else ctypes.POINTER(ctypes.c_int32)(),
            c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            m_dim,
            k_dim,
            n_dim,
            zp_a,
            zp_b,
            ctypes.byref(stats),
        )
        if st != OB_OK:
            raise RuntimeError(f"ob_gemm_matmul_int8 failed: {self.bridge.status_str(st)}")
        return c_out, stats

    def matmul_f32_quant_batch(
        self,
        a_mk: np.ndarray,
        weights_nk_list: Sequence[np.ndarray],
        activations: Sequence[int],
    ) -> Tuple[list[np.ndarray], ObInvokeStats]:
        a = np.ascontiguousarray(a_mk, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D [M, K] input tensor, got shape {a.shape}")
        m_dim, k_dim = a.shape
        k_pad = ((k_dim + OB_QK8_0 - 1) // OB_QK8_0) * OB_QK8_0
        if k_pad != k_dim:
            a_pad = np.zeros((m_dim, k_pad), dtype=np.float32)
            a_pad[:, :k_dim] = a
            a = a_pad

        num_jobs = len(weights_nk_list)
        if num_jobs != len(activations):
            raise ValueError("weights_nk_list and activations length mismatch")

        packed_buffers: list[bytes] = []
        outputs: list[np.ndarray] = []
        jobs = (ObGemmF32QuantJob * num_jobs)()
        for idx, (w_nk, act) in enumerate(zip(weights_nk_list, activations)):
            w_arr = np.asarray(w_nk, dtype=np.float32)
            if w_arr.ndim != 2:
                raise ValueError(f"Expected 2D [N, K] weight array, got shape {w_arr.shape}")
            if w_arr.shape[1] != k_dim:
                raise ValueError(f"Weight K mismatch: expected {k_dim}, got {w_arr.shape[1]}")
            q_bytes, w_k_pad = pack_f32_to_q8_0_bytes(w_arr)
            if w_k_pad != k_pad:
                raise ValueError(f"Weight K mismatch: expected {k_dim}, got {w_arr.shape[1]}")
            packed_buffers.append(q_bytes)
            n_dim = w_arr.shape[0]
            out_arr = np.zeros((m_dim, n_dim), dtype=np.float32)
            outputs.append(out_arr)
            jobs[idx].b_nk = ctypes.cast(ctypes.c_char_p(q_bytes), ctypes.c_void_p)
            jobs[idx].c_mn = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            jobs[idx].n_dim = n_dim
            jobs[idx].activation = int(act)
            jobs[idx].is_q4_0 = 0

        stats = ObInvokeStats()
        st = self.bridge.lib.ob_gemm_matmul_f32_quant_batch(
            self.ctx,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m_dim,
            k_pad,
            jobs,
            num_jobs,
            ctypes.byref(stats),
        )
        if st != OB_OK:
            raise RuntimeError(
                f"ob_gemm_matmul_f32_quant_batch failed: {self.bridge.status_str(st)}"
            )
        return outputs, stats

    def synthesize_dgc0(
        self,
        *,
        m_dim: int,
        k_dim: int,
        n_dim: int,
        weights_nk: np.ndarray,
        bias_n: Optional[np.ndarray],
        s_in: float,
        zp_in: int,
        s_w: float,
        s_out: float,
        zp_out: int,
        activation: int,
    ) -> bytes:
        if m_dim <= 0 or k_dim <= 0 or n_dim <= 0:
            raise ValueError(
                f"synthesize_dgc0 dimensions must be positive, got m={m_dim}, k={k_dim}, n={n_dim}"
            )
        w = np.ascontiguousarray(weights_nk, dtype=np.int8)
        if w.shape != (n_dim, k_dim):
            raise ValueError(
                f"weights_nk must have shape ({n_dim}, {k_dim}), got {w.shape}"
            )
        if bias_n is not None:
            b = np.ascontiguousarray(bias_n, dtype=np.int32)
            if b.shape != (n_dim,):
                raise ValueError(
                    f"bias_n must have shape ({n_dim},), got {b.shape}"
                )
        else:
            b = np.zeros((n_dim,), dtype=np.int32)
        if not self.emitter_registered:
            raise RuntimeError(
                "ob_gemm_synthesize_dgc0 requires a loaded RV64 DIVE interpreter image "
                "(set OPENBUENOS_DIVE_INTERP_PATH)"
            )
        desc = ObGemmDesc()
        desc.m_dim = m_dim
        desc.k_dim = k_dim
        desc.n_dim = n_dim
        desc.s_in = float(s_in)
        desc.zp_in = int(zp_in)
        desc.s_w = float(s_w)
        desc.zp_w = 0
        desc.s_out = float(s_out)
        desc.zp_out = int(zp_out)
        desc.activation = int(activation)
        desc.weights_nk = w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        desc.bias_n = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        req = ctypes.c_size_t(0)
        st = self.bridge.lib.ob_gemm_synthesize_dgc0(
            self.ctx, ctypes.byref(desc), None, 0, ctypes.byref(req)
        )
        if st != OB_ERR_OVERFLOW or req.value == 0:
            raise RuntimeError(f"ob_gemm_synthesize_dgc0 size probe failed: {self.bridge.status_str(st)}")
        buf = (ctypes.c_uint8 * req.value)()
        st = self.bridge.lib.ob_gemm_synthesize_dgc0(
            self.ctx, ctypes.byref(desc), buf, req.value, ctypes.byref(req)
        )
        if st != OB_OK:
            raise RuntimeError(f"ob_gemm_synthesize_dgc0 failed: {self.bridge.status_str(st)}")
        return bytes(buf[: req.value])
