import functools
import gc
import importlib.metadata
import json
import math
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ---------------------------------------------------------------------------
# Exact Llama-3-8B Geometry & Hardware Constants (TPU v5e-1)
# ---------------------------------------------------------------------------
D_MODEL = 4096
N_LAYERS = 32
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024
FF_DIM = 14336
KV_SEQ = 256
try:
    _TPU_INFO = pltpu.get_tpu_info()
    PEAK_HBM_GBS = round(float(getattr(_TPU_INFO, "mem_bw_bytes_per_second", 819e9)) / 1e9, 2)
except Exception:
    _TPU_INFO = None
    PEAK_HBM_GBS = 819.0
RMS_EPS = 1e-5

PROJECTIONS = [
    ("QKV", D_MODEL, D_MODEL + 2 * KV_DIM),  # 4096 -> 6144
    ("O", D_MODEL, D_MODEL),                 # 4096 -> 4096
    ("Gate+Up", D_MODEL, 2 * FF_DIM),        # 4096 -> 28672
    ("Down", FF_DIM, D_MODEL),               # 14336 -> 4096
]

CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(
    pltpu, "TPUCompilerParams"
)


def pkg_ver(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def get_hbm_stats():
    dev = jax.devices()[0]
    try:
        ms = dev.memory_stats()
        limit = int(ms.get("bytes_limit", 0))
        used = int(ms.get("bytes_in_use", 0))
        peak = int(ms.get("peak_bytes_in_use", 0))
        return {
            "bytes_in_use": used,
            "gib_in_use": round(used / (1024**3), 3),
            "peak_gib": round(peak / (1024**3), 3),
            "limit_gib": round(limit / (1024**3), 3),
            "largest_free_gib": round(
                int(ms.get("largest_free_block_bytes", 0)) / (1024**3), 3
            ),
        }
    except Exception as e:
        return {"err": repr(e)}


def bench_fn(fn, *args, warmup=4, iters=20):
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def bench_donated_decode(fn, weights, x, k_cache, v_cache, warmup=3, iters=12):
    for i in range(warmup):
        x_out, k_cache, v_cache = fn(weights, x, k_cache, v_cache, jnp.int32(i % KV_SEQ))
        jax.block_until_ready((x_out, k_cache, v_cache))
    ts = []
    for i in range(iters):
        pos = jnp.int32((warmup + i) % KV_SEQ)
        t0 = time.perf_counter()
        x_out, k_cache, v_cache = fn(weights, x, k_cache, v_cache, pos)
        jax.block_until_ready((x_out, k_cache, v_cache))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), x_out, k_cache, v_cache


print(f"=== LLAMA-3-8B W8A16 & W4A16 DECODE BENCHMARK ({jax.devices()[0].device_kind}, peak_hbm_gbs={PEAK_HBM_GBS}) ===")
print(
    "VERSIONS:",
    json.dumps(
        {
            k: v
            for k, v in {
                "jax": pkg_ver("jax"),
                "jaxlib": pkg_ver("jaxlib"),
                "libtpu": pkg_ver("libtpu"),
            }.items()
            if v is not None
        }
    ),
)
print("HBM_INIT:", json.dumps(get_hbm_stats()))


# ---------------------------------------------------------------------------
# Triton-like Decorator & Startup Probes for Mosaic TPU Capabilities
# ---------------------------------------------------------------------------
def tpu_kernel(fn):
    """Minimal Triton/CUDA-like marker decorator for Pallas-backed TPU ops."""
    return fn


def _make_probe_bf16_matmul(m_pad, bk=512, bn=512):
    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = jnp.dot(
            x_ref[...], w_ref[...], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

    @jax.jit
    def run(x, w):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m_pad, bn), jnp.bfloat16),
            grid=(1, 1),
            in_specs=[
                pl.BlockSpec((m_pad, bk), lambda i, j: (i, 0)),
                pl.BlockSpec((bk, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((m_pad, bn), lambda i, j: (i, j)),
            compiler_params=CompilerParams(dimension_semantics=("parallel", "parallel")),
        )(x, w)

    return run


def probe_min_m_pad():
    results = {}
    selected = None
    for m_try in (1, 8, 128):
        try:
            x = jax.random.normal(jax.random.PRNGKey(0), (m_try, 512), dtype=jnp.bfloat16)
            w = jax.random.normal(jax.random.PRNGKey(1), (512, 512), dtype=jnp.bfloat16) * (
                1.0 / math.sqrt(512)
            )
            fn = _make_probe_bf16_matmul(m_try, 512, 512)
            out = fn(x, w)
            out.block_until_ready()
            ref = jax.lax.dot_general(
                x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16)
            err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
            results[str(m_try)] = {"ok": True, "max_abs_err": err}
            if selected is None and m_try >= 8:
                selected = m_try
        except Exception as e:
            results[str(m_try)] = {"ok": False, "error": repr(e)[:220]}
    if selected is None:
        selected = 8 if results.get("8", {}).get("ok") else 128
    print("PROBE_MIN_M:", json.dumps({"results": results, "selected_min_m_pad": selected}))
    return selected


MIN_M_PAD = probe_min_m_pad()


def pad_m_dim(m):
    if m <= MIN_M_PAD:
        return MIN_M_PAD
    align = max(8, MIN_M_PAD) if MIN_M_PAD >= 128 else 8
    return ((m + align - 1) // align) * align


# ---------------------------------------------------------------------------
# W8A16 (BF16) Kernel + Probe
# ---------------------------------------------------------------------------
def _make_pallas_w8a16_fn(m_pad, k_dim, n_dim, bk, bn, cast_mode, scale_mode):
    num_k = k_dim // bk
    bm = m_pad

    def kernel(x_ref, w_ref, s_ref, o_ref, acc_ref):
        k_idx = pl.program_id(2)
        x_tile = x_ref[...]
        w_raw = w_ref[...]
        if cast_mode == "direct_bf16":
            w_f = w_raw.astype(jnp.bfloat16)
        elif cast_mode == "via_int32":
            w_f = w_raw.astype(jnp.int32).astype(jnp.bfloat16)
        else:
            w_f = w_raw.astype(jnp.float32).astype(jnp.bfloat16)

        s_tile = s_ref[...]
        if scale_mode == "1n_mul":
            w_deq = w_f * s_tile
        elif scale_mode == "1n_bcast":
            w_deq = w_f * jnp.broadcast_to(s_tile, (bk, bn))
        else:  # "8n_bcast"
            w_deq = w_f * jnp.repeat(s_tile, bk // 8, axis=0)

        prod = jnp.dot(x_tile, w_deq, preferred_element_type=jnp.float32)

        @pl.when(k_idx == 0)
        def _init():
            acc_ref[...] = prod

        @pl.when(k_idx > 0)
        def _acc():
            acc_ref[...] += prod

        @pl.when(k_idx == num_k - 1)
        def _epilogue():
            o_ref[...] = acc_ref[...].astype(jnp.bfloat16)

    scale_block = (1, bn) if scale_mode.startswith("1n") else (8, bn)

    @jax.jit
    def run(x_padded, w_int8, scales_2d):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m_pad, n_dim), jnp.bfloat16),
            grid=(1, n_dim // bn, num_k),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec(scale_block, lambda i, j, k: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                disable_bounds_checks=True,
            ),
        )(x_padded, w_int8, scales_2d)

    return run


def probe_w8a16_mode():
    k_t, n_t = 512, 512
    x = jax.random.normal(jax.random.PRNGKey(10), (MIN_M_PAD, k_t), dtype=jnp.bfloat16)
    w = jax.random.randint(jax.random.PRNGKey(11), (k_t, n_t), -127, 128, dtype=jnp.int8)
    s1 = (
        jax.random.uniform(jax.random.PRNGKey(12), (1, n_t), dtype=jnp.bfloat16) * 0.01
    )
    s8 = jnp.broadcast_to(s1, (8, n_t))
    ref = jax.lax.dot_general(
        x,
        w.astype(jnp.bfloat16) * s1,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    ref.block_until_ready()

    candidates = [
        ("direct_bf16", "1n_mul"),
        ("direct_bf16", "1n_bcast"),
        ("direct_bf16", "8n_bcast"),
        ("via_int32", "1n_mul"),
        ("via_f32", "1n_mul"),
    ]
    probe_log = []
    chosen = None
    for cm, sm in candidates:
        s_arg = s1 if sm.startswith("1n") else s8
        try:
            fn = _make_pallas_w8a16_fn(MIN_M_PAD, k_t, n_t, 512, 512, cm, sm)
            has_cc = "tpu_custom_call" in fn.lower(x, w, s_arg).as_text()
            out = fn(x, w, s_arg)
            out.block_until_ready()
            err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
            rec = {"cast": cm, "scale": sm, "ok": True, "tpu_custom_call": has_cc, "max_abs_err": err}
            probe_log.append(rec)
            if chosen is None and has_cc and err <= 0.05:
                chosen = (cm, sm)
        except Exception as e:
            probe_log.append({"cast": cm, "scale": sm, "ok": False, "error": repr(e)[:220]})

    print("PROBE_W8A16:", json.dumps({"chosen": chosen, "attempts": probe_log}))
    return chosen


W8A16_MODE = probe_w8a16_mode()


# ---------------------------------------------------------------------------
# W4A16 (BF16) Kernels + Probe (uint8 2-nibble vs int32 8-nibble)
# ---------------------------------------------------------------------------
def _make_pallas_w4_uint8_fn(m_pad, k_dim, n_dim, bk, bn_packed):
    """W4A16 packed in uint8 [K, N//2] -> unpacked in VMEM via int32 shifts (strictly 2D refs)."""
    num_k = k_dim // bk
    bm = m_pad
    n_half = n_dim // 2

    def kernel(x_ref, w_u8_ref, s0_ref, s1_ref, o_ref, acc0_ref, acc1_ref):
        k_idx = pl.program_id(2)
        x_tile = x_ref[...]
        wp = w_u8_ref[...].astype(jnp.int32)
        nib0 = (wp & 0xF) - 8
        nib1 = ((wp >> 4) & 0xF) - 8
        w0 = nib0.astype(jnp.bfloat16) * s0_ref[...]
        w1 = nib1.astype(jnp.bfloat16) * s1_ref[...]

        p0 = jnp.dot(x_tile, w0, preferred_element_type=jnp.float32)
        p1 = jnp.dot(x_tile, w1, preferred_element_type=jnp.float32)

        @pl.when(k_idx == 0)
        def _init():
            acc0_ref[...] = p0
            acc1_ref[...] = p1

        @pl.when(k_idx > 0)
        def _acc():
            acc0_ref[...] += p0
            acc1_ref[...] += p1

        @pl.when(k_idx == num_k - 1)
        def _epilogue():
            o_ref[0:bm, :] = acc0_ref[...].astype(jnp.bfloat16)
            o_ref[bm : 2 * bm, :] = acc1_ref[...].astype(jnp.bfloat16)

    @jax.jit
    def run(x_padded, w_u8, s_2x):
        s0_1n = s_2x[0:1, :]
        s1_1n = s_2x[1:2, :]
        out_split = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((2 * m_pad, n_half), jnp.bfloat16),
            grid=(1, n_half // bn_packed, num_k),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn_packed), lambda i, j, k: (k, j)),
                pl.BlockSpec((1, bn_packed), lambda i, j, k: (0, j)),
                pl.BlockSpec((1, bn_packed), lambda i, j, k: (0, j)),
            ],
            out_specs=pl.BlockSpec((2 * bm, bn_packed), lambda i, j, k: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((bm, bn_packed), jnp.float32),
                pltpu.VMEM((bm, bn_packed), jnp.float32),
            ],
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                disable_bounds_checks=True,
            ),
        )(x_padded, w_u8, s0_1n, s1_1n)
        return out_split.reshape(2, m_pad, n_half).transpose(1, 0, 2).reshape(m_pad, n_dim)

    return run


def _make_pallas_w4_int32_split8_fn(m_pad, k_dim, n_dim, bk, bn_packed, direct_2d=False):
    """W4A16 packed 8x4-bit nibbles per int32 [K, N//8] -> 0.5 B/weight (strictly 2D 1xN scales)."""
    num_k = k_dim // bk
    bm = m_pad
    n_packed = n_dim // 8

    if direct_2d:
        bn_full = 8 * bn_packed

        def kernel_2d(
            x_ref,
            w_i32_ref,
            s0_ref,
            s1_ref,
            s2_ref,
            s3_ref,
            s4_ref,
            s5_ref,
            s6_ref,
            s7_ref,
            o_ref,
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
        ):
            k_idx = pl.program_id(2)
            x_tile = x_ref[...]
            wp = w_i32_ref[...]
            s_refs = (s0_ref, s1_ref, s2_ref, s3_ref, s4_ref, s5_ref, s6_ref, s7_ref)
            acc_refs = (a0, a1, a2, a3, a4, a5, a6, a7)
            prods = []
            for s in range(8):
                nib = ((wp >> (4 * s)) & 0xF) - 8
                w_bf16 = nib.astype(jnp.bfloat16) * s_refs[s][...]
                prods.append(jnp.dot(x_tile, w_bf16, preferred_element_type=jnp.float32))

            @pl.when(k_idx == 0)
            def _init():
                for s in range(8):
                    acc_refs[s][...] = prods[s]

            @pl.when(k_idx > 0)
            def _acc():
                for s in range(8):
                    acc_refs[s][...] += prods[s]

            @pl.when(k_idx == num_k - 1)
            def _epilogue():
                for s in range(8):
                    o_ref[:, s * bn_packed : (s + 1) * bn_packed] = acc_refs[s][
                        ...
                    ].astype(jnp.bfloat16)

        @jax.jit
        def run_2d(x_padded, w_i32, s_8x):
            s_slices = [s_8x[s : s + 1, :] for s in range(8)]
            return pl.pallas_call(
                kernel_2d,
                out_shape=jax.ShapeDtypeStruct((m_pad, n_dim), jnp.bfloat16),
                grid=(1, n_packed // bn_packed, num_k),
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn_packed), lambda i, j, k: (k, j)),
                ]
                + [pl.BlockSpec((1, bn_packed), lambda i, j, k: (0, j))] * 8,
                out_specs=pl.BlockSpec((bm, bn_full), lambda i, j, k: (i, j)),
                scratch_shapes=[pltpu.VMEM((bm, bn_packed), jnp.float32)] * 8,
                compiler_params=CompilerParams(
                    dimension_semantics=("parallel", "parallel", "arbitrary"),
                    disable_bounds_checks=True,
                ),
            )(x_padded, w_i32, *s_slices)

        return run_2d

    def kernel_3d(
        x_ref,
        w_i32_ref,
        s0_ref,
        s1_ref,
        s2_ref,
        s3_ref,
        s4_ref,
        s5_ref,
        s6_ref,
        s7_ref,
        o_ref,
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
    ):
        k_idx = pl.program_id(2)
        x_tile = x_ref[...]
        wp = w_i32_ref[...]
        s_refs = (s0_ref, s1_ref, s2_ref, s3_ref, s4_ref, s5_ref, s6_ref, s7_ref)
        acc_refs = (a0, a1, a2, a3, a4, a5, a6, a7)
        prods = []
        for s in range(8):
            nib = ((wp >> (4 * s)) & 0xF) - 8
            w_bf16 = nib.astype(jnp.bfloat16) * s_refs[s][...]
            prods.append(jnp.dot(x_tile, w_bf16, preferred_element_type=jnp.float32))

        @pl.when(k_idx == 0)
        def _init():
            for s in range(8):
                acc_refs[s][...] = prods[s]

        @pl.when(k_idx > 0)
        def _acc():
            for s in range(8):
                acc_refs[s][...] += prods[s]

        @pl.when(k_idx == num_k - 1)
        def _epilogue():
            for s in range(8):
                o_ref[s * bm : (s + 1) * bm, :] = acc_refs[s][...].astype(jnp.bfloat16)

    @jax.jit
    def run_3d(x_padded, w_i32, s_8x):
        s_slices = [s_8x[s : s + 1, :] for s in range(8)]
        out_split = pl.pallas_call(
            kernel_3d,
            out_shape=jax.ShapeDtypeStruct((8 * m_pad, n_packed), jnp.bfloat16),
            grid=(1, n_packed // bn_packed, num_k),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn_packed), lambda i, j, k: (k, j)),
            ]
            + [pl.BlockSpec((1, bn_packed), lambda i, j, k: (0, j))] * 8,
            out_specs=pl.BlockSpec((8 * bm, bn_packed), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn_packed), jnp.float32)] * 8,
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                disable_bounds_checks=True,
            ),
        )(x_padded, w_i32, *s_slices)
        return out_split.reshape(8, m_pad, n_packed).transpose(1, 0, 2).reshape(m_pad, n_dim)

    return run_3d


@jax.jit
def xla_w4_uint8_ref(x, w_u8, s_2x):
    wp = w_u8.astype(jnp.int32)
    w0 = ((wp & 0xF) - 8).astype(jnp.bfloat16) * s_2x[0][None, :]
    w1 = (((wp >> 4) & 0xF) - 8).astype(jnp.bfloat16) * s_2x[1][None, :]
    w_bf16 = jnp.stack([w0, w1], axis=1).reshape(w_u8.shape[0], -1)
    return jax.lax.dot_general(
        x, w_bf16, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("tile_packed",))
def xla_w4_int32_ref(x, w_i32, s_8x, tile_packed=False):
    slices = [
        (((w_i32 >> (4 * s)) & 0xF) - 8).astype(jnp.bfloat16) * s_8x[s][None, :]
        for s in range(8)
    ]
    if tile_packed:
        # w_i32 columns are organized as (N // 1024, 128) and s=0..7 within each 1024 tile
        k_dim = w_i32.shape[0]
        n_packed = w_i32.shape[1]
        bn_p = 128
        num_tiles = n_packed // bn_p
        # slices: 8 * (K, num_tiles * bn_p) -> (K, num_tiles, 8, bn_p)
        stacked = jnp.stack([sl.reshape(k_dim, num_tiles, bn_p) for sl in slices], axis=2)
        w_bf16 = stacked.reshape(k_dim, -1)
    else:
        w_bf16 = jnp.stack(slices, axis=1).reshape(w_i32.shape[0], -1)
    return jax.lax.dot_general(
        x, w_bf16, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


def probe_w4a16_mode():
    k_t, n_t = 512, 1024
    x = jax.random.normal(jax.random.PRNGKey(20), (MIN_M_PAD, k_t), dtype=jnp.bfloat16)
    probe_log = []
    chosen = None

    # 1. Try uint8 (2 nibbles/byte)
    try:
        w_u8 = jax.random.randint(
            jax.random.PRNGKey(21), (k_t, n_t // 2), 0, 256, dtype=jnp.int32
        ).astype(jnp.uint8)
        s_2x = (
            jax.random.uniform(jax.random.PRNGKey(22), (2, n_t // 2), dtype=jnp.bfloat16)
            * 0.05
        )
        ref_u8 = xla_w4_uint8_ref(x, w_u8, s_2x)
        fn_u8 = _make_pallas_w4_uint8_fn(MIN_M_PAD, k_t, n_t, 512, 256)
        has_cc = "tpu_custom_call" in fn_u8.lower(x, w_u8, s_2x).as_text()
        out_u8 = fn_u8(x, w_u8, s_2x)
        out_u8.block_until_ready()
        err_u8 = float(jnp.max(jnp.abs(out_u8.astype(jnp.float32) - ref_u8.astype(jnp.float32))))
        med_u8, _ = bench_fn(fn_u8, x, w_u8, s_2x, warmup=3, iters=15)
        rec = {
            "mode": "uint8_split2",
            "ok": True,
            "tpu_custom_call": has_cc,
            "max_abs_err": err_u8,
            "probe_us": round(med_u8 * 1e6, 2),
        }
        probe_log.append(rec)
    except Exception as e:
        probe_log.append({"mode": "uint8_split2", "ok": False, "error": repr(e)[:240]})

    # 2. Try int32 8-nibble direct_2d & split8_3d
    w_i32 = jax.random.bits(jax.random.PRNGKey(23), (k_t, n_t // 8), dtype=jnp.uint32).astype(
        jnp.int32
    )
    s_8x = (
        jax.random.uniform(jax.random.PRNGKey(24), (8, n_t // 8), dtype=jnp.bfloat16)
        * 0.05
    )

    for mode_name, direct_2d in [("int32_direct2d", True), ("int32_split8_3d", False)]:
        try:
            ref_i32 = xla_w4_int32_ref(x, w_i32, s_8x, tile_packed=direct_2d)
            fn_i32 = _make_pallas_w4_int32_split8_fn(
                MIN_M_PAD, k_t, n_t, 512, 128, direct_2d=direct_2d
            )
            has_cc = "tpu_custom_call" in fn_i32.lower(x, w_i32, s_8x).as_text()
            out_i32 = fn_i32(x, w_i32, s_8x)
            out_i32.block_until_ready()
            err_i32 = float(
                jnp.max(jnp.abs(out_i32.astype(jnp.float32) - ref_i32.astype(jnp.float32)))
            )
            med_i32, _ = bench_fn(fn_i32, x, w_i32, s_8x, warmup=3, iters=15)
            rec = {
                "mode": mode_name,
                "ok": True,
                "tpu_custom_call": has_cc,
                "max_abs_err": err_i32,
                "probe_us": round(med_i32 * 1e6, 2),
            }
            probe_log.append(rec)
        except Exception as e:
            probe_log.append({"mode": mode_name, "ok": False, "error": repr(e)[:240]})

    ok_modes = [
        r
        for r in probe_log
        if r.get("ok") and r.get("tpu_custom_call") and r["max_abs_err"] <= 0.05
    ]
    if ok_modes:
        chosen = min(ok_modes, key=lambda r: r["probe_us"])["mode"]
    else:
        print("PROBE_W4A16_FAILED:", json.dumps({"attempts": probe_log}))

    print("PROBE_W4A16:", json.dumps({"chosen": chosen, "attempts": probe_log}))
    return chosen


W4A16_MODE = probe_w4a16_mode()


# ---------------------------------------------------------------------------
# Fused RMSNorm & Fused SwiGLU Pallas Kernels + Probe
# ---------------------------------------------------------------------------
def _make_pallas_rmsnorm_fn(m_pad, d_model=D_MODEL, bcast_mode="direct"):
    def kernel(x_ref, g_ref, o_ref):
        xf = x_ref[...].astype(jnp.float32)
        ms = jnp.mean(xf * xf, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(ms + RMS_EPS)
        if bcast_mode == "direct":
            normed = xf * inv
        elif bcast_mode == "broadcast_in_dim":
            normed = xf * jax.lax.broadcast_in_dim(inv, (m_pad, d_model), (0, 1))
        else:
            normed = xf * jnp.broadcast_to(inv, (m_pad, d_model))
        o_ref[...] = (normed * g_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)

    @jax.jit
    def run(x_padded, g_1n):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m_pad, d_model), jnp.bfloat16),
            grid=(1,),
            in_specs=[
                pl.BlockSpec((m_pad, d_model), lambda i: (0, 0)),
                pl.BlockSpec((1, d_model), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((m_pad, d_model), lambda i: (0, 0)),
            compiler_params=CompilerParams(
                dimension_semantics=("parallel",),
                disable_bounds_checks=True,
            ),
        )(x_padded, g_1n)

    return run


@jax.jit
def xla_rmsnorm(x, gamma_1d):
    xf = x.astype(jnp.float32)
    ms = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(ms + RMS_EPS) * gamma_1d.astype(jnp.float32)[None, :]).astype(
        jnp.bfloat16
    )


def _make_pallas_swiglu_fn(m_pad, ff_dim=FF_DIM, block_n=1024):
    def kernel(gate_ref, up_ref, o_ref):
        x = gate_ref[...].astype(jnp.float32)
        u = up_ref[...].astype(jnp.float32)
        o_ref[...] = ((x * jax.nn.sigmoid(x)) * u).astype(jnp.bfloat16)

    @jax.jit
    def run(gate_padded, up_padded):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m_pad, ff_dim), jnp.bfloat16),
            grid=(1, ff_dim // block_n),
            in_specs=[
                pl.BlockSpec((m_pad, block_n), lambda i, j: (0, j)),
                pl.BlockSpec((m_pad, block_n), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((m_pad, block_n), lambda i, j: (0, j)),
            compiler_params=CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                disable_bounds_checks=True,
            ),
        )(gate_padded, up_padded)

    return run


@jax.jit
def xla_swiglu(gate, up):
    gf = gate.astype(jnp.float32)
    uf = up.astype(jnp.float32)
    return ((gf * jax.nn.sigmoid(gf)) * uf).astype(jnp.bfloat16)


def probe_rmsnorm_swiglu():
    rms_mode = None
    g_1d = jax.random.normal(jax.random.PRNGKey(31), (D_MODEL,), dtype=jnp.bfloat16) * 0.1 + 1.0
    g_1n = g_1d[None, :]
    rms_log = []
    for bm_mode in ("direct", "broadcast_in_dim", "broadcast_to"):
        try:
            mode_ok = True
            errs = {}
            has_ccs = {}
            for m_test in sorted({MIN_M_PAD, 32}):
                x_rms = jax.random.normal(
                    jax.random.PRNGKey(30 + m_test), (m_test, D_MODEL), dtype=jnp.bfloat16
                )
                rms_ref = xla_rmsnorm(x_rms, g_1d)
                fn = _make_pallas_rmsnorm_fn(m_test, D_MODEL, bcast_mode=bm_mode)
                has_cc = "tpu_custom_call" in fn.lower(x_rms, g_1n).as_text()
                out = fn(x_rms, g_1n)
                out.block_until_ready()
                err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - rms_ref.astype(jnp.float32))))
                errs[str(m_test)] = err
                has_ccs[str(m_test)] = has_cc
                if not (has_cc and err <= 0.05):
                    mode_ok = False
            rms_log.append({"bcast": bm_mode, "ok": mode_ok, "tpu_custom_call": has_ccs, "max_abs_err": errs})
            if rms_mode is None and mode_ok:
                rms_mode = bm_mode
        except Exception as e:
            rms_log.append({"bcast": bm_mode, "ok": False, "error": repr(e)[:220]})

    sw_ok = False
    sw_log = {}
    try:
        gx = jax.random.normal(jax.random.PRNGKey(32), (MIN_M_PAD, FF_DIM), dtype=jnp.bfloat16)
        ux = jax.random.normal(jax.random.PRNGKey(33), (MIN_M_PAD, FF_DIM), dtype=jnp.bfloat16)
        fn_sw = _make_pallas_swiglu_fn(MIN_M_PAD, FF_DIM, 1024)
        has_cc = "tpu_custom_call" in fn_sw.lower(gx, ux).as_text()
        out_sw = fn_sw(gx, ux)
        out_sw.block_until_ready()
        ref_sw = xla_swiglu(gx, ux)
        err_sw = float(jnp.max(jnp.abs(out_sw.astype(jnp.float32) - ref_sw.astype(jnp.float32))))
        sw_ok = has_cc and err_sw <= 0.05
        sw_log = {"ok": True, "tpu_custom_call": has_cc, "max_abs_err": err_sw}
    except Exception as e:
        sw_log = {"ok": False, "error": repr(e)[:220]}

    print("PROBE_RMSNORM_SWIGLU:", json.dumps({"rmsnorm_mode": rms_mode, "rmsnorm": rms_log, "swiglu": sw_log}))
    return rms_mode, sw_ok


RMSNORM_MODE, SWIGLU_OK = probe_rmsnorm_swiglu()


# ---------------------------------------------------------------------------
# Clean Triton-like High-Level Wrappers (with automatic padding & caching)
# ---------------------------------------------------------------------------
_KERNEL_CACHE = {}


def _get_w8_kernel(m_pad, k_dim, n_dim, bk, bn):
    key_t = ("w8", m_pad, k_dim, n_dim, bk, bn)
    if key_t not in _KERNEL_CACHE:
        cm, sm = W8A16_MODE
        _KERNEL_CACHE[key_t] = _make_pallas_w8a16_fn(m_pad, k_dim, n_dim, bk, bn, cm, sm)
    return _KERNEL_CACHE[key_t]


def _get_w4_kernel(m_pad, k_dim, n_dim, bk, bn_packed):
    key_t = ("w4", W4A16_MODE, m_pad, k_dim, n_dim, bk, bn_packed)
    if key_t not in _KERNEL_CACHE:
        if W4A16_MODE == "uint8_split2":
            _KERNEL_CACHE[key_t] = _make_pallas_w4_uint8_fn(m_pad, k_dim, n_dim, bk, bn_packed)
        elif W4A16_MODE == "int32_direct2d":
            _KERNEL_CACHE[key_t] = _make_pallas_w4_int32_split8_fn(
                m_pad, k_dim, n_dim, bk, bn_packed, direct_2d=True
            )
        else:
            _KERNEL_CACHE[key_t] = _make_pallas_w4_int32_split8_fn(
                m_pad, k_dim, n_dim, bk, bn_packed, direct_2d=False
            )
    return _KERNEL_CACHE[key_t]


@tpu_kernel
def tpu_w8a16_linear(x, w_int8, scales, *, bk=1024, bn=1024):
    m_orig, k_dim = x.shape
    n_dim = w_int8.shape[1]
    if W8A16_MODE is None:
        s_1d = scales[0] if scales.ndim == 2 else scales
        return xla_w8a16_linear(x, w_int8, s_1d)
    m_pad = pad_m_dim(m_orig)
    x_pad = x if m_pad == m_orig else jnp.pad(x, ((0, m_pad - m_orig), (0, 0)))
    fn = _get_w8_kernel(m_pad, k_dim, n_dim, bk, bn)
    out = fn(x_pad, w_int8, scales)
    return out[:m_orig] if m_pad != m_orig else out


@jax.jit
def xla_w8a16_linear(x, w_int8, scales_1d):
    w_bf16 = w_int8.astype(jnp.bfloat16) * scales_1d[None, :]
    return jax.lax.dot_general(
        x, w_bf16, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


@tpu_kernel
def tpu_w4a16_linear(x, w_packed, scales_pack, n_dim, *, bk=1024, bn_packed=128):
    m_orig, k_dim = x.shape
    if W4A16_MODE is None:
        if w_packed.dtype == jnp.uint8:
            return xla_w4_uint8_ref(x, w_packed, scales_pack)
        return xla_w4_int32_ref(x, w_packed, scales_pack, tile_packed=False)
    m_pad = pad_m_dim(m_orig)
    x_pad = x if m_pad == m_orig else jnp.pad(x, ((0, m_pad - m_orig), (0, 0)))
    fn = _get_w4_kernel(m_pad, k_dim, n_dim, bk, bn_packed)
    out = fn(x_pad, w_packed, scales_pack)
    return out[:m_orig] if m_pad != m_orig else out


@tpu_kernel
def tpu_rmsnorm(x, gamma_1n, gamma_1d):
    m_orig, d_model = x.shape
    if RMSNORM_MODE is None:
        return xla_rmsnorm(x, gamma_1d)
    m_pad = pad_m_dim(m_orig)
    x_pad = x if m_pad == m_orig else jnp.pad(x, ((0, m_pad - m_orig), (0, 0)))
    key_t = ("rms", m_pad, d_model)
    if key_t not in _KERNEL_CACHE:
        _KERNEL_CACHE[key_t] = _make_pallas_rmsnorm_fn(m_pad, d_model, bcast_mode=RMSNORM_MODE)
    out = _KERNEL_CACHE[key_t](x_pad, gamma_1n)
    return out[:m_orig] if m_pad != m_orig else out


@tpu_kernel
def tpu_swiglu(gate, up):
    m_orig, ff_dim = gate.shape
    if not SWIGLU_OK:
        return xla_swiglu(gate, up)
    m_pad = pad_m_dim(m_orig)
    g_pad = gate if m_pad == m_orig else jnp.pad(gate, ((0, m_pad - m_orig), (0, 0)))
    u_pad = up if m_pad == m_orig else jnp.pad(up, ((0, m_pad - m_orig), (0, 0)))
    key_t = ("swiglu", m_pad, ff_dim)
    if key_t not in _KERNEL_CACHE:
        _KERNEL_CACHE[key_t] = _make_pallas_swiglu_fn(m_pad, ff_dim, 1024)
    out = _KERNEL_CACHE[key_t](g_pad, u_pad)
    return out[:m_orig] if m_pad != m_orig else out


# ---------------------------------------------------------------------------
# Auto-Tune Tile Sizes on QKV Projection Before Microbench & 32-Layer Run
# ---------------------------------------------------------------------------
def tune_w8_tile():
    if W8A16_MODE is None:
        return (1024, 1024)
    k_d, n_d = 4096, 6144
    m_pad = pad_m_dim(1)
    x = jax.random.normal(jax.random.PRNGKey(40), (m_pad, k_d), dtype=jnp.bfloat16)
    w = jax.random.randint(jax.random.PRNGKey(41), (k_d, n_d), -127, 128, dtype=jnp.int8)
    s1 = jax.random.uniform(jax.random.PRNGKey(42), (1, n_d), dtype=jnp.bfloat16) * (
        0.01 / math.sqrt(k_d)
    )
    s_arg = s1 if W8A16_MODE[1].startswith("1n") else jnp.broadcast_to(s1, (8, n_d))
    best_tile = (1024, 1024)
    best_us = float("inf")
    for bk, bn in [(1024, 1024), (1024, 2048), (2048, 1024), (512, 1024)]:
        try:
            fn = _get_w8_kernel(m_pad, k_d, n_d, bk, bn)
            out = fn(x, w, s_arg)
            out.block_until_ready()
            med_s, _ = bench_fn(fn, x, w, s_arg, warmup=3, iters=15)
            us = med_s * 1e6
            print("TUNE_W8:", json.dumps({"tile": f"{bk}x{bn}", "median_us": round(us, 2)}))
            if us < best_us:
                best_us = us
                best_tile = (bk, bn)
        except Exception as e:
            print("TUNE_W8_ERR:", json.dumps({"tile": f"{bk}x{bn}", "err": repr(e)[:180]}))
    return best_tile


def tune_w4_tile():
    if W4A16_MODE is None:
        return (1024, 128)
    k_d, n_d = 4096, 6144
    m_pad = pad_m_dim(1)
    x = jax.random.normal(jax.random.PRNGKey(50), (m_pad, k_d), dtype=jnp.bfloat16)
    best_tile = (1024, 128)
    best_us = float("inf")
    if W4A16_MODE == "uint8_split2":
        w = jax.random.randint(
            jax.random.PRNGKey(51), (k_d, n_d // 2), 0, 256, dtype=jnp.int32
        ).astype(jnp.uint8)
        s = (
            jax.random.uniform(
                jax.random.PRNGKey(52), (2, n_d // 2), dtype=jnp.bfloat16
            )
            * (0.1 / math.sqrt(k_d))
        )
        cands = [(1024, 256), (1024, 512), (2048, 512), (512, 256)]
    elif W4A16_MODE == "int32_direct2d":
        w = jax.random.bits(
            jax.random.PRNGKey(51), (k_d, n_d // 8), dtype=jnp.uint32
        ).astype(jnp.int32)
        s = (
            jax.random.uniform(
                jax.random.PRNGKey(52), (8, n_d // 8), dtype=jnp.bfloat16
            )
            * (0.1 / math.sqrt(k_d))
        )
        # direct_2d packs 8 * 128 = 1024 columns per bn_packed=128 tile
        cands = [(1024, 128), (2048, 128), (512, 128)]
    else:
        w = jax.random.bits(
            jax.random.PRNGKey(51), (k_d, n_d // 8), dtype=jnp.uint32
        ).astype(jnp.int32)
        s = (
            jax.random.uniform(
                jax.random.PRNGKey(52), (8, n_d // 8), dtype=jnp.bfloat16
            )
            * (0.1 / math.sqrt(k_d))
        )
        cands = [(1024, 128), (1024, 256), (2048, 256), (512, 128)]

    for bk, bn_p in cands:
        try:
            fn = _get_w4_kernel(m_pad, k_d, n_d, bk, bn_p)
            out = fn(x, w, s)
            out.block_until_ready()
            med_s, _ = bench_fn(fn, x, w, s, warmup=3, iters=15)
            us = med_s * 1e6
            print("TUNE_W4:", json.dumps({"mode": W4A16_MODE, "tile": f"{bk}x{bn_p}", "median_us": round(us, 2)}))
            if us < best_us:
                best_us = us
                best_tile = (bk, bn_p)
        except Exception as e:
            print("TUNE_W4_ERR:", json.dumps({"mode": W4A16_MODE, "tile": f"{bk}x{bn_p}", "err": repr(e)[:180]}))
    return best_tile


BEST_W8_BK, BEST_W8_BN = tune_w8_tile()
BEST_W4_BK, BEST_W4_BNP = tune_w4_tile()
_KERNEL_CACHE.clear()
jax.clear_caches()


# ---------------------------------------------------------------------------
# Helpers to Allocate W8A16 / W4A16 Projection Weights Fast & Cleanly
# ---------------------------------------------------------------------------
def alloc_w8_proj(key, k_dim, n_dim, resid_scale=1.0):
    k1, k2 = jax.random.split(key)
    bits = jax.random.bits(k1, (k_dim, n_dim // 4), dtype=jnp.uint32)
    w_i8 = jax.lax.bitcast_convert_type(bits, jnp.int8).reshape(k_dim, n_dim)
    s_1d = (
        jax.random.uniform(k2, (n_dim,), minval=0.5, maxval=1.5, dtype=jnp.bfloat16)
        * jnp.bfloat16(resid_scale / (128.0 * math.sqrt(k_dim)))
    )
    if W8A16_MODE is not None and not W8A16_MODE[1].startswith("1n"):
        s_kern = jnp.broadcast_to(s_1d[None, :], (8, n_dim))
    else:
        s_kern = s_1d[None, :]
    jax.block_until_ready((w_i8, s_1d, s_kern))
    return w_i8, s_1d, s_kern


def alloc_w4_proj(key, k_dim, n_dim, resid_scale=1.0):
    k1, k2 = jax.random.split(key)
    scale_mag = resid_scale / (8.0 * math.sqrt(k_dim))
    if W4A16_MODE == "uint8_split2":
        bits = jax.random.bits(k1, (k_dim, n_dim // 8), dtype=jnp.uint32)
        w_packed = jax.lax.bitcast_convert_type(bits, jnp.uint8).reshape(k_dim, n_dim // 2)
        s_base = (
            jax.random.uniform(k2, (2, n_dim // 2), minval=0.5, maxval=1.5, dtype=jnp.bfloat16)
            * jnp.bfloat16(scale_mag)
        )
    else:
        w_packed = jax.random.bits(k1, (k_dim, n_dim // 8), dtype=jnp.uint32).astype(jnp.int32)
        s_base = (
            jax.random.uniform(k2, (8, n_dim // 8), minval=0.5, maxval=1.5, dtype=jnp.bfloat16)
            * jnp.bfloat16(scale_mag)
        )
    jax.block_until_ready((w_packed, s_base))
    return w_packed, s_base, s_base


# ---------------------------------------------------------------------------
# 1. Single-Projection Microbenchmarks (Batch=1 & Batch=32, W8A16 & W4A16)
# ---------------------------------------------------------------------------
print("\n=== PART 1: SINGLE-PROJECTION MICROBENCHMARKS (PALLAS VS XLA) ===")
key_root = jax.random.PRNGKey(2026)

for proj_name, k_dim, n_dim in PROJECTIONS:
    w_i8, s8_1d, s8_kern = alloc_w8_proj(key_root, k_dim, n_dim)
    w_i4, s4_base, s4_kern = alloc_w4_proj(jax.random.fold_in(key_root, 1), k_dim, n_dim)

    for batch in (1, 32):
        x_in = jax.random.normal(jax.random.PRNGKey(batch), (batch, k_dim), dtype=jnp.bfloat16)
        x_in.block_until_ready()

        # --- W8A16 ---
        pallas_w8 = jax.jit(
            lambda xx, ww, ss: tpu_w8a16_linear(xx, ww, ss, bk=BEST_W8_BK, bn=BEST_W8_BN)
        )
        xla_w8 = jax.jit(lambda xx, ww, ss: xla_w8a16_linear(xx, ww, ss))

        has_cc_w8 = "tpu_custom_call" in pallas_w8.lower(x_in, w_i8, s8_kern).as_text()
        out_p8 = pallas_w8(x_in, w_i8, s8_kern)
        out_x8 = xla_w8(x_in, w_i8, s8_1d)
        jax.block_until_ready((out_p8, out_x8))
        err_w8 = float(jnp.max(jnp.abs(out_p8.astype(jnp.float32) - out_x8.astype(jnp.float32))))

        p8_med_s, p8_min_s = bench_fn(pallas_w8, x_in, w_i8, s8_kern, warmup=3, iters=12)
        x8_med_s, x8_min_s = bench_fn(xla_w8, x_in, w_i8, s8_1d, warmup=3, iters=12)
        bytes_w8 = float(w_i8.nbytes + s8_kern.nbytes + x_in.nbytes + out_p8.nbytes)
        p8_gbs = bytes_w8 / (p8_med_s * 1e9)
        x8_gbs = bytes_w8 / (x8_med_s * 1e9)

        print(
            "PROJ_MICROBENCH:",
            json.dumps(
                {
                    "quant": "W8A16",
                    "proj": proj_name,
                    "shape": f"{batch}x{k_dim}->{n_dim}",
                    "tile": f"{BEST_W8_BK}x{BEST_W8_BN}",
                    "tpu_custom_call": has_cc_w8,
                    "max_abs_err": err_w8,
                    "bytes_moved_mb": round(bytes_w8 / 1e6, 2),
                    "pallas_median_us": round(p8_med_s * 1e6, 2),
                    "pallas_min_us": round(p8_min_s * 1e6, 2),
                    "pallas_gb_s": round(p8_gbs, 2),
                    "pallas_pct_819": round(100.0 * p8_gbs / PEAK_HBM_GBS, 2),
                    "xla_median_us": round(x8_med_s * 1e6, 2),
                    "xla_gb_s": round(x8_gbs, 2),
                    "xla_pct_819": round(100.0 * x8_gbs / PEAK_HBM_GBS, 2),
                    "speedup": round(x8_med_s / p8_med_s, 2),
                }
            ),
        )

        # --- W4A16 ---
        pallas_w4 = jax.jit(
            lambda xx, ww, ss: tpu_w4a16_linear(
                xx, ww, ss, n_dim, bk=BEST_W4_BK, bn_packed=BEST_W4_BNP
            )
        )
        if W4A16_MODE == "uint8_split2":
            xla_w4 = jax.jit(lambda xx, ww, ss: xla_w4_uint8_ref(xx, ww, ss))
        else:
            xla_w4 = jax.jit(
                lambda xx, ww, ss: xla_w4_int32_ref(
                    xx, ww, ss, tile_packed=(W4A16_MODE == "int32_direct2d")
                )
            )

        has_cc_w4 = "tpu_custom_call" in pallas_w4.lower(x_in, w_i4, s4_kern).as_text()
        out_p4 = pallas_w4(x_in, w_i4, s4_kern)
        out_x4 = xla_w4(x_in, w_i4, s4_base)
        jax.block_until_ready((out_p4, out_x4))
        err_w4 = float(jnp.max(jnp.abs(out_p4.astype(jnp.float32) - out_x4.astype(jnp.float32))))

        p4_med_s, p4_min_s = bench_fn(pallas_w4, x_in, w_i4, s4_kern, warmup=3, iters=12)
        x4_med_s, x4_min_s = bench_fn(xla_w4, x_in, w_i4, s4_base, warmup=3, iters=12)
        bytes_w4 = float(w_i4.nbytes + s4_kern.nbytes + x_in.nbytes + out_p4.nbytes)
        p4_gbs = bytes_w4 / (p4_med_s * 1e9)
        x4_gbs = bytes_w4 / (x4_med_s * 1e9)

        print(
            "PROJ_MICROBENCH:",
            json.dumps(
                {
                    "quant": f"W4A16({W4A16_MODE or 'fallback_xla'})",
                    "pallas_fallback_to_xla": W4A16_MODE is None,
                    "proj": proj_name,
                    "shape": f"{batch}x{k_dim}->{n_dim}",
                    "tile": f"{BEST_W4_BK}x{BEST_W4_BNP}",
                    "tpu_custom_call": has_cc_w4,
                    "max_abs_err": err_w4,
                    "bytes_moved_mb": round(bytes_w4 / 1e6, 2),
                    "pallas_median_us": round(p4_med_s * 1e6, 2),
                    "pallas_min_us": round(p4_min_s * 1e6, 2),
                    "pallas_gb_s": round(p4_gbs, 2),
                    "pallas_pct_819": round(100.0 * p4_gbs / PEAK_HBM_GBS, 2),
                    "xla_median_us": round(x4_med_s * 1e6, 2),
                    "xla_gb_s": round(x4_gbs, 2),
                    "xla_pct_819": round(100.0 * x4_gbs / PEAK_HBM_GBS, 2),
                    "speedup": round(x4_med_s / p4_med_s, 2),
                }
            ),
        )

        del x_in, pallas_w8, xla_w8, out_p8, out_x8, pallas_w4, xla_w4, out_p4, out_x4
    del w_i8, s8_1d, s8_kern, w_i4, s4_base, s4_kern
    gc.collect()

# Microbenchmark RMSNorm & SwiGLU at B=1 and B=32
for batch in (1, 32):
    xr = jax.random.normal(jax.random.PRNGKey(60 + batch), (batch, D_MODEL), dtype=jnp.bfloat16)
    g1 = jax.random.uniform(jax.random.PRNGKey(61), (D_MODEL,), minval=0.8, maxval=1.2, dtype=jnp.bfloat16)
    g1n = g1[None, :]
    pr_fn = jax.jit(lambda a, gn, g: tpu_rmsnorm(a, gn, g))
    xr_fn = jax.jit(lambda a, g: xla_rmsnorm(a, g))
    pr_out = pr_fn(xr, g1n, g1)
    xr_out = xr_fn(xr, g1)
    jax.block_until_ready((pr_out, xr_out))
    rms_err = float(jnp.max(jnp.abs(pr_out.astype(jnp.float32) - xr_out.astype(jnp.float32))))
    pr_med, _ = bench_fn(pr_fn, xr, g1n, g1, warmup=3, iters=12)
    xr_med, _ = bench_fn(xr_fn, xr, g1, warmup=3, iters=12)
    print(
        "OP_MICROBENCH:",
        json.dumps(
            {
                "op": "RMSNorm",
                "shape": f"{batch}x{D_MODEL}",
                "tpu_custom_call": "tpu_custom_call" in pr_fn.lower(xr, g1n, g1).as_text(),
                "max_abs_err": rms_err,
                "pallas_us": round(pr_med * 1e6, 2),
                "xla_us": round(xr_med * 1e6, 2),
            }
        ),
    )

    gs = jax.random.normal(jax.random.PRNGKey(62 + batch), (batch, FF_DIM), dtype=jnp.bfloat16)
    us = jax.random.normal(jax.random.PRNGKey(63 + batch), (batch, FF_DIM), dtype=jnp.bfloat16)
    ps_fn = jax.jit(lambda a, b: tpu_swiglu(a, b))
    xs_fn = jax.jit(lambda a, b: xla_swiglu(a, b))
    ps_out = ps_fn(gs, us)
    xs_out = xs_fn(gs, us)
    jax.block_until_ready((ps_out, xs_out))
    sw_err = float(jnp.max(jnp.abs(ps_out.astype(jnp.float32) - xs_out.astype(jnp.float32))))
    ps_med, _ = bench_fn(ps_fn, gs, us, warmup=3, iters=12)
    xs_med, _ = bench_fn(xs_fn, gs, us, warmup=3, iters=12)
    print(
        "OP_MICROBENCH:",
        json.dumps(
            {
                "op": "SwiGLU",
                "shape": f"{batch}x{FF_DIM}",
                "tpu_custom_call": "tpu_custom_call" in ps_fn.lower(gs, us).as_text(),
                "max_abs_err": sw_err,
                "pallas_us": round(ps_med * 1e6, 2),
                "xla_us": round(xs_med * 1e6, 2),
            }
        ),
    )
    del xr, g1, g1n, pr_fn, xr_fn, pr_out, xr_out, gs, us, ps_fn, xs_fn, ps_out, xs_out

_KERNEL_CACHE.clear()
gc.collect()
jax.clear_caches()


# ---------------------------------------------------------------------------
# Shared Transformer Layer Decode Math (GQA 32Q / 8KV + Residuals)
# ---------------------------------------------------------------------------
def alloc_kv_cache(batch, seed=900):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + batch))
    kc = jax.random.uniform(
        k1,
        (N_LAYERS, batch, N_KV_HEADS, KV_SEQ, HEAD_DIM),
        minval=-0.2,
        maxval=0.2,
        dtype=jnp.bfloat16,
    )
    vc = jax.random.uniform(
        k2,
        (N_LAYERS, batch, N_KV_HEADS, KV_SEQ, HEAD_DIM),
        minval=-0.2,
        maxval=0.2,
        dtype=jnp.bfloat16,
    )
    jax.block_until_ready((kc, vc))
    return kc, vc


def gqa_decode_attn_layer(q_flat, k_new, v_new, k_l, v_l, pos):
    batch = q_flat.shape[0]
    k_l = k_l.at[:, :, pos, :].set(k_new)
    v_l = v_l.at[:, :, pos, :].set(v_new)
    q_gqa = q_flat.reshape(batch, N_KV_HEADS, N_HEADS // N_KV_HEADS, HEAD_DIM)
    scores = (
        jnp.einsum("bghd,bgsd->bghs", q_gqa.astype(jnp.float32), k_l.astype(jnp.float32))
        * (1.0 / math.sqrt(HEAD_DIM))
    )
    probs = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum("bghs,bgsd->bghd", probs, v_l.astype(jnp.float32))
    return attn.reshape(batch, D_MODEL).astype(jnp.bfloat16), k_l, v_l


def _gen_stacked_int8(key, n_layers, k_dim, n_dim):
    @jax.jit
    def _build(k):
        bits = jax.random.bits(k, (n_layers, k_dim, n_dim // 4), dtype=jnp.uint32)
        return jax.lax.bitcast_convert_type(bits, jnp.int8).reshape(n_layers, k_dim, n_dim)

    out = _build(key)
    out.block_until_ready()
    return out


def _gen_stacked_uint8(key, n_layers, k_dim, n_dim_half):
    @jax.jit
    def _build(k):
        bits = jax.random.bits(k, (n_layers, k_dim, n_dim_half // 4), dtype=jnp.uint32)
        return jax.lax.bitcast_convert_type(bits, jnp.uint8).reshape(
            n_layers, k_dim, n_dim_half
        )

    out = _build(key)
    out.block_until_ready()
    return out


def _gen_stacked_int32(key, n_layers, k_dim, n_dim_packed):
    @jax.jit
    def _build(k):
        return jax.random.bits(k, (n_layers, k_dim, n_dim_packed), dtype=jnp.uint32).astype(
            jnp.int32
        )

    out = _build(key)
    out.block_until_ready()
    return out


def alloc_stacked_w8_weights():
    resid_scale = 1.0 / math.sqrt(2.0 * N_LAYERS)
    k_root = jax.random.PRNGKey(1000)
    keys = jax.random.split(k_root, 10)
    # Allocate largest weight tensors first to minimize HBM fragmentation
    wgu = _gen_stacked_int8(keys[0], N_LAYERS, D_MODEL, 2 * FF_DIM)
    wd = _gen_stacked_int8(keys[1], N_LAYERS, FF_DIM, D_MODEL)
    wqkv = _gen_stacked_int8(keys[2], N_LAYERS, D_MODEL, D_MODEL + 2 * KV_DIM)
    wo = _gen_stacked_int8(keys[3], N_LAYERS, D_MODEL, D_MODEL)

    sgu = jax.random.uniform(
        keys[4], (N_LAYERS, 2 * FF_DIM), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
    ) * jnp.bfloat16(1.0 / (128.0 * math.sqrt(D_MODEL)))
    sd = jax.random.uniform(
        keys[5], (N_LAYERS, D_MODEL), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
    ) * jnp.bfloat16(resid_scale / (128.0 * math.sqrt(FF_DIM)))
    sqkv = jax.random.uniform(
        keys[6], (N_LAYERS, D_MODEL + 2 * KV_DIM), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
    ) * jnp.bfloat16(1.0 / (128.0 * math.sqrt(D_MODEL)))
    so = jax.random.uniform(
        keys[7], (N_LAYERS, D_MODEL), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
    ) * jnp.bfloat16(resid_scale / (128.0 * math.sqrt(D_MODEL)))
    r1 = jax.random.uniform(keys[8], (N_LAYERS, D_MODEL), minval=0.9, maxval=1.1, dtype=jnp.bfloat16)
    r2 = jax.random.uniform(keys[9], (N_LAYERS, D_MODEL), minval=0.9, maxval=1.1, dtype=jnp.bfloat16)
    jax.block_until_ready((wgu, wd, wqkv, wo, sgu, sd, sqkv, so, r1, r2))
    return {
        "wqkv": wqkv,
        "sqkv": sqkv,
        "wo": wo,
        "so": so,
        "wgu": wgu,
        "sgu": sgu,
        "wd": wd,
        "sd": sd,
        "r1": r1,
        "r2": r2,
    }


def alloc_stacked_w4_weights():
    resid_scale = 1.0 / math.sqrt(2.0 * N_LAYERS)
    k_root = jax.random.PRNGKey(5000)
    keys = jax.random.split(k_root, 10)
    if W4A16_MODE == "uint8_split2":
        wgu = _gen_stacked_uint8(keys[0], N_LAYERS, D_MODEL, FF_DIM)
        wd = _gen_stacked_uint8(keys[1], N_LAYERS, FF_DIM, D_MODEL // 2)
        wqkv = _gen_stacked_uint8(keys[2], N_LAYERS, D_MODEL, (D_MODEL + 2 * KV_DIM) // 2)
        wo = _gen_stacked_uint8(keys[3], N_LAYERS, D_MODEL, D_MODEL // 2)
        sgu = jax.random.uniform(
            keys[4], (N_LAYERS, 2, FF_DIM), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(1.0 / (8.0 * math.sqrt(D_MODEL)))
        sd = jax.random.uniform(
            keys[5], (N_LAYERS, 2, D_MODEL // 2), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(resid_scale / (8.0 * math.sqrt(FF_DIM)))
        sqkv = jax.random.uniform(
            keys[6],
            (N_LAYERS, 2, (D_MODEL + 2 * KV_DIM) // 2),
            minval=0.5,
            maxval=1.5,
            dtype=jnp.bfloat16,
        ) * jnp.bfloat16(1.0 / (8.0 * math.sqrt(D_MODEL)))
        so = jax.random.uniform(
            keys[7], (N_LAYERS, 2, D_MODEL // 2), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(resid_scale / (8.0 * math.sqrt(D_MODEL)))
    else:
        wgu = _gen_stacked_int32(keys[0], N_LAYERS, D_MODEL, (2 * FF_DIM) // 8)
        wd = _gen_stacked_int32(keys[1], N_LAYERS, FF_DIM, D_MODEL // 8)
        wqkv = _gen_stacked_int32(keys[2], N_LAYERS, D_MODEL, (D_MODEL + 2 * KV_DIM) // 8)
        wo = _gen_stacked_int32(keys[3], N_LAYERS, D_MODEL, D_MODEL // 8)
        sgu = jax.random.uniform(
            keys[4], (N_LAYERS, 8, (2 * FF_DIM) // 8), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(1.0 / (8.0 * math.sqrt(D_MODEL)))
        sd = jax.random.uniform(
            keys[5], (N_LAYERS, 8, D_MODEL // 8), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(resid_scale / (8.0 * math.sqrt(FF_DIM)))
        sqkv = jax.random.uniform(
            keys[6],
            (N_LAYERS, 8, (D_MODEL + 2 * KV_DIM) // 8),
            minval=0.5,
            maxval=1.5,
            dtype=jnp.bfloat16,
        ) * jnp.bfloat16(1.0 / (8.0 * math.sqrt(D_MODEL)))
        so = jax.random.uniform(
            keys[7], (N_LAYERS, 8, D_MODEL // 8), minval=0.5, maxval=1.5, dtype=jnp.bfloat16
        ) * jnp.bfloat16(resid_scale / (8.0 * math.sqrt(D_MODEL)))
    r1 = jax.random.uniform(keys[8], (N_LAYERS, D_MODEL), minval=0.9, maxval=1.1, dtype=jnp.bfloat16)
    r2 = jax.random.uniform(keys[9], (N_LAYERS, D_MODEL), minval=0.9, maxval=1.1, dtype=jnp.bfloat16)
    jax.block_until_ready((wgu, wd, wqkv, wo, sgu, sd, sqkv, so, r1, r2))
    return {
        "wqkv": wqkv,
        "sqkv": sqkv,
        "wo": wo,
        "so": so,
        "wgu": wgu,
        "sgu": sgu,
        "wd": wd,
        "sd": sd,
        "r1": r1,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# 2. Full 32-Layer Resident Llama-3-8B Decode: W8A16 (~6.50 GiB / ~7.0 GB)
# ---------------------------------------------------------------------------
print("\n=== PART 2: FULL 32-LAYER LLAMA-3-8B W8A16 DECODE (RESIDENT IN HBM, jax.lax.scan) ===")
w8_weights = alloc_stacked_w8_weights()
w8_weight_bytes = sum(
    int(w8_weights[k].nbytes) for k in ("wqkv", "wo", "wgu", "wd")
)
w8_scale_bytes = sum(
    int(w8_weights[k].nbytes) for k in ("sqkv", "so", "sgu", "sd", "r1", "r2")
)

print(
    "W8A16_ALLOC:",
    json.dumps(
        {
            "layers": N_LAYERS,
            "weights_gib": round(w8_weight_bytes / (1024**3), 4),
            "weights_gb": round(w8_weight_bytes / 1e9, 4),
            "scales_mb": round(w8_scale_bytes / 1e6, 2),
            "hbm": get_hbm_stats(),
        }
    ),
)


def _w8_scale_for_kernel(s_1d):
    if W8A16_MODE is not None and not W8A16_MODE[1].startswith("1n"):
        return jnp.broadcast_to(s_1d[None, :], (8, s_1d.shape[0]))
    return s_1d[None, :]


def make_w8_32l_step(use_pallas):
    @functools.partial(jax.jit, donate_argnums=(2, 3))
    def step(weights, x, k_cache, v_cache, pos):
        def layer_body(x_carry, xs_layer):
            wqkv, sqkv, wo, so, wgu, sgu, wd, sd, r1, r2, k_l, v_l = xs_layer
            h = tpu_rmsnorm(x_carry, r1[None, :], r1) if use_pallas else xla_rmsnorm(x_carry, r1)
            qkv = (
                tpu_w8a16_linear(h, wqkv, _w8_scale_for_kernel(sqkv), bk=BEST_W8_BK, bn=BEST_W8_BN)
                if use_pallas
                else xla_w8a16_linear(h, wqkv, sqkv)
            )
            q = qkv[:, :D_MODEL]
            k_new = qkv[:, D_MODEL : D_MODEL + KV_DIM].reshape(
                x_carry.shape[0], N_KV_HEADS, HEAD_DIM
            )
            v_new = qkv[:, D_MODEL + KV_DIM :].reshape(
                x_carry.shape[0], N_KV_HEADS, HEAD_DIM
            )
            attn, k_l_out, v_l_out = gqa_decode_attn_layer(q, k_new, v_new, k_l, v_l, pos)
            o_out = (
                tpu_w8a16_linear(attn, wo, _w8_scale_for_kernel(so), bk=BEST_W8_BK, bn=BEST_W8_BN)
                if use_pallas
                else xla_w8a16_linear(attn, wo, so)
            )
            x_mid = x_carry + o_out

            h2 = tpu_rmsnorm(x_mid, r2[None, :], r2) if use_pallas else xla_rmsnorm(x_mid, r2)
            gu = (
                tpu_w8a16_linear(h2, wgu, _w8_scale_for_kernel(sgu), bk=BEST_W8_BK, bn=BEST_W8_BN)
                if use_pallas
                else xla_w8a16_linear(h2, wgu, sgu)
            )
            mlp_act = (
                tpu_swiglu(gu[:, :FF_DIM], gu[:, FF_DIM:])
                if use_pallas
                else xla_swiglu(gu[:, :FF_DIM], gu[:, FF_DIM:])
            )
            down = (
                tpu_w8a16_linear(mlp_act, wd, _w8_scale_for_kernel(sd), bk=BEST_W8_BK, bn=BEST_W8_BN)
                if use_pallas
                else xla_w8a16_linear(mlp_act, wd, sd)
            )
            return x_mid + down, (k_l_out, v_l_out)

        xs = (
            weights["wqkv"],
            weights["sqkv"],
            weights["wo"],
            weights["so"],
            weights["wgu"],
            weights["sgu"],
            weights["wd"],
            weights["sd"],
            weights["r1"],
            weights["r2"],
            k_cache,
            v_cache,
        )
        x_out, (kc_out, vc_out) = jax.lax.scan(layer_body, x, xs)
        return x_out, kc_out, vc_out

    return step


for batch in (1, 32):
    x_tok = jax.random.normal(jax.random.PRNGKey(300 + batch), (batch, D_MODEL), dtype=jnp.bfloat16)
    kc, vc = alloc_kv_cache(batch, seed=900)
    jax.block_until_ready(x_tok)
    kv_bytes = float(kc.nbytes + vc.nbytes)
    total_bytes = float(w8_weight_bytes + w8_scale_bytes + kv_bytes)

    w8_pallas_fn = make_w8_32l_step(use_pallas=True)
    p_cc_count = (
        w8_pallas_fn.lower(w8_weights, x_tok, kc, vc, jnp.int32(0))
        .as_text()
        .count("tpu_custom_call")
    )
    t_c0 = time.perf_counter()
    p_x0, kc, vc = w8_pallas_fn(w8_weights, x_tok, kc, vc, jnp.int32(0))
    jax.block_until_ready((p_x0, kc, vc))
    p_compile_s = time.perf_counter() - t_c0
    p_med_s, p_min_s, p_x_last, kc, vc = bench_donated_decode(
        w8_pallas_fn, w8_weights, x_tok, kc, vc
    )

    del kc, vc, w8_pallas_fn
    gc.collect()
    kc, vc = alloc_kv_cache(batch, seed=900)

    w8_xla_fn = make_w8_32l_step(use_pallas=False)
    t_c1 = time.perf_counter()
    x_x0, kc, vc = w8_xla_fn(w8_weights, x_tok, kc, vc, jnp.int32(0))
    jax.block_until_ready((x_x0, kc, vc))
    x_compile_s = time.perf_counter() - t_c1
    step0_err = float(jnp.max(jnp.abs(p_x0.astype(jnp.float32) - x_x0.astype(jnp.float32))))
    x_med_s, x_min_s, _, kc, vc = bench_donated_decode(
        w8_xla_fn, w8_weights, x_tok, kc, vc
    )

    p_gbs = total_bytes / (p_med_s * 1e9)
    x_gbs = total_bytes / (x_med_s * 1e9)
    print(
        "LLAMA8B_32L_W8A16:",
        json.dumps(
            {
                "batch": batch,
                "kv_cache_shape": [N_LAYERS, batch, N_KV_HEADS, KV_SEQ, HEAD_DIM],
                "total_bytes_read_gb": round(total_bytes / 1e9, 4),
                "weight_gb": round(w8_weight_bytes / 1e9, 4),
                "kv_mb": round(kv_bytes / 1e6, 2),
                "scan_tpu_custom_call_count": p_cc_count,
                "step0_max_abs_err": step0_err,
                "pallas_compile_s": round(p_compile_s, 2),
                "pallas_ms_per_step": round(p_med_s * 1e3, 3),
                "pallas_tok_per_s": round(batch / p_med_s, 2),
                "pallas_gb_s": round(p_gbs, 2),
                "pallas_pct_819": round(100.0 * p_gbs / PEAK_HBM_GBS, 2),
                "xla_compile_s": round(x_compile_s, 2),
                "xla_ms_per_step": round(x_med_s * 1e3, 3),
                "xla_tok_per_s": round(batch / x_med_s, 2),
                "xla_gb_s": round(x_gbs, 2),
                "xla_pct_819": round(100.0 * x_gbs / PEAK_HBM_GBS, 2),
                "speedup": round(x_med_s / p_med_s, 2),
                "hbm": get_hbm_stats(),
            }
        ),
    )
    del x_tok, kc, vc, p_x0, p_x_last, x_x0, w8_xla_fn
    gc.collect()
    jax.clear_caches()

# Explicit HBM cleanup BEFORE allocating 32-layer W4A16 weights!
del w8_weights
_KERNEL_CACHE.clear()
gc.collect()
jax.clear_caches()
print("HBM_AFTER_W8A16_FREE:", json.dumps(get_hbm_stats()))


# ---------------------------------------------------------------------------
# 3. Full 32-Layer Resident Llama-3-8B Decode: W4A16 (~3.25 GiB / ~3.5 GB)
# ---------------------------------------------------------------------------
print("\n=== PART 3: FULL 32-LAYER LLAMA-3-8B W4A16 DECODE (RESIDENT IN HBM, jax.lax.scan) ===")
w4_weights = alloc_stacked_w4_weights()
w4_weight_bytes = sum(
    int(w4_weights[k].nbytes) for k in ("wqkv", "wo", "wgu", "wd")
)
w4_scale_bytes = sum(
    int(w4_weights[k].nbytes) for k in ("sqkv", "so", "sgu", "sd", "r1", "r2")
)

print(
    "W4A16_ALLOC:",
    json.dumps(
        {
            "mode": W4A16_MODE or "fallback_xla",
            "layers": N_LAYERS,
            "weights_gib": round(w4_weight_bytes / (1024**3), 4),
            "weights_gb": round(w4_weight_bytes / 1e9, 4),
            "scales_mb": round(w4_scale_bytes / 1e6, 2),
            "hbm": get_hbm_stats(),
        }
    ),
)


def _xla_w4_linear(x, w_p, s_b):
    if W4A16_MODE == "uint8_split2":
        return xla_w4_uint8_ref(x, w_p, s_b)
    return xla_w4_int32_ref(x, w_p, s_b, tile_packed=(W4A16_MODE == "int32_direct2d"))


def make_w4_32l_step(use_pallas):
    @functools.partial(jax.jit, donate_argnums=(2, 3))
    def step(weights, x, k_cache, v_cache, pos):
        def layer_body(x_carry, xs_layer):
            wqkv, sqkv, wo, so, wgu, sgu, wd, sd, r1, r2, k_l, v_l = xs_layer
            h = tpu_rmsnorm(x_carry, r1[None, :], r1) if use_pallas else xla_rmsnorm(x_carry, r1)
            qkv = (
                tpu_w4a16_linear(
                    h, wqkv, sqkv, D_MODEL + 2 * KV_DIM, bk=BEST_W4_BK, bn_packed=BEST_W4_BNP
                )
                if use_pallas
                else _xla_w4_linear(h, wqkv, sqkv)
            )
            q = qkv[:, :D_MODEL]
            k_new = qkv[:, D_MODEL : D_MODEL + KV_DIM].reshape(
                x_carry.shape[0], N_KV_HEADS, HEAD_DIM
            )
            v_new = qkv[:, D_MODEL + KV_DIM :].reshape(
                x_carry.shape[0], N_KV_HEADS, HEAD_DIM
            )
            attn, k_l_out, v_l_out = gqa_decode_attn_layer(q, k_new, v_new, k_l, v_l, pos)
            o_out = (
                tpu_w4a16_linear(attn, wo, so, D_MODEL, bk=BEST_W4_BK, bn_packed=BEST_W4_BNP)
                if use_pallas
                else _xla_w4_linear(attn, wo, so)
            )
            x_mid = x_carry + o_out

            h2 = tpu_rmsnorm(x_mid, r2[None, :], r2) if use_pallas else xla_rmsnorm(x_mid, r2)
            gu = (
                tpu_w4a16_linear(
                    h2, wgu, sgu, 2 * FF_DIM, bk=BEST_W4_BK, bn_packed=BEST_W4_BNP
                )
                if use_pallas
                else _xla_w4_linear(h2, wgu, sgu)
            )
            mlp_act = (
                tpu_swiglu(gu[:, :FF_DIM], gu[:, FF_DIM:])
                if use_pallas
                else xla_swiglu(gu[:, :FF_DIM], gu[:, FF_DIM:])
            )
            down = (
                tpu_w4a16_linear(mlp_act, wd, sd, D_MODEL, bk=BEST_W4_BK, bn_packed=BEST_W4_BNP)
                if use_pallas
                else _xla_w4_linear(mlp_act, wd, sd)
            )
            return x_mid + down, (k_l_out, v_l_out)

        xs = (
            weights["wqkv"],
            weights["sqkv"],
            weights["wo"],
            weights["so"],
            weights["wgu"],
            weights["sgu"],
            weights["wd"],
            weights["sd"],
            weights["r1"],
            weights["r2"],
            k_cache,
            v_cache,
        )
        x_out, (kc_out, vc_out) = jax.lax.scan(layer_body, x, xs)
        return x_out, kc_out, vc_out

    return step


for batch in (1, 32):
    x_tok = jax.random.normal(jax.random.PRNGKey(600 + batch), (batch, D_MODEL), dtype=jnp.bfloat16)
    kc, vc = alloc_kv_cache(batch, seed=1900)
    jax.block_until_ready(x_tok)
    kv_bytes = float(kc.nbytes + vc.nbytes)
    total_bytes = float(w4_weight_bytes + w4_scale_bytes + kv_bytes)

    w4_pallas_fn = make_w4_32l_step(use_pallas=True)
    p_cc_count = (
        w4_pallas_fn.lower(w4_weights, x_tok, kc, vc, jnp.int32(0))
        .as_text()
        .count("tpu_custom_call")
    )
    t_c0 = time.perf_counter()
    p_x0, kc, vc = w4_pallas_fn(w4_weights, x_tok, kc, vc, jnp.int32(0))
    jax.block_until_ready((p_x0, kc, vc))
    p_compile_s = time.perf_counter() - t_c0
    p_med_s, p_min_s, p_x_last, kc, vc = bench_donated_decode(
        w4_pallas_fn, w4_weights, x_tok, kc, vc
    )

    del kc, vc, w4_pallas_fn
    gc.collect()
    kc, vc = alloc_kv_cache(batch, seed=1900)

    w4_xla_fn = make_w4_32l_step(use_pallas=False)
    t_c1 = time.perf_counter()
    x_x0, kc, vc = w4_xla_fn(w4_weights, x_tok, kc, vc, jnp.int32(0))
    jax.block_until_ready((x_x0, kc, vc))
    x_compile_s = time.perf_counter() - t_c1
    step0_err = float(jnp.max(jnp.abs(p_x0.astype(jnp.float32) - x_x0.astype(jnp.float32))))
    x_med_s, x_min_s, _, kc, vc = bench_donated_decode(
        w4_xla_fn, w4_weights, x_tok, kc, vc
    )

    p_gbs = total_bytes / (p_med_s * 1e9)
    x_gbs = total_bytes / (x_med_s * 1e9)
    print(
        "LLAMA8B_32L_W4A16:",
        json.dumps(
            {
                "mode": W4A16_MODE or "fallback_xla",
                "pallas_fallback_to_xla": W4A16_MODE is None,
                "batch": batch,
                "kv_cache_shape": [N_LAYERS, batch, N_KV_HEADS, KV_SEQ, HEAD_DIM],
                "total_bytes_read_gb": round(total_bytes / 1e9, 4),
                "weight_gb": round(w4_weight_bytes / 1e9, 4),
                "kv_mb": round(kv_bytes / 1e6, 2),
                "scan_tpu_custom_call_count": p_cc_count,
                "step0_max_abs_err": step0_err,
                "pallas_compile_s": round(p_compile_s, 2),
                "pallas_ms_per_step": round(p_med_s * 1e3, 3),
                "pallas_tok_per_s": round(batch / p_med_s, 2),
                "pallas_gb_s": round(p_gbs, 2),
                "pallas_pct_819": round(100.0 * p_gbs / PEAK_HBM_GBS, 2),
                "xla_compile_s": round(x_compile_s, 2),
                "xla_ms_per_step": round(x_med_s * 1e3, 3),
                "xla_tok_per_s": round(batch / x_med_s, 2),
                "xla_gb_s": round(x_gbs, 2),
                "xla_pct_819": round(100.0 * x_gbs / PEAK_HBM_GBS, 2),
                "speedup": round(x_med_s / p_med_s, 2),
                "hbm": get_hbm_stats(),
            }
        ),
    )
    del x_tok, kc, vc, p_x0, p_x_last, x_x0, w4_xla_fn
    gc.collect()
    jax.clear_caches()

del w4_weights
_KERNEL_CACHE.clear()
gc.collect()
jax.clear_caches()
print("HBM_END:", json.dumps(get_hbm_stats()))
print("=== LLAMA8B_COMPLETE ===")
