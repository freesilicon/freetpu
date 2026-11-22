import functools
import gc
import importlib.metadata
import inspect
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
from jax.experimental.pallas.ops.tpu import flash_attention


def pkg_ver(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def get_hbm_stats():
    dev = jax.devices()[0]
    try:
        ms = dev.memory_stats()
        return {
            "bytes_in_use": int(ms.get("bytes_in_use", 0)),
            "peak_bytes_in_use": int(ms.get("peak_bytes_in_use", 0)),
            "bytes_limit": int(ms.get("bytes_limit", 0)),
        }
    except Exception as e:
        return {"err": repr(e)}


def bench_fn(fn, *args, warmup=5, iters=30):
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


print("=== PALLAS / MOSAIC OPTIMIZATION + FLASH ATTENTION SUITE ===")
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
print("DEVICE:", str(jax.devices()[0]), "KIND:", jax.devices()[0].device_kind)
try:
    tpu_info_val = pltpu.get_tpu_info()
    print("TPU_INFO:", repr(tpu_info_val))
except Exception as e:
    print("TPU_INFO_ERR:", repr(e))

try:
    print("EMIT_PIPELINE_SIG:", str(inspect.signature(pltpu.emit_pipeline)))
except Exception as e:
    print("EMIT_PIPELINE_SIG_ERR:", repr(e))

CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(
    pltpu, "TPUCompilerParams"
)
print("COMPILER_PARAMS_SIG:", str(inspect.signature(CompilerParams)))

# Explicit opt_level is SparseCore-only on TensorCore in JAX 0.11.2 (lowering rejects it).
opt_levels_available = [None]


# ---------------------------------------------------------------------------
# Part A: Blocked 4096x4096x4096 BF16 GEMM + SiLU Optimization Sweep
# ---------------------------------------------------------------------------
print("\n=== PART A: 4096x4096x4096 BF16 GEMM + SILU SWEEP ===")
MM, KK, NN = 4096, 4096, 4096
FLOPS_GEMM = 2.0 * MM * KK * NN

key = jax.random.PRNGKey(0)
a_mat = jax.random.normal(key, (MM, KK), dtype=jnp.bfloat16) * (1.0 / math.sqrt(KK))
b_mat = jax.random.normal(jax.random.PRNGKey(1), (KK, NN), dtype=jnp.bfloat16) * (
    1.0 / math.sqrt(KK)
)
b_mat_t = jnp.array(jnp.swapaxes(b_mat, 0, 1), copy=True)
jax.block_until_ready((a_mat, b_mat, b_mat_t))


@jax.jit
def xla_matmul_silu(a, b):
    v = jax.lax.dot_general(
        a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    return (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)


@jax.jit
def xla_matmul_silu_rhs_t(a, b_t):
    v = jax.lax.dot_general(
        a, b_t, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    return (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)


xla_ref = xla_matmul_silu(a_mat, b_mat)
xla_ref.block_until_ready()
if not bool(jnp.all(jnp.isfinite(xla_ref)).item()):
    raise RuntimeError("Part A XLA baseline produced nonfinite outputs")
xla_med_s, xla_min_s = bench_fn(xla_matmul_silu, a_mat, b_mat, warmup=5, iters=30)
xla_t_med_s, xla_t_min_s = bench_fn(
    xla_matmul_silu_rhs_t, a_mat, b_mat_t, warmup=5, iters=30
)

xla_summary = {
    "variant": "xla_dot_general_silu",
    "shape": f"{MM}x{KK}x{NN}",
    "median_us": round(xla_med_s * 1e6, 2),
    "min_us": round(xla_min_s * 1e6, 2),
    "median_tflops": round(FLOPS_GEMM / (xla_med_s * 1e12), 2),
    "peak_tflops": round(FLOPS_GEMM / (xla_min_s * 1e12), 2),
    "rhs_t_median_us": round(xla_t_med_s * 1e6, 2),
    "rhs_t_median_tflops": round(FLOPS_GEMM / (xla_t_med_s * 1e12), 2),
}
print("XLA_GEMM_SILU:", json.dumps(xla_summary))


def tpu_gemm_silu_kernel_fn(
    bm,
    bn,
    bk,
    *,
    num_k,
    overwrite_k0=True,
    rhs_transposed=False,
):
    def kernel(a_ref, b_ref, o_ref, acc_ref):
        k_idx = pl.program_id(2)
        a_tile = a_ref[...]
        b_tile = b_ref[...]
        if rhs_transposed:
            prod = jax.lax.dot_general(
                a_tile,
                b_tile,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            prod = jnp.dot(a_tile, b_tile, preferred_element_type=jnp.float32)

        if overwrite_k0:
            @pl.when(k_idx == 0)
            def _init():
                acc_ref[...] = prod

            @pl.when(k_idx > 0)
            def _acc():
                acc_ref[...] += prod
        else:
            @pl.when(k_idx == 0)
            def _zero():
                acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

            acc_ref[...] += prod

        @pl.when(k_idx == num_k - 1)
        def _epilogue():
            v = acc_ref[...]
            o_ref[...] = (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)

    return kernel


def make_pallas_3d_gemm_silu(
    bm,
    bn,
    bk,
    *,
    overwrite_k0=True,
    rhs_transposed=False,
    vmem_limit_bytes=None,
    opt_level=None,
    allow_input_fusion=None,
    disable_bounds_checks=False,
):
    num_k = KK // bk
    kernel = tpu_gemm_silu_kernel_fn(
        bm,
        bn,
        bk,
        num_k=num_k,
        overwrite_k0=overwrite_k0,
        rhs_transposed=rhs_transposed,
    )
    cp_kwargs = {
        "dimension_semantics": ("parallel", "parallel", "arbitrary"),
    }
    if vmem_limit_bytes is not None:
        cp_kwargs["vmem_limit_bytes"] = vmem_limit_bytes
    if opt_level is not None:
        cp_kwargs["opt_level"] = opt_level
    if allow_input_fusion is not None:
        cp_kwargs["allow_input_fusion"] = allow_input_fusion
    if disable_bounds_checks:
        cp_kwargs["disable_bounds_checks"] = True

    b_spec = (
        pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
        if rhs_transposed
        else pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    )

    @jax.jit
    def run(a, b):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((MM, NN), jnp.bfloat16),
            grid=(MM // bm, NN // bn, num_k),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                b_spec,
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            compiler_params=CompilerParams(**cp_kwargs),
        )(a, b)

    return run


def make_emit_pipeline_gemm_silu(
    bm,
    bn,
    bk,
    *,
    rhs_transposed=False,
    vmem_limit_bytes=None,
    opt_level=None,
    disable_bounds_checks=False,
):
    num_k = KK // bk

    def outer_kernel(a_hbm_ref, b_hbm_ref, o_ref, acc_ref):
        i = pl.program_id(0)
        j = pl.program_id(1)
        acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

        def step_body(a_tile_ref, b_tile_ref):
            if rhs_transposed:
                acc_ref[...] += jax.lax.dot_general(
                    a_tile_ref[...],
                    b_tile_ref[...],
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            else:
                acc_ref[...] += jnp.dot(
                    a_tile_ref[...],
                    b_tile_ref[...],
                    preferred_element_type=jnp.float32,
                )

        b_inner_spec = (
            pl.BlockSpec((bn, bk), lambda k: (j, k))
            if rhs_transposed
            else pl.BlockSpec((bk, bn), lambda k: (k, j))
        )
        pltpu.emit_pipeline(
            step_body,
            grid=(num_k,),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda k: (i, k)),
                b_inner_spec,
            ],
            out_specs=(),
        )(a_hbm_ref, b_hbm_ref)

        v = acc_ref[...]
        o_ref[...] = (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)

    cp_kwargs = {
        "dimension_semantics": ("parallel", "parallel"),
    }
    if vmem_limit_bytes is not None:
        cp_kwargs["vmem_limit_bytes"] = vmem_limit_bytes
    if opt_level is not None:
        cp_kwargs["opt_level"] = opt_level
    if disable_bounds_checks:
        cp_kwargs["disable_bounds_checks"] = True

    @jax.jit
    def run(a, b):
        return pl.pallas_call(
            outer_kernel,
            out_shape=jax.ShapeDtypeStruct((MM, NN), jnp.bfloat16),
            grid=(MM // bm, NN // bn),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            compiler_params=CompilerParams(**cp_kwargs),
        )(a, b)

    return run


def make_async_copy_db_gemm_silu(
    bm,
    bn,
    bk,
    *,
    vmem_limit_bytes=None,
    disable_bounds_checks=False,
):
    num_k = KK // bk
    assert num_k >= 2

    def outer_kernel(
        a_hbm_ref, b_hbm_ref, o_ref, a_bufs, b_bufs, acc_ref, a_sems, b_sems
    ):
        i = pl.program_id(0)
        j = pl.program_id(1)
        m_off = i * bm
        n_off = j * bn

        def start_copy(k_idx, slot):
            k_off = k_idx * bk
            a_cp = pltpu.make_async_copy(
                a_hbm_ref.at[pl.ds(m_off, bm), pl.ds(k_off, bk)],
                a_bufs.at[slot],
                a_sems.at[slot],
            )
            b_cp = pltpu.make_async_copy(
                b_hbm_ref.at[pl.ds(k_off, bk), pl.ds(n_off, bn)],
                b_bufs.at[slot],
                b_sems.at[slot],
            )
            a_cp.start()
            b_cp.start()
            return a_cp, b_cp

        # Static-unrolled double-buffered K-loop (zero dynamic Ref slot indexing)
        pending = {0: start_copy(0, 0)}
        for k_idx in range(num_k):
            cur_slot = k_idx % 2
            if k_idx + 1 < num_k:
                nxt_slot = (k_idx + 1) % 2
                pending[nxt_slot] = start_copy(k_idx + 1, nxt_slot)
            a_cp, b_cp = pending[cur_slot]
            a_cp.wait()
            b_cp.wait()
            prod = jnp.dot(
                a_bufs[cur_slot], b_bufs[cur_slot], preferred_element_type=jnp.float32
            )
            if k_idx == 0:
                acc_ref[...] = prod
            else:
                acc_ref[...] += prod

        v = acc_ref[...]
        o_ref[...] = (v * jax.nn.sigmoid(v)).astype(jnp.bfloat16)

    cp_kwargs = {"dimension_semantics": ("parallel", "parallel")}
    if vmem_limit_bytes is not None:
        cp_kwargs["vmem_limit_bytes"] = vmem_limit_bytes
    if disable_bounds_checks:
        cp_kwargs["disable_bounds_checks"] = True

    @jax.jit
    def run(a, b):
        return pl.pallas_call(
            outer_kernel,
            out_shape=jax.ShapeDtypeStruct((MM, NN), jnp.bfloat16),
            grid=(MM // bm, NN // bn),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((2, bm, bk), jnp.bfloat16),
                pltpu.VMEM((2, bk, bn), jnp.bfloat16),
                pltpu.VMEM((bm, bn), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            compiler_params=CompilerParams(**cp_kwargs),
        )(a, b)

    return run


PART_A_MAX_ABS_ERR_TOL = 0.05
PART_B_MAX_ABS_ERR_TOL = 0.1


def run_gemm_silu_cfg(label, fn, lhs, rhs):
    try:
        lowered_txt = fn.lower(lhs, rhs).as_text()
        has_tpu_custom_call = "tpu_custom_call" in lowered_txt
        out = fn(lhs, rhs)
        out.block_until_ready()
        all_finite = bool(jnp.all(jnp.isfinite(out)).item())
        max_abs_err = float(
            jnp.max(jnp.abs(out.astype(jnp.float32) - xla_ref.astype(jnp.float32)))
        )
        parity_ok = (
            all_finite
            and math.isfinite(max_abs_err)
            and max_abs_err <= PART_A_MAX_ABS_ERR_TOL
        )
        med_s, min_s = bench_fn(fn, lhs, rhs, warmup=4, iters=25)
        med_tflops = FLOPS_GEMM / (med_s * 1e12)
        peak_tflops = FLOPS_GEMM / (min_s * 1e12)
        rec = {
            "config": label,
            "status": "ok" if parity_ok else "parity_fail",
            "finite": all_finite,
            "parity_ok": parity_ok,
            "tpu_custom_call": has_tpu_custom_call,
            "max_abs_err": max_abs_err,
            "median_us": round(med_s * 1e6, 2),
            "min_us": round(min_s * 1e6, 2),
            "median_tflops": round(med_tflops, 2),
            "peak_tflops": round(peak_tflops, 2),
            "pct_of_xla": round(100.0 * (xla_med_s / med_s), 2),
        }
        print("PALLAS_GEMM_SILU:", json.dumps(rec))
        return rec
    except Exception as e:
        err_rec = {
            "config": label,
            "status": "error",
            "error": repr(e)[:400],
        }
        print("PALLAS_GEMM_SILU_ERR:", json.dumps(err_rec))
        return err_rec


part_a_results = []

# 1. Tile sweep with pallas_call 3D grid
tile_candidates = [
    (512, 512, 256),
    (512, 512, 512),
    (1024, 512, 512),
    (512, 1024, 512),
    (1024, 1024, 256),
    (1024, 1024, 512),
    (1024, 512, 1024),
    (512, 1024, 1024),
    (2048, 512, 512),
    (512, 2048, 512),
    (1024, 1024, 1024),  # Expected VMEM stress/overflow test
]

for bm, bn, bk in tile_candidates:
    part_a_results.append(
        run_gemm_silu_cfg(
            f"pallas3d_{bm}x{bn}x{bk}",
            make_pallas_3d_gemm_silu(bm, bn, bk, overwrite_k0=True),
            a_mat,
            b_mat,
        )
    )

# 2. CompilerParams & transposed RHS sweep on strong tile shapes
cp_configs = [
    ("pallas3d_1024x512x512_vmem16m_dbc", 1024, 512, 512, False, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_1024x1024x256_vmem16m_dbc", 1024, 1024, 256, False, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_1024x1024x512_vmem16m_dbc", 1024, 1024, 512, False, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_512x1024x1024_vmem16m_dbc", 512, 1024, 1024, False, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_1024x512x512_rhsT", 1024, 512, 512, True, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_1024x1024x512_rhsT", 1024, 1024, 512, True, 16 * 1024 * 1024, None, (True, True), True),
    ("pallas3d_512x1024x1024_rhsT", 512, 1024, 1024, True, 16 * 1024 * 1024, None, (True, True), True),
]

for label, bm, bn, bk, rhs_t, vmem_b, opt_lvl, aif, dbc in cp_configs:
    rhs_arg = b_mat_t if rhs_t else b_mat
    part_a_results.append(
        run_gemm_silu_cfg(
            label,
            make_pallas_3d_gemm_silu(
                bm,
                bn,
                bk,
                overwrite_k0=True,
                rhs_transposed=rhs_t,
                vmem_limit_bytes=vmem_b,
                opt_level=opt_lvl,
                allow_input_fusion=aif,
                disable_bounds_checks=dbc,
            ),
            a_mat,
            rhs_arg,
        )
    )

# 3. Explicit pltpu.emit_pipeline double-buffered variants inside 16 MiB VMEM
emit_candidates = [
    (512, 512, 512, False),
    (1024, 512, 512, False),
    (512, 1024, 512, False),
    (1024, 1024, 256, False),
    (1024, 1024, 512, False),
    (512, 1024, 1024, False),
    (1024, 512, 1024, False),
    (1024, 1024, 512, True),
]
for bm, bn, bk, rhs_t in emit_candidates:
    rhs_arg = b_mat_t if rhs_t else b_mat
    tag = "rhsT" if rhs_t else "std"
    part_a_results.append(
        run_gemm_silu_cfg(
            f"emit_pipeline_{bm}x{bn}x{bk}_{tag}",
            make_emit_pipeline_gemm_silu(
                bm,
                bn,
                bk,
                rhs_transposed=rhs_t,
                vmem_limit_bytes=16 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
            a_mat,
            rhs_arg,
        )
    )

# 4. Explicit pltpu.make_async_copy manual double-buffered K-loop
for bm, bn, bk in [(1024, 512, 512), (1024, 1024, 256), (512, 1024, 512)]:
    part_a_results.append(
        run_gemm_silu_cfg(
            f"async_copy_db_{bm}x{bn}x{bk}",
            make_async_copy_db_gemm_silu(
                bm,
                bn,
                bk,
                vmem_limit_bytes=16 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
            a_mat,
            b_mat,
        )
    )

ok_part_a = [
    r for r in part_a_results
    if r.get("status") == "ok" and r.get("parity_ok") and r.get("finite")
]
if ok_part_a:
    best_a = min(ok_part_a, key=lambda r: r["median_us"])
    print(
        "PALLAS_OPT_PART_A_BEST:",
        json.dumps(
            {
                "best_pallas": best_a,
                "xla_ref": xla_summary,
                "gap_ratio_xla_over_pallas": round(
                    xla_summary["median_us"] / best_a["median_us"], 3
                ),
            }
        ),
    )
else:
    print("PALLAS_OPT_PART_A_FAIL:", json.dumps({"reason": "no finite parity-passing Pallas GEMM+SiLU config"}))

# Free Part A tensors before Part B FlashAttention
del a_mat, b_mat, b_mat_t, xla_ref
gc.collect()
jax.clear_caches()


# ---------------------------------------------------------------------------
# Part B: Triton/CUDA-like FlashAttention Wrapper vs Dense XLA Causal Attn
# ---------------------------------------------------------------------------
print("\n=== PART B: @tpu_flash_attn (PALLAS FLASH ATTENTION) VS XLA ===")


def tpu_flash_attn(causal=True, sm_scale=None, block_sizes=None):
    """Triton/CUDA-style decorator wrapping Pallas TPU FlashAttention."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(q, k, v):
            scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(q.shape[-1]))
            return fn(q, k, v, causal=causal, sm_scale=scale, block_sizes=block_sizes)

        return jax.jit(wrapped)

    return decorator


def tpu_flash_attention(q, k, v, *, causal=True, sm_scale=None, block_sizes=None):
    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(q.shape[-1]))
    return flash_attention.flash_attention(
        q, k, v, causal=causal, sm_scale=scale, block_sizes=block_sizes
    )


@jax.jit
def xla_causal_attention(q, k, v, sm_scale):
    logits = (
        jnp.einsum(
            "bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)
        )
        * sm_scale
    )
    q_len, k_len = q.shape[2], k.shape[2]
    q_idx = jnp.arange(q_len)[:, None]
    k_idx = jnp.arange(k_len)[None, :]
    causal_mask = q_idx >= k_idx
    logits = jnp.where(causal_mask[None, None, :, :], logits, -1e30)
    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v.astype(jnp.float32))
    return out.astype(jnp.bfloat16)


attn_cases = [
    (2, 32, 2048, 2048, 128),
    (1, 32, 4096, 4096, 128),
]

block_size_candidates = [
    ("default", None),
    ("bs_512_512_128", (512, 512, 128, 1)),
    ("bs_512_512_256", (512, 512, 256, 1)),
    ("bs_256_512_256", (256, 512, 256, 1)),
    ("bs_512_1024_256", (512, 1024, 256, 1)),
    ("bs_128_256_128", (128, 256, 128, 1)),
]

part_b_case_valid = []
for B, H, Q, K, D in attn_cases:
    print(f"\n--- FlashAttention Case: B={B}, H={H}, Q={Q}, K={K}, D={D}, causal=True ---")
    sm_scale = 1.0 / math.sqrt(D)
    # Causal attention FLOPs: 0.5 * (2*B*H*Q*K*D [QK^T] + 2*B*H*Q*K*D [PV]) = 2*B*H*Q*K*D
    causal_flops = 2.0 * B * H * Q * K * D

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(100 + Q), 3)
    q_arr = jax.random.normal(k1, (B, H, Q, D), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, H, K, D), dtype=jnp.bfloat16)
    v_arr = jax.random.normal(k3, (B, H, K, D), dtype=jnp.bfloat16)
    jax.block_until_ready((q_arr, k_arr, v_arr))

    xla_attn_fn = jax.jit(lambda qq, kk, vv: xla_causal_attention(qq, kk, vv, sm_scale))
    xla_out = None
    xa_med_s = None
    try:
        cand_xla_out = xla_attn_fn(q_arr, k_arr, v_arr)
        cand_xla_out.block_until_ready()
        if not bool(jnp.all(jnp.isfinite(cand_xla_out)).item()):
            raise RuntimeError("dense XLA causal attention produced nonfinite values")
        xla_out = cand_xla_out
        xa_med_s, xa_min_s = bench_fn(xla_attn_fn, q_arr, k_arr, v_arr, warmup=3, iters=15)
        xa_rec = {
            "shape": f"B={B},H={H},Q={Q},K={K},D={D}",
            "causal": True,
            "variant": "xla_dense_causal",
            "median_us": round(xa_med_s * 1e6, 2),
            "min_us": round(xa_min_s * 1e6, 2),
            "effective_tflops_median": round(causal_flops / (xa_med_s * 1e12), 2),
            "effective_tflops_peak": round(causal_flops / (xa_min_s * 1e12), 2),
        }
        print("XLA_ATTN:", json.dumps(xa_rec))
    except Exception as e:
        print("XLA_ATTN_ERR:", json.dumps({"shape": f"B={B},H={H},Q={Q},K={K},D={D}", "error": repr(e)[:400]}))

    fa_results = []
    for bs_name, bs_tuple in block_size_candidates:
        try:
            bs_obj = (
                None
                if bs_tuple is None
                else flash_attention.BlockSizes(
                    block_q=bs_tuple[0],
                    block_k_major=bs_tuple[1],
                    block_k=bs_tuple[2],
                    block_b=bs_tuple[3],
                )
            )

            @tpu_flash_attn(causal=True, sm_scale=sm_scale, block_sizes=bs_obj)
            def run_fa(qq, kk, vv, *, causal, sm_scale, block_sizes):
                return tpu_flash_attention(
                    qq, kk, vv, causal=causal, sm_scale=sm_scale, block_sizes=block_sizes
                )

            hlo_txt = run_fa.lower(q_arr, k_arr, v_arr).as_text()
            has_tpu_cc = "tpu_custom_call" in hlo_txt

            fa_out = run_fa(q_arr, k_arr, v_arr)
            fa_out.block_until_ready()
            fa_finite = bool(jnp.all(jnp.isfinite(fa_out)).item())
            err = (
                float(
                    jnp.max(
                        jnp.abs(fa_out.astype(jnp.float32) - xla_out.astype(jnp.float32))
                    )
                )
                if xla_out is not None
                else None
            )
            parity_ok = (
                xla_out is not None
                and fa_finite
                and err is not None
                and math.isfinite(err)
                and err <= PART_B_MAX_ABS_ERR_TOL
            )
            fa_med_s, fa_min_s = bench_fn(
                run_fa, q_arr, k_arr, v_arr, warmup=3, iters=20
            )
            status = (
                "ok"
                if parity_ok
                else ("baseline_missing" if xla_out is None else "parity_fail")
            )
            rec = {
                "shape": f"B={B},H={H},Q={Q},K={K},D={D}",
                "causal": True,
                "block_sizes": bs_name,
                "status": status,
                "finite": fa_finite,
                "parity_ok": parity_ok,
                "tpu_custom_call": has_tpu_cc,
                "max_abs_err": err,
                "median_us": round(fa_med_s * 1e6, 2),
                "min_us": round(fa_min_s * 1e6, 2),
                "effective_tflops_median": round(causal_flops / (fa_med_s * 1e12), 2),
                "effective_tflops_peak": round(causal_flops / (fa_min_s * 1e12), 2),
                "speedup_vs_xla": round(xa_med_s / fa_med_s, 2) if xa_med_s else None,
            }
            print("FLASH_ATTN:", json.dumps(rec))
            fa_results.append(rec)
        except Exception as e:
            err_rec = {
                "shape": f"B={B},H={H},Q={Q},K={K},D={D}",
                "block_sizes": bs_name,
                "status": "error",
                "error": repr(e)[:400],
            }
            print("FLASH_ATTN_ERR:", json.dumps(err_rec))

    valid_fa = [
        r for r in fa_results
        if r.get("status") == "ok" and r.get("parity_ok") and r.get("finite")
    ]
    if valid_fa and xa_med_s is not None:
        best_fa = min(valid_fa, key=lambda r: r["median_us"])
        print("FLASH_ATTN_BEST:", json.dumps(best_fa))
        part_b_case_valid.append(True)
    else:
        print(
            "FLASH_ATTN_FAIL:",
            json.dumps({
                "shape": f"B={B},H={H},Q={Q},K={K},D={D}",
                "xla_ok": xa_med_s is not None,
                "valid_candidates": len(valid_fa),
            }),
        )
        part_b_case_valid.append(False)

    del q_arr, k_arr, v_arr, xla_out, xla_attn_fn
    gc.collect()
    jax.clear_caches()

print("HBM_END:", json.dumps(get_hbm_stats()))
if not ok_part_a or not all(part_b_case_valid):
    raise RuntimeError(
        f"Pallas suite incomplete: part_a_valid={len(ok_part_a)}, "
        f"part_b_cases={part_b_case_valid}"
    )
print("=== PALLAS_OPT_COMPLETE ===")
