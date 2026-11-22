/*
 * Fallback offline mock shim for openbuenos.h when libopenbuenos.a is not yet built.
 * Replaced automatically at link time once libopenbuenos/build/{android-arm64,host}/libopenbuenos.a exists.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "openbuenos.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct ob_context {
    int  mock_mode;
    char last_error[128];
};

static uint64_t ob_shim_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static float ob_shim_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)(h & 0x03ffu);
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

ob_config_t ob_config_default(void) {
    ob_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size           = (uint32_t)sizeof(ob_config_t);
    cfg.mock_mode             = 1;
    cfg.max_inflight          = OB_MAX_CREDITS;
    cfg.graph_cache_capacity  = 64u;
    cfg.buffer_pool_max_bytes = 64u * 1024u * 1024u;
    cfg.timeout_ms            = 5000u;
    cfg.core_power_state      = 0x0bu;
    cfg.mem_power_state       = 0x03u;
    cfg.qos_class             = 0x01u;
    cfg.cluster_id            = 0x02u;
    return cfg;
}

ob_status_t ob_context_create(const ob_config_t * config, ob_context_t ** out_ctx) {
    if (!out_ctx) {
        return OB_ERR_INVALID_ARG;
    }
    ob_context_t * ctx = (ob_context_t *)calloc(1, sizeof(ob_context_t));
    if (!ctx) {
        return OB_ERR_NO_MEMORY;
    }
    ctx->mock_mode = config ? config->mock_mode : 1;
    if (!ctx->mock_mode) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "shim build has no /dev/edgetpu-soc driver");
        free(ctx);
        return OB_ERR_DEVICE;
    }
    *out_ctx = ctx;
    return OB_OK;
}

void ob_context_destroy(ob_context_t * ctx) {
    free(ctx);
}

const char * ob_status_string(ob_status_t status) {
    switch (status) {
        case OB_OK:              return "OB_OK";
        case OB_ERR_INVALID_ARG: return "OB_ERR_INVALID_ARG";
        case OB_ERR_NO_MEMORY:   return "OB_ERR_NO_MEMORY";
        case OB_ERR_DEVICE:      return "OB_ERR_DEVICE";
        case OB_ERR_IOCTL:       return "OB_ERR_IOCTL";
        case OB_ERR_TIMEOUT:     return "OB_ERR_TIMEOUT";
        case OB_ERR_BAD_DGC0:    return "OB_ERR_BAD_DGC0";
        case OB_ERR_BUSY:        return "OB_ERR_BUSY";
        case OB_ERR_NOT_FOUND:   return "OB_ERR_NOT_FOUND";
        case OB_ERR_UNSUPPORTED: return "OB_ERR_UNSUPPORTED";
        case OB_ERR_FW_ERROR:    return "OB_ERR_FW_ERROR";
        case OB_ERR_OVERFLOW:    return "OB_ERR_OVERFLOW";
        case OB_ERR_IO:          return "OB_ERR_IO";
        default:                 return "OB_ERR_UNKNOWN";
    }
}

const char * ob_context_last_error(const ob_context_t * ctx) {
    return ctx ? ctx->last_error : "";
}

uint32_t ob_context_credit_size(const ob_context_t * ctx) {
    (void)ctx;
    return OB_MAX_CREDITS;
}

int ob_context_is_mock(const ob_context_t * ctx) {
    return ctx ? ctx->mock_mode : 1;
}

ob_status_t ob_context_set_perf_mode(ob_context_t * ctx, uint32_t perf_mode) {
    (void)perf_mode;
    return ctx ? OB_OK : OB_ERR_INVALID_ARG;
}

ob_status_t ob_context_buffer_pool_stats(const ob_context_t * ctx,
                                         size_t *             out_pooled_bytes,
                                         size_t *             out_pooled_count,
                                         uint64_t *           out_pool_hits,
                                         uint64_t *           out_pool_misses) {
    if (!ctx) return OB_ERR_INVALID_ARG;
    if (out_pooled_bytes) *out_pooled_bytes = 0u;
    if (out_pooled_count) *out_pooled_count = 0u;
    if (out_pool_hits)    *out_pool_hits    = 0u;
    if (out_pool_misses)  *out_pool_misses  = 0u;
    return OB_OK;
}

ob_status_t ob_context_graph_cache_stats(const ob_context_t * ctx,
                                         size_t *             out_entries,
                                         uint64_t *           out_hits,
                                         uint64_t *           out_misses,
                                         uint64_t *           out_evictions) {
    if (!ctx) return OB_ERR_INVALID_ARG;
    if (out_entries)   *out_entries   = 0u;
    if (out_hits)      *out_hits      = 0u;
    if (out_misses)    *out_misses    = 0u;
    if (out_evictions) *out_evictions = 0u;
    return OB_OK;
}

ob_status_t ob_gemm_register_builtin_emitter(ob_context_t * ctx,
                                             const char *   dive_interp_path) {
    (void)dive_interp_path;
    if (!ctx) return OB_ERR_INVALID_ARG;
    snprintf(ctx->last_error, sizeof(ctx->last_error), "shim build has no built-in TPU GEMM emitter");
    return OB_ERR_UNSUPPORTED;
}

ob_status_t ob_gemm_set_tile_limits(ob_context_t * ctx,
                                    uint32_t       max_m,
                                    uint32_t       max_k,
                                    uint32_t       max_n) {
    if (!ctx || max_m == 0u || max_k == 0u || max_n == 0u) {
        return OB_ERR_INVALID_ARG;
    }
    return OB_OK;
}

size_t ob_gemm_packed_weight_bytes(uint32_t k_dim, uint32_t n_dim) {
    const size_t n_pad = ((size_t)n_dim + 7u) & ~((size_t)7u);
    const size_t k_pad = ((size_t)k_dim + 3u) & ~((size_t)3u);
    return n_pad * sizeof(int32_t) + (k_pad / 4u) * n_pad * 4u;
}

ob_status_t ob_gemm_pack_weights_int8(const int8_t *  weights_nk,
                                      const int32_t * bias_n,
                                      uint32_t        k_dim,
                                      uint32_t        n_dim,
                                      uint8_t *       out_packed,
                                      size_t          out_cap) {
    if (!weights_nk || !out_packed || k_dim == 0 || n_dim == 0) {
        return OB_ERR_INVALID_ARG;
    }
    const size_t required = ob_gemm_packed_weight_bytes(k_dim, n_dim);
    if (out_cap < required) {
        return OB_ERR_OVERFLOW;
    }
    const size_t n_pad = ((size_t)n_dim + 7u) & ~((size_t)7u);
    const size_t k_pad = ((size_t)k_dim + 3u) & ~((size_t)3u);

    int32_t * bias_out = (int32_t *)out_packed;
    for (size_t ni = 0; ni < n_pad; ++ni) {
        bias_out[ni] = (bias_n && ni < n_dim) ? bias_n[ni] : 0;
    }
    uint8_t * w_out = out_packed + n_pad * sizeof(int32_t);
    size_t pos = 0;
    for (size_t k4 = 0; k4 < k_pad / 4u; ++k4) {
        for (size_t ni = 0; ni < n_pad; ++ni) {
            for (size_t lane = 0; lane < 4u; ++lane) {
                if (ni >= n_dim) {
                    w_out[pos++] = 0u;
                } else {
                    const size_t ki = k4 * 4u + lane;
                    if (ki < k_dim) {
                        const int32_t w = (int32_t)weights_nk[ni * (size_t)k_dim + ki];
                        w_out[pos++] = (uint8_t)((w + 128) & 0xff);
                    } else {
                        w_out[pos++] = 128u;
                    }
                }
            }
        }
    }
    return OB_OK;
}

ob_status_t ob_gemm_matmul_int8(ob_context_t *      ctx,
                                const int8_t *      a_mk,
                                const int8_t *      b_nk,
                                const int32_t *     bias_n,
                                int32_t *           c_mn,
                                uint32_t            m_dim,
                                uint32_t            k_dim,
                                uint32_t            n_dim,
                                int32_t             zp_a,
                                int32_t             zp_b,
                                ob_invoke_stats_t * out_stats) {
    if (!ctx || !a_mk || !b_nk || !c_mn || m_dim == 0 || k_dim == 0 || n_dim == 0) {
        return OB_ERR_INVALID_ARG;
    }
    const uint64_t t0 = ob_shim_now_us();
    for (uint32_t m = 0; m < m_dim; ++m) {
        const int8_t * a_row = a_mk + (size_t)m * k_dim;
        int32_t *      c_row = c_mn + (size_t)m * n_dim;
        for (uint32_t n = 0; n < n_dim; ++n) {
            const int8_t * b_row = b_nk + (size_t)n * k_dim;
            int32_t acc = bias_n ? bias_n[n] : 0;
            for (uint32_t k = 0; k < k_dim; ++k) {
                acc += ((int32_t)a_row[k] - zp_a) * ((int32_t)b_row[k] - zp_b);
            }
            c_row[n] = acc;
        }
    }
    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->wall_time_us     = ob_shim_now_us() - t0;
        out_stats->flags            = (uint16_t)(OB_STAT_FLAG_MOCK | OB_STAT_FLAG_CPU_FALLBACK);
        out_stats->tiles_dispatched = 1u;
    }
    return OB_OK;
}

ob_status_t ob_gemm_matmul_f32_q8_0(ob_context_t *      ctx,
                                    const float *       a_mk,
                                    const void *        b_nk_q8_0,
                                    float *             c_mn,
                                    uint32_t            m_dim,
                                    uint32_t            k_dim,
                                    uint32_t            n_dim,
                                    ob_invoke_stats_t * out_stats) {
    if (!ctx || !a_mk || !b_nk_q8_0 || !c_mn || m_dim == 0 || k_dim == 0 || n_dim == 0 || (k_dim % OB_QK8_0) != 0) {
        return OB_ERR_INVALID_ARG;
    }
    const uint64_t t0 = ob_shim_now_us();
    const uint32_t n_blocks = k_dim / OB_QK8_0;
    const ob_block_q8_0_t * blocks = (const ob_block_q8_0_t *)b_nk_q8_0;

    for (uint32_t m = 0; m < m_dim; ++m) {
        const float * a_row = a_mk + (size_t)m * k_dim;
        float *       c_row = c_mn + (size_t)m * n_dim;
        for (uint32_t n = 0; n < n_dim; ++n) {
            const ob_block_q8_0_t * b_row = blocks + (size_t)n * n_blocks;
            float sum = 0.0f;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                const float d = ob_shim_fp16_to_fp32(b_row[b].d);
                const int8_t * qs = b_row[b].qs;
                const float * a_blk = a_row + (size_t)b * OB_QK8_0;
                float bsum = 0.0f;
                for (uint32_t j = 0; j < OB_QK8_0; ++j) {
                    bsum += a_blk[j] * (float)qs[j];
                }
                sum += d * bsum;
            }
            c_row[n] = sum;
        }
    }
    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->wall_time_us     = ob_shim_now_us() - t0;
        out_stats->flags            = (uint16_t)(OB_STAT_FLAG_MOCK | OB_STAT_FLAG_CPU_FALLBACK);
        out_stats->tiles_dispatched = 1u;
    }
    return OB_OK;
}

ob_status_t ob_gemm_matmul_f32_quant_batch(ob_context_t *                  ctx,
                                           const float *                   a_mk,
                                           uint32_t                        m_dim,
                                           uint32_t                        k_dim,
                                           const ob_gemm_f32_quant_job_t * jobs,
                                           uint32_t                        num_jobs,
                                           ob_invoke_stats_t *             out_stats) {
    if (!ctx || !a_mk || !jobs || num_jobs == 0u || m_dim == 0u || k_dim == 0u || (k_dim % OB_QK8_0) != 0u) {
        return OB_ERR_INVALID_ARG;
    }
    const uint64_t t0 = ob_shim_now_us();
    const uint32_t n_blocks = k_dim / OB_QK8_0;
    for (uint32_t j = 0; j < num_jobs; ++j) {
        const ob_gemm_f32_quant_job_t * job = &jobs[j];
        if (!job->b_nk || !job->c_mn || job->n_dim == 0u) {
            return OB_ERR_INVALID_ARG;
        }
        if (job->is_q4_0 == 0u) {
            ob_status_t st = ob_gemm_matmul_f32_q8_0(ctx, a_mk, job->b_nk, job->c_mn, m_dim, k_dim, job->n_dim, NULL);
            if (st != OB_OK) return st;
        } else {
            const ob_block_q4_0_t * blocks = (const ob_block_q4_0_t *)job->b_nk;
            for (uint32_t m = 0; m < m_dim; ++m) {
                const float * a_row = a_mk + (size_t)m * k_dim;
                float *       c_row = job->c_mn + (size_t)m * job->n_dim;
                for (uint32_t n = 0; n < job->n_dim; ++n) {
                    const ob_block_q4_0_t * b_row = blocks + (size_t)n * n_blocks;
                    float sum = 0.0f;
                    for (uint32_t b = 0; b < n_blocks; ++b) {
                        const float d = ob_shim_fp16_to_fp32(b_row[b].d);
                        const float * a_blk = a_row + (size_t)b * OB_QK8_0;
                        float bsum = 0.0f;
                        for (uint32_t i = 0; i < OB_QK8_0 / 2u; ++i) {
                            const uint8_t byte = b_row[b].qs[i];
                            bsum += a_blk[i] * (float)((int)(byte & 0x0Fu) - 8);
                            bsum += a_blk[i + OB_QK8_0 / 2u] * (float)((int)(byte >> 4) - 8);
                        }
                        sum += d * bsum;
                    }
                    c_row[n] = sum;
                }
            }
        }
    }
    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->wall_time_us     = ob_shim_now_us() - t0;
        out_stats->flags            = (uint16_t)(OB_STAT_FLAG_MOCK | OB_STAT_FLAG_CPU_FALLBACK);
        out_stats->tiles_dispatched = num_jobs;
    }
    return OB_OK;
}
