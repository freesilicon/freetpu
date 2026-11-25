/*
 * Fallback offline mock shim for openbuenos.h when libopenbuenos.a is not yet built.
 * Replaced automatically at link time once libopenbuenos/build/{android-arm64,host}/libopenbuenos.a exists.
 */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "openbuenos.h"

#include <math.h>
#include <stdint.h>
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

ob_status_t ob_context_set_profile(ob_context_t * ctx, int enable) {
    (void)enable;
    return ctx ? OB_OK : OB_ERR_INVALID_ARG;
}

ob_status_t ob_context_get_profile(const ob_context_t * ctx, ob_profile_stats_t * out_profile) {
    if (!ctx || !out_profile) return OB_ERR_INVALID_ARG;
    memset(out_profile, 0, sizeof(*out_profile));
    return OB_OK;
}

ob_status_t ob_context_reset_profile(ob_context_t * ctx) {
    return ctx ? OB_OK : OB_ERR_INVALID_ARG;
}

int ob_context_get_freetile(const ob_context_t * ctx) {
    (void)ctx;
    return 0;
}

ob_status_t ob_context_set_freetile(ob_context_t * ctx, int enable) {
    if (!ctx) return OB_ERR_INVALID_ARG;
    if (enable != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "shim build has no FreeTile BUENOS_A0 emitter");
        return OB_ERR_UNSUPPORTED;
    }
    return OB_OK;
}

ob_status_t ob_context_freetile_stats(const ob_context_t * ctx,
                                      uint64_t *           out_emitted,
                                      uint64_t *           out_fallbacks) {
    if (!ctx) return OB_ERR_INVALID_ARG;
    if (out_emitted)   *out_emitted   = 0u;
    if (out_fallbacks) *out_fallbacks = 0u;
    return OB_OK;
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
        out_stats->tiles_dispatched = 0u;
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
        out_stats->tiles_dispatched = 0u;
    }
    return OB_OK;
}

static const int8_t ob_shim_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static uint16_t ob_shim_load_u16(const void * p) {
    uint16_t v = 0u;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void ob_shim_get_scale_min_k4(uint32_t j, const uint8_t scales[OB_K_SCALE_SIZE],
                                     uint8_t * out_sc, uint8_t * out_m) {
    if (j < 4u) {
        *out_sc = (uint8_t)(scales[j] & 63u);
        *out_m  = (uint8_t)(scales[j + 4u] & 63u);
    } else {
        *out_sc = (uint8_t)((scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4));
        *out_m  = (uint8_t)((scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4));
    }
}

uint32_t ob_quant_block_elements(ob_quant_type_t type) {
    switch (type) {
        case OB_QUANT_Q8_0:   return OB_QK8_0;
        case OB_QUANT_Q4_0:   return OB_QK8_0;
        case OB_QUANT_Q4_K:   return OB_QK_K;
        case OB_QUANT_Q6_K:   return OB_QK_K;
        case OB_QUANT_IQ4_NL: return OB_QK4_NL;
        case OB_QUANT_IQ4_XS: return OB_QK_K;
        default:              return 0u;
    }
}

size_t ob_quant_block_bytes(ob_quant_type_t type) {
    switch (type) {
        case OB_QUANT_Q8_0:   return sizeof(ob_block_q8_0_t);
        case OB_QUANT_Q4_0:   return sizeof(ob_block_q4_0_t);
        case OB_QUANT_Q4_K:   return sizeof(ob_block_q4_k_t);
        case OB_QUANT_Q6_K:   return sizeof(ob_block_q6_k_t);
        case OB_QUANT_IQ4_NL: return sizeof(ob_block_iq4_nl_t);
        case OB_QUANT_IQ4_XS: return sizeof(ob_block_iq4_xs_t);
        default:              return 0u;
    }
}

const char * ob_quant_type_name(ob_quant_type_t type) {
    switch (type) {
        case OB_QUANT_Q8_0:   return "Q8_0";
        case OB_QUANT_Q4_0:   return "Q4_0";
        case OB_QUANT_Q4_K:   return "Q4_K";
        case OB_QUANT_Q6_K:   return "Q6_K";
        case OB_QUANT_IQ4_NL: return "IQ4_NL";
        case OB_QUANT_IQ4_XS: return "IQ4_XS";
        default:              return "UNKNOWN";
    }
}

ob_status_t ob_quant_dequantize_row_f32(ob_quant_type_t type,
                                        const void *    src_blocks,
                                        uint32_t        k_dim,
                                        float *         dst_f32) {
    const uint32_t qk = ob_quant_block_elements(type);
    const size_t blk_sz = ob_quant_block_bytes(type);
    if (qk == 0u || blk_sz == 0u || !src_blocks || !dst_f32 || k_dim == 0u || (k_dim % qk) != 0u) {
        return OB_ERR_INVALID_ARG;
    }
    const uint32_t nb = k_dim / qk;
    const uint8_t * base = (const uint8_t *)src_blocks;
    for (uint32_t bi = 0u; bi < nb; ++bi) {
        const uint8_t * bp = base + (size_t)bi * blk_sz;
        float * out = dst_f32 + (size_t)bi * qk;
        switch (type) {
            case OB_QUANT_Q8_0: {
                const ob_block_q8_0_t * blk = (const ob_block_q8_0_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                for (uint32_t j = 0u; j < OB_QK8_0; ++j) {
                    out[j] = d * (float)blk->qs[j];
                }
                break;
            }
            case OB_QUANT_Q4_0: {
                const ob_block_q4_0_t * blk = (const ob_block_q4_0_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                for (uint32_t j = 0u; j < OB_QK8_0 / 2u; ++j) {
                    const uint8_t byte = blk->qs[j];
                    out[j] = d * (float)((int)(byte & 0x0Fu) - 8);
                    out[j + OB_QK8_0 / 2u] = d * (float)((int)(byte >> 4) - 8);
                }
                break;
            }
            case OB_QUANT_Q4_K: {
                const ob_block_q4_k_t * blk = (const ob_block_q4_k_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                const float dmin = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->dmin));
                const uint8_t * q = blk->qs;
                uint32_t is = 0u;
                for (uint32_t j = 0u; j < OB_QK_K; j += 64u) {
                    uint8_t sc1 = 0u, m1_u = 0u, sc2 = 0u, m2_u = 0u;
                    ob_shim_get_scale_min_k4(is + 0u, blk->scales, &sc1, &m1_u);
                    ob_shim_get_scale_min_k4(is + 1u, blk->scales, &sc2, &m2_u);
                    const float d1 = d * (float)sc1;
                    const float m1 = dmin * (float)m1_u;
                    const float d2 = d * (float)sc2;
                    const float m2 = dmin * (float)m2_u;
                    for (uint32_t l = 0u; l < 32u; ++l) {
                        out[j + l] = d1 * (float)(q[l] & 0x0Fu) - m1;
                    }
                    for (uint32_t l = 0u; l < 32u; ++l) {
                        out[j + 32u + l] = d2 * (float)(q[l] >> 4) - m2;
                    }
                    q += 32u;
                    is += 2u;
                }
                break;
            }
            case OB_QUANT_Q6_K: {
                const ob_block_q6_k_t * blk = (const ob_block_q6_k_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                const uint8_t * ql = blk->ql;
                const uint8_t * qh = blk->qh;
                const int8_t * sc = blk->scales;
                float * y = out;
                for (uint32_t n = 0u; n < OB_QK_K; n += 128u) {
                    for (uint32_t l = 0u; l < 32u; ++l) {
                        const uint32_t is = l / 16u;
                        const int q1 = (int)((ql[l + 0u] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                        const int q3 = (int)((ql[l + 0u] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                        y[l + 0u]  = (d * (float)sc[is + 0u]) * (float)q1;
                        y[l + 32u] = (d * (float)sc[is + 2u]) * (float)q2;
                        y[l + 64u] = (d * (float)sc[is + 4u]) * (float)q3;
                        y[l + 96u] = (d * (float)sc[is + 6u]) * (float)q4;
                    }
                    y += 128u;
                    ql += 64u;
                    qh += 32u;
                    sc += 8u;
                }
                break;
            }
            case OB_QUANT_IQ4_NL: {
                const ob_block_iq4_nl_t * blk = (const ob_block_iq4_nl_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                for (uint32_t j = 0u; j < OB_QK4_NL / 2u; ++j) {
                    const uint8_t byte = blk->qs[j];
                    out[j] = d * (float)ob_shim_kvalues_iq4nl[byte & 0x0Fu];
                    out[j + OB_QK4_NL / 2u] = d * (float)ob_shim_kvalues_iq4nl[byte >> 4];
                }
                break;
            }
            case OB_QUANT_IQ4_XS: {
                const ob_block_iq4_xs_t * blk = (const ob_block_iq4_xs_t *)bp;
                const float d = ob_shim_fp16_to_fp32(ob_shim_load_u16(&blk->d));
                const uint16_t scales_h = ob_shim_load_u16(&blk->scales_h);
                const uint8_t * qs = blk->qs;
                float * y = out;
                for (uint32_t ib = 0u; ib < OB_QK_K / 32u; ++ib) {
                    const int ls = (int)(((blk->scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0Fu) |
                                         (((scales_h >> (2u * ib)) & 3u) << 4));
                    const float dl = d * (float)(ls - 32);
                    for (uint32_t j = 0u; j < 16u; ++j) {
                        const uint8_t byte = qs[j];
                        y[j] = dl * (float)ob_shim_kvalues_iq4nl[byte & 0x0Fu];
                        y[j + 16u] = dl * (float)ob_shim_kvalues_iq4nl[byte >> 4];
                    }
                    y += 32u;
                    qs += 16u;
                }
                break;
            }
            default:
                return OB_ERR_INVALID_ARG;
        }
    }
    return OB_OK;
}

static void ob_shim_apply_activation(float * data, size_t count, uint32_t act) {
    if (!data || act == OB_ACT_NONE) return;
    for (size_t i = 0u; i < count; ++i) {
        const float x = data[i];
        if (act == OB_ACT_RELU) {
            data[i] = (x < 0.0f) ? 0.0f : x;
        } else if (act == OB_ACT_RELU6) {
            data[i] = (x < 0.0f) ? 0.0f : ((x > 6.0f) ? 6.0f : x);
        } else if (act == OB_ACT_SILU) {
            data[i] = x / (1.0f + expf(-x));
        } else if (act == OB_ACT_GELU) {
            const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            data[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
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
    for (uint32_t j = 0u; j < num_jobs; ++j) {
        const ob_gemm_f32_quant_job_t * job = &jobs[j];
        if (!job->b_nk || !job->c_mn || job->n_dim == 0u || job->is_q4_0 > (uint8_t)OB_QUANT_IQ4_XS) {
            return OB_ERR_INVALID_ARG;
        }
        const uint32_t qk = ob_quant_block_elements((ob_quant_type_t)job->is_q4_0);
        if (qk == 0u || (k_dim % qk) != 0u) {
            return OB_ERR_INVALID_ARG;
        }
    }

    const uint64_t t0 = ob_shim_now_us();
    float * w_row = (float *)malloc((size_t)k_dim * sizeof(float));
    if (!w_row) {
        return OB_ERR_NO_MEMORY;
    }

    for (uint32_t j = 0u; j < num_jobs; ++j) {
        const ob_gemm_f32_quant_job_t * job = &jobs[j];
        const ob_quant_type_t qtype = (ob_quant_type_t)job->is_q4_0;
        const uint32_t qk = ob_quant_block_elements(qtype);
        const size_t row_bytes = (size_t)(k_dim / qk) * ob_quant_block_bytes(qtype);
        const uint8_t * b_bytes = (const uint8_t *)job->b_nk;

        for (uint32_t n = 0u; n < job->n_dim; ++n) {
            ob_status_t st = ob_quant_dequantize_row_f32(
                qtype, b_bytes + (size_t)n * row_bytes, k_dim, w_row);
            if (st != OB_OK) {
                free(w_row);
                return st;
            }
            for (uint32_t m = 0u; m < m_dim; ++m) {
                const float * a_row = a_mk + (size_t)m * k_dim;
                double acc = 0.0;
                for (uint32_t k = 0u; k < k_dim; ++k) {
                    acc += (double)a_row[k] * (double)w_row[k];
                }
                job->c_mn[(size_t)m * job->n_dim + n] = (float)acc;
            }
        }
        if (job->activation != OB_ACT_NONE) {
            ob_shim_apply_activation(job->c_mn, (size_t)m_dim * (size_t)job->n_dim, job->activation);
        }
    }
    free(w_row);

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->wall_time_us     = ob_shim_now_us() - t0;
        out_stats->flags            = (uint16_t)(OB_STAT_FLAG_MOCK | OB_STAT_FLAG_CPU_FALLBACK);
        out_stats->tiles_dispatched = 0u;
    }
    return OB_OK;
}

ob_status_t ob_gemm_register_immutable_weights(ob_context_t * ctx,
                                               const void *   base,
                                               size_t         size,
                                               uint64_t       version) {
    if (!ctx || !base || size == 0u) {
        return OB_ERR_INVALID_ARG;
    }
    (void)version;
    return OB_OK;
}

ob_status_t ob_gemm_unregister_immutable_weights(ob_context_t * ctx,
                                                 const void *   base,
                                                 size_t         size) {
    if (!ctx || (!base && size != 0u)) {
        return OB_ERR_INVALID_ARG;
    }
    return OB_OK;
}

ob_status_t ob_gemm_swiglu_f32_quant(ob_context_t *                      ctx,
                                     const float *                       a_mk,
                                     const ob_gemm_swiglu_quant_desc_t * desc,
                                     float *                             c_mn,
                                     ob_invoke_stats_t *                 out_stats) {
    if (!ctx || !a_mk || !desc || !c_mn || !desc->w_gate_hk || !desc->w_up_hk || !desc->w_down_nh ||
        desc->m_dim == 0u || desc->k_dim == 0u || desc->h_dim == 0u || desc->n_dim == 0u ||
        desc->gate_is_q4_0 > (uint8_t)OB_QUANT_IQ4_XS ||
        desc->up_is_q4_0   > (uint8_t)OB_QUANT_IQ4_XS ||
        desc->down_is_q4_0 > (uint8_t)OB_QUANT_IQ4_XS) {
        return OB_ERR_INVALID_ARG;
    }
    if ((desc->k_dim % ob_quant_block_elements((ob_quant_type_t)desc->gate_is_q4_0)) != 0u ||
        (desc->k_dim % ob_quant_block_elements((ob_quant_type_t)desc->up_is_q4_0))   != 0u ||
        (desc->h_dim % ob_quant_block_elements((ob_quant_type_t)desc->down_is_q4_0)) != 0u) {
        return OB_ERR_INVALID_ARG;
    }
    const uint64_t t0 = ob_shim_now_us();
    const size_t m_sz = (size_t)desc->m_dim;
    const size_t h_sz = (size_t)desc->h_dim;
    if (m_sz > (SIZE_MAX / sizeof(float)) / h_sz) {
        return OB_ERR_INVALID_ARG;
    }
    const size_t mh = m_sz * h_sz;
    const size_t scratch_bytes = mh * sizeof(float);
    float * gate_buf = (float *)malloc(scratch_bytes);
    float * up_buf   = (float *)malloc(scratch_bytes);
    if (!gate_buf || !up_buf) {
        free(gate_buf);
        free(up_buf);
        return OB_ERR_NO_MEMORY;
    }
    ob_gemm_f32_quant_job_t gu_jobs[2];
    memset(gu_jobs, 0, sizeof(gu_jobs));
    gu_jobs[0].b_nk    = desc->w_gate_hk;
    gu_jobs[0].c_mn    = gate_buf;
    gu_jobs[0].n_dim   = desc->h_dim;
    gu_jobs[0].is_q4_0 = desc->gate_is_q4_0;
    gu_jobs[1].b_nk    = desc->w_up_hk;
    gu_jobs[1].c_mn    = up_buf;
    gu_jobs[1].n_dim   = desc->h_dim;
    gu_jobs[1].is_q4_0 = desc->up_is_q4_0;

    ob_status_t st = ob_gemm_matmul_f32_quant_batch(ctx, a_mk, desc->m_dim, desc->k_dim, gu_jobs, 2u, NULL);
    if (st == OB_OK) {
        for (size_t i = 0; i < mh; ++i) {
            const float g = gate_buf[i];
            gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
        }
        ob_gemm_f32_quant_job_t down_job;
        memset(&down_job, 0, sizeof(down_job));
        down_job.b_nk    = desc->w_down_nh;
        down_job.c_mn    = c_mn;
        down_job.n_dim   = desc->n_dim;
        down_job.is_q4_0 = desc->down_is_q4_0;
        st = ob_gemm_matmul_f32_quant_batch(ctx, gate_buf, desc->m_dim, desc->h_dim, &down_job, 1u, NULL);
    }
    free(gate_buf);
    free(up_buf);
    if (st != OB_OK) {
        return st;
    }
    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->wall_time_us     = ob_shim_now_us() - t0;
        out_stats->flags            = (uint16_t)(OB_STAT_FLAG_MOCK | OB_STAT_FLAG_CPU_FALLBACK);
        out_stats->tiles_dispatched = 0u;
    }
    return OB_OK;
}
