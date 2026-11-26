/* SPDX-License-Identifier: Apache-2.0 */
/*
 * FreeTile Digital Twin (freetile-twin) — C11 / C++17 Public API
 *
 * Deterministic offline schedule scorer, SRAM budget verifier, .dgc0 wire
 * integrity checker, 0x640 minimax spline validator, and libopenbuenos
 * mock_mode=1 numerical parity oracle for Pixel 10 (buenos-a0) and
 * Pixel 11 (malibu).
 *
 * Emits verdicts conforming to freesilicon.twin.v1 (REWARD-SCHEMA.json).
 */
#ifndef FREETILE_TWIN_H_
#define FREETILE_TWIN_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FT_TWIN_SCHEMA_VERSION  "freesilicon.twin.v1"
#define FT_TWIN_MAX_DIAGNOSTICS 16u

typedef enum ft_twin_status {
  FT_TWIN_OK              = 0,
  FT_TWIN_ERR_INVALID_ARG = -1,
  FT_TWIN_ERR_IO          = -2,
  FT_TWIN_ERR_OVERFLOW    = -3,
  FT_TWIN_ERR_INTERNAL    = -4
} ft_twin_status_t;

typedef enum ft_twin_target {
  FT_TWIN_TARGET_BUENOS_A0 = 0,
  FT_TWIN_TARGET_MALIBU    = 1
} ft_twin_target_t;

typedef enum ft_twin_workload {
  FT_TWIN_WORKLOAD_GEMM         = 0,
  FT_TWIN_WORKLOAD_SWIGLU       = 1,
  FT_TWIN_WORKLOAD_FLASH_ATTN   = 2,
  FT_TWIN_WORKLOAD_QUANT_W8A16  = 3,
  FT_TWIN_WORKLOAD_QUANT_Q4_K   = 4,
  FT_TWIN_WORKLOAD_QUANT_Q6_K   = 5,
  FT_TWIN_WORKLOAD_QUANT_IQ4_NL = 6,
  FT_TWIN_WORKLOAD_QUANT_IQ4_XS = 7,
  FT_TWIN_WORKLOAD_QUANT_Q8_0   = 8,
  FT_TWIN_WORKLOAD_QUANT_Q4_0   = 9,
  FT_TWIN_WORKLOAD_SPLINE_0X640 = 10,
  FT_TWIN_WORKLOAD_DGC0         = 11
} ft_twin_workload_t;

typedef enum ft_twin_dtype {
  FT_TWIN_DTYPE_BF16 = 0,
  FT_TWIN_DTYPE_FP32 = 1,
  FT_TWIN_DTYPE_FP16 = 2,
  FT_TWIN_DTYPE_INT8 = 3
} ft_twin_dtype_t;

typedef enum ft_twin_activation {
  FT_TWIN_ACT_NONE  = 0,
  FT_TWIN_ACT_RELU  = 1,
  FT_TWIN_ACT_RELU6 = 3,
  FT_TWIN_ACT_SILU  = 4,
  FT_TWIN_ACT_GELU  = 5
} ft_twin_activation_t;

typedef enum ft_twin_spline_kind {
  FT_TWIN_SPLINE_EXP_NEG12_96 = 0,
  FT_TWIN_SPLINE_EXP_NEG11    = 1,
  FT_TWIN_SPLINE_RSQRT        = 2,
  FT_TWIN_SPLINE_RECIP        = 3,
  FT_TWIN_SPLINE_SILU         = 4,
  FT_TWIN_SPLINE_GELU_TANH    = 5,
  FT_TWIN_SPLINE_GELU_ERF     = 6
} ft_twin_spline_kind_t;

typedef enum ft_twin_fa_phase {
  FT_TWIN_FA_DECODE  = 0,
  FT_TWIN_FA_PREFILL = 1
} ft_twin_fa_phase_t;

typedef enum ft_twin_fa_buf_policy {
  FT_TWIN_FA_PING_PONG  = 0,
  FT_TWIN_FA_KV_DB_FP32 = 1,
  FT_TWIN_FA_SINGLE_BUF = 2
} ft_twin_fa_buf_policy_t;

typedef enum ft_twin_kv_format {
  FT_TWIN_KV_Q8_0 = 0,
  FT_TWIN_KV_F16  = 1,
  FT_TWIN_KV_Q4_0 = 2
} ft_twin_kv_format_t;

typedef enum ft_twin_diag_severity {
  FT_TWIN_DIAG_INFO    = 0,
  FT_TWIN_DIAG_WARNING = 1,
  FT_TWIN_DIAG_ERROR   = 2
} ft_twin_diag_severity_t;

typedef struct ft_twin_diagnostic {
  char                    code[48];
  ft_twin_diag_severity_t severity;
  int                     has_offset;
  int64_t                 offset;
  char                    field[48];
  char                    message[192];
} ft_twin_diagnostic_t;

typedef struct ft_twin_request {
  uint32_t                struct_size;
  ft_twin_target_t        target;
  ft_twin_workload_t      workload;
  const char*             workload_label; /* Optional custom workload string */

  /* GEMM / SwiGLU / Quantized matmul shape */
  uint32_t                m;
  uint32_t                k;
  uint32_t                n;
  uint32_t                h; /* SwiGLU hidden dim */

  /* Candidate tile schedule overrides (0 = planner auto-select) */
  uint32_t                bm;
  uint32_t                bn;
  uint32_t                bk;
  int                     double_buffered; /* -1 = auto, 0 = off, 1 = on */

  ft_twin_dtype_t         act_dtype;
  ft_twin_dtype_t         out_dtype;
  ft_twin_activation_t    activation;

  /* SwiGLU options */
  int                     swiglu_11chunk; /* 1 = 11-chunk in-SRAM chained INT8 SwiGLU */
  uint32_t                quant_type;     /* ob_quant_type_t (0..5) for quantized SwiGLU */

  /* FlashAttention options */
  ft_twin_fa_phase_t      fa_phase;
  uint32_t                head_dim;
  uint32_t                num_q_heads;
  uint32_t                num_kv_heads;
  uint32_t                seq_len;
  uint32_t                br;             /* 0 = auto */
  uint32_t                bc;             /* 0 = auto */
  uint32_t                kv_tile_align;  /* 0 = default 64 */
  ft_twin_fa_buf_policy_t fa_buf_policy;
  ft_twin_kv_format_t     kv_format;

  /* 0x640 Spline options */
  ft_twin_spline_kind_t   spline_kind;
  uint32_t                spline_samples; /* 0 = default 2049 */

  /* Hardware budget & numerical tolerances (0 = target/workload default) */
  uint64_t                sram_limit_override_bytes;
  double                  max_nmse;
  double                  max_abs_err_limit;

  /* Malibu hybrid profile options */
  uint32_t                malibu_threads;
  uint32_t                malibu_active_engine_mask;
  int                     malibu_mesa_pco_ready;

  /* Reproducibility & execution flags */
  uint32_t                seed;
  int                     verify_dgc0; /* 1 = emit & validate DGC0 when applicable */

  /* Raw .dgc0 payload inspection (FT_TWIN_WORKLOAD_DGC0) */
  const uint8_t*          dgc0_bytes;
  size_t                  dgc0_size;
  const char*             dgc0_path;
  const char*             expected_sha256; /* Optional 64-hex-char SHA-256 check */
} ft_twin_request_t;

typedef struct ft_twin_result {
  char     schema_version[32];
  char     domain[16];
  char     target[32];
  char     workload[128];

  int      pass;
  double   reward;

  int      has_nmse;
  double   nmse;
  int      has_max_abs_err;
  double   max_abs_err;

  int      has_sram_bytes;
  uint64_t sram_bytes;
  int      has_sram_limit_bytes;
  uint64_t sram_limit_bytes;

  int      has_est_hw_us;
  double   est_hw_us;
  double   eval_ms;

  int      has_fault;
  char     fault_reason[256];

  /* Structured check tri-states: -1 = N/A, 0 = fail, 1 = pass */
  int      check_sram_budget_ok;
  int      check_dgc0_wire_ok;
  int      check_spline_0x640_ok;
  int      check_reference_parity_ok;
  int      check_profile_routing_ok;
  int      check_schedule_valid;

  /* GRPO scalar reward components */
  double   rb_gate_pass;
  double   rb_accuracy_score;
  double   rb_sram_efficiency;
  double   rb_latency_score;

  /* Rich execution & wire telemetry */
  uint32_t dgc0_bytes_checked;
  uint32_t dgc0_num_inputs;
  uint32_t dgc0_num_outputs;
  uint32_t dgc0_cluster_id;
  char     dgc0_sha256[65];
  char     selected_engine[32];
  uint32_t tiles_dispatched;
  uint32_t verified_elements;
  double   spline_max_err;
  uint32_t resolved_bm;
  uint32_t resolved_bn;
  uint32_t resolved_bk;

  uint32_t             num_diagnostics;
  ft_twin_diagnostic_t diagnostics[FT_TWIN_MAX_DIAGNOSTICS];
} ft_twin_result_t;

/* Opaque reusable session holding a warm mock_mode=1 libopenbuenos context */
typedef struct ft_twin_session ft_twin_session_t;

/* Default-initialized request */
ft_twin_request_t ft_twin_request_default(void);

/* Session lifecycle (reuse across batch / RL inner loops for maximum throughput) */
ft_twin_status_t ft_twin_session_create(ft_twin_session_t** out_session);
void             ft_twin_session_destroy(ft_twin_session_t* session);

/* Evaluate a candidate FreeTile schedule, kernel, spline, or .dgc0 payload */
ft_twin_status_t ft_twin_session_eval(ft_twin_session_t*     session,
                                      const ft_twin_request_t* req,
                                      ft_twin_result_t*        out_res);

/* Stateless convenience wrapper (creates/reuses an internal mock context) */
ft_twin_status_t ft_twin_eval(const ft_twin_request_t* req,
                              ft_twin_result_t*        out_res);

/* Direct raw .dgc0 wire integrity & SRAM budget validation helpers */
ft_twin_status_t ft_twin_eval_dgc0_buffer(ft_twin_target_t  target,
                                          const uint8_t*    data,
                                          size_t            size,
                                          const char*       expected_sha256,
                                          uint64_t          sram_limit_override_bytes,
                                          ft_twin_result_t* out_res);

ft_twin_status_t ft_twin_eval_dgc0_file(ft_twin_target_t  target,
                                        const char*       path,
                                        const char*       expected_sha256,
                                        uint64_t          sram_limit_override_bytes,
                                        ft_twin_result_t* out_res);

/*
 * Emit a valid deterministic .dgc0 executable payload for a supported workload
 * (GEMM W8A8, W8A16, or 11-chunk SwiGLU) using a synthetic RV64 DIVE ext[1] stub.
 * Useful for generating offline fixtures and negative-testing .dgc0 wire checks.
 */
ft_twin_status_t ft_twin_emit_sample_dgc0(const ft_twin_request_t* req,
                                          uint8_t*                 out_buf,
                                          size_t                   out_cap,
                                          size_t*                  out_size);

/* Serialize ft_twin_result_t to deterministic freesilicon.twin.v1 JSON */
ft_twin_status_t ft_twin_result_to_json(const ft_twin_result_t* res,
                                        int                     pretty,
                                        char*                   out_buf,
                                        size_t                  out_cap,
                                        size_t*                 out_len);

/* Name helpers */
const char* ft_twin_target_name(ft_twin_target_t target);
const char* ft_twin_workload_name(ft_twin_workload_t workload);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FREETILE_TWIN_H_ */
