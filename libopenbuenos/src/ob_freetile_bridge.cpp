// SPDX-License-Identifier: Apache-2.0
// Bridge between libopenbuenos (ob_gemm_desc_t / W8A16 / dyn-weights) and the
// clean-room zero-dependency C++17 FreeTile BUENOS_A0 VLIW/DGC0 emitter.

#include "ob_internal.h"

#if defined(OPENBUENOS_WITH_FREETILE) && OPENBUENOS_WITH_FREETILE
#include "freetile/buenos_emitter.hpp"
#endif

namespace ob {

bool freetile_is_compiled_in() {
#if defined(OPENBUENOS_WITH_FREETILE) && OPENBUENOS_WITH_FREETILE
  return true;
#else
  return false;
#endif
}

#if defined(OPENBUENOS_WITH_FREETILE) && OPENBUENOS_WITH_FREETILE

namespace {

bool map_ob_activation(uint32_t ob_act, freetile::EpilogueActivation* out_act) {
  if (!out_act) return false;
  switch (ob_act) {
    case OB_ACT_NONE:
      *out_act = freetile::EpilogueActivation::NONE;
      return true;
    case OB_ACT_RELU:
      *out_act = freetile::EpilogueActivation::RELU;
      return true;
    case OB_ACT_RELU6:
      *out_act = freetile::EpilogueActivation::RELU6;
      return true;
    case OB_ACT_SILU:
      *out_act = freetile::EpilogueActivation::SILU;
      return true;
    case OB_ACT_GELU:
      *out_act = freetile::EpilogueActivation::GELU;
      return true;
    default:
      return false;
  }
}

}  // namespace

ob_status_t freetile_try_emit_w8a8_dgc0(const ob_gemm_desc_t& desc,
                                        const uint8_t* dive_interp,
                                        size_t dive_interp_size,
                                        std::vector<uint8_t>* out_dgc0,
                                        std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  freetile::EpilogueActivation act = freetile::EpilogueActivation::NONE;
  if (!map_ob_activation(desc.activation, &act)) {
    if (err) *err = "Unsupported activation for FreeTile BUENOS_A0 emitter";
    return OB_ERR_UNSUPPORTED;
  }

  freetile::TileKernel kernel;
  std::string reason;
  if (!freetile::plan_buenos_matmul_kernel(static_cast<int64_t>(desc.m_dim),
                                           static_cast<int64_t>(desc.k_dim),
                                           static_cast<int64_t>(desc.n_dim),
                                           freetile::QuantFormat::Q8_0,
                                           freetile::DType::S8,
                                           freetile::DType::S8,
                                           act,
                                           &kernel,
                                           &reason)) {
    if (err) *err = reason;
    return OB_ERR_UNSUPPORTED;
  }

  freetile::BuenosEmitOptions opts;
  opts.mode = freetile::BuenosKernelMode::W8A8_INT8;
  opts.s_in = desc.s_in;
  opts.zp_in = desc.zp_in;
  opts.s_w = desc.s_w;
  opts.zp_w = desc.zp_w;
  opts.s_out = desc.s_out;
  opts.zp_out = desc.zp_out;
  opts.weights_nk = desc.weights_nk;
  opts.bias_n = desc.bias_n;
  opts.packed_params = desc.packed_params;
  opts.packed_params_bytes = desc.packed_params_bytes;
  opts.dive_interp = dive_interp;
  opts.dive_interp_size = dive_interp_size;

  freetile::BuenosEmitResult res;
  if (!freetile::try_emit_buenos_kernel(kernel, opts, &res, &reason) || res.dgc0.empty()) {
    if (err) *err = reason.empty() ? "FreeTile W8A8 emission returned empty DGC0" : reason;
    return OB_ERR_UNSUPPORTED;
  }

  *out_dgc0 = std::move(res.dgc0);
  return OB_OK;
}

ob_status_t freetile_try_emit_w8a16_dgc0(uint32_t K,
                                         uint32_t N,
                                         float pt_scale,
                                         const int8_t* weights_nk,
                                         const uint8_t* dive_interp,
                                         size_t dive_interp_size,
                                         std::vector<uint8_t>* out_dgc0,
                                         std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  freetile::TileKernel kernel;
  std::string reason;
  if (!freetile::plan_buenos_matmul_kernel(1,
                                           static_cast<int64_t>(K),
                                           static_cast<int64_t>(N),
                                           freetile::QuantFormat::Q8_0,
                                           freetile::DType::F16,
                                           freetile::DType::F32,
                                           freetile::EpilogueActivation::NONE,
                                           &kernel,
                                           &reason)) {
    if (err) *err = reason;
    return OB_ERR_UNSUPPORTED;
  }

  freetile::BuenosEmitOptions opts;
  opts.mode = freetile::BuenosKernelMode::W8A16_PT_1X;
  opts.w8a16_pt_scale = pt_scale;
  opts.weights_nk = weights_nk;
  opts.dive_interp = dive_interp;
  opts.dive_interp_size = dive_interp_size;

  freetile::BuenosEmitResult res;
  if (!freetile::try_emit_buenos_kernel(kernel, opts, &res, &reason) || res.dgc0.empty()) {
    if (err) *err = reason.empty() ? "FreeTile W8A16 emission returned empty DGC0" : reason;
    return OB_ERR_UNSUPPORTED;
  }

  *out_dgc0 = std::move(res.dgc0);
  return OB_OK;
}

ob_status_t freetile_try_emit_w8a16_64x_dgc0(uint32_t K,
                                             uint32_t N,
                                             float pt_scale,
                                             const int8_t* weights_nk,
                                             const uint8_t* dive_interp,
                                             size_t dive_interp_size,
                                             std::vector<uint8_t>* out_dgc0,
                                             std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  freetile::TileKernel kernel;
  std::string reason;
  if (!freetile::plan_buenos_matmul_kernel(64,
                                           static_cast<int64_t>(K),
                                           static_cast<int64_t>(N),
                                           freetile::QuantFormat::Q8_0,
                                           freetile::DType::F16,
                                           freetile::DType::F32,
                                           freetile::EpilogueActivation::NONE,
                                           &kernel,
                                           &reason)) {
    if (err) *err = reason;
    return OB_ERR_UNSUPPORTED;
  }

  freetile::BuenosEmitOptions opts;
  opts.mode = freetile::BuenosKernelMode::W8A16_PT_64X;
  opts.w8a16_pt_scale = pt_scale;
  opts.weights_nk = weights_nk;
  opts.dive_interp = dive_interp;
  opts.dive_interp_size = dive_interp_size;

  freetile::BuenosEmitResult res;
  if (!freetile::try_emit_buenos_kernel(kernel, opts, &res, &reason) || res.dgc0.empty()) {
    if (err) *err = reason.empty() ? "FreeTile W8A16_PT_64X emission returned empty DGC0" : reason;
    return OB_ERR_UNSUPPORTED;
  }

  *out_dgc0 = std::move(res.dgc0);
  return OB_OK;
}

ob_status_t freetile_try_emit_swiglu_dgc0(const ob_swiglu_int8_desc_t& desc,
                                          const uint8_t* dive_interp,
                                          size_t dive_interp_size,
                                          std::vector<uint8_t>* out_dgc0,
                                          std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  freetile::BuenosSwigluDesc ft_desc{};
  ft_desc.m_dim = desc.m_dim;
  ft_desc.k_dim = desc.k_dim;
  ft_desc.h_dim = desc.h_dim;
  ft_desc.n_dim = desc.n_dim;
  ft_desc.wg_hk = desc.wg_hk;
  ft_desc.bg_h = desc.bg_h;
  ft_desc.wu_hk = desc.wu_hk;
  ft_desc.bu_h = desc.bu_h;
  ft_desc.wd_nh = desc.wd_nh;
  ft_desc.bd_n = desc.bd_n;
  ft_desc.s_in = desc.s_in;
  ft_desc.s_wg = desc.s_wg;
  ft_desc.s_gate = desc.s_gate;
  ft_desc.s_wu = desc.s_wu;
  ft_desc.s_up = desc.s_up;
  ft_desc.s_silu = desc.s_silu;
  ft_desc.s_prod = desc.s_prod;
  ft_desc.s_wd = desc.s_wd;
  ft_desc.s_out = desc.s_out;
  ft_desc.dive_interp = dive_interp;
  ft_desc.dive_interp_size = dive_interp_size;

  freetile::BuenosSwigluResult res{};
  std::string reason;
  if (!freetile::try_emit_buenos_swiglu(ft_desc, &res, &reason) || res.dgc0.empty()) {
    if (err) *err = reason.empty() ? "FreeTile SwiGLU emission returned empty DGC0" : reason;
    return freetile::is_buenos_swiglu_shape(desc.m_dim, desc.k_dim, desc.h_dim, desc.n_dim)
               ? OB_ERR_INVALID_ARG
               : OB_ERR_UNSUPPORTED;
  }

  *out_dgc0 = std::move(res.dgc0);
  return OB_OK;
}

ob_status_t freetile_try_emit_dyn_weights_dgc0(const ob_gemm_desc_t& desc,
                                               const uint8_t* dive_interp,
                                               size_t dive_interp_size,
                                               std::vector<uint8_t>* out_dgc0,
                                               uint32_t* out_ext2_bytes,
                                               std::string* err) {
  if (!dive_interp || dive_interp_size == 0u || !out_dgc0) {
    if (err) *err = "Missing external RV64 DIVE interpreter image (ext[1])";
    return OB_ERR_INVALID_ARG;
  }

  freetile::EpilogueActivation act = freetile::EpilogueActivation::NONE;
  if (!map_ob_activation(desc.activation, &act)) {
    if (err) *err = "Unsupported activation for FreeTile BUENOS_A0 dyn-weights emitter";
    return OB_ERR_UNSUPPORTED;
  }

  freetile::TileKernel kernel;
  std::string reason;
  if (!freetile::plan_buenos_matmul_kernel(static_cast<int64_t>(desc.m_dim),
                                           static_cast<int64_t>(desc.k_dim),
                                           static_cast<int64_t>(desc.n_dim),
                                           freetile::QuantFormat::Q8_0,
                                           freetile::DType::S8,
                                           freetile::DType::S8,
                                           act,
                                           &kernel,
                                           &reason)) {
    if (err) *err = reason;
    return OB_ERR_UNSUPPORTED;
  }

  freetile::BuenosEmitOptions opts;
  opts.mode = freetile::BuenosKernelMode::W8A8_DYN_WEIGHTS;
  opts.s_in = desc.s_in;
  opts.zp_in = desc.zp_in;
  opts.s_w = desc.s_w;
  opts.zp_w = desc.zp_w;
  opts.s_out = desc.s_out;
  opts.zp_out = desc.zp_out;
  opts.bias_n = desc.bias_n;
  opts.dive_interp = dive_interp;
  opts.dive_interp_size = dive_interp_size;

  freetile::BuenosEmitResult res;
  if (!freetile::try_emit_buenos_kernel(kernel, opts, &res, &reason) || res.dgc0.empty()) {
    if (err) *err = reason.empty() ? "FreeTile dyn-weights emission returned empty DGC0" : reason;
    return OB_ERR_UNSUPPORTED;
  }

  if (out_ext2_bytes) *out_ext2_bytes = res.ext2_bytes;
  *out_dgc0 = std::move(res.dgc0);
  return OB_OK;
}

#else

ob_status_t freetile_try_emit_w8a8_dgc0(const ob_gemm_desc_t&,
                                        const uint8_t*,
                                        size_t,
                                        std::vector<uint8_t>*,
                                        std::string* err) {
  if (err) *err = "FreeTile BUENOS_A0 emitter not compiled into libopenbuenos";
  return OB_ERR_UNSUPPORTED;
}

ob_status_t freetile_try_emit_w8a16_dgc0(uint32_t,
                                         uint32_t,
                                         float,
                                         const int8_t*,
                                         const uint8_t*,
                                         size_t,
                                         std::vector<uint8_t>*,
                                         std::string* err) {
  if (err) *err = "FreeTile BUENOS_A0 emitter not compiled into libopenbuenos";
  return OB_ERR_UNSUPPORTED;
}

ob_status_t freetile_try_emit_w8a16_64x_dgc0(uint32_t,
                                             uint32_t,
                                             float,
                                             const int8_t*,
                                             const uint8_t*,
                                             size_t,
                                             std::vector<uint8_t>*,
                                             std::string* err) {
  if (err) *err = "FreeTile BUENOS_A0 emitter not compiled into libopenbuenos";
  return OB_ERR_UNSUPPORTED;
}

ob_status_t freetile_try_emit_swiglu_dgc0(const ob_swiglu_int8_desc_t&,
                                          const uint8_t*,
                                          size_t,
                                          std::vector<uint8_t>*,
                                          std::string* err) {
  if (err) *err = "FreeTile BUENOS_A0 emitter not compiled into libopenbuenos";
  return OB_ERR_UNSUPPORTED;
}

ob_status_t freetile_try_emit_dyn_weights_dgc0(const ob_gemm_desc_t&,
                                               const uint8_t*,
                                               size_t,
                                               std::vector<uint8_t>*,
                                               uint32_t*,
                                               std::string* err) {
  if (err) *err = "FreeTile BUENOS_A0 emitter not compiled into libopenbuenos";
  return OB_ERR_UNSUPPORTED;
}

#endif

}  // namespace ob
