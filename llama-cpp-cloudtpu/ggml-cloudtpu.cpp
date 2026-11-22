#include "ggml-cloudtpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t CLOUDTPU_BUFFER_ALIGNMENT = 64;

struct cloudtpu_env_config {
    bool mock_mode = false;
    int verbose = 0;
    bool fuse_graphs = true;
    uint32_t cache_cap = 4096;
    uint64_t min_elements = 0;
    std::string libtpu_path;
};

static cloudtpu_env_config load_env_config() {
    cloudtpu_env_config cfg;
#if defined(__linux__)
    cfg.mock_mode = false;
#else
    cfg.mock_mode = true;
#endif
    if (const char* v = std::getenv("CLOUDTPU_MOCK")) {
        cfg.mock_mode = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_VERBOSE")) {
        cfg.verbose = std::atoi(v);
    }
    if (const char* v = std::getenv("CLOUDTPU_FUSE_GRAPHS")) {
        cfg.fuse_graphs = (std::atoi(v) != 0);
    }
    if (const char* v = std::getenv("CLOUDTPU_CACHE_CAP")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0 && parsed <= 1048576LL) {
            cfg.cache_cap = static_cast<uint32_t>(parsed);
        }
    }
    if (const char* v = std::getenv("CLOUDTPU_MIN_ELEMENTS")) {
        const long long parsed = std::atoll(v);
        if (parsed > 0) {
            cfg.min_elements = static_cast<uint64_t>(parsed);
        }
    }
    if (const char* v = std::getenv("LIBTPU_PATH")) {
        cfg.libtpu_path = v;
    } else if (const char* v2 = std::getenv("TPU_LIBRARY_PATH")) {
        cfg.libtpu_path = v2;
    }
    return cfg;
}

struct cloudtpu_hot_env_cache {
    std::atomic<bool> initialized{false};
    std::atomic<bool> mock_mode{false};
    std::atomic<int> verbose{0};
    std::atomic<bool> fuse_graphs{true};
    std::atomic<uint64_t> min_elements{0};

    void store(const cloudtpu_env_config& cfg) {
        mock_mode.store(cfg.mock_mode, std::memory_order_relaxed);
        verbose.store(cfg.verbose, std::memory_order_relaxed);
        fuse_graphs.store(cfg.fuse_graphs, std::memory_order_relaxed);
        min_elements.store(cfg.min_elements, std::memory_order_relaxed);
        initialized.store(true, std::memory_order_release);
    }

    void ensure_initialized() {
        if (!initialized.load(std::memory_order_acquire)) {
            store(load_env_config());
        }
    }
};

static cloudtpu_hot_env_cache g_hot_env;

static cloudtpu::PjrtRuntime& ensure_runtime_initialized() {
    g_hot_env.ensure_initialized();
    const cloudtpu_env_config env = load_env_config();
    cloudtpu::RuntimeConfig rt_cfg;
    rt_cfg.mock_mode = env.mock_mode;
    rt_cfg.verbose = env.verbose;
    rt_cfg.exec_cache_cap = env.cache_cap;
    rt_cfg.libtpu_path = env.libtpu_path;
    return cloudtpu::shared_pjrt_runtime(&rt_cfg);
}

static cloudtpu::PjrtDType ggml_type_to_pjrt(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return cloudtpu::PjrtDType::F32;
        case GGML_TYPE_F16: return cloudtpu::PjrtDType::F16;
        case GGML_TYPE_BF16: return cloudtpu::PjrtDType::BF16;
        case GGML_TYPE_I8: return cloudtpu::PjrtDType::S8;
        case GGML_TYPE_I16: return cloudtpu::PjrtDType::S16;
        case GGML_TYPE_I32: return cloudtpu::PjrtDType::S32;
        default: return cloudtpu::PjrtDType::INVALID;
    }
}

static const struct ggml_tensor* unwrap_view_src(const struct ggml_tensor* t) {
    while (t && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static bool is_dense_supported_type(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_I32;
}

static bool is_quant_split_type(enum ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
}

static bool is_supported_operand_layout(const struct ggml_tensor* t) {
    if (!t || !ggml_is_contiguous(t)) {
        return false;
    }
    if (t->view_src == nullptr) {
        return true;
    }
    // Accept zero-offset element-preserving reshape/view chains; reject offset-slices and permutes
    // until M2 adds general slice/permute HBM view lowering.
    if (t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
        return false;
    }
    if (t->view_offs != 0 || ggml_nelements(t) != ggml_nelements(t->view_src) ||
        t->type != t->view_src->type) {
        return false;
    }
    return is_supported_operand_layout(t->view_src);
}

// =========================================================================
// HBM Buffer Management (PJRT_Buffer* Tensor Slot Table)
// =========================================================================

struct CloudTpuTensorSlot {
    size_t offset = 0;
    size_t nbytes = 0;
    enum ggml_type ggml_dt = GGML_TYPE_F32;
    cloudtpu::TensorStorageKind storage = cloudtpu::TensorStorageKind::DENSE;
    cloudtpu::PjrtDType dtype = cloudtpu::PjrtDType::F32;
    int64_t dims[4] = {1, 1, 1, 1};
    cloudtpu::PjrtBufferPtr hbm_primary;
    cloudtpu::PjrtBufferPtr hbm_scales;
    std::vector<uint8_t> staging;
    bool staging_dirty = false;
};

struct ggml_backend_cloudtpu_buffer_context {
    int device_index = 0;
    uintptr_t virtual_base = 0;
    size_t size = 0;

    std::mutex mu;
    std::unordered_map<size_t, CloudTpuTensorSlot> slots;  // keyed by byte offset from virtual_base

    void evict_overlapping_locked(size_t offset, size_t nbytes, size_t keep_exact_offset = SIZE_MAX) {
        const size_t end = offset + nbytes;
        for (auto it = slots.begin(); it != slots.end();) {
            if (it->first == keep_exact_offset) {
                ++it;
                continue;
            }
            const size_t s0 = it->second.offset;
            const size_t s1 = s0 + it->second.nbytes;
            if (s0 < end && offset < s1) {
                it = slots.erase(it);
            } else {
                ++it;
            }
        }
    }
};

static std::atomic<uintptr_t> g_next_virtual_base{0x200000000ULL};

static bool is_cloudtpu_buft(ggml_backend_buffer_type_t buft);

static bool is_cloudtpu_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && is_cloudtpu_buft(buffer->buft);
}

static size_t tensor_offset_in_buffer(const ggml_backend_cloudtpu_buffer_context* bctx,
                                      const struct ggml_tensor* tensor) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(addr >= bctx->virtual_base);
    return static_cast<size_t>(addr - bctx->virtual_base);
}

static void unpack_q8_0_to_split(const block_q8_0* src,
                                 size_t num_rows,
                                 size_t nb,
                                 std::vector<int8_t>& out_qs,
                                 std::vector<uint16_t>& out_d) {
    const size_t total_blocks = num_rows * nb;
    out_qs.resize(total_blocks * QK8_0);
    out_d.resize(total_blocks);
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&out_d[b], &src[b].d, sizeof(uint16_t));
        std::memcpy(out_qs.data() + b * QK8_0, src[b].qs, QK8_0);
    }
}

static void pack_split_to_q8_0(const int8_t* qs,
                               const uint16_t* d,
                               size_t total_blocks,
                               block_q8_0* dst) {
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&dst[b].d, &d[b], sizeof(uint16_t));
        std::memcpy(dst[b].qs, qs + b * QK8_0, QK8_0);
    }
}

static void unpack_q4_0_to_split(const block_q4_0* src,
                                 size_t num_rows,
                                 size_t nb,
                                 std::vector<uint8_t>& out_qs,
                                 std::vector<uint16_t>& out_d) {
    const size_t total_blocks = num_rows * nb;
    out_qs.resize(total_blocks * (QK4_0 / 2));
    out_d.resize(total_blocks);
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&out_d[b], &src[b].d, sizeof(uint16_t));
        std::memcpy(out_qs.data() + b * (QK4_0 / 2), src[b].qs, QK4_0 / 2);
    }
}

static void pack_split_to_q4_0(const uint8_t* qs,
                               const uint16_t* d,
                               size_t total_blocks,
                               block_q4_0* dst) {
    for (size_t b = 0; b < total_blocks; ++b) {
        std::memcpy(&dst[b].d, &d[b], sizeof(uint16_t));
        std::memcpy(dst[b].qs, qs + b * (QK4_0 / 2), QK4_0 / 2);
    }
}

static void upload_raw_tensor_to_slot(cloudtpu::PjrtRuntime& rt,
                                      int device_index,
                                      CloudTpuTensorSlot& slot,
                                      const struct ggml_tensor* tensor,
                                      const void* raw_data) {
    slot.ggml_dt = tensor->type;
    slot.dims[0] = tensor->ne[3];
    slot.dims[1] = tensor->ne[2];
    slot.dims[2] = tensor->ne[1];
    slot.dims[3] = tensor->ne[0];

    const int64_t B3 = tensor->ne[3];
    const int64_t B2 = tensor->ne[2];
    const int64_t N = tensor->ne[1];
    const int64_t K = tensor->ne[0];

    if (tensor->type == GGML_TYPE_Q8_0 && (K % QK8_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK8_0);
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        std::vector<int8_t> qs;
        std::vector<uint16_t> d;
        unpack_q8_0_to_split(static_cast<const block_q8_0*>(raw_data), nrows, nb, qs, d);
        slot.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::S8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::S8, {B3, B2, N, static_cast<int64_t>(nb), QK8_0});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d.data(), cloudtpu::PjrtDType::F16, {B3, B2, N, static_cast<int64_t>(nb)});
    } else if (tensor->type == GGML_TYPE_Q4_0 && (K % QK4_0) == 0) {
        const size_t nb = static_cast<size_t>(K / QK4_0);
        const size_t nrows = static_cast<size_t>(B3 * B2 * N);
        std::vector<uint8_t> qs;
        std::vector<uint16_t> d;
        unpack_q4_0_to_split(static_cast<const block_q4_0*>(raw_data), nrows, nb, qs, d);
        slot.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
        slot.dtype = cloudtpu::PjrtDType::U8;
        slot.hbm_primary = rt.buffer_from_host(
            device_index, qs.data(), cloudtpu::PjrtDType::U8, {B3, B2, N, static_cast<int64_t>(nb), QK4_0 / 2});
        slot.hbm_scales = rt.buffer_from_host(
            device_index, d.data(), cloudtpu::PjrtDType::F16, {B3, B2, N, static_cast<int64_t>(nb)});
    } else {
        const cloudtpu::PjrtDType pdt = ggml_type_to_pjrt(tensor->type);
        if (pdt != cloudtpu::PjrtDType::INVALID && ggml_is_contiguous(tensor)) {
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.dtype = pdt;
            slot.hbm_primary = rt.buffer_from_host(device_index, raw_data, pdt, {B3, B2, N, K});
            slot.hbm_scales.reset();
        } else {
            // Raw byte fallback for unsupported types (keeps tensor data intact for get_tensor / CPU fallback).
            slot.storage = cloudtpu::TensorStorageKind::DENSE;
            slot.dtype = cloudtpu::PjrtDType::U8;
            slot.hbm_primary = rt.buffer_from_host(
                device_index, raw_data, cloudtpu::PjrtDType::U8, {static_cast<int64_t>(slot.nbytes)});
            slot.hbm_scales.reset();
        }
    }
    std::vector<uint8_t>().swap(slot.staging);
    slot.staging_dirty = false;
}

static void download_slot_to_host(cloudtpu::PjrtRuntime& rt,
                                  const CloudTpuTensorSlot& slot,
                                  void* dst_full) {
    if (slot.staging_dirty && !slot.staging.empty()) {
        std::memcpy(dst_full, slot.staging.data(), slot.nbytes);
        return;
    }
    if (slot.storage == cloudtpu::TensorStorageKind::Q8_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales->num_elements();
        std::vector<int8_t> qs(total_blocks * QK8_0);
        std::vector<uint16_t> d(total_blocks);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        rt.buffer_to_host(slot.hbm_scales->handle.get(), d.data(), d.size() * sizeof(uint16_t));
        pack_split_to_q8_0(qs.data(), d.data(), total_blocks, static_cast<block_q8_0*>(dst_full));
    } else if (slot.storage == cloudtpu::TensorStorageKind::Q4_0_SPLIT) {
        const size_t total_blocks = slot.hbm_scales->num_elements();
        std::vector<uint8_t> qs(total_blocks * (QK4_0 / 2));
        std::vector<uint16_t> d(total_blocks);
        rt.buffer_to_host(slot.hbm_primary->handle.get(), qs.data(), qs.size());
        rt.buffer_to_host(slot.hbm_scales->handle.get(), d.data(), d.size() * sizeof(uint16_t));
        pack_split_to_q4_0(qs.data(), d.data(), total_blocks, static_cast<block_q4_0*>(dst_full));
    } else if (slot.hbm_primary) {
        rt.buffer_to_host(slot.hbm_primary->handle.get(), dst_full, slot.nbytes);
    }
}

static void ggml_backend_cloudtpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    delete bctx;
}

static void* ggml_backend_cloudtpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    return reinterpret_cast<void*>(bctx->virtual_base);
}

static void ggml_backend_cloudtpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       struct ggml_tensor* tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
    CloudTpuTensorSlot& slot = bctx->slots[t_off];
    slot.offset = t_off;
    slot.nbytes = t_bytes;

    if (offset == 0 && size == t_bytes) {
        std::vector<uint8_t> tmp(t_bytes, value);
        upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, tmp.data());
    } else {
        if (!slot.staging_dirty) {
            slot.staging.assign(t_bytes, 0);
            if (slot.hbm_primary) {
                download_slot_to_host(rt, slot, slot.staging.data());
            }
            slot.staging_dirty = true;
        }
        std::memset(slot.staging.data() + offset, value, size);
    }
}

static void ggml_backend_cloudtpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    struct ggml_tensor* tensor,
                                                    const void* data,
                                                    size_t offset,
                                                    size_t size) {
    if (size == 0) {
        return;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
    CloudTpuTensorSlot& slot = bctx->slots[t_off];
    slot.offset = t_off;
    slot.nbytes = t_bytes;

    if (offset == 0 && size == t_bytes) {
        upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, data);
    } else {
        if (!slot.staging_dirty) {
            slot.staging.assign(t_bytes, 0);
            if (slot.hbm_primary) {
                download_slot_to_host(rt, slot, slot.staging.data());
            }
            slot.staging_dirty = true;
        }
        std::memcpy(slot.staging.data() + offset, data, size);
        // If full buffer has been populated via tail write, flush to HBM immediately.
        if (offset + size == t_bytes) {
            upload_raw_tensor_to_slot(rt, bctx->device_index, slot, tensor, slot.staging.data());
        }
    }
}

static void ggml_backend_cloudtpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const struct ggml_tensor* tensor,
                                                    void* data,
                                                    size_t offset,
                                                    size_t size) {
    if (size == 0) {
        return;
    }
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    const size_t t_off = tensor_offset_in_buffer(bctx, tensor);
    const size_t t_bytes = ggml_nbytes(tensor);

    std::lock_guard<std::mutex> lock(bctx->mu);
    auto it = bctx->slots.find(t_off);
    if (it != bctx->slots.end() && it->second.nbytes >= offset + size) {
        const CloudTpuTensorSlot& slot = it->second;
        if (offset == 0 && size == slot.nbytes && slot.storage == cloudtpu::TensorStorageKind::DENSE &&
            !slot.staging_dirty && slot.hbm_primary) {
            rt.buffer_to_host(slot.hbm_primary->handle.get(), data, size);
            return;
        }
        std::vector<uint8_t> tmp(slot.nbytes);
        download_slot_to_host(rt, slot, tmp.data());
        std::memcpy(data, tmp.data() + offset, size);
        return;
    }

    // Check if tensor is a sub-slice view inside a parent slot.
    for (const auto& kv : bctx->slots) {
        const CloudTpuTensorSlot& parent = kv.second;
        if (t_off >= parent.offset && t_off + t_bytes <= parent.offset + parent.nbytes) {
            std::vector<uint8_t> tmp(parent.nbytes);
            download_slot_to_host(rt, parent, tmp.data());
            const size_t rel = (t_off - parent.offset) + offset;
            std::memcpy(data, tmp.data() + rel, size);
            return;
        }
    }

    // Zero-initialized fallback if tensor was never explicitly written.
    std::memset(data, 0, size);
}

static bool ggml_backend_cloudtpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const struct ggml_tensor* src,
                                                    struct ggml_tensor* dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_cloudtpu_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    if (is_cloudtpu_buffer(src->buffer) && src->type == dst->type && ggml_nbytes(src) == ggml_nbytes(dst)) {
        auto* src_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(src->buffer->context);
        auto* dst_bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
        if (src_bctx->device_index == dst_bctx->device_index) {
            const size_t src_off = tensor_offset_in_buffer(src_bctx, src);
            const size_t dst_off = tensor_offset_in_buffer(dst_bctx, dst);
            if (src_bctx == dst_bctx) {
                std::lock_guard<std::mutex> lock(src_bctx->mu);
                auto sit = src_bctx->slots.find(src_off);
                if (sit != src_bctx->slots.end() && !sit->second.staging_dirty && sit->second.hbm_primary) {
                    CloudTpuTensorSlot copy = sit->second;
                    copy.offset = dst_off;
                    src_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                    src_bctx->slots[dst_off] = std::move(copy);
                    return true;
                }
            } else {
                CloudTpuTensorSlot copy;
                bool ok = false;
                {
                    std::lock_guard<std::mutex> slock(src_bctx->mu);
                    auto sit = src_bctx->slots.find(src_off);
                    if (sit != src_bctx->slots.end() && !sit->second.staging_dirty && sit->second.hbm_primary) {
                        copy = sit->second;
                        ok = true;
                    }
                }
                if (ok) {
                    std::lock_guard<std::mutex> dlock(dst_bctx->mu);
                    copy.offset = dst_off;
                    dst_bctx->evict_overlapping_locked(dst_off, copy.nbytes, dst_off);
                    dst_bctx->slots[dst_off] = std::move(copy);
                    return true;
                }
            }
        }
    }
    return false;
}

static void ggml_backend_cloudtpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(value);
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->slots.clear();
}

static void ggml_backend_cloudtpu_buffer_reset(ggml_backend_buffer_t buffer) {
    auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(buffer->context);
    std::lock_guard<std::mutex> lock(bctx->mu);
    bctx->slots.clear();
}

static const struct ggml_backend_buffer_i ggml_backend_cloudtpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_cloudtpu_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_cloudtpu_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_cloudtpu_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_cloudtpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_cloudtpu_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_cloudtpu_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_cloudtpu_buffer_clear,
    /* .reset         = */ ggml_backend_cloudtpu_buffer_reset,
    /* .set_tensor_n  = */ NULL,
    /* .get_tensor_n  = */ NULL,
    /* .cpy_tensor_n  = */ NULL,
};

struct cloudtpu_buft_context {
    int device_index = 0;
    std::string name;
};

static const char* ggml_backend_cloudtpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    return btc->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cloudtpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t size) {
    auto* btc = static_cast<cloudtpu_buft_context*>(buft->context);
    ensure_runtime_initialized();

    const size_t aligned_span = std::max<size_t>(
        CLOUDTPU_BUFFER_ALIGNMENT,
        ((size + CLOUDTPU_BUFFER_ALIGNMENT - 1) & ~(CLOUDTPU_BUFFER_ALIGNMENT - 1)) + 4096);

    auto* bctx = new ggml_backend_cloudtpu_buffer_context();
    bctx->device_index = btc->device_index;
    bctx->virtual_base = g_next_virtual_base.fetch_add(aligned_span, std::memory_order_relaxed);
    bctx->size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_cloudtpu_buffer_i, bctx, size);
}

static size_t ggml_backend_cloudtpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return CLOUDTPU_BUFFER_ALIGNMENT;
}

static bool ggml_backend_cloudtpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static bool is_cloudtpu_buft(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_cloudtpu_buffer_type_get_name;
}

// =========================================================================
// Backend Stream Context & StableHLO Subgraph Dispatch
// =========================================================================

struct ggml_backend_cloudtpu_context {
    int device_index = 0;
    cloudtpu_env_config env;
    uint64_t total_graph_evals = 0;
    uint64_t total_subgraphs = 0;
    uint64_t total_ops_executed = 0;

    ~ggml_backend_cloudtpu_context() {
        if (env.verbose > 0) {
            cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
            if (rt.is_initialized()) {
                const cloudtpu::CacheStats st = rt.cache_stats();
                GGML_LOG_INFO(
                    "ggml-cloudtpu: summary device=%d mock=%d maps_libpython=%zu evals=%llu subgraphs=%llu ops=%llu "
                    "cache_entries=%zu cache_hits=%llu cache_misses=%llu cache_evictions=%llu "
                    "compile_us=%llu execute_us=%llu\n",
                    device_index,
                    rt.is_mock() ? 1 : 0,
                    cloudtpu::count_libpython_mappings(),
                    static_cast<unsigned long long>(total_graph_evals),
                    static_cast<unsigned long long>(total_subgraphs),
                    static_cast<unsigned long long>(total_ops_executed),
                    st.entries,
                    static_cast<unsigned long long>(st.hits),
                    static_cast<unsigned long long>(st.misses),
                    static_cast<unsigned long long>(st.evictions),
                    static_cast<unsigned long long>(st.compile_us),
                    static_cast<unsigned long long>(st.execute_us));
            }
        }
    }
};

static bool is_view_noop_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool ggml_backend_cloudtpu_device_supports_op_impl(const struct ggml_tensor* op) {
    if (!op) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(src1) ||
                !is_supported_operand_layout(op)) {
                return false;
            }
            const bool src0_ok = is_dense_supported_type(src0->type) ||
                                 (is_quant_split_type(src0->type) && (src0->ne[0] % QK8_0) == 0);
            const bool src1_ok = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                                  src1->type == GGML_TYPE_BF16);
            if (!src0_ok || !src1_ok || src0->type == GGML_TYPE_I32) {
                return false;
            }
            const int64_t K = src0->ne[0];
            const int64_t N = src0->ne[1];
            const int64_t M = src1->ne[1];
            if (K <= 0 || N <= 0 || M <= 0 || src1->ne[0] != K) {
                return false;
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return false;
            }
            g_hot_env.ensure_initialized();
            const uint64_t min_elements = g_hot_env.min_elements.load(std::memory_order_relaxed);
            if (min_elements > 0) {
                const uint64_t work = static_cast<uint64_t>(M) * static_cast<uint64_t>(K) *
                                      static_cast<uint64_t>(N) * static_cast<uint64_t>(src1->ne[2]) *
                                      static_cast<uint64_t>(src1->ne[3]);
                if (work < min_elements) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_RMS_NORM: {
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            return is_supported_operand_layout(src0) && is_supported_operand_layout(op);
        }

        case GGML_OP_ROPE: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            const struct ggml_tensor* src2 = op->src[2];
            if (!src0 || !src1) {
                return false;
            }
            if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) ||
                op->type != src0->type || src1->type != GGML_TYPE_I32) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(src1) ||
                !is_supported_operand_layout(op)) {
                return false;
            }
            const int32_t n_dims = static_cast<const int32_t*>(op->op_params)[1];
            const int32_t mode = static_cast<const int32_t*>(op->op_params)[2];
            const int32_t n_offs = static_cast<const int32_t*>(op->op_params)[15];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false;
            }
            if (n_dims <= 0 || (n_dims % 2) != 0 || n_offs < 0 || (n_offs % 2) != 0 ||
                n_offs + n_dims > src0->ne[0]) {
                return false;
            }
            if (src1->ne[0] != src0->ne[2]) {
                return false;
            }
            if (src2 != nullptr &&
                (src2->type != GGML_TYPE_F32 || !is_supported_operand_layout(src2) || src2->ne[0] < n_dims / 2)) {
                return false;
            }
            return true;
        }

        case GGML_OP_UNARY: {
            if (ggml_get_unary_op(op) != GGML_UNARY_OP_SILU) {
                return false;
            }
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || op->type != src0->type) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
                return false;
            }
            return is_supported_operand_layout(src0) && is_supported_operand_layout(op);
        }

        case GGML_OP_GLU: {
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) {
                return false;
            }
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || op->type != src0->type) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(op)) {
                return false;
            }
            if (src1) {
                if (src1->type != src0->type || !is_supported_operand_layout(src1) ||
                    !ggml_are_same_shape(src0, src1)) {
                    return false;
                }
            } else if ((src0->ne[0] % 2) != 0) {
                return false;
            }
            return true;
        }

        case GGML_OP_SOFT_MAX: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            const struct ggml_tensor* src2 = op->src[2];
            if (!src0 || src2 != nullptr || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(op)) {
                return false;
            }
            if (src1) {
                if ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                    !is_supported_operand_layout(src1)) {
                    return false;
                }
                if (src1->ne[0] != src0->ne[0] || src1->ne[1] < src0->ne[1] ||
                    (src0->ne[2] % src1->ne[2]) != 0 || (src0->ne[3] % src1->ne[3]) != 0) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1 || src1->type != GGML_TYPE_I32 || src1->ne[3] != 1) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(src1) ||
                !is_supported_operand_layout(op)) {
                return false;
            }
            if (src0->ne[2] != src1->ne[1] || src0->ne[3] != src1->ne[2]) {
                return false;
            }
            const bool src0_ok = is_dense_supported_type(src0->type) ||
                                 (is_quant_split_type(src0->type) && (src0->ne[0] % QK8_0) == 0);
            if (!src0_ok) {
                return false;
            }
            if (src0->type == GGML_TYPE_I32) {
                return op->type == GGML_TYPE_I32;
            }
            return op->type == GGML_TYPE_F32;
        }

        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const struct ggml_tensor* src0 = op->src[0];
            const struct ggml_tensor* src1 = op->src[1];
            if (!src0 || !src1) {
                return false;
            }
            if (src0->type != op->type || src1->type != op->type) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(src1) ||
                !is_supported_operand_layout(op)) {
                return false;
            }
            for (int d = 0; d < 4; ++d) {
                if (src1->ne[d] <= 0 || (src0->ne[d] % src1->ne[d]) != 0 || op->ne[d] != src0->ne[d]) {
                    return false;
                }
            }
            return true;
        }

        case GGML_OP_CPY:
        case GGML_OP_CONT: {
            const struct ggml_tensor* src0 = op->src[0];
            if (!src0 || !is_dense_supported_type(src0->type) || !is_dense_supported_type(op->type)) {
                return false;
            }
            if (!is_supported_operand_layout(src0) || !is_supported_operand_layout(op)) {
                return false;
            }
            return ggml_nelements(src0) == ggml_nelements(op);
        }

        default:
            return false;
    }
}

static cloudtpu::TensorSpec tensor_to_spec(const struct ggml_tensor* t, uint64_t id) {
    cloudtpu::TensorSpec ts;
    ts.id = id;
    ts.dims[0] = t->ne[3];
    ts.dims[1] = t->ne[2];
    ts.dims[2] = t->ne[1];
    ts.dims[3] = t->ne[0];
    if (t->type == GGML_TYPE_Q8_0) {
        ts.storage = cloudtpu::TensorStorageKind::Q8_0_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::S8;
    } else if (t->type == GGML_TYPE_Q4_0) {
        ts.storage = cloudtpu::TensorStorageKind::Q4_0_SPLIT;
        ts.dtype = cloudtpu::PjrtDType::U8;
    } else {
        ts.storage = cloudtpu::TensorStorageKind::DENSE;
        ts.dtype = ggml_type_to_pjrt(t->type);
    }
    return ts;
}

// Resolves a GGML tensor to HBM buffers (fetching from CloudTpuBuffer or uploading transient host tensor).
static void resolve_tensor_hbm(cloudtpu::PjrtRuntime& rt,
                               int device_index,
                               const struct ggml_tensor* t,
                               cloudtpu::PjrtBufferPtr& out_primary,
                               cloudtpu::PjrtBufferPtr& out_scales) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        std::lock_guard<std::mutex> lock(bctx->mu);
        auto it = bctx->slots.find(t_off);
        if (it != bctx->slots.end()) {
            CloudTpuTensorSlot& slot = it->second;
            if (slot.staging_dirty && !slot.staging.empty()) {
                upload_raw_tensor_to_slot(rt, bctx->device_index, slot, t, slot.staging.data());
            }
            if (slot.hbm_primary && slot.ggml_dt == t->type &&
                slot.dims[0] == t->ne[3] && slot.dims[1] == t->ne[2] &&
                slot.dims[2] == t->ne[1] && slot.dims[3] == t->ne[0]) {
                out_primary = slot.hbm_primary;
                out_scales = slot.hbm_scales;
                return;
            }
        }
        // Check if tensor is a contiguous view aliasing a parent slot at offset 0 with matching element count.
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t)) {
            const struct ggml_tensor* base = unwrap_view_src(t);
            const size_t base_off = tensor_offset_in_buffer(bctx, base);
            auto bit = bctx->slots.find(base_off);
            if (bit != bctx->slots.end() && bit->second.hbm_primary) {
                out_primary = bit->second.hbm_primary;
                out_scales = bit->second.hbm_scales;
                return;
            }
        }
        throw std::runtime_error("ggml-cloudtpu: missing HBM slot for input tensor '" +
                                 std::string(t->name) + "'");
    }

    // Host tensor fallback: upload transient copy to device HBM.
    if (!t->data) {
        throw std::runtime_error("ggml-cloudtpu: null host data pointer for tensor '" +
                                 std::string(t->name) + "'");
    }
    CloudTpuTensorSlot tmp_slot;
    tmp_slot.nbytes = ggml_nbytes(t);
    upload_raw_tensor_to_slot(rt, device_index, tmp_slot, t, t->data);
    out_primary = std::move(tmp_slot.hbm_primary);
    out_scales = std::move(tmp_slot.hbm_scales);
}

static void store_tensor_hbm(cloudtpu::PjrtRuntime& rt,
                             struct ggml_tensor* t,
                             const cloudtpu::PjrtBufferPtr& buf) {
    if (t->buffer && is_cloudtpu_buffer(t->buffer)) {
        auto* bctx = static_cast<ggml_backend_cloudtpu_buffer_context*>(t->buffer->context);
        const size_t t_off = tensor_offset_in_buffer(bctx, t);
        const size_t t_bytes = ggml_nbytes(t);
        std::lock_guard<std::mutex> lock(bctx->mu);
        bctx->evict_overlapping_locked(t_off, t_bytes, t_off);
        CloudTpuTensorSlot& slot = bctx->slots[t_off];
        slot.offset = t_off;
        slot.nbytes = t_bytes;
        slot.ggml_dt = t->type;
        slot.storage = cloudtpu::TensorStorageKind::DENSE;
        slot.dtype = buf->dtype;
        slot.dims[0] = t->ne[3];
        slot.dims[1] = t->ne[2];
        slot.dims[2] = t->ne[1];
        slot.dims[3] = t->ne[0];
        slot.hbm_primary = buf;
        slot.hbm_scales.reset();
        std::vector<uint8_t>().swap(slot.staging);
        slot.staging_dirty = false;
        return;
    }

    if (t->data) {
        rt.buffer_to_host(buf->handle.get(), t->data, ggml_nbytes(t));
    }
}

static cloudtpu::OpNodeSpec build_op_node_spec(const struct ggml_tensor* node,
                                               uint64_t dst_id,
                                               const std::vector<uint64_t>& src_ids) {
    cloudtpu::OpNodeSpec spec;
    spec.src_ids = src_ids;
    spec.dst = tensor_to_spec(node, dst_id);

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            spec.kind = cloudtpu::CloudOpKind::MUL_MAT;
            const enum ggml_prec prec = ggml_prec_get_src(node, 1);
            spec.high_precision = (prec == GGML_PREC_F32);
            break;
        }
        case GGML_OP_RMS_NORM:
            spec.kind = cloudtpu::CloudOpKind::RMS_NORM;
            std::memcpy(&spec.eps, node->op_params, sizeof(float));
            break;
        case GGML_OP_ROPE: {
            spec.kind = cloudtpu::CloudOpKind::ROPE;
            const int32_t* p = node->op_params;
            spec.rope.n_dims = p[1];
            spec.rope.mode = p[2];
            spec.rope.n_ctx_orig = p[4];
            spec.rope.truncate = ggml_rope_get_truncate(node);
            std::memcpy(&spec.rope.freq_base, p + 5, sizeof(float));
            std::memcpy(&spec.rope.freq_scale, p + 6, sizeof(float));
            std::memcpy(&spec.rope.ext_factor, p + 7, sizeof(float));
            std::memcpy(&spec.rope.attn_factor, p + 8, sizeof(float));
            std::memcpy(&spec.rope.beta_fast, p + 9, sizeof(float));
            std::memcpy(&spec.rope.beta_slow, p + 10, sizeof(float));
            spec.rope.n_offs = p[15];
            break;
        }
        case GGML_OP_UNARY:
            spec.kind = cloudtpu::CloudOpKind::SILU;
            break;
        case GGML_OP_GLU:
            spec.kind = cloudtpu::CloudOpKind::SWIGLU;
            spec.glu_swapped = ggml_get_op_params_i32(node, 1);
            break;
        case GGML_OP_SOFT_MAX:
            spec.kind = cloudtpu::CloudOpKind::SOFT_MAX;
            std::memcpy(&spec.soft_max.scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
            std::memcpy(&spec.soft_max.max_bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
            break;
        case GGML_OP_GET_ROWS:
            spec.kind = cloudtpu::CloudOpKind::GET_ROWS;
            break;
        case GGML_OP_ADD:
            spec.kind = cloudtpu::CloudOpKind::ADD;
            break;
        case GGML_OP_MUL:
            spec.kind = cloudtpu::CloudOpKind::MUL;
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            spec.kind = cloudtpu::CloudOpKind::CPY;
            break;
        default:
            GGML_ABORT("ggml-cloudtpu: unexpected op %s", ggml_op_desc(node));
    }
    return spec;
}

static enum ggml_status execute_subgraph(ggml_backend_cloudtpu_context* ctx,
                                         cloudtpu::PjrtRuntime& rt,
                                         struct ggml_cgraph* cgraph,
                                         int start_idx,
                                         int end_idx) {
    // Identify which tensors produced inside [start_idx, end_idx) are consumed inside vs outside.
    std::unordered_set<const struct ggml_tensor*> produced_in_slice;
    std::unordered_set<const struct ggml_tensor*> consumed_inside;
    std::unordered_set<const struct ggml_tensor*> needed_outside;
    for (int i = start_idx; i < end_idx; ++i) {
        produced_in_slice.insert(cgraph->nodes[i]);
    }
    for (int i = start_idx; i < end_idx; ++i) {
        const struct ggml_tensor* n = cgraph->nodes[i];
        if (is_view_noop_op(n->op)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] && produced_in_slice.count(n->src[s])) {
                consumed_inside.insert(n->src[s]);
            }
            if (n->src[s] && n->src[s]->view_src && produced_in_slice.count(n->src[s]->view_src)) {
                consumed_inside.insert(n->src[s]->view_src);
            }
        }
    }
    for (int i = end_idx; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor* n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] && produced_in_slice.count(n->src[s])) {
                needed_outside.insert(n->src[s]);
            }
        }
        if (n->view_src && produced_in_slice.count(n->view_src)) {
            needed_outside.insert(n->view_src);
        }
    }

    cloudtpu::GraphProgramSpec prog_spec;
    std::unordered_map<const struct ggml_tensor*, uint64_t> tensor_to_id;
    std::unordered_map<uint64_t, const struct ggml_tensor*> id_to_input_tensor;
    std::unordered_map<uint64_t, struct ggml_tensor*> id_to_output_tensor;
    uint64_t next_id = 1;

    auto get_or_register_input = [&](auto& self, const struct ggml_tensor* t) -> uint64_t {
        if (!t) return 0;
        auto it = tensor_to_id.find(t);
        if (it != tensor_to_id.end()) {
            return it->second;
        }
        // Contiguous reshape/view at offset 0 with matching element count & type:
        // bind the underlying view_src tensor (matching its HBM PJRT_Buffer shape) and emit RESHAPE.
        if (t->view_src != nullptr && t->view_offs == 0 && ggml_is_contiguous(t) &&
            ggml_nelements(t) == ggml_nelements(t->view_src) &&
            is_dense_supported_type(t->type) && t->type == t->view_src->type) {
            const uint64_t base_id = self(self, t->view_src);
            const uint64_t rid = next_id++;
            cloudtpu::OpNodeSpec r_op;
            r_op.kind = cloudtpu::CloudOpKind::RESHAPE;
            r_op.src_ids = {base_id};
            r_op.dst = tensor_to_spec(t, rid);
            prog_spec.ops.push_back(std::move(r_op));
            tensor_to_id[t] = rid;
            return rid;
        }
        const uint64_t id = next_id++;
        tensor_to_id[t] = id;
        id_to_input_tensor[id] = t;
        prog_spec.inputs.push_back(tensor_to_spec(t, id));
        return id;
    };

    for (int i = start_idx; i < end_idx; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (is_view_noop_op(node->op)) {
            continue;
        }

        std::vector<uint64_t> src_ids;
        if (node->op == GGML_OP_ROPE || node->op == GGML_OP_SOFT_MAX) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            if (node->src[2]) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[2]));
            }
        } else if (node->op == GGML_OP_GLU) {
            src_ids.push_back(get_or_register_input(get_or_register_input, node->src[0]));
            if (node->src[1]) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[1]));
            }
        } else {
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                src_ids.push_back(get_or_register_input(get_or_register_input, node->src[s]));
            }
        }

        const uint64_t dst_id = next_id++;
        prog_spec.ops.push_back(build_op_node_spec(node, dst_id, src_ids));
        tensor_to_id[node] = dst_id;

        const bool is_live_out =
            (i == end_idx - 1) ||
            (consumed_inside.count(node) == 0) ||
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            needed_outside.count(node) != 0 ||
            (!node->buffer || !is_cloudtpu_buffer(node->buffer));
        if (is_live_out) {
            prog_spec.output_ids.push_back(dst_id);
            id_to_output_tensor[dst_id] = node;
        }
    }

    if (prog_spec.ops.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    const cloudtpu::EmittedProgram emitted = cloudtpu::emit_stablehlo_program(prog_spec);
    if (ctx->env.verbose >= 3) {
        GGML_LOG_INFO("ggml-cloudtpu: subgraph [%d..%d) inputs=%zu ops=%zu outs=%zu\n%s\n",
                      start_idx, end_idx,
                      emitted.input_bindings.size(), prog_spec.ops.size(), emitted.output_bindings.size(),
                      emitted.mlir_text.c_str());
    }
    const cloudtpu::CachedExecutable* exec = rt.compile_or_get(emitted.cache_key, emitted.mlir_text);

    // Resolve HBM PJRT_Buffer* inputs (keeping transient buffers alive through execute).
    std::vector<cloudtpu::PjrtBufferPtr> held_inputs;
    std::vector<PJRT_Buffer*> raw_inputs;
    held_inputs.reserve(emitted.input_bindings.size());
    raw_inputs.reserve(emitted.input_bindings.size());

    std::unordered_map<uint64_t, std::pair<cloudtpu::PjrtBufferPtr, cloudtpu::PjrtBufferPtr>> resolved_cache;
    for (const auto& ib : emitted.input_bindings) {
        auto rit = resolved_cache.find(ib.tensor_id);
        if (rit == resolved_cache.end()) {
            const struct ggml_tensor* t = id_to_input_tensor.at(ib.tensor_id);
            cloudtpu::PjrtBufferPtr p, s;
            resolve_tensor_hbm(rt, ctx->device_index, t, p, s);
            rit = resolved_cache.emplace(ib.tensor_id, std::make_pair(std::move(p), std::move(s))).first;
        }
        cloudtpu::PjrtBufferPtr chosen = (ib.part == 0) ? rit->second.first : rit->second.second;
        raw_inputs.push_back(chosen->handle.get());
        held_inputs.push_back(std::move(chosen));
    }

    std::vector<cloudtpu::PjrtDType> out_dtypes;
    std::vector<std::vector<int64_t>> out_dims;
    out_dtypes.reserve(emitted.output_bindings.size());
    out_dims.reserve(emitted.output_bindings.size());
    for (const auto& ob : emitted.output_bindings) {
        out_dtypes.push_back(ob.dtype);
        out_dims.push_back(ob.dims);
    }

    std::vector<cloudtpu::PjrtBufferPtr> outputs;
    const bool sync_exec = (end_idx == cgraph->n_nodes) || !ctx->env.fuse_graphs;
    rt.execute(exec,
               ctx->device_index,
               raw_inputs.data(),
               raw_inputs.size(),
               outputs,
               out_dtypes,
               out_dims,
               sync_exec);

    for (size_t oi = 0; oi < emitted.output_bindings.size(); ++oi) {
        struct ggml_tensor* out_t = id_to_output_tensor.at(emitted.output_bindings[oi].tensor_id);
        store_tensor_hbm(rt, out_t, outputs[oi]);
    }

    ctx->total_subgraphs += 1;
    ctx->total_ops_executed += prog_spec.ops.size();
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cloudtpu_graph_compute(ggml_backend_t backend,
                                                            struct ggml_cgraph* cgraph) {
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    ctx->total_graph_evals += 1;

    try {
        int i = 0;
        while (i < cgraph->n_nodes) {
            if (is_view_noop_op(cgraph->nodes[i]->op)) {
                ++i;
                continue;
            }
            if (!ggml_backend_cloudtpu_device_supports_op_impl(cgraph->nodes[i])) {
                GGML_LOG_ERROR("ggml-cloudtpu: unsupported op %s in graph_compute\n",
                               ggml_op_desc(cgraph->nodes[i]));
                return GGML_STATUS_FAILED;
            }
            int j = i + 1;
            if (ctx->env.fuse_graphs) {
                while (j < cgraph->n_nodes &&
                       ggml_backend_cloudtpu_device_supports_op_impl(cgraph->nodes[j])) {
                    // Stop fusing at a view_noop only if it breaks contiguous zero-offset aliasing;
                    // RESHAPE/VIEW/etc. without matching simple reshape can stay outside or break cleanly.
                    if (is_view_noop_op(cgraph->nodes[j]->op)) {
                        const struct ggml_tensor* vn = cgraph->nodes[j];
                        if (vn->op != GGML_OP_NONE && vn->op != GGML_OP_RESHAPE &&
                            !(vn->op == GGML_OP_VIEW && vn->view_offs == 0 && ggml_is_contiguous(vn))) {
                            break;
                        }
                    }
                    ++j;
                }
                while (j > i + 1 && is_view_noop_op(cgraph->nodes[j - 1]->op)) {
                    --j;
                }
            }
            const enum ggml_status st = execute_subgraph(ctx, rt, cgraph, i, j);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
            i = j;
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: graph_compute error: %s\n", e.what());
        return GGML_STATUS_FAILED;
    }
}

static const char* ggml_backend_cloudtpu_get_name(ggml_backend_t backend) {
    return ggml_backend_dev_name(backend->device);
}

static void ggml_backend_cloudtpu_free(ggml_backend_t backend) {
    auto* ctx = static_cast<ggml_backend_cloudtpu_context*>(backend->context);
    delete ctx;
    delete backend;
}

static struct ggml_backend_i ggml_backend_cloudtpu_i = {
    /* .get_name            = */ ggml_backend_cloudtpu_get_name,
    /* .free                = */ ggml_backend_cloudtpu_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_cloudtpu_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_cloudtpu_guid(void) {
    static const uint8_t guid[16] = {
        0x43, 0x4c, 0x4f, 0x55, 0x44, 0x54, 0x50, 0x55, // "CLOUDTPU"
        0x50, 0x4a, 0x52, 0x54, 0x00, 0x00, 0x00, 0x01  // "PJRT\0\0\0\1"
    };
    return const_cast<ggml_guid_t>(&guid);
}

// =========================================================================
// Device & Registry Interface
// =========================================================================

struct cloudtpu_dev_context {
    int device_index = 0;
    std::string name;
    std::string description;
};

static const char* ggml_backend_cloudtpu_device_get_name(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return dc->name.c_str();
}

static const char* ggml_backend_cloudtpu_device_get_description(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return dc->description.c_str();
}

static void ggml_backend_cloudtpu_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
    if (!rt.query_memory_stats(dc->device_index, free, total)) {
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_cloudtpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cloudtpu_device_get_props(ggml_backend_dev_t dev,
                                                   struct ggml_backend_dev_props* props) {
    props->name = ggml_backend_cloudtpu_device_get_name(dev);
    props->description = ggml_backend_cloudtpu_device_get_description(dev);
    props->type = ggml_backend_cloudtpu_device_get_type(dev);
    ggml_backend_cloudtpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_cloudtpu_device_init_backend(ggml_backend_dev_t dev,
                                                                const char* params) {
    GGML_UNUSED(params);
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return ggml_backend_cloudtpu_init_device(dc->device_index);
}

static ggml_backend_buffer_type_t ggml_backend_cloudtpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return ggml_backend_cloudtpu_buffer_type(dc->device_index);
}

static bool ggml_backend_cloudtpu_device_supports_op(ggml_backend_dev_t dev,
                                                     const struct ggml_tensor* op) {
    GGML_UNUSED(dev);
    return ggml_backend_cloudtpu_device_supports_op_impl(op);
}

static bool ggml_backend_cloudtpu_device_supports_buft(ggml_backend_dev_t dev,
                                                       ggml_backend_buffer_type_t buft) {
    auto* dc = static_cast<cloudtpu_dev_context*>(dev->context);
    return buft == ggml_backend_cloudtpu_buffer_type(dc->device_index);
}

static const struct ggml_backend_device_i ggml_backend_cloudtpu_device_i = {
    /* .get_name             = */ ggml_backend_cloudtpu_device_get_name,
    /* .get_description      = */ ggml_backend_cloudtpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cloudtpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cloudtpu_device_get_type,
    /* .get_props            = */ ggml_backend_cloudtpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cloudtpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cloudtpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_cloudtpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cloudtpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct cloudtpu_reg_state {
    std::mutex mu;
    bool initialized = false;
    std::vector<std::unique_ptr<cloudtpu_dev_context>> dev_ctxs;
    std::vector<ggml_backend_device> devices;
    std::vector<std::unique_ptr<cloudtpu_buft_context>> buft_ctxs;
    std::vector<ggml_backend_buffer_type> bufts;
};

static cloudtpu_reg_state& get_reg_state(ggml_backend_reg_t reg) {
    static cloudtpu_reg_state state;
    std::lock_guard<std::mutex> lock(state.mu);
    if (!state.initialized) {
        try {
            cloudtpu::PjrtRuntime& rt = ensure_runtime_initialized();
            const int count = std::max(1, rt.device_count());
            state.dev_ctxs.reserve(static_cast<size_t>(count));
            state.devices.resize(static_cast<size_t>(count));
            state.buft_ctxs.reserve(static_cast<size_t>(count));
            state.bufts.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                auto dc = std::make_unique<cloudtpu_dev_context>();
                dc->device_index = i;
                dc->name = (i == 0) ? "CLOUDTPU" : ("CLOUDTPU" + std::to_string(i));
                if (i < rt.device_count()) {
                    dc->description = rt.device_info(i).kind;
                } else {
                    dc->description = "Google Cloud TPU";
                }

                state.devices[static_cast<size_t>(i)] = {
                    /* .iface   = */ ggml_backend_cloudtpu_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ dc.get(),
                };
                state.dev_ctxs.push_back(std::move(dc));

                auto btc = std::make_unique<cloudtpu_buft_context>();
                btc->device_index = i;
                btc->name = state.dev_ctxs[static_cast<size_t>(i)]->name;
                state.bufts[static_cast<size_t>(i)] = {
                    /* .iface   = */ {
                        /* .get_name         = */ ggml_backend_cloudtpu_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_cloudtpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n   = */ NULL,
                        /* .get_alignment    = */ ggml_backend_cloudtpu_buffer_type_get_alignment,
                        /* .get_max_size     = */ NULL,
                        /* .get_alloc_size   = */ NULL,
                        /* .get_alloc_size_n = */ NULL,
                        /* .is_host          = */ ggml_backend_cloudtpu_buffer_type_is_host,
                    },
                    /* .device  = */ &state.devices[static_cast<size_t>(i)],
                    /* .context = */ btc.get(),
                };
                state.buft_ctxs.push_back(std::move(btc));
            }
        } catch (const std::exception& e) {
            GGML_LOG_ERROR("ggml-cloudtpu: failed to initialize PJRT runtime (%s); 0 devices exposed\n",
                           e.what());
            state.devices.clear();
            state.bufts.clear();
        }
        state.initialized = true;
    }
    return state;
}

static const char* ggml_backend_cloudtpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "CLOUDTPU";
}

static size_t ggml_backend_cloudtpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return get_reg_state(reg).devices.size();
}

static ggml_backend_dev_t ggml_backend_cloudtpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    cloudtpu_reg_state& st = get_reg_state(reg);
    GGML_ASSERT(index < st.devices.size());
    return &st.devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_cloudtpu_reg_i = {
    /* .get_name         = */ ggml_backend_cloudtpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cloudtpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cloudtpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

}  // namespace

// =========================================================================
// Public C API
// =========================================================================

int ggml_backend_cloudtpu_get_device_count(void) {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_cloudtpu_reg()));
}

ggml_backend_buffer_type_t ggml_backend_cloudtpu_buffer_type(int device_index) {
    cloudtpu_reg_state& st = get_reg_state(ggml_backend_cloudtpu_reg());
    if (device_index < 0 || static_cast<size_t>(device_index) >= st.bufts.size()) {
        return nullptr;
    }
    return &st.bufts[static_cast<size_t>(device_index)];
}

ggml_backend_t ggml_backend_cloudtpu_init_device(int device_index) {
    const cloudtpu_env_config env = load_env_config();
    g_hot_env.store(env);

    cloudtpu::PjrtRuntime* rt_ptr = nullptr;
    try {
        rt_ptr = &ensure_runtime_initialized();
    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ggml-cloudtpu: runtime init failed: %s\n", e.what());
        return nullptr;
    }
    cloudtpu::PjrtRuntime& rt = *rt_ptr;
    if (device_index < 0 || device_index >= rt.device_count()) {
        GGML_LOG_ERROR("ggml-cloudtpu: invalid device_index=%d (device_count=%d)\n",
                       device_index, rt.device_count());
        return nullptr;
    }

    auto* ctx = new ggml_backend_cloudtpu_context();
    ctx->device_index = device_index;
    ctx->env = env;

    if (env.verbose > 0) {
        const auto& info = rt.device_info(device_index);
        GGML_LOG_INFO(
            "ggml-cloudtpu: initialized device %d (%s, platform=%s %s, mock=%d, fuse=%d, hbm_limit=%lld)\n",
            device_index,
            info.kind.c_str(),
            rt.platform_name().c_str(),
            rt.platform_version().c_str(),
            rt.is_mock() ? 1 : 0,
            env.fuse_graphs ? 1 : 0,
            static_cast<long long>(info.hbm_bytes_limit));
    }

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_cloudtpu_guid(),
        /* .iface   = */ ggml_backend_cloudtpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cloudtpu_reg(), static_cast<size_t>(device_index)),
        /* .context = */ ctx,
    };
    return backend;
}

ggml_backend_t ggml_backend_cloudtpu_init(void) {
    return ggml_backend_cloudtpu_init_device(0);
}

bool ggml_backend_is_cloudtpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cloudtpu_guid());
}

bool ggml_backend_cloudtpu_is_mock(ggml_backend_t backend) {
    if (!ggml_backend_is_cloudtpu(backend)) {
        return true;
    }
    cloudtpu::PjrtRuntime& rt = cloudtpu::shared_pjrt_runtime(nullptr);
    return !rt.is_initialized() || rt.is_mock();
}

ggml_backend_reg_t ggml_backend_cloudtpu_reg(void) {
    static struct ggml_backend_reg ggml_backend_cloudtpu_reg_s = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cloudtpu_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_cloudtpu_reg_s;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cloudtpu_reg)
