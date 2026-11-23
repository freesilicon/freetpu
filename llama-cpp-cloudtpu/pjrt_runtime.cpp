#include "pjrt_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace cloudtpu {

#define PJRT_API_HAS(api, field)                                            \
    ((api) != nullptr &&                                                    \
     (offsetof(PJRT_Api, field) + sizeof((api)->field) <= (api)->struct_size) && \
     ((api)->field != nullptr))

// =========================================================================
// Scalar Conversions & Type Helpers
// =========================================================================

uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<uint16_t>((u >> 16) | 0x0040U);
    }
    const uint32_t lsb = (u >> 16) & 1U;
    u += 0x7fffU + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    uint32_t mant = x & 0x007fffffU;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;

    if (((x >> 23) & 0xffU) == 0xffU) {
        if (mant != 0) {
            return static_cast<uint16_t>(sign | 0x7e00U | (mant >> 13));
        }
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t shift = static_cast<uint32_t>(1 - exp);
        const uint32_t full = mant | 0x00800000U;
        const uint32_t sticky = ((full & ((1U << shift) - 1U)) != 0U) ? 1U : 0U;
        mant = (full >> shift) | sticky;
        const uint32_t round_bits = mant & 0x00001fffU;
        const uint32_t lsb = (mant >> 13) & 1U;
        if (round_bits > 0x00001000U || (round_bits == 0x00001000U && lsb != 0)) {
            mant += 0x00002000U;
        }
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    const uint32_t round_bits = mant & 0x00001fffU;
    const uint32_t lsb = (mant >> 13) & 1U;
    if (round_bits > 0x00001000U || (round_bits == 0x00001000U && lsb != 0)) {
        mant += 0x00002000U;
        if ((mant & 0x00800000U) != 0) {
            mant = 0;
            exp += 1;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00U);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1fU;
    uint32_t mant = h & 0x03ffU;
    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ffU;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint8_t f32_to_f8e4m3fn(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint8_t sign = static_cast<uint8_t>((x >> 24) & 0x80U);
    const uint32_t abs_x = x & 0x7fffffffU;
    // NaN -> 0x7F (preserving sign)
    if (abs_x > 0x7f800000U) {
        return static_cast<uint8_t>(sign | 0x7fU);
    }
    // Inf or magnitude >= 448.0 saturates to finite max 0x7E (448.0) per OCP E4M3FN clamping.
    if (abs_x >= 0x43e00000U) {
        return static_cast<uint8_t>(sign | 0x7eU);
    }
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 7;
    uint32_t mant = x & 0x007fffffU;
    if (exp <= 0) {
        // Subnormal or underflow to zero (min positive subnormal = 2^-9 = 0x3b000000).
        if (exp < -3) {
            return sign;
        }
        // value = full * 2^(exp - 30) = (full / 2^(21 - exp)) * 2^-9, with total_shift in [21, 24].
        const uint32_t total_shift = static_cast<uint32_t>(21 - exp);
        const uint32_t full = mant | 0x00800000U;
        uint32_t q_mant = full >> total_shift;
        const uint32_t rem = full & ((1U << total_shift) - 1U);
        const uint32_t half = 1U << (total_shift - 1U);
        if (rem > half || (rem == half && (q_mant & 1U) != 0U)) {
            q_mant += 1U;
        }
        if (q_mant >= 8U) {
            // Rounded up from largest subnormal into smallest normal (exp=1, mant=0).
            return static_cast<uint8_t>(sign | 0x08U);
        }
        return static_cast<uint8_t>(sign | q_mant);
    }
    const uint32_t round_bits = mant & 0x000fffffU;
    uint32_t q_mant = mant >> 20;
    const uint32_t lsb = q_mant & 1U;
    if (round_bits > 0x00080000U || (round_bits == 0x00080000U && lsb != 0U)) {
        q_mant += 1U;
        if (q_mant >= 8U) {
            q_mant = 0U;
            exp += 1;
        }
    }
    if (exp > 15 || (exp == 15 && q_mant >= 7U)) {
        return static_cast<uint8_t>(sign | 0x7eU);
    }
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | q_mant);
}

float f8e4m3fn_to_f32(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80U) << 24;
    uint32_t exp = (b >> 3) & 0x0fU;
    uint32_t mant = b & 0x07U;
    uint32_t out = 0;
    if (exp == 0x0fU && mant == 0x07U) {
        out = sign | 0x7fc00000U;
    } else if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else {
            exp = 1U;
            while ((mant & 0x08U) == 0U) {
                mant <<= 1;
                exp -= 1U;
            }
            mant &= 0x07U;
            out = sign | ((exp + 127U - 7U) << 23) | (mant << 20);
        }
    } else {
        out = sign | ((exp + 127U - 7U) << 23) | (mant << 20);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

size_t count_libpython_mappings() {
    std::ifstream ifs("/proc/self/maps");
    if (!ifs.is_open()) {
        return 0;
    }
    std::string line;
    size_t count = 0;
    while (std::getline(ifs, line)) {
        if (line.find("libpython") != std::string::npos) {
            ++count;
        }
    }
    return count;
}

PJRT_Buffer_Type to_pjrt_buffer_type(PjrtDType dt) {
    switch (dt) {
        case PjrtDType::PRED: return PJRT_Buffer_Type_PRED;
        case PjrtDType::F32: return PJRT_Buffer_Type_F32;
        case PjrtDType::F16: return PJRT_Buffer_Type_F16;
        case PjrtDType::BF16: return PJRT_Buffer_Type_BF16;
        case PjrtDType::F8E4M3FN: return PJRT_Buffer_Type_F8E4M3FN;
        case PjrtDType::S8: return PJRT_Buffer_Type_S8;
        case PjrtDType::U8: return PJRT_Buffer_Type_U8;
        case PjrtDType::S4: return PJRT_Buffer_Type_S4;
        case PjrtDType::U4: return PJRT_Buffer_Type_U4;
        case PjrtDType::S16: return PJRT_Buffer_Type_S16;
        case PjrtDType::S32: return PJRT_Buffer_Type_S32;
        case PjrtDType::U32: return PJRT_Buffer_Type_U32;
        case PjrtDType::S64: return PJRT_Buffer_Type_S64;
        default: return PJRT_Buffer_Type_INVALID;
    }
}

PjrtDType from_pjrt_buffer_type(PJRT_Buffer_Type dt) {
    switch (dt) {
        case PJRT_Buffer_Type_PRED: return PjrtDType::PRED;
        case PJRT_Buffer_Type_F32: return PjrtDType::F32;
        case PJRT_Buffer_Type_F16: return PjrtDType::F16;
        case PJRT_Buffer_Type_BF16: return PjrtDType::BF16;
        case PJRT_Buffer_Type_F8E4M3FN: return PjrtDType::F8E4M3FN;
        case PJRT_Buffer_Type_S8: return PjrtDType::S8;
        case PJRT_Buffer_Type_U8: return PjrtDType::U8;
        case PJRT_Buffer_Type_S4: return PjrtDType::S4;
        case PJRT_Buffer_Type_U4: return PjrtDType::U4;
        case PJRT_Buffer_Type_S16: return PjrtDType::S16;
        case PJRT_Buffer_Type_S32: return PjrtDType::S32;
        case PJRT_Buffer_Type_U32: return PjrtDType::U32;
        case PJRT_Buffer_Type_S64: return PjrtDType::S64;
        default: return PjrtDType::INVALID;
    }
}

const char* pjrt_dtype_mlir_name(PjrtDType dt) {
    switch (dt) {
        case PjrtDType::PRED: return "i1";
        case PjrtDType::F32: return "f32";
        case PjrtDType::F16: return "f16";
        case PjrtDType::BF16: return "bf16";
        case PjrtDType::F8E4M3FN: return "f8E4M3FN";
        case PjrtDType::S8: return "i8";
        case PjrtDType::U8: return "ui8";
        case PjrtDType::S4: return "i4";
        case PjrtDType::U4: return "ui4";
        case PjrtDType::S16: return "i16";
        case PjrtDType::S32: return "i32";
        case PjrtDType::U32: return "ui32";
        case PjrtDType::S64: return "i64";
        default: return "invalid";
    }
}

size_t pjrt_dtype_bits(PjrtDType dt) {
    switch (dt) {
        case PjrtDType::S64: return 64;
        case PjrtDType::F32:
        case PjrtDType::S32:
        case PjrtDType::U32: return 32;
        case PjrtDType::F16:
        case PjrtDType::BF16:
        case PjrtDType::S16: return 16;
        case PjrtDType::PRED:
        case PjrtDType::F8E4M3FN:
        case PjrtDType::S8:
        case PjrtDType::U8: return 8;
        case PjrtDType::S4:
        case PjrtDType::U4: return 4;
        default: return 0;
    }
}

size_t pjrt_dtype_size(PjrtDType dt) {
    return pjrt_dtype_bits(dt) / 8;
}

size_t pjrt_dtype_nbytes(PjrtDType dt, size_t num_elements) {
    const size_t bits = pjrt_dtype_bits(dt);
    if (bits == 0 || num_elements == 0) {
        return 0;
    }
    return (num_elements / 8) * bits + ((num_elements % 8) * bits + 7) / 8;
}

size_t pjrt_host_nbytes(PjrtDType dt, size_t num_elements) {
    const size_t bits = pjrt_dtype_bits(dt);
    if (bits == 0 || num_elements == 0) {
        return 0;
    }
    // OpenXLA PJRT BufferFromHostBuffer(byte_strides=nullptr) reads 1 byte per element for S4/U4.
    const size_t host_elem_bytes = (bits < 8) ? 1 : (bits / 8);
    return num_elements * host_elem_bytes;
}

static inline int8_t sign_extend_nibble(uint8_t nib) {
    const uint8_t u = nib & 0x0FU;
    return static_cast<int8_t>((u & 0x08U) ? (static_cast<int>(u) - 16) : static_cast<int>(u));
}

void pack_u4_from_u8(const uint8_t* src, size_t nelems, uint8_t* dst_packed) {
    const size_t pairs = nelems / 2;
    for (size_t i = 0; i < pairs; ++i) {
        const uint8_t lo = src[2 * i] & 0x0FU;
        const uint8_t hi = src[2 * i + 1] & 0x0FU;
        dst_packed[i] = static_cast<uint8_t>(lo | (hi << 4));
    }
    if ((nelems & 1U) != 0U) {
        dst_packed[pairs] = src[nelems - 1] & 0x0FU;
    }
}

void unpack_u4_to_u8(const uint8_t* src_packed, size_t nelems, uint8_t* dst) {
    const size_t pairs = nelems / 2;
    for (size_t i = 0; i < pairs; ++i) {
        const uint8_t b = src_packed[i];
        dst[2 * i] = b & 0x0FU;
        dst[2 * i + 1] = (b >> 4) & 0x0FU;
    }
    if ((nelems & 1U) != 0U) {
        dst[nelems - 1] = src_packed[pairs] & 0x0FU;
    }
}

void pack_s4_from_i8(const int8_t* src, size_t nelems, uint8_t* dst_packed) {
    pack_u4_from_u8(reinterpret_cast<const uint8_t*>(src), nelems, dst_packed);
}

void unpack_s4_to_i8(const uint8_t* src_packed, size_t nelems, int8_t* dst) {
    const size_t pairs = nelems / 2;
    for (size_t i = 0; i < pairs; ++i) {
        const uint8_t b = src_packed[i];
        dst[2 * i] = sign_extend_nibble(b);
        dst[2 * i + 1] = sign_extend_nibble(b >> 4);
    }
    if ((nelems & 1U) != 0U) {
        dst[nelems - 1] = sign_extend_nibble(src_packed[pairs]);
    }
}

// =========================================================================
// Pure C++ CompileOptionsProto Encoder
// =========================================================================

namespace {

void append_varint(std::string& out, uint64_t val) {
    while (val >= 0x80ULL) {
        out.push_back(static_cast<char>((val & 0x7fULL) | 0x80ULL));
        val >>= 7;
    }
    out.push_back(static_cast<char>(val));
}

void append_tag_varint(std::string& out, uint32_t field_number, uint64_t val) {
    append_varint(out, (static_cast<uint64_t>(field_number) << 3) | 0ULL);
    append_varint(out, val);
}

void append_tag_bytes(std::string& out, uint32_t field_number, const std::string& payload) {
    append_varint(out, (static_cast<uint64_t>(field_number) << 3) | 2ULL);
    append_varint(out, payload.size());
    out.append(payload);
}

void append_tag_fixed64(std::string& out, uint32_t field_number, uint64_t val) {
    append_varint(out, (static_cast<uint64_t>(field_number) << 3) | 1ULL);
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((val >> (i * 8)) & 0xFFU));
    }
}

}  // namespace

std::string encode_compile_options_proto(const PjrtCompileOptions& opts) {
    // xla::ExecutableBuildOptionsProto (openxla/xla/xla/pjrt/proto/compile_options.proto)
    // Intentionally omit field 3 (debug_options) so OpenXLA preserves compiler defaults
    // and XLA_FLAGS (e.g. --xla_dump_to); pass flag overrides via CompileOptionsProto
    // field 7 (env_option_overrides) instead.
    std::string ebo;
    if (opts.num_replicas > 0) {
        append_tag_varint(ebo, 4, static_cast<uint64_t>(opts.num_replicas));
    }
    if (opts.num_partitions > 0) {
        append_tag_varint(ebo, 5, static_cast<uint64_t>(opts.num_partitions));
    }
    if (opts.use_spmd_partitioning) {
        append_tag_varint(ebo, 6, 1);
    }
    if (!opts.device_assignment_ids.empty()) {
        // xla::DeviceAssignmentProto (field 9 of ExecutableBuildOptionsProto):
        //   int32 replica_count = 1;
        //   int32 computation_count = 2;
        //   repeated ComputationDevice computation_devices = 3;
        //     message ComputationDevice { repeated int64 replica_device_ids = 1; }
        const int32_t replicas = std::max<int32_t>(1, opts.num_replicas);
        const int32_t computations = std::max<int32_t>(1, opts.num_partitions);
        if (opts.device_assignment_ids.size() !=
            static_cast<size_t>(replicas) * static_cast<size_t>(computations)) {
            throw std::runtime_error("encode_compile_options_proto: device_assignment_ids size mismatch");
        }
        std::string da;
        append_tag_varint(da, 1, static_cast<uint64_t>(replicas));
        append_tag_varint(da, 2, static_cast<uint64_t>(computations));
        for (int32_t p = 0; p < computations; ++p) {
            std::string packed_ids;
            for (int32_t r = 0; r < replicas; ++r) {
                const int64_t dev_id = opts.device_assignment_ids[static_cast<size_t>(p * replicas + r)];
                append_varint(packed_ids, static_cast<uint64_t>(dev_id));
            }
            std::string comp_dev;
            append_tag_bytes(comp_dev, 1, packed_ids);
            append_tag_bytes(da, 3, comp_dev);
        }
        append_tag_bytes(ebo, 9, da);
    }
    if (opts.alias_passthrough_params) {
        append_tag_varint(ebo, 10, 1);
    }
    if (opts.use_shardy_partitioner) {
        append_tag_varint(ebo, 19, 1);
    }

    // xla::CompileOptionsProto {
    //   ExecutableBuildOptionsProto executable_build_options = 3;
    //   map<string, OptionOverrideProto> env_option_overrides = 7;
    // }
    std::string out;
    append_tag_bytes(out, 3, ebo);
    for (const auto& ov : opts.env_option_overrides) {
        // OptionOverrideProto {
        //   oneof value { string string_field = 1; bool bool_field = 2; int64 int_field = 3; double double_field = 4; }
        // }
        std::string override_val;
        switch (ov.kind) {
            case PjrtOptionOverride::Kind::STRING:
                append_tag_bytes(override_val, 1, ov.string_val);
                break;
            case PjrtOptionOverride::Kind::BOOL:
                append_tag_varint(override_val, 2, ov.bool_val ? 1ULL : 0ULL);
                break;
            case PjrtOptionOverride::Kind::INT64:
                append_tag_varint(override_val, 3, static_cast<uint64_t>(ov.int_val));
                break;
            case PjrtOptionOverride::Kind::DOUBLE: {
                uint64_t bits = 0;
                std::memcpy(&bits, &ov.double_val, sizeof(bits));
                append_tag_fixed64(override_val, 4, bits);
                break;
            }
        }
        // map<string, OptionOverrideProto> entry { string key = 1; OptionOverrideProto value = 2; }
        std::string entry;
        append_tag_bytes(entry, 1, ov.key);
        append_tag_bytes(entry, 2, override_val);
        append_tag_bytes(out, 7, entry);
    }
    return out;
}

std::string append_async_collective_libtpu_init_args(const std::string& existing_args) {
    // Documented TPU async collective overlap flags for LIBTPU_INIT_ARGS (TorchTPU / MaxText
    // v5e/v6e reference). These are libtpu initialization flags, NOT xla::DebugOptions fields,
    // so they must be supplied via LIBTPU_INIT_ARGS before libtpu.so loads rather than via
    // CompileOptionsProto.env_option_overrides.
    static const std::pair<const char*, const char*> kFlags[] = {
        {"xla_tpu_enable_async_collective_fusion", "true"},
        {"xla_tpu_enable_async_collective_fusion_fuse_all_gather", "true"},
        {"xla_tpu_enable_async_collective_fusion_multiple_steps", "true"},
        {"xla_tpu_overlap_compute_collective_tc", "true"},
        {"xla_enable_async_all_gather", "true"},
    };

    std::unordered_set<std::string> present;
    {
        std::istringstream iss(existing_args);
        std::string tok;
        while (iss >> tok) {
            size_t start = 0;
            while (start < tok.size() && tok[start] == '-') {
                ++start;
            }
            if (start == 0 || start >= tok.size()) {
                continue;
            }
            const size_t eq = tok.find('=', start);
            std::string name = (eq == std::string::npos) ? tok.substr(start) : tok.substr(start, eq - start);
            present.insert(name);
            if (name.size() > 2 && name.compare(0, 2, "no") == 0) {
                present.insert(name.substr(2));
            }
        }
    }

    std::string merged = existing_args;
    while (!merged.empty() && std::isspace(static_cast<unsigned char>(merged.back()))) {
        merged.pop_back();
    }
    for (const auto& kv : kFlags) {
        if (present.count(kv.first) != 0) {
            continue;
        }
        if (!merged.empty()) {
            merged.push_back(' ');
        }
        merged.append("--");
        merged.append(kv.first);
        merged.push_back('=');
        merged.append(kv.second);
        present.insert(kv.first);
    }
    return merged;
}

// =========================================================================
// PjrtBuffer & PjrtLoadedExecutable Helpers
// =========================================================================

void PjrtLoadedExecutableDeleter::operator()(PJRT_LoadedExecutable* exec) const noexcept {
    if (!exec || !api || !PJRT_API_HAS(api, PJRT_LoadedExecutable_Destroy)) {
        return;
    }
    PJRT_LoadedExecutable_Destroy_Args args = {};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = exec;
    PJRT_Error* err = api->PJRT_LoadedExecutable_Destroy(&args);
    if (err && PJRT_API_HAS(api, PJRT_Error_Destroy)) {
        PJRT_Error_Destroy_Args d_args = {};
        d_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        d_args.error = err;
        api->PJRT_Error_Destroy(&d_args);
    }
}

void PjrtBufferDeleter::operator()(PJRT_Buffer* buf) const noexcept {
    if (!buf || !api || !PJRT_API_HAS(api, PJRT_Buffer_Destroy)) {
        return;
    }
    PJRT_Buffer_Destroy_Args args = {};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buf;
    PJRT_Error* err = api->PJRT_Buffer_Destroy(&args);
    if (err && PJRT_API_HAS(api, PJRT_Error_Destroy)) {
        PJRT_Error_Destroy_Args d_args = {};
        d_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        d_args.error = err;
        api->PJRT_Error_Destroy(&d_args);
    }
}

void PjrtEventDeleter::operator()(PJRT_Event* ev) const noexcept {
    if (!ev || !api || !PJRT_API_HAS(api, PJRT_Event_Destroy)) {
        return;
    }
    PJRT_Event_Destroy_Args args = {};
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.event = ev;
    PJRT_Error* err = api->PJRT_Event_Destroy(&args);
    if (err && PJRT_API_HAS(api, PJRT_Error_Destroy)) {
        PJRT_Error_Destroy_Args d_args = {};
        d_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        d_args.error = err;
        api->PJRT_Error_Destroy(&d_args);
    }
}

size_t PjrtBuffer::num_elements() const {
    size_t n = 1;
    for (int64_t d : dims) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

size_t PjrtBuffer::num_bytes() const {
    return pjrt_dtype_nbytes(dtype, num_elements());
}

MockTopologyProfile parse_mock_topology_profile(const char* val) {
    if (!val || val[0] == '\0') {
        return MockTopologyProfile::AUTO;
    }
    std::string s;
    for (const char* p = val; *p; ++p) {
        s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
    if (s == "auto") {
        return MockTopologyProfile::AUTO;
    }
    if (s == "v5e" || s == "v5e-1" || s == "v5e_1" || s == "v5litepod-1") {
        return MockTopologyProfile::V5E_1;
    }
    if (s == "v6e" || s == "v6e-8" || s == "v6e_8") {
        return MockTopologyProfile::V6E_8;
    }
    if (s == "tpu7x-1t" || s == "tpu7x_1t" || s == "tpu7x-standard-1t" || s == "1t" ||
        s == "ironwood-1t") {
        return MockTopologyProfile::TPU7X_1T;
    }
    if (s == "tpu7x-4t" || s == "tpu7x_4t" || s == "tpu7x-standard-4t" || s == "4t" ||
        s == "tpu7x" || s == "ironwood" || s == "ironwood-4t") {
        return MockTopologyProfile::TPU7X_4T;
    }
    return MockTopologyProfile::AUTO;
}

bool is_tpu7x_device_kind(const std::string& kind) {
    std::string lower;
    lower.reserve(kind.size());
    for (unsigned char c : kind) {
        lower.push_back(static_cast<char>(std::tolower(c)));
    }
    return lower.find("tpu7x") != std::string::npos ||
           lower.find("tpu v7") != std::string::npos ||
           lower.find("tpu7") != std::string::npos ||
           lower.find("ironwood") != std::string::npos;
}

std::vector<int> compute_d2d_hamiltonian_order(const std::vector<PjrtDeviceInfo>& infos) {
    const int n = static_cast<int>(infos.size());
    std::vector<int> identity;
    identity.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        identity.push_back(i);
    }
    if (n <= 1) {
        return identity;
    }

    bool any_coords = false;
    bool any_core = false;
    bool any_tpu7x = false;
    for (const auto& info : infos) {
        if (!info.coords.empty()) {
            any_coords = true;
        }
        if (info.core_on_chip >= 0 || info.is_dual_chiplet) {
            any_core = true;
        }
        if (is_tpu7x_device_kind(info.kind)) {
            any_tpu7x = true;
        }
    }

    struct PkgKey {
        std::vector<int64_t> coords;
        bool operator<(const PkgKey& o) const { return coords < o.coords; }
        bool operator==(const PkgKey& o) const { return coords == o.coords; }
    };

    // Group logical devices into physical chip packages so intra-package chiplets
    // remain consecutive D2D pairs (0,1), (2,3), (4,5), (6,7), avoiding the 6x
    // cross-package ICI penalty when pairing ranks.
    struct PkgGroup {
        std::vector<int64_t> coords;
        std::vector<int> member_indices;
    };
    std::vector<PkgGroup> groups;

    auto find_or_add_group = [&](const std::vector<int64_t>& key) -> PkgGroup& {
        for (auto& g : groups) {
            if (g.coords == key) {
                return g;
            }
        }
        groups.push_back(PkgGroup{key, {}});
        return groups.back();
    };

    for (int i = 0; i < n; ++i) {
        const auto& info = infos[static_cast<size_t>(i)];
        std::vector<int64_t> key;
        if (any_coords && !info.coords.empty()) {
            key = info.coords;
        } else if (any_tpu7x || any_core) {
            key = {info.id / 2};
        } else {
            key = {info.id};
        }
        find_or_add_group(key).member_indices.push_back(i);
    }

    bool multi_member_group = false;
    for (const auto& g : groups) {
        if (g.member_indices.size() >= 2) {
            multi_member_group = true;
            break;
        }
    }

    // If hardware didn't populate coordinates/core_on_chip yet for tpu7x, pair by id/2.
    if (!multi_member_group && (any_tpu7x || any_core) && (n % 2 == 0)) {
        groups.clear();
        for (int i = 0; i < n; ++i) {
            const auto& info = infos[static_cast<size_t>(i)];
            find_or_add_group({info.id / 2}).member_indices.push_back(i);
        }
    }

    // Order physical packages along a serpentine Hamiltonian torus path:
    // For (x, y, z): sort by z ascending; within z, flip y on odd z; within (y, z), flip x on odd (y ^ z).
    std::sort(groups.begin(), groups.end(), [&](const PkgGroup& a, const PkgGroup& b) {
        const size_t dims = std::min(a.coords.size(), b.coords.size());
        if (dims >= 3) {
            const int64_t az = a.coords[2];
            const int64_t bz = b.coords[2];
            if (az != bz) return az < bz;
            const int64_t ay = a.coords[1];
            const int64_t by = b.coords[1];
            if (ay != by) return ((az & 1) == 0) ? (ay < by) : (ay > by);
            const int64_t ax = a.coords[0];
            const int64_t bx = b.coords[0];
            if (ax != bx) return (((az ^ ay) & 1) == 0) ? (ax < bx) : (ax > bx);
        } else if (dims == 2) {
            const int64_t ay = a.coords[1];
            const int64_t by = b.coords[1];
            if (ay != by) return ay < by;
            const int64_t ax = a.coords[0];
            const int64_t bx = b.coords[0];
            if (ax != bx) return ((ay & 1) == 0) ? (ax < bx) : (ax > bx);
        }
        return a.coords < b.coords;
    });

    std::vector<int> order;
    order.reserve(static_cast<size_t>(n));
    for (auto& g : groups) {
        std::sort(g.member_indices.begin(), g.member_indices.end(), [&](int ia, int ib) {
            const auto& a = infos[static_cast<size_t>(ia)];
            const auto& b = infos[static_cast<size_t>(ib)];
            if (a.core_on_chip != b.core_on_chip && a.core_on_chip >= 0 && b.core_on_chip >= 0) {
                return a.core_on_chip < b.core_on_chip;
            }
            if (a.id != b.id) {
                return a.id < b.id;
            }
            return ia < ib;
        });
        for (int idx : g.member_indices) {
            order.push_back(idx);
        }
    }
    return order;
}

// =========================================================================
// In-Process Host Mock PJRT C API + StableHLO Textual Evaluator
// =========================================================================

namespace {

struct MockTensor {
    PjrtDType dtype = PjrtDType::INVALID;
    std::vector<int64_t> dims;
    std::vector<uint8_t> bytes;

    size_t num_elements() const {
        size_t n = 1;
        for (int64_t d : dims) {
            n *= static_cast<size_t>(d);
        }
        return n;
    }

    void allocate(PjrtDType dt, std::vector<int64_t> shape) {
        dtype = dt;
        dims = std::move(shape);
        bytes.assign(pjrt_dtype_nbytes(dt, num_elements()), 0);
    }

    float read_f32(size_t idx) const {
        switch (dtype) {
            case PjrtDType::F32: {
                float v = 0.0f;
                std::memcpy(&v, bytes.data() + idx * 4, 4);
                return v;
            }
            case PjrtDType::F16: {
                uint16_t h = 0;
                std::memcpy(&h, bytes.data() + idx * 2, 2);
                return f16_to_f32(h);
            }
            case PjrtDType::BF16: {
                uint16_t b = 0;
                std::memcpy(&b, bytes.data() + idx * 2, 2);
                return bf16_to_f32(b);
            }
            case PjrtDType::F8E4M3FN:
                return f8e4m3fn_to_f32(bytes[idx]);
            case PjrtDType::S8:
                return static_cast<float>(reinterpret_cast<const int8_t*>(bytes.data())[idx]);
            case PjrtDType::U8:
                return static_cast<float>(bytes[idx]);
            case PjrtDType::S4:
            case PjrtDType::U4:
                return static_cast<float>(read_i64(idx));
            case PjrtDType::S16: {
                int16_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 2, 2);
                return static_cast<float>(v);
            }
            case PjrtDType::S32: {
                int32_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 4, 4);
                return static_cast<float>(v);
            }
            case PjrtDType::U32: {
                uint32_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 4, 4);
                return static_cast<float>(v);
            }
            case PjrtDType::S64: {
                int64_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 8, 8);
                return static_cast<float>(v);
            }
            default:
                return 0.0f;
        }
    }

    void write_f32(size_t idx, float v) {
        switch (dtype) {
            case PjrtDType::F32:
                std::memcpy(bytes.data() + idx * 4, &v, 4);
                break;
            case PjrtDType::F16: {
                const uint16_t h = f32_to_f16(v);
                std::memcpy(bytes.data() + idx * 2, &h, 2);
                break;
            }
            case PjrtDType::BF16: {
                const uint16_t b = f32_to_bf16(v);
                std::memcpy(bytes.data() + idx * 2, &b, 2);
                break;
            }
            case PjrtDType::F8E4M3FN:
                bytes[idx] = f32_to_f8e4m3fn(v);
                break;
            case PjrtDType::S8: {
                const int8_t iv = static_cast<int8_t>(std::lround(v));
                std::memcpy(bytes.data() + idx, &iv, 1);
                break;
            }
            case PjrtDType::U8: {
                const uint8_t uv = static_cast<uint8_t>(std::lround(v));
                bytes[idx] = uv;
                break;
            }
            case PjrtDType::S4:
            case PjrtDType::U4:
                write_i64(idx, static_cast<int64_t>(std::lround(v)));
                break;
            case PjrtDType::S16: {
                const int16_t iv = static_cast<int16_t>(std::lround(v));
                std::memcpy(bytes.data() + idx * 2, &iv, 2);
                break;
            }
            case PjrtDType::S32: {
                const int32_t iv = static_cast<int32_t>(std::lround(v));
                std::memcpy(bytes.data() + idx * 4, &iv, 4);
                break;
            }
            case PjrtDType::U32: {
                const uint32_t uv = static_cast<uint32_t>(std::lround(v));
                std::memcpy(bytes.data() + idx * 4, &uv, 4);
                break;
            }
            case PjrtDType::S64: {
                const int64_t iv = static_cast<int64_t>(std::llround(v));
                std::memcpy(bytes.data() + idx * 8, &iv, 8);
                break;
            }
            default:
                break;
        }
    }

    int64_t read_i64(size_t idx) const {
        switch (dtype) {
            case PjrtDType::S64: {
                int64_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 8, 8);
                return v;
            }
            case PjrtDType::S32: {
                int32_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 4, 4);
                return v;
            }
            case PjrtDType::U32: {
                uint32_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 4, 4);
                return static_cast<int64_t>(v);
            }
            case PjrtDType::S16: {
                int16_t v = 0;
                std::memcpy(&v, bytes.data() + idx * 2, 2);
                return v;
            }
            case PjrtDType::S8:
                return reinterpret_cast<const int8_t*>(bytes.data())[idx];
            case PjrtDType::PRED:
            case PjrtDType::U8:
                return static_cast<int64_t>(bytes[idx]);
            case PjrtDType::S4: {
                const uint8_t b = bytes[idx >> 1];
                const uint8_t nib = ((idx & 1U) == 0U) ? (b & 0x0FU) : ((b >> 4) & 0x0FU);
                return static_cast<int64_t>(sign_extend_nibble(nib));
            }
            case PjrtDType::U4: {
                const uint8_t b = bytes[idx >> 1];
                return static_cast<int64_t>(((idx & 1U) == 0U) ? (b & 0x0FU) : ((b >> 4) & 0x0FU));
            }
            default:
                return static_cast<int64_t>(read_f32(idx));
        }
    }

    void write_i64(size_t idx, int64_t v) {
        switch (dtype) {
            case PjrtDType::S64:
                std::memcpy(bytes.data() + idx * 8, &v, 8);
                break;
            case PjrtDType::S32: {
                const int32_t iv = static_cast<int32_t>(v);
                std::memcpy(bytes.data() + idx * 4, &iv, 4);
                break;
            }
            case PjrtDType::U32: {
                const uint32_t uv = static_cast<uint32_t>(v);
                std::memcpy(bytes.data() + idx * 4, &uv, 4);
                break;
            }
            case PjrtDType::S16: {
                const int16_t iv = static_cast<int16_t>(v);
                std::memcpy(bytes.data() + idx * 2, &iv, 2);
                break;
            }
            case PjrtDType::S8: {
                const int8_t iv = static_cast<int8_t>(v);
                std::memcpy(bytes.data() + idx, &iv, 1);
                break;
            }
            case PjrtDType::PRED:
                bytes[idx] = (v != 0) ? 1U : 0U;
                break;
            case PjrtDType::U8:
                bytes[idx] = static_cast<uint8_t>(v);
                break;
            case PjrtDType::S4:
            case PjrtDType::U4: {
                const size_t bi = idx >> 1;
                const uint8_t nib = static_cast<uint8_t>(v) & 0x0FU;
                if ((idx & 1U) == 0U) {
                    bytes[bi] = static_cast<uint8_t>((bytes[bi] & 0xF0U) | nib);
                } else {
                    bytes[bi] = static_cast<uint8_t>((bytes[bi] & 0x0FU) | (nib << 4));
                }
                break;
            }
            default:
                write_f32(idx, static_cast<float>(v));
                break;
        }
    }
};

static inline void copy_mock_elem(MockTensor& dst, size_t dst_idx, const MockTensor& src, size_t src_idx) {
    const size_t bits = pjrt_dtype_bits(dst.dtype);
    if (bits < 8) {
        dst.write_i64(dst_idx, src.read_i64(src_idx));
    } else {
        const size_t elem_sz = bits / 8;
        std::memcpy(dst.bytes.data() + dst_idx * elem_sz,
                    src.bytes.data() + src_idx * elem_sz,
                    elem_sz);
    }
}

static bool is_mock_integer_dtype(PjrtDType dt) {
    return dt == PjrtDType::PRED || dt == PjrtDType::S4 || dt == PjrtDType::U4 ||
           dt == PjrtDType::S8 || dt == PjrtDType::U8 || dt == PjrtDType::S16 ||
           dt == PjrtDType::S32 || dt == PjrtDType::U32 || dt == PjrtDType::S64;
}

struct MockTypeSpec {
    PjrtDType dtype = PjrtDType::INVALID;
    std::vector<int64_t> dims;
};

static std::string trim_copy(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static PjrtDType parse_mlir_dtype(const std::string& tok) {
    if (tok == "i1") return PjrtDType::PRED;
    if (tok == "f32") return PjrtDType::F32;
    if (tok == "f16") return PjrtDType::F16;
    if (tok == "bf16") return PjrtDType::BF16;
    if (tok == "f8E4M3FN" || tok == "f8e4m3fn") return PjrtDType::F8E4M3FN;
    if (tok == "i8" || tok == "si8") return PjrtDType::S8;
    if (tok == "ui8") return PjrtDType::U8;
    if (tok == "i4" || tok == "si4") return PjrtDType::S4;
    if (tok == "ui4") return PjrtDType::U4;
    if (tok == "i16" || tok == "si16") return PjrtDType::S16;
    if (tok == "i32" || tok == "si32") return PjrtDType::S32;
    if (tok == "ui32") return PjrtDType::U32;
    if (tok == "i64" || tok == "si64") return PjrtDType::S64;
    return PjrtDType::INVALID;
}

static MockTypeSpec parse_tensor_type(const std::string& spec) {
    const size_t lt = spec.find("tensor<");
    const size_t gt = spec.rfind('>');
    if (lt == std::string::npos || gt == std::string::npos || gt <= lt + 7) {
        throw std::runtime_error("MockPJRT: invalid tensor type: " + spec);
    }
    const std::string inner = spec.substr(lt + 7, gt - (lt + 7));
    MockTypeSpec out;
    size_t pos = 0;
    std::vector<std::string> parts;
    while (pos <= inner.size()) {
        const size_t x = inner.find('x', pos);
        if (x == std::string::npos) {
            parts.push_back(inner.substr(pos));
            break;
        }
        parts.push_back(inner.substr(pos, x - pos));
        pos = x + 1;
    }
    if (parts.empty()) {
        throw std::runtime_error("MockPJRT: empty tensor type: " + spec);
    }
    out.dtype = parse_mlir_dtype(parts.back());
    if (out.dtype == PjrtDType::INVALID) {
        throw std::runtime_error("MockPJRT: unsupported dtype in: " + spec);
    }
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        out.dims.push_back(std::stoll(parts[i]));
    }
    return out;
}

static std::vector<int64_t> parse_int_list(const std::string& s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '-' || std::isdigit(static_cast<unsigned char>(s[i]))) {
            size_t end = i + 1;
            while (end < s.size() && std::isdigit(static_cast<unsigned char>(s[end]))) ++end;
            out.push_back(std::stoll(s.substr(i, end - i)));
            i = end;
        } else {
            ++i;
        }
    }
    return out;
}

static std::vector<std::string> extract_ssa_refs(const std::string& s) {
    std::vector<std::string> refs;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%') {
            size_t j = i + 1;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) {
                ++j;
            }
            if (j > i + 1) {
                refs.push_back(s.substr(i, j - i));
            }
            i = j;
        }
    }
    return refs;
}

static std::vector<size_t> compute_row_major_strides(const std::vector<int64_t>& dims) {
    std::vector<size_t> strides(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<size_t>(dims[i + 1]);
    }
    return strides;
}

struct MockOp {
    enum class Kind {
        CONSTANT,
        IOTA,
        CONVERT,
        BITCAST_CONVERT,
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        MAXIMUM,
        MINIMUM,
        BIT_AND,
        SHR_LOGICAL,
        RSQRT,
        COSINE,
        SINE,
        LOGISTIC,
        EXPONENTIAL,
        TANH,
        BROADCAST_IN_DIM,
        RESHAPE,
        TRANSPOSE,
        SLICE,
        CONCATENATE,
        DOT_GENERAL,
        REDUCE_ADD,
        REDUCE_MAX,
        GATHER,
        SCATTER_3D_ROWS,
        COMPARE,
        SELECT,
        REPLICA_ID,
        DYNAMIC_SLICE,
        DYNAMIC_UPDATE_SLICE,
        ALL_REDUCE_ADD,
        ALL_REDUCE_MAX,
        ALL_GATHER,
        REDUCE_SCATTER_ADD,
        TPU_CUSTOM_CALL,
    };

    Kind kind;
    std::string out_name;
    MockTypeSpec out_type;
    std::vector<std::string> inputs;
    std::string cmp_dir;

    // Constant values
    bool is_splat = true;
    double scalar_f = 0.0;
    int64_t scalar_i = 0;
    std::vector<double> array_f;
    std::vector<int64_t> array_i;

    // Parameters
    int64_t iota_dim = 0;
    int64_t concat_dim = 0;
    int64_t collective_dim = 0;
    std::vector<int64_t> dims_attr;
    std::vector<int64_t> lhs_batch;
    std::vector<int64_t> rhs_batch;
    std::vector<int64_t> lhs_contract;
    std::vector<int64_t> rhs_contract;
    std::vector<int64_t> slice_starts;
    std::vector<int64_t> slice_limits;
    std::vector<int64_t> slice_strides;
    std::vector<int64_t> dynamic_slice_sizes;
    std::vector<std::vector<int64_t>> replica_groups;
    // FreeTile Mosaic custom_call attrs
    std::string custom_call_kernel_name;
    int64_t custom_call_mosaic_version = 17;
    int64_t custom_call_blk = 32;
    bool custom_call_has_min = false;
    double custom_call_q4_shift = 0.0;
    // Gather attrs
    std::vector<int64_t> gather_offset_dims;
    std::vector<int64_t> gather_collapsed_dims;
    std::vector<int64_t> gather_start_index_map;
    int64_t gather_index_vector_dim = 0;
};

struct MockProgram {
    std::vector<std::pair<std::string, MockTypeSpec>> args;
    std::vector<MockOp> ops;
    std::vector<std::string> return_names;
    std::string sdy_mesh_line;
};

static MockProgram parse_stablehlo_mlir(const std::string& mlir) {
    MockProgram prog;
    std::istringstream iss(mlir);
    std::string raw_line;
    std::vector<std::string> lines;
    while (std::getline(iss, raw_line)) {
        const std::string t = trim_copy(raw_line);
        if (!t.empty()) lines.push_back(t);
    }

    for (size_t li = 0; li < lines.size(); ++li) {
        const std::string& line = lines[li];
        if (line.rfind("func.func", 0) == 0) {
            const size_t p0 = line.find('(');
            const size_t p1 = line.find(") ->", p0);
            if (p0 == std::string::npos || p1 == std::string::npos) {
                throw std::runtime_error("MockPJRT: failed to parse func.func header");
            }
            const std::string arg_str = line.substr(p0 + 1, p1 - (p0 + 1));
            size_t pos = 0;
            while (pos < arg_str.size()) {
                const size_t pct = arg_str.find('%', pos);
                if (pct == std::string::npos) break;
                const size_t col = arg_str.find(':', pct);
                const size_t gt = arg_str.find('>', col);
                const std::string name = trim_copy(arg_str.substr(pct, col - pct));
                const MockTypeSpec ts = parse_tensor_type(arg_str.substr(col + 1, gt - col));
                prog.args.push_back({name, ts});
                pos = gt + 1;
            }
            continue;
        }

        if (line.rfind("return ", 0) == 0) {
            const size_t col = line.find(':');
            const std::string ret_part = (col == std::string::npos) ? line.substr(7) : line.substr(7, col - 7);
            prog.return_names = extract_ssa_refs(ret_part);
            continue;
        }

        if (line == "}" || line.rfind("module ", 0) == 0) {
            continue;
        }
        if (line.rfind("sdy.mesh ", 0) == 0) {
            prog.sdy_mesh_line = line;
            continue;
        }

        // Accumulate multi-line ops (e.g. reduce, gather, scatter, collectives, dynamic_slice)
        std::string stmt = line;
        if (stmt.find("\"stablehlo.reduce\"") != std::string::npos ||
            stmt.find("\"stablehlo.gather\"") != std::string::npos ||
            stmt.find("\"stablehlo.scatter\"") != std::string::npos ||
            stmt.find("\"stablehlo.all_reduce\"") != std::string::npos ||
            stmt.find("\"stablehlo.all_gather\"") != std::string::npos ||
            stmt.find("\"stablehlo.reduce_scatter\"") != std::string::npos ||
            stmt.find("\"stablehlo.dynamic_slice\"") != std::string::npos ||
            stmt.find("\"stablehlo.dynamic_update_slice\"") != std::string::npos) {
            while (li + 1 < lines.size() &&
                   (stmt.find("->") == std::string::npos || stmt.rfind('>') < stmt.find("->"))) {
                stmt += " " + lines[++li];
            }
        }

        const size_t eq = stmt.find(" = ");
        if (eq == std::string::npos) continue;
        MockOp op;
        op.out_name = trim_copy(stmt.substr(0, eq));
        const std::string rhs = trim_copy(stmt.substr(eq + 3));

        const size_t arrow = rhs.rfind("->");
        const size_t colon = rhs.rfind(':');
        if (arrow != std::string::npos) {
            op.out_type = parse_tensor_type(rhs.substr(arrow + 2));
        } else if (colon != std::string::npos) {
            op.out_type = parse_tensor_type(rhs.substr(colon + 1));
        }

        if (rhs.rfind("stablehlo.constant ", 0) == 0) {
            op.kind = MockOp::Kind::CONSTANT;
            const size_t d0 = rhs.find("dense<");
            if (d0 == std::string::npos) {
                throw std::runtime_error("MockPJRT: missing dense< literal in stablehlo.constant");
            }
            const size_t d1 = rhs.find('>', d0 + 6);
            if (d1 == std::string::npos) {
                throw std::runtime_error("MockPJRT: unterminated dense< literal in stablehlo.constant");
            }
            const std::string lit = trim_copy(rhs.substr(d0 + 6, d1 - (d0 + 6)));
            const bool is_int = is_mock_integer_dtype(op.out_type.dtype);
            auto parse_mlir_float = [](const std::string& s) -> double {
                if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
                    size_t idx = 0;
                    const uint32_t bits = static_cast<uint32_t>(std::stoul(s, &idx, 16));
                    if (idx != s.size()) {
                        throw std::runtime_error("MockPJRT: invalid hex float literal: " + s);
                    }
                    float f = 0.0f;
                    std::memcpy(&f, &bits, sizeof(f));
                    return static_cast<double>(f);
                }
                std::istringstream iss(s);
                iss.imbue(std::locale::classic());
                double val = 0.0;
                if (!(iss >> val) || !(iss >> std::ws).eof()) {
                    throw std::runtime_error("MockPJRT: invalid float literal: " + s);
                }
                return val;
            };
            auto parse_mlir_int = [](const std::string& s) -> int64_t {
                std::istringstream iss(s);
                iss.imbue(std::locale::classic());
                int64_t val = 0;
                if (!(iss >> val) || !(iss >> std::ws).eof()) {
                    throw std::runtime_error("MockPJRT: invalid int literal: " + s);
                }
                return val;
            };
            if (!lit.empty() && lit.front() == '[') {
                op.is_splat = false;
                std::string inner = lit.substr(1, lit.size() - 2);
                std::istringstream vs(inner);
                std::string tok;
                while (std::getline(vs, tok, ',')) {
                    tok = trim_copy(tok);
                    if (tok.empty()) continue;
                    if (is_int) op.array_i.push_back(parse_mlir_int(tok));
                    else op.array_f.push_back(parse_mlir_float(tok));
                }
            } else {
                op.is_splat = true;
                if (is_int) op.scalar_i = parse_mlir_int(lit);
                else op.scalar_f = parse_mlir_float(lit);
            }
        } else if (rhs.rfind("stablehlo.iota ", 0) == 0) {
            op.kind = MockOp::Kind::IOTA;
            const size_t dp = rhs.find("dim = ");
            op.iota_dim = std::stoll(rhs.substr(dp + 6));
        } else if (rhs.rfind("stablehlo.convert ", 0) == 0) {
            op.kind = MockOp::Kind::CONVERT;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.bitcast_convert ", 0) == 0) {
            op.kind = MockOp::Kind::BITCAST_CONVERT;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.add ", 0) == 0) {
            op.kind = MockOp::Kind::ADD;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.subtract ", 0) == 0) {
            op.kind = MockOp::Kind::SUBTRACT;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.multiply ", 0) == 0) {
            op.kind = MockOp::Kind::MULTIPLY;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.divide ", 0) == 0) {
            op.kind = MockOp::Kind::DIVIDE;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.maximum ", 0) == 0) {
            op.kind = MockOp::Kind::MAXIMUM;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.minimum ", 0) == 0) {
            op.kind = MockOp::Kind::MINIMUM;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.and ", 0) == 0) {
            op.kind = MockOp::Kind::BIT_AND;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.shift_right_logical ", 0) == 0) {
            op.kind = MockOp::Kind::SHR_LOGICAL;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.rsqrt ", 0) == 0) {
            op.kind = MockOp::Kind::RSQRT;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.cosine ", 0) == 0) {
            op.kind = MockOp::Kind::COSINE;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.sine ", 0) == 0) {
            op.kind = MockOp::Kind::SINE;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.logistic ", 0) == 0) {
            op.kind = MockOp::Kind::LOGISTIC;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.exponential ", 0) == 0) {
            op.kind = MockOp::Kind::EXPONENTIAL;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.tanh ", 0) == 0) {
            op.kind = MockOp::Kind::TANH;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.broadcast_in_dim ", 0) == 0) {
            op.kind = MockOp::Kind::BROADCAST_IN_DIM;
            const size_t dp = rhs.find("dims = [");
            const size_t de = rhs.find(']', dp);
            op.inputs = extract_ssa_refs(rhs.substr(0, dp));
            op.dims_attr = parse_int_list(rhs.substr(dp + 8, de - (dp + 8)));
        } else if (rhs.rfind("stablehlo.reshape ", 0) == 0) {
            op.kind = MockOp::Kind::RESHAPE;
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("stablehlo.transpose ", 0) == 0) {
            op.kind = MockOp::Kind::TRANSPOSE;
            const size_t dp = rhs.find("dims = [");
            const size_t de = rhs.find(']', dp);
            op.inputs = extract_ssa_refs(rhs.substr(0, dp));
            op.dims_attr = parse_int_list(rhs.substr(dp + 8, de - (dp + 8)));
        } else if (rhs.rfind("stablehlo.slice ", 0) == 0) {
            op.kind = MockOp::Kind::SLICE;
            const size_t lb = rhs.find('[');
            const size_t rb = rhs.find(']', lb);
            op.inputs = extract_ssa_refs(rhs.substr(0, lb));
            std::string body = rhs.substr(lb + 1, rb - (lb + 1));
            std::istringstream ss(body);
            std::string item;
            while (std::getline(ss, item, ',')) {
                auto vals = parse_int_list(item);
                if (vals.size() >= 3) {
                    op.slice_starts.push_back(vals[0]);
                    op.slice_limits.push_back(vals[1]);
                    op.slice_strides.push_back(vals[2]);
                }
            }
        } else if (rhs.rfind("stablehlo.concatenate ", 0) == 0) {
            op.kind = MockOp::Kind::CONCATENATE;
            const size_t dp = rhs.find("dim = ");
            op.inputs = extract_ssa_refs(rhs.substr(0, dp));
            op.concat_dim = std::stoll(rhs.substr(dp + 6));
        } else if (rhs.rfind("stablehlo.dot_general ", 0) == 0) {
            op.kind = MockOp::Kind::DOT_GENERAL;
            const size_t bp = rhs.find("batching_dims =");
            op.inputs = extract_ssa_refs(rhs.substr(0, bp));
            const size_t b_l0 = rhs.find('[', bp);
            const size_t b_l1 = rhs.find(']', b_l0);
            const size_t b_r0 = rhs.find('[', b_l1);
            const size_t b_r1 = rhs.find(']', b_r0);
            op.lhs_batch = parse_int_list(rhs.substr(b_l0 + 1, b_l1 - b_l0 - 1));
            op.rhs_batch = parse_int_list(rhs.substr(b_r0 + 1, b_r1 - b_r0 - 1));
            const size_t cp = rhs.find("contracting_dims =", b_r1);
            const size_t c_l0 = rhs.find('[', cp);
            const size_t c_l1 = rhs.find(']', c_l0);
            const size_t c_r0 = rhs.find('[', c_l1);
            const size_t c_r1 = rhs.find(']', c_r0);
            op.lhs_contract = parse_int_list(rhs.substr(c_l0 + 1, c_l1 - c_l0 - 1));
            op.rhs_contract = parse_int_list(rhs.substr(c_r0 + 1, c_r1 - c_r0 - 1));
        } else if (rhs.rfind("\"stablehlo.reduce\"", 0) == 0) {
            op.kind = (rhs.find("stablehlo.maximum") != std::string::npos)
                          ? MockOp::Kind::REDUCE_MAX
                          : MockOp::Kind::REDUCE_ADD;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
            const size_t dp = rhs.find("dimensions = array<i64:");
            const size_t de = rhs.find('>', dp);
            op.dims_attr = parse_int_list(rhs.substr(dp + 23, de - (dp + 23)));
        } else if (rhs.rfind("\"stablehlo.gather\"", 0) == 0) {
            op.kind = MockOp::Kind::GATHER;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
            auto parse_named_arr = [&](const char* key) {
                const size_t p = rhs.find(key);
                const size_t lb = rhs.find('[', p);
                const size_t rb = rhs.find(']', lb);
                return parse_int_list(rhs.substr(lb + 1, rb - lb - 1));
            };
            op.gather_offset_dims = parse_named_arr("offset_dims");
            op.gather_collapsed_dims = parse_named_arr("collapsed_slice_dims");
            op.gather_start_index_map = parse_named_arr("start_index_map");
            const size_t iv = rhs.find("index_vector_dim =");
            op.gather_index_vector_dim = std::stoll(rhs.substr(iv + 18));
        } else if (rhs.rfind("stablehlo.compare ", 0) == 0) {
            op.kind = MockOp::Kind::COMPARE;
            const size_t dir_start = 18;
            const size_t comma = rhs.find(',', dir_start);
            op.cmp_dir = trim_copy(rhs.substr(dir_start, comma - dir_start));
            op.inputs = extract_ssa_refs(rhs.substr(0, colon));
        } else if (rhs.rfind("\"stablehlo.select\"", 0) == 0) {
            op.kind = MockOp::Kind::SELECT;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
        } else if (rhs.rfind("\"stablehlo.scatter\"", 0) == 0) {
            op.kind = MockOp::Kind::SCATTER_3D_ROWS;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
        } else if (rhs.rfind("\"stablehlo.replica_id\"", 0) == 0) {
            op.kind = MockOp::Kind::REPLICA_ID;
        } else if (rhs.rfind("\"stablehlo.dynamic_slice\"", 0) == 0) {
            op.kind = MockOp::Kind::DYNAMIC_SLICE;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
            const size_t sp = rhs.find("slice_sizes = array<i64:");
            const size_t se = rhs.find('>', sp);
            op.dynamic_slice_sizes = parse_int_list(rhs.substr(sp + 24, se - (sp + 24)));
        } else if (rhs.rfind("\"stablehlo.dynamic_update_slice\"", 0) == 0) {
            op.kind = MockOp::Kind::DYNAMIC_UPDATE_SLICE;
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
        } else if (rhs.rfind("\"stablehlo.all_reduce\"", 0) == 0 ||
                   rhs.rfind("\"stablehlo.all_gather\"", 0) == 0 ||
                   rhs.rfind("\"stablehlo.reduce_scatter\"", 0) == 0) {
            if (rhs.rfind("\"stablehlo.all_reduce\"", 0) == 0) {
                op.kind = (rhs.find("stablehlo.maximum") != std::string::npos)
                              ? MockOp::Kind::ALL_REDUCE_MAX
                              : MockOp::Kind::ALL_REDUCE_ADD;
            } else if (rhs.rfind("\"stablehlo.all_gather\"", 0) == 0) {
                op.kind = MockOp::Kind::ALL_GATHER;
                const size_t dp = rhs.find("all_gather_dim =");
                op.collective_dim = std::stoll(rhs.substr(dp + 16));
            } else {
                op.kind = MockOp::Kind::REDUCE_SCATTER_ADD;
                const size_t dp = rhs.find("scatter_dimension =");
                op.collective_dim = std::stoll(rhs.substr(dp + 19));
            }
            const size_t lp = rhs.find('(');
            const size_t rp = rhs.find(')', lp);
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
            const size_t rg = rhs.find("replica_groups = dense<");
            if (rg != std::string::npos) {
                const size_t rg_end = rhs.find('>', rg + 23);
                const std::string body = rhs.substr(rg + 23, rg_end - (rg + 23));
                // Parse nested [[g0...], [g1...]] or scalar [0]
                size_t p = 0;
                while (p < body.size()) {
                    const size_t lb = body.find('[', p);
                    if (lb == std::string::npos) break;
                    if (lb + 1 < body.size() && body[lb + 1] == '[') {
                        p = lb + 1;
                        continue;
                    }
                    const size_t rb = body.find(']', lb);
                    if (rb == std::string::npos) break;
                    auto grp = parse_int_list(body.substr(lb + 1, rb - lb - 1));
                    if (!grp.empty()) {
                        op.replica_groups.push_back(std::move(grp));
                    }
                    p = rb + 1;
                }
            }
        } else if (rhs.rfind("stablehlo.custom_call @tpu_custom_call(", 0) == 0) {
            op.kind = MockOp::Kind::TPU_CUSTOM_CALL;
            const size_t lp = rhs.find('(');
            const size_t rp = (lp != std::string::npos) ? rhs.find(')', lp) : std::string::npos;
            if (lp == std::string::npos || rp == std::string::npos) {
                throw std::runtime_error("MockPJRT: malformed tpu_custom_call operand list");
            }
            op.inputs = extract_ssa_refs(rhs.substr(lp + 1, rp - lp - 1));
            const size_t knp = rhs.find("kernel_name = \"");
            if (knp != std::string::npos) {
                const size_t kne = rhs.find('"', knp + 15);
                if (kne == std::string::npos) {
                    throw std::runtime_error("MockPJRT: unterminated tpu_custom_call kernel_name");
                }
                op.custom_call_kernel_name = rhs.substr(knp + 15, kne - (knp + 15));
            }
            const size_t bcp = rhs.find("backend_config = \"");
            if (bcp == std::string::npos) {
                throw std::runtime_error("MockPJRT: tpu_custom_call missing backend_config");
            }
            size_t cur = bcp + 18;
            std::string cfg_json;
            bool closed_quote = false;
            while (cur < rhs.size()) {
                if (rhs[cur] == '\\' && cur + 1 < rhs.size()) {
                    char nx = rhs[cur + 1];
                    if (nx == 'n') cfg_json.push_back('\n');
                    else cfg_json.push_back(nx);
                    cur += 2;
                } else if (rhs[cur] == '"') {
                    closed_quote = true;
                    break;
                } else {
                    cfg_json.push_back(rhs[cur++]);
                }
            }
            if (!closed_quote) {
                throw std::runtime_error("MockPJRT: unterminated tpu_custom_call backend_config");
            }
            const size_t sfp = cfg_json.find("\"serialization_format\"");
            if (sfp == std::string::npos) {
                throw std::runtime_error("MockPJRT: missing tpu_custom_call serialization_format");
            }
            size_t sf_cur = sfp + 22;
            while (sf_cur < cfg_json.size() && std::isspace(static_cast<unsigned char>(cfg_json[sf_cur]))) {
                ++sf_cur;
            }
            if (sf_cur >= cfg_json.size() || cfg_json[sf_cur] != ':') {
                throw std::runtime_error("MockPJRT: malformed tpu_custom_call serialization_format");
            }
            ++sf_cur;
            while (sf_cur < cfg_json.size() && std::isspace(static_cast<unsigned char>(cfg_json[sf_cur]))) {
                ++sf_cur;
            }
            const size_t sf_num_start = sf_cur;
            if (sf_cur < cfg_json.size() && (cfg_json[sf_cur] == '-' || cfg_json[sf_cur] == '+')) {
                ++sf_cur;
            }
            const size_t sf_digits_start = sf_cur;
            while (sf_cur < cfg_json.size() && std::isdigit(static_cast<unsigned char>(cfg_json[sf_cur]))) {
                ++sf_cur;
            }
            const bool sf_valid_delim =
                (sf_cur == cfg_json.size()) || cfg_json[sf_cur] == ',' || cfg_json[sf_cur] == '}' ||
                std::isspace(static_cast<unsigned char>(cfg_json[sf_cur]));
            if (sf_cur == sf_digits_start || !sf_valid_delim) {
                throw std::runtime_error("MockPJRT: invalid tpu_custom_call serialization_format value");
            }
            const int64_t ser_fmt = std::stoll(cfg_json.substr(sf_num_start, sf_cur - sf_num_start));
            if (ser_fmt != 1) {
                throw std::runtime_error("MockPJRT: unsupported tpu_custom_call serialization_format: " +
                                         std::to_string(ser_fmt));
            }
            const size_t bp = cfg_json.find("\"body\": \"");
            if (bp == std::string::npos) {
                throw std::runtime_error("MockPJRT: tpu_custom_call missing body");
            }
            const size_t be = cfg_json.find('"', bp + 9);
            if (be == std::string::npos) {
                throw std::runtime_error("MockPJRT: unterminated tpu_custom_call body");
            }
            const std::string b64 = cfg_json.substr(bp + 9, be - (bp + 9));
            auto b64_val = [](unsigned char c) -> int {
                if (c >= 'A' && c <= 'Z') return c - 'A';
                if (c >= 'a' && c <= 'z') return c - 'a' + 26;
                if (c >= '0' && c <= '9') return c - '0' + 52;
                if (c == '+') return 62;
                if (c == '/') return 63;
                return -1;
            };
            std::string mosaic_txt;
            uint32_t acc_bits = 0;
            int acc_cnt = 0;
            for (unsigned char c : b64) {
                if (c == '=') break;
                int v = b64_val(c);
                if (v < 0) continue;
                acc_bits = (acc_bits << 6) | static_cast<uint32_t>(v);
                acc_cnt += 6;
                if (acc_cnt >= 8) {
                    acc_cnt -= 8;
                    mosaic_txt.push_back(static_cast<char>((acc_bits >> acc_cnt) & 0xFF));
                }
            }
            const size_t vp = mosaic_txt.find("stable_mosaic.version = ");
            if (vp == std::string::npos) {
                throw std::runtime_error("MockPJRT: Mosaic module missing stable_mosaic.version");
            }
            op.custom_call_mosaic_version = std::stoll(mosaic_txt.substr(vp + 24));
            if (op.custom_call_mosaic_version != 17) {
                throw std::runtime_error("MockPJRT: unsupported stable_mosaic.version: " +
                                         std::to_string(op.custom_call_mosaic_version));
            }
            if (mosaic_txt.find("xui4") != std::string::npos ||
                mosaic_txt.find("xui8") != std::string::npos) {
                throw std::runtime_error(
                    "MockPJRT: Mosaic module must use signless integer types (i4/i8), not ui4/ui8");
            }
            const bool has_const_8 = (mosaic_txt.find("value = 8.000000e+00 : bf16") != std::string::npos);
            if (op.custom_call_kernel_name.rfind("ft_q6_k_", 0) == 0) {
                op.custom_call_blk = 16;
                op.custom_call_has_min = false;
                op.custom_call_q4_shift = 0.0;
            } else if (op.custom_call_kernel_name.rfind("ft_q4_k_", 0) == 0) {
                op.custom_call_blk = 32;
                op.custom_call_has_min = true;
                op.custom_call_q4_shift =
                    (mosaic_txt.find("stable_mosaic.arith.sitofp") != std::string::npos &&
                     has_const_8 &&
                     mosaic_txt.find("stable_mosaic.arith.addf") != std::string::npos)
                        ? 8.0
                        : 0.0;
            } else if (op.custom_call_kernel_name.rfind("ft_q4_0_", 0) == 0) {
                op.custom_call_blk = 32;
                op.custom_call_has_min = false;
                op.custom_call_q4_shift =
                    (mosaic_txt.find("stable_mosaic.arith.uitofp") != std::string::npos &&
                     has_const_8 &&
                     mosaic_txt.find("stable_mosaic.arith.subf") != std::string::npos)
                        ? -8.0
                        : 0.0;
            } else if (op.custom_call_kernel_name.rfind("ft_q8_0_", 0) == 0) {
                op.custom_call_blk = 32;
                op.custom_call_has_min = false;
                op.custom_call_q4_shift = 0.0;
            } else {
                op.custom_call_blk = 1;
                op.custom_call_has_min = false;
                op.custom_call_q4_shift = 0.0;
            }
        } else {
            throw std::runtime_error("MockPJRT: unsupported StableHLO op: " + rhs);
        }

        prog.ops.push_back(std::move(op));
    }

    return prog;
}

static MockTensor eval_mock_op(const MockOp& op,
                               const std::unordered_map<std::string, MockTensor>& env,
                               int64_t replica_id = 0) {
    MockTensor out;
    out.allocate(op.out_type.dtype, op.out_type.dims);
    const size_t n_out = out.num_elements();

    const bool is_int = is_mock_integer_dtype(out.dtype);

    switch (op.kind) {
        case MockOp::Kind::CONSTANT: {
            if (op.is_splat) {
                for (size_t i = 0; i < n_out; ++i) {
                    if (is_int) out.write_i64(i, op.scalar_i);
                    else out.write_f32(i, static_cast<float>(op.scalar_f));
                }
            } else {
                for (size_t i = 0; i < n_out; ++i) {
                    if (is_int) out.write_i64(i, op.array_i[i]);
                    else out.write_f32(i, static_cast<float>(op.array_f[i]));
                }
            }
            break;
        }
        case MockOp::Kind::IOTA: {
            const auto strides = compute_row_major_strides(out.dims);
            const int64_t dim_sz = out.dims[static_cast<size_t>(op.iota_dim)];
            const size_t stride = strides[static_cast<size_t>(op.iota_dim)];
            for (size_t i = 0; i < n_out; ++i) {
                const int64_t coord = static_cast<int64_t>((i / stride) % static_cast<size_t>(dim_sz));
                if (is_int) out.write_i64(i, coord);
                else out.write_f32(i, static_cast<float>(coord));
            }
            break;
        }
        case MockOp::Kind::CONVERT: {
            const MockTensor& a = env.at(op.inputs[0]);
            const bool src_int = is_mock_integer_dtype(a.dtype);
            for (size_t i = 0; i < n_out; ++i) {
                if (is_int && src_int) out.write_i64(i, a.read_i64(i));
                else if (is_int) out.write_i64(i, static_cast<int64_t>(a.read_f32(i)));
                else if (src_int) out.write_f32(i, static_cast<float>(a.read_i64(i)));
                else out.write_f32(i, a.read_f32(i));
            }
            break;
        }
        case MockOp::Kind::BITCAST_CONVERT: {
            const MockTensor& a = env.at(op.inputs[0]);
            const size_t bits_in = pjrt_dtype_bits(a.dtype);
            const size_t bits_out = pjrt_dtype_bits(out.dtype);
            if (bits_in == 0 || bits_out == 0 ||
                a.num_elements() * bits_in != n_out * bits_out ||
                a.bytes.size() != out.bytes.size()) {
                throw std::runtime_error("MockPJRT: bitcast_convert total bitwidth mismatch");
            }
            out.bytes = a.bytes;
            break;
        }
        case MockOp::Kind::ADD:
        case MockOp::Kind::SUBTRACT:
        case MockOp::Kind::MULTIPLY:
        case MockOp::Kind::DIVIDE:
        case MockOp::Kind::MAXIMUM:
        case MockOp::Kind::MINIMUM:
        case MockOp::Kind::BIT_AND:
        case MockOp::Kind::SHR_LOGICAL: {
            const MockTensor& a = env.at(op.inputs[0]);
            const MockTensor& b = env.at(op.inputs[1]);
            const size_t out_bits = pjrt_dtype_bits(out.dtype);
            const uint64_t bit_mask = (out_bits >= 64) ? ~0ULL : ((1ULL << out_bits) - 1ULL);
            for (size_t i = 0; i < n_out; ++i) {
                if (is_int) {
                    const int64_t va = a.read_i64(i);
                    const int64_t vb = b.read_i64(i);
                    int64_t vr = 0;
                    if (op.kind == MockOp::Kind::ADD) vr = static_cast<int64_t>(static_cast<uint64_t>(va) + static_cast<uint64_t>(vb));
                    else if (op.kind == MockOp::Kind::SUBTRACT) vr = static_cast<int64_t>(static_cast<uint64_t>(va) - static_cast<uint64_t>(vb));
                    else if (op.kind == MockOp::Kind::MULTIPLY) vr = static_cast<int64_t>(static_cast<uint64_t>(va) * static_cast<uint64_t>(vb));
                    else if (op.kind == MockOp::Kind::DIVIDE) vr = (vb != 0) ? (va / vb) : 0;
                    else if (op.kind == MockOp::Kind::MAXIMUM) vr = std::max(va, vb);
                    else if (op.kind == MockOp::Kind::MINIMUM) vr = std::min(va, vb);
                    else if (op.kind == MockOp::Kind::BIT_AND) vr = (va & vb);
                    else if (op.kind == MockOp::Kind::SHR_LOGICAL) {
                        const uint64_t ua = static_cast<uint64_t>(va) & bit_mask;
                        vr = (vb < 0 || static_cast<uint64_t>(vb) >= out_bits)
                                 ? 0
                                 : static_cast<int64_t>(ua >> static_cast<unsigned>(vb));
                    }
                    out.write_i64(i, vr);
                } else {
                    const float va = a.read_f32(i);
                    const float vb = b.read_f32(i);
                    float vr = 0.0f;
                    if (op.kind == MockOp::Kind::ADD) vr = va + vb;
                    else if (op.kind == MockOp::Kind::SUBTRACT) vr = va - vb;
                    else if (op.kind == MockOp::Kind::MULTIPLY) vr = va * vb;
                    else if (op.kind == MockOp::Kind::DIVIDE) vr = va / vb;
                    else if (op.kind == MockOp::Kind::MAXIMUM) vr = std::max(va, vb);
                    else if (op.kind == MockOp::Kind::MINIMUM) vr = std::min(va, vb);
                    out.write_f32(i, vr);
                }
            }
            break;
        }
        case MockOp::Kind::RSQRT:
        case MockOp::Kind::COSINE:
        case MockOp::Kind::SINE:
        case MockOp::Kind::LOGISTIC:
        case MockOp::Kind::EXPONENTIAL:
        case MockOp::Kind::TANH: {
            const MockTensor& a = env.at(op.inputs[0]);
            for (size_t i = 0; i < n_out; ++i) {
                const float va = a.read_f32(i);
                float vr = 0.0f;
                if (op.kind == MockOp::Kind::RSQRT) vr = 1.0f / std::sqrt(va);
                else if (op.kind == MockOp::Kind::COSINE) vr = std::cos(va);
                else if (op.kind == MockOp::Kind::SINE) vr = std::sin(va);
                else if (op.kind == MockOp::Kind::LOGISTIC) vr = 1.0f / (1.0f + std::exp(-va));
                else if (op.kind == MockOp::Kind::EXPONENTIAL) vr = std::exp(va);
                else if (op.kind == MockOp::Kind::TANH) vr = std::tanh(va);
                out.write_f32(i, vr);
            }
            break;
        }
        case MockOp::Kind::RESHAPE: {
            const MockTensor& a = env.at(op.inputs[0]);
            out.bytes = a.bytes;
            break;
        }
        case MockOp::Kind::BROADCAST_IN_DIM: {
            const MockTensor& a = env.at(op.inputs[0]);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            for (size_t i = 0; i < n_out; ++i) {
                size_t rem = i;
                size_t src_idx = 0;
                std::vector<int64_t> out_coord(out.dims.size(), 0);
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    out_coord[d] = static_cast<int64_t>(rem / out_strides[d]);
                    rem %= out_strides[d];
                }
                for (size_t sd = 0; sd < a.dims.size(); ++sd) {
                    const size_t od = static_cast<size_t>(op.dims_attr[sd]);
                    const int64_t c = (a.dims[sd] == 1) ? 0 : out_coord[od];
                    src_idx += static_cast<size_t>(c) * in_strides[sd];
                }
                copy_mock_elem(out, i, a, src_idx);
            }
            break;
        }
        case MockOp::Kind::TRANSPOSE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            for (size_t i = 0; i < n_out; ++i) {
                size_t rem = i;
                size_t src_idx = 0;
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    const size_t c = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_idx += c * in_strides[static_cast<size_t>(op.dims_attr[d])];
                }
                copy_mock_elem(out, i, a, src_idx);
            }
            break;
        }
        case MockOp::Kind::SLICE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            for (size_t i = 0; i < n_out; ++i) {
                size_t rem = i;
                size_t src_idx = 0;
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    const size_t c = rem / out_strides[d];
                    rem %= out_strides[d];
                    const size_t sc = static_cast<size_t>(op.slice_starts[d]) +
                                      c * static_cast<size_t>(op.slice_strides[d]);
                    src_idx += sc * in_strides[d];
                }
                copy_mock_elem(out, i, a, src_idx);
            }
            break;
        }
        case MockOp::Kind::CONCATENATE: {
            const auto out_strides = compute_row_major_strides(out.dims);
            const size_t cdim = static_cast<size_t>(op.concat_dim);
            int64_t offset = 0;
            for (const std::string& in_name : op.inputs) {
                const MockTensor& a = env.at(in_name);
                const auto in_strides = compute_row_major_strides(a.dims);
                const size_t n_in = a.num_elements();
                for (size_t i = 0; i < n_in; ++i) {
                    size_t rem = i;
                    size_t dst_idx = 0;
                    for (size_t d = 0; d < a.dims.size(); ++d) {
                        const size_t c = rem / in_strides[d];
                        rem %= in_strides[d];
                        const size_t dc = (d == cdim) ? (c + static_cast<size_t>(offset)) : c;
                        dst_idx += dc * out_strides[d];
                    }
                    copy_mock_elem(out, dst_idx, a, i);
                }
                offset += a.dims[cdim];
            }
            break;
        }
        case MockOp::Kind::DOT_GENERAL: {
            const MockTensor& lhs = env.at(op.inputs[0]);
            const MockTensor& rhs = env.at(op.inputs[1]);
            // Supports standard 4D [B3, B2, M, K] x [B3, B2, N, K] contracting [3]x[3], batch [0,1]x[0,1]
            // as well as general arbitrary batch/contract dims.
            std::vector<bool> lhs_used(lhs.dims.size(), false);
            std::vector<bool> rhs_used(rhs.dims.size(), false);
            for (int64_t d : op.lhs_batch) lhs_used[static_cast<size_t>(d)] = true;
            for (int64_t d : op.lhs_contract) lhs_used[static_cast<size_t>(d)] = true;
            for (int64_t d : op.rhs_batch) rhs_used[static_cast<size_t>(d)] = true;
            for (int64_t d : op.rhs_contract) rhs_used[static_cast<size_t>(d)] = true;

            std::vector<size_t> lhs_free_dims, rhs_free_dims;
            for (size_t d = 0; d < lhs.dims.size(); ++d) if (!lhs_used[d]) lhs_free_dims.push_back(d);
            for (size_t d = 0; d < rhs.dims.size(); ++d) if (!rhs_used[d]) rhs_free_dims.push_back(d);

            const auto lhs_strides = compute_row_major_strides(lhs.dims);
            const auto rhs_strides = compute_row_major_strides(rhs.dims);
            const auto out_strides = compute_row_major_strides(out.dims);

            size_t contract_total = 1;
            for (int64_t d : op.lhs_contract) contract_total *= static_cast<size_t>(lhs.dims[static_cast<size_t>(d)]);

            std::vector<int64_t> contract_shape;
            for (int64_t d : op.lhs_contract) contract_shape.push_back(lhs.dims[static_cast<size_t>(d)]);
            const auto c_strides = compute_row_major_strides(contract_shape);

            for (size_t oi = 0; oi < n_out; ++oi) {
                size_t rem = oi;
                std::vector<size_t> out_coord(out.dims.size(), 0);
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    out_coord[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }
                size_t lhs_base = 0;
                size_t rhs_base = 0;
                size_t od = 0;
                for (size_t bi = 0; bi < op.lhs_batch.size(); ++bi, ++od) {
                    lhs_base += out_coord[od] * lhs_strides[static_cast<size_t>(op.lhs_batch[bi])];
                    rhs_base += out_coord[od] * rhs_strides[static_cast<size_t>(op.rhs_batch[bi])];
                }
                for (size_t fd : lhs_free_dims) {
                    lhs_base += out_coord[od++] * lhs_strides[fd];
                }
                for (size_t fd : rhs_free_dims) {
                    rhs_base += out_coord[od++] * rhs_strides[fd];
                }

                double acc = 0.0;
                for (size_t ci = 0; ci < contract_total; ++ci) {
                    size_t crem = ci;
                    size_t l_off = 0;
                    size_t r_off = 0;
                    for (size_t k = 0; k < contract_shape.size(); ++k) {
                        const size_t cc = crem / c_strides[k];
                        crem %= c_strides[k];
                        l_off += cc * lhs_strides[static_cast<size_t>(op.lhs_contract[k])];
                        r_off += cc * rhs_strides[static_cast<size_t>(op.rhs_contract[k])];
                    }
                    acc += static_cast<double>(lhs.read_f32(lhs_base + l_off)) *
                           static_cast<double>(rhs.read_f32(rhs_base + r_off));
                }
                out.write_f32(oi, static_cast<float>(acc));
            }
            break;
        }
        case MockOp::Kind::REDUCE_ADD:
        case MockOp::Kind::REDUCE_MAX: {
            const MockTensor& x = env.at(op.inputs[0]);
            const MockTensor& init = env.at(op.inputs[1]);
            const float init_val = init.read_f32(0);
            for (size_t i = 0; i < n_out; ++i) {
                out.write_f32(i, init_val);
            }
            std::vector<bool> is_red(x.dims.size(), false);
            for (int64_t d : op.dims_attr) is_red[static_cast<size_t>(d)] = true;
            const auto in_strides = compute_row_major_strides(x.dims);
            const auto out_strides = compute_row_major_strides(out.dims);
            const size_t n_in = x.num_elements();
            for (size_t i = 0; i < n_in; ++i) {
                size_t rem = i;
                size_t dst_idx = 0;
                size_t od = 0;
                for (size_t d = 0; d < x.dims.size(); ++d) {
                    const size_t c = rem / in_strides[d];
                    rem %= in_strides[d];
                    if (!is_red[d]) {
                        dst_idx += c * out_strides[od++];
                    }
                }
                const float vx = x.read_f32(i);
                const float vo = out.read_f32(dst_idx);
                if (op.kind == MockOp::Kind::REDUCE_ADD) out.write_f32(dst_idx, vo + vx);
                else out.write_f32(dst_idx, std::max(vo, vx));
            }
            break;
        }
        case MockOp::Kind::GATHER: {
            const MockTensor& operand = env.at(op.inputs[0]);
            const MockTensor& indices = env.at(op.inputs[1]);
            const auto op_strides = compute_row_major_strides(operand.dims);
            const auto idx_strides = compute_row_major_strides(indices.dims);
            const auto out_strides = compute_row_major_strides(out.dims);

            std::vector<bool> is_offset_dim(out.dims.size(), false);
            for (int64_t d : op.gather_offset_dims) is_offset_dim[static_cast<size_t>(d)] = true;
            std::vector<bool> is_collapsed(operand.dims.size(), false);
            for (int64_t d : op.gather_collapsed_dims) is_collapsed[static_cast<size_t>(d)] = true;
            std::vector<size_t> uncollapsed_op_dims;
            for (size_t d = 0; d < operand.dims.size(); ++d) {
                if (!is_collapsed[d]) uncollapsed_op_dims.push_back(d);
            }

            const size_t ivec_dim = static_cast<size_t>(op.gather_index_vector_dim);
            const bool has_ivec_dim = (ivec_dim < indices.dims.size());

            for (size_t oi = 0; oi < n_out; ++oi) {
                size_t rem = oi;
                std::vector<size_t> out_coord(out.dims.size(), 0);
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    out_coord[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }

                std::vector<size_t> batch_coords;
                std::vector<size_t> offset_coords;
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    if (is_offset_dim[d]) offset_coords.push_back(out_coord[d]);
                    else batch_coords.push_back(out_coord[d]);
                }

                size_t idx_base = 0;
                size_t b_ptr = 0;
                for (size_t d = 0; d < indices.dims.size(); ++d) {
                    if (has_ivec_dim && d == ivec_dim) continue;
                    idx_base += batch_coords[b_ptr++] * idx_strides[d];
                }

                std::vector<int64_t> op_coord(operand.dims.size(), 0);
                for (size_t k = 0; k < op.gather_start_index_map.size(); ++k) {
                    const size_t stride_k = has_ivec_dim ? idx_strides[ivec_dim] : 0;
                    int64_t idx_val = indices.read_i64(idx_base + k * stride_k);
                    const size_t od = static_cast<size_t>(op.gather_start_index_map[k]);
                    idx_val = std::max<int64_t>(0, std::min<int64_t>(operand.dims[od] - 1, idx_val));
                    op_coord[od] = idx_val;
                }
                for (size_t k = 0; k < offset_coords.size(); ++k) {
                    op_coord[uncollapsed_op_dims[k]] += static_cast<int64_t>(offset_coords[k]);
                }

                size_t src_idx = 0;
                for (size_t d = 0; d < operand.dims.size(); ++d) {
                    src_idx += static_cast<size_t>(op_coord[d]) * op_strides[d];
                }
                copy_mock_elem(out, oi, operand, src_idx);
            }
            break;
        }
        case MockOp::Kind::SCATTER_3D_ROWS: {
            // operand: [B3, B2, R, D], scatter_indices: [B3, B2, N_upd, 3], updates: [B3, B2, N_upd, D]
            const MockTensor& operand = env.at(op.inputs[0]);
            const MockTensor& indices = env.at(op.inputs[1]);
            const MockTensor& updates = env.at(op.inputs[2]);
            out.bytes = operand.bytes;
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto upd_strides = compute_row_major_strides(updates.dims);
            const auto idx_strides = compute_row_major_strides(indices.dims);
            const int64_t B3 = updates.dims[0];
            const int64_t B2 = updates.dims[1];
            const int64_t N_upd = updates.dims[2];
            const int64_t D = updates.dims[3];
            for (int64_t b3 = 0; b3 < B3; ++b3) {
                for (int64_t b2 = 0; b2 < B2; ++b2) {
                    for (int64_t ni = 0; ni < N_upd; ++ni) {
                        const size_t ibase = static_cast<size_t>(b3) * idx_strides[0] +
                                             static_cast<size_t>(b2) * idx_strides[1] +
                                             static_cast<size_t>(ni) * idx_strides[2];
                        const int64_t c0 = indices.read_i64(ibase + 0);
                        const int64_t c1 = indices.read_i64(ibase + 1);
                        const int64_t c2 = indices.read_i64(ibase + 2);
                        if (c0 < 0 || c0 >= out.dims[0] || c1 < 0 || c1 >= out.dims[1] || c2 < 0 || c2 >= out.dims[2]) {
                            continue;
                        }
                        const size_t dst_row = static_cast<size_t>(c0) * out_strides[0] +
                                               static_cast<size_t>(c1) * out_strides[1] +
                                               static_cast<size_t>(c2) * out_strides[2];
                        const size_t src_row = static_cast<size_t>(b3) * upd_strides[0] +
                                               static_cast<size_t>(b2) * upd_strides[1] +
                                               static_cast<size_t>(ni) * upd_strides[2];
                        if (elem_sz == 0) {
                            for (int64_t d = 0; d < D; ++d) {
                                copy_mock_elem(out, dst_row + static_cast<size_t>(d),
                                               updates, src_row + static_cast<size_t>(d));
                            }
                        } else {
                            std::memcpy(out.bytes.data() + dst_row * elem_sz,
                                        updates.bytes.data() + src_row * elem_sz,
                                        static_cast<size_t>(D) * elem_sz);
                        }
                    }
                }
            }
            break;
        }
        case MockOp::Kind::COMPARE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const MockTensor& b = env.at(op.inputs[1]);
            const bool a_int = is_mock_integer_dtype(a.dtype);
            for (size_t i = 0; i < n_out; ++i) {
                bool r = false;
                if (a_int) {
                    const int64_t va = a.read_i64(i);
                    const int64_t vb = b.read_i64(i);
                    if (op.cmp_dir == "EQ") r = (va == vb);
                    else if (op.cmp_dir == "NE") r = (va != vb);
                    else if (op.cmp_dir == "GT") r = (va > vb);
                    else if (op.cmp_dir == "GE") r = (va >= vb);
                    else if (op.cmp_dir == "LT") r = (va < vb);
                    else if (op.cmp_dir == "LE") r = (va <= vb);
                } else {
                    const float va = a.read_f32(i);
                    const float vb = b.read_f32(i);
                    if (op.cmp_dir == "EQ") r = (va == vb);
                    else if (op.cmp_dir == "NE") r = (va != vb);
                    else if (op.cmp_dir == "GT") r = (va > vb);
                    else if (op.cmp_dir == "GE") r = (va >= vb);
                    else if (op.cmp_dir == "LT") r = (va < vb);
                    else if (op.cmp_dir == "LE") r = (va <= vb);
                }
                out.write_i64(i, r ? 1 : 0);
            }
            break;
        }
        case MockOp::Kind::SELECT: {
            const MockTensor& p = env.at(op.inputs[0]);
            const MockTensor& t = env.at(op.inputs[1]);
            const MockTensor& f = env.at(op.inputs[2]);
            const bool scalar_pred = (p.num_elements() == 1);
            for (size_t i = 0; i < n_out; ++i) {
                const bool take_t = (p.read_i64(scalar_pred ? 0 : i) != 0);
                copy_mock_elem(out, i, take_t ? t : f, i);
            }
            break;
        }
        case MockOp::Kind::REPLICA_ID: {
            out.write_i64(0, replica_id);
            break;
        }
        case MockOp::Kind::DYNAMIC_SLICE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const size_t rank = a.dims.size();
            std::vector<int64_t> starts(rank, 0);
            for (size_t d = 0; d < rank && (d + 1) < op.inputs.size(); ++d) {
                int64_t st = env.at(op.inputs[d + 1]).read_i64(0);
                const int64_t max_st = std::max<int64_t>(0, a.dims[d] - out.dims[d]);
                starts[d] = std::max<int64_t>(0, std::min<int64_t>(max_st, st));
            }
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            for (size_t i = 0; i < n_out; ++i) {
                size_t rem = i;
                size_t src_idx = 0;
                for (size_t d = 0; d < rank; ++d) {
                    const size_t c = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_idx += (static_cast<size_t>(starts[d]) + c) * in_strides[d];
                }
                copy_mock_elem(out, i, a, src_idx);
            }
            break;
        }
        case MockOp::Kind::DYNAMIC_UPDATE_SLICE: {
            if (op.inputs.size() < 2) {
                throw std::runtime_error("MockPJRT: dynamic_update_slice requires operand and update");
            }
            const MockTensor& operand = env.at(op.inputs[0]);
            const MockTensor& update = env.at(op.inputs[1]);
            const size_t rank = operand.dims.size();
            if (operand.dtype != out.dtype || update.dtype != operand.dtype ||
                operand.dims != out.dims || update.dims.size() != rank) {
                throw std::runtime_error("MockPJRT: dynamic_update_slice operand/update type or rank mismatch");
            }
            for (size_t d = 0; d < rank; ++d) {
                if (update.dims[d] > operand.dims[d]) {
                    throw std::runtime_error("MockPJRT: dynamic_update_slice update dim exceeds operand dim");
                }
            }
            out.bytes = operand.bytes;
            std::vector<int64_t> starts(rank, 0);
            for (size_t d = 0; d < rank && (d + 2) < op.inputs.size(); ++d) {
                const int64_t st = env.at(op.inputs[d + 2]).read_i64(0);
                const int64_t max_st = std::max<int64_t>(0, operand.dims[d] - update.dims[d]);
                starts[d] = std::max<int64_t>(0, std::min<int64_t>(max_st, st));
            }
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto upd_strides = compute_row_major_strides(update.dims);
            const size_t n_upd = update.num_elements();
            for (size_t i = 0; i < n_upd; ++i) {
                size_t rem = i;
                size_t dst_idx = 0;
                for (size_t d = 0; d < rank; ++d) {
                    const size_t c = rem / upd_strides[d];
                    rem %= upd_strides[d];
                    dst_idx += (static_cast<size_t>(starts[d]) + c) * out_strides[d];
                }
                copy_mock_elem(out, dst_idx, update, i);
            }
            break;
        }
        case MockOp::Kind::TPU_CUSTOM_CALL: {
            if (op.inputs.size() < 2) {
                throw std::runtime_error("MockPJRT: tpu_custom_call requires at least 2 operands");
            }
            const MockTensor& x = env.at(op.inputs.at(0));
            const MockTensor& w = env.at(op.inputs.at(1));
            if (x.dims.size() != 2 || w.dims.size() != 2 || out.dims.size() != 2 ||
                x.dims[1] != w.dims[0] || out.dims[0] != x.dims[0] || out.dims[1] != w.dims[1]) {
                throw std::runtime_error("MockPJRT: tpu_custom_call operand/output shape mismatch");
            }
            const int64_t M = x.dims[0];
            const int64_t K = x.dims[1];
            const int64_t N = w.dims[1];
            const bool has_scales = (op.inputs.size() >= 3);
            const MockTensor* d_ptr = has_scales ? &env.at(op.inputs.at(2)) : nullptr;
            const MockTensor* m_ptr =
                (op.custom_call_has_min && op.inputs.size() >= 4) ? &env.at(op.inputs.at(3)) : nullptr;
            const int64_t blk = std::max<int64_t>(1, op.custom_call_blk);
            if (d_ptr != nullptr) {
                if ((K % blk) != 0 || d_ptr->dims.size() != 2 ||
                    d_ptr->dims[0] != (K / blk) || d_ptr->dims[1] != N) {
                    throw std::runtime_error("MockPJRT: tpu_custom_call scale shape mismatch");
                }
            }
            if (m_ptr != nullptr) {
                if (d_ptr == nullptr || m_ptr->dims != d_ptr->dims) {
                    throw std::runtime_error("MockPJRT: tpu_custom_call min shape mismatch");
                }
            }

            for (int64_t mi = 0; mi < M; ++mi) {
                for (int64_t ni = 0; ni < N; ++ni) {
                    double acc = 0.0;
                    for (int64_t ki = 0; ki < K; ++ki) {
                        const double xv = static_cast<double>(x.read_f32(static_cast<size_t>(mi * K + ki)));
                        double wv = static_cast<double>(w.read_f32(static_cast<size_t>(ki * N + ni)));
                        wv += op.custom_call_q4_shift;
                        if (d_ptr != nullptr) {
                            const int64_t bi = ki / blk;
                            const double dv = static_cast<double>(d_ptr->read_f32(static_cast<size_t>(bi * N + ni)));
                            wv *= dv;
                            if (m_ptr != nullptr) {
                                const double mv = static_cast<double>(m_ptr->read_f32(static_cast<size_t>(bi * N + ni)));
                                wv += mv;
                            }
                        }
                        acc += xv * wv;
                    }
                    out.write_f32(static_cast<size_t>(mi * N + ni), static_cast<float>(acc));
                }
            }
            break;
        }
        case MockOp::Kind::ALL_REDUCE_ADD:
        case MockOp::Kind::ALL_REDUCE_MAX:
        case MockOp::Kind::ALL_GATHER:
        case MockOp::Kind::REDUCE_SCATTER_ADD:
            throw std::runtime_error("MockPJRT: collective op must be evaluated in SPMD step");
    }
    return out;
}

static bool is_mock_collective_op(MockOp::Kind k) {
    return k == MockOp::Kind::ALL_REDUCE_ADD || k == MockOp::Kind::ALL_REDUCE_MAX ||
           k == MockOp::Kind::ALL_GATHER || k == MockOp::Kind::REDUCE_SCATTER_ADD;
}

static void eval_mock_collective(const MockOp& op,
                                 std::vector<std::unordered_map<std::string, MockTensor>>& envs) {
    const size_t P = envs.size();
    std::vector<std::vector<int64_t>> groups = op.replica_groups;
    if (groups.empty()) {
        std::vector<int64_t> all(P);
        for (size_t r = 0; r < P; ++r) all[r] = static_cast<int64_t>(r);
        groups.push_back(std::move(all));
    }

    for (const auto& grp : groups) {
        if (grp.empty()) continue;
        if (op.kind == MockOp::Kind::ALL_REDUCE_ADD || op.kind == MockOp::Kind::ALL_REDUCE_MAX) {
            MockTensor acc;
            acc.allocate(op.out_type.dtype, op.out_type.dims);
            const size_t n = acc.num_elements();
            const bool int_dt = is_mock_integer_dtype(acc.dtype);
            for (size_t gi = 0; gi < grp.size(); ++gi) {
                const size_t r = static_cast<size_t>(grp[gi]);
                const MockTensor& src = envs.at(r).at(op.inputs[0]);
                for (size_t i = 0; i < n; ++i) {
                    if (int_dt) {
                        const int64_t v = src.read_i64(i);
                        if (gi == 0) {
                            acc.write_i64(i, v);
                        } else if (op.kind == MockOp::Kind::ALL_REDUCE_ADD) {
                            acc.write_i64(
                                i, static_cast<int64_t>(static_cast<uint64_t>(acc.read_i64(i)) +
                                                        static_cast<uint64_t>(v)));
                        } else {
                            acc.write_i64(i, std::max(acc.read_i64(i), v));
                        }
                    } else {
                        const float v = src.read_f32(i);
                        if (gi == 0) {
                            acc.write_f32(i, v);
                        } else if (op.kind == MockOp::Kind::ALL_REDUCE_ADD) {
                            acc.write_f32(i, acc.read_f32(i) + v);
                        } else {
                            acc.write_f32(i, std::max(acc.read_f32(i), v));
                        }
                    }
                }
            }
            for (int64_t r : grp) {
                envs.at(static_cast<size_t>(r))[op.out_name] = acc;
            }
        } else if (op.kind == MockOp::Kind::ALL_GATHER) {
            MockTensor gathered;
            gathered.allocate(op.out_type.dtype, op.out_type.dims);
            const size_t cdim = static_cast<size_t>(op.collective_dim);
            const auto out_strides = compute_row_major_strides(gathered.dims);
            int64_t offset = 0;
            for (int64_t r : grp) {
                const MockTensor& src = envs.at(static_cast<size_t>(r)).at(op.inputs[0]);
                const auto in_strides = compute_row_major_strides(src.dims);
                const size_t n_in = src.num_elements();
                for (size_t i = 0; i < n_in; ++i) {
                    size_t rem = i;
                    size_t dst_idx = 0;
                    for (size_t d = 0; d < src.dims.size(); ++d) {
                        const size_t c = rem / in_strides[d];
                        rem %= in_strides[d];
                        const size_t dc = (d == cdim) ? (c + static_cast<size_t>(offset)) : c;
                        dst_idx += dc * out_strides[d];
                    }
                    copy_mock_elem(gathered, dst_idx, src, i);
                }
                offset += src.dims[cdim];
            }
            for (int64_t r : grp) {
                envs.at(static_cast<size_t>(r))[op.out_name] = gathered;
            }
        } else if (op.kind == MockOp::Kind::REDUCE_SCATTER_ADD) {
            const MockTensor& first = envs.at(static_cast<size_t>(grp[0])).at(op.inputs[0]);
            MockTensor full;
            full.allocate(first.dtype, first.dims);
            const size_t n_full = full.num_elements();
            const bool int_dt = is_mock_integer_dtype(full.dtype);
            for (size_t gi = 0; gi < grp.size(); ++gi) {
                const MockTensor& src = envs.at(static_cast<size_t>(grp[gi])).at(op.inputs[0]);
                for (size_t i = 0; i < n_full; ++i) {
                    if (int_dt) {
                        const int64_t v = src.read_i64(i);
                        full.write_i64(
                            i, (gi == 0) ? v
                                         : static_cast<int64_t>(static_cast<uint64_t>(full.read_i64(i)) +
                                                                static_cast<uint64_t>(v)));
                    } else {
                        const float v = src.read_f32(i);
                        full.write_f32(i, (gi == 0) ? v : (full.read_f32(i) + v));
                    }
                }
            }
            const size_t sdim = static_cast<size_t>(op.collective_dim);
            const int64_t chunk = op.out_type.dims[sdim];
            const auto full_strides = compute_row_major_strides(full.dims);
            const auto out_strides = compute_row_major_strides(op.out_type.dims);
            for (size_t gi = 0; gi < grp.size(); ++gi) {
                MockTensor shard;
                shard.allocate(op.out_type.dtype, op.out_type.dims);
                const size_t n_sh = shard.num_elements();
                const int64_t start = static_cast<int64_t>(gi) * chunk;
                for (size_t i = 0; i < n_sh; ++i) {
                    size_t rem = i;
                    size_t src_idx = 0;
                    for (size_t d = 0; d < shard.dims.size(); ++d) {
                        const size_t c = rem / out_strides[d];
                        rem %= out_strides[d];
                        const size_t sc = (d == sdim) ? (c + static_cast<size_t>(start)) : c;
                        src_idx += sc * full_strides[d];
                    }
                    copy_mock_elem(shard, i, full, src_idx);
                }
                envs.at(static_cast<size_t>(grp[gi]))[op.out_name] = std::move(shard);
            }
        }
    }
}

// Mock PJRT C API opaque structs
struct MockError {
    std::string message;
};
struct MockEvent {};
struct MockClient;
struct MockDeviceDescription {
    int id = 0;
    std::string kind;
    std::string to_string;
    std::vector<int64_t> coords;
    int64_t core_on_chip = -1;
    std::vector<PJRT_NamedValue> attrs;
};
struct MockDevice {
    MockClient* client = nullptr;
    int id = 0;
    MockDeviceDescription desc;
    // Shared ownership so orphan MockBuffers outliving MockClient::destroy never UAF.
    std::shared_ptr<std::atomic<int64_t>> allocated_bytes = std::make_shared<std::atomic<int64_t>>(0);
    int64_t hbm_limit = 16909336064LL;  // 15.748 GiB (v5e-1 default)
};
struct MockClient {
    std::vector<std::unique_ptr<MockDevice>> devices;
    std::vector<PJRT_Device*> device_ptrs;

    MockClient(int n_devs,
               MockTopologyProfile profile = MockTopologyProfile::AUTO,
               bool scramble_order = false) {
        int profile_devs = 0;
        switch (profile) {
            case MockTopologyProfile::V5E_1:    profile_devs = 1; break;
            case MockTopologyProfile::TPU7X_1T: profile_devs = 2; break;
            case MockTopologyProfile::V6E_8:    profile_devs = 8; break;
            case MockTopologyProfile::TPU7X_4T: profile_devs = 8; break;
            case MockTopologyProfile::AUTO:     profile_devs = 0; break;
        }
        if (profile_devs > 0) {
            if (n_devs > 1 && n_devs != profile_devs) {
                std::fprintf(stderr,
                             "ggml-cloudtpu: CLOUDTPU_MOCK_DEVICES=%d conflicts with named "
                             "CLOUDTPU_MOCK_PROFILE (expects %d logical devices); using %d\n",
                             n_devs, profile_devs, profile_devs);
            }
            n_devs = profile_devs;
        }
        n_devs = std::max(1, std::min(16, n_devs));

        std::string kind_str = "Cloud TPU Host Mock (v5e-1)";
        int64_t per_chip_hbm = 16909336064LL;  // 15.748 GiB (v5e-1)
        bool dual_chiplet = false;
        if (profile == MockTopologyProfile::TPU7X_1T ||
            (profile == MockTopologyProfile::AUTO && n_devs == 2)) {
            kind_str = "Cloud TPU Host Mock (tpu7x-1t)";
            per_chip_hbm = 103079215104LL;  // 96 GiB/logical chiplet (192 GiB/package)
            dual_chiplet = true;
        } else if (profile == MockTopologyProfile::TPU7X_4T) {
            kind_str = "Cloud TPU Host Mock (tpu7x-4t)";
            per_chip_hbm = 103079215104LL;  // 96 GiB/logical chiplet (768 GiB total across 8 logical)
            dual_chiplet = true;
        } else if (profile == MockTopologyProfile::V6E_8 ||
                   (profile == MockTopologyProfile::AUTO && n_devs == 8)) {
            kind_str = "Cloud TPU Host Mock (v6e-8)";
            per_chip_hbm = 34359738368LL;  // 32 GiB/chip (256 GiB total)
        } else if (n_devs == 4) {
            kind_str = "Cloud TPU Host Mock (4-chip)";
            per_chip_hbm = 34359738368LL;  // 32 GiB/chip
        } else if (n_devs > 1) {
            kind_str = "Cloud TPU Host Mock (" + std::to_string(n_devs) + "-chip)";
            per_chip_hbm = 34359738368LL;
        }

        // Build logical device layout, optionally scrambled (all chiplet-0s then chiplet-1s +
        // non-Hamiltonian 2x2 torus order) so tests can verify PjrtRuntime D2D Hamiltonian sorting.
        struct DevEntry {
            int hw_id;
            std::vector<int64_t> coords;
            int64_t core_on_chip;
        };
        std::vector<DevEntry> entries;
        entries.reserve(static_cast<size_t>(n_devs));
        if (dual_chiplet && (n_devs % 2) == 0) {
            const int n_pkgs = n_devs / 2;
            const int grid_x = (n_pkgs == 4) ? 2 : n_pkgs;
            if (scramble_order) {
                for (int core = 1; core >= 0; --core) {
                    for (int pkg = n_pkgs - 1; pkg >= 0; --pkg) {
                        const int x = pkg % grid_x;
                        const int y = pkg / grid_x;
                        const int hw_id = pkg * 2 + core;
                        entries.push_back({hw_id, {x, y, 0}, core});
                    }
                }
            } else {
                for (int i = 0; i < n_devs; ++i) {
                    const int pkg = i / 2;
                    const int core = i % 2;
                    const int x = pkg % grid_x;
                    const int y = pkg / grid_x;
                    entries.push_back({i, {x, y, 0}, core});
                }
            }
        } else {
            for (int i = 0; i < n_devs; ++i) {
                entries.push_back({i, {i, 0, 0}, -1});
            }
        }

        devices.reserve(static_cast<size_t>(n_devs));
        device_ptrs.reserve(static_cast<size_t>(n_devs));
        for (int i = 0; i < n_devs; ++i) {
            const auto& e = entries[static_cast<size_t>(i)];
            auto d = std::make_unique<MockDevice>();
            d->client = this;
            d->id = e.hw_id;
            d->hbm_limit = per_chip_hbm;
            d->desc.id = e.hw_id;
            d->desc.kind = kind_str;
            d->desc.to_string = "CLOUDTPU_MOCK(id=" + std::to_string(e.hw_id) + ")";
            d->desc.coords = e.coords;
            d->desc.core_on_chip = e.core_on_chip;

            PJRT_NamedValue coords_attr = {};
            coords_attr.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            coords_attr.name = "coords";
            coords_attr.name_size = 6;
            coords_attr.type = PJRT_NamedValue_kInt64List;
            coords_attr.int64_array_value = d->desc.coords.data();
            coords_attr.value_size = d->desc.coords.size();
            d->desc.attrs.push_back(coords_attr);

            if (d->desc.core_on_chip >= 0) {
                PJRT_NamedValue coc_attr = {};
                coc_attr.struct_size = PJRT_NamedValue_STRUCT_SIZE;
                coc_attr.name = "core_on_chip";
                coc_attr.name_size = 12;
                coc_attr.type = PJRT_NamedValue_kInt64;
                coc_attr.int64_value = d->desc.core_on_chip;
                coc_attr.value_size = 1;
                d->desc.attrs.push_back(coc_attr);
            }

            device_ptrs.push_back(reinterpret_cast<PJRT_Device*>(d.get()));
            devices.push_back(std::move(d));
        }
    }
};
struct MockBuffer {
    MockTensor tensor;
    std::shared_ptr<std::atomic<int64_t>> allocated_bytes;
    int device_id = 0;
};
struct MockExecutable {
    MockProgram program;
};
struct MockLoadedExecutable {
    MockExecutable exec;
    std::vector<std::shared_ptr<std::atomic<int64_t>>> device_allocated_bytes;
    std::vector<int> assigned_device_ids;
};

static bool pb_read_varint(const uint8_t* data, size_t size, size_t& pos, uint64_t& out) {
    out = 0;
    unsigned shift = 0;
    while (pos < size && shift < 64) {
        const uint8_t b = data[pos++];
        out |= static_cast<uint64_t>(b & 0x7fU) << shift;
        if ((b & 0x80U) == 0) {
            return true;
        }
        shift += 7;
    }
    return false;
}

static bool pb_skip_field(const uint8_t* data, size_t size, size_t& pos, uint64_t wire_type) {
    uint64_t v = 0;
    switch (wire_type) {
        case 0:
            return pb_read_varint(data, size, pos, v);
        case 1:
            if (size - pos < 8) return false;
            pos += 8;
            return true;
        case 2:
            if (!pb_read_varint(data, size, pos, v) || v > size - pos) return false;
            pos += static_cast<size_t>(v);
            return true;
        case 5:
            if (size - pos < 4) return false;
            pos += 4;
            return true;
        default:
            return false;
    }
}

static std::vector<int> parse_assigned_device_ids_from_compile_options(const char* buf, size_t len) {
    std::vector<int> ids;
    const auto* data = reinterpret_cast<const uint8_t*>(buf);
    size_t pos = 0;
    while (pos < len) {
        uint64_t tag = 0;
        if (!pb_read_varint(data, len, pos, tag)) break;
        const uint64_t field = tag >> 3;
        const uint64_t wt = tag & 7U;
        if (field == 3 && wt == 2) {  // executable_build_options
            uint64_t ebo_len = 0;
            if (!pb_read_varint(data, len, pos, ebo_len) || ebo_len > len - pos) break;
            const uint8_t* ebo = data + pos;
            size_t epos = 0;
            while (epos < ebo_len) {
                uint64_t etag = 0;
                if (!pb_read_varint(ebo, static_cast<size_t>(ebo_len), epos, etag)) break;
                const uint64_t efield = etag >> 3;
                const uint64_t ewt = etag & 7U;
                if (efield == 9 && ewt == 2) {  // device_assignment
                    uint64_t da_len = 0;
                    if (!pb_read_varint(ebo, static_cast<size_t>(ebo_len), epos, da_len) ||
                        da_len > ebo_len - epos) break;
                    const uint8_t* da = ebo + epos;
                    size_t dpos = 0;
                    while (dpos < da_len) {
                        uint64_t dtag = 0;
                        if (!pb_read_varint(da, static_cast<size_t>(da_len), dpos, dtag)) break;
                        const uint64_t dfield = dtag >> 3;
                        const uint64_t dwt = dtag & 7U;
                        if (dfield == 3 && dwt == 2) {  // computation_devices
                            uint64_t cd_len = 0;
                            if (!pb_read_varint(da, static_cast<size_t>(da_len), dpos, cd_len) ||
                                cd_len > da_len - dpos) break;
                            const uint8_t* cd = da + dpos;
                            size_t cpos = 0;
                            while (cpos < cd_len) {
                                uint64_t ctag = 0;
                                if (!pb_read_varint(cd, static_cast<size_t>(cd_len), cpos, ctag)) break;
                                const uint64_t cfield = ctag >> 3;
                                const uint64_t cwt = ctag & 7U;
                                if (cfield == 1 && cwt == 2) {  // packed replica_device_ids
                                    uint64_t ids_len = 0;
                                    if (!pb_read_varint(cd, static_cast<size_t>(cd_len), cpos, ids_len) ||
                                        ids_len > cd_len - cpos) break;
                                    size_t ipos = 0;
                                    while (ipos < ids_len) {
                                        uint64_t dev_id = 0;
                                        if (!pb_read_varint(cd + cpos, static_cast<size_t>(ids_len), ipos, dev_id)) {
                                            break;
                                        }
                                        ids.push_back(static_cast<int>(dev_id));
                                    }
                                    cpos += static_cast<size_t>(ids_len);
                                } else if (cfield == 1 && cwt == 0) {
                                    uint64_t dev_id = 0;
                                    if (pb_read_varint(cd, static_cast<size_t>(cd_len), cpos, dev_id)) {
                                        ids.push_back(static_cast<int>(dev_id));
                                    }
                                } else if (!pb_skip_field(cd, static_cast<size_t>(cd_len), cpos, cwt)) {
                                    break;
                                }
                            }
                            dpos += static_cast<size_t>(cd_len);
                        } else if (!pb_skip_field(da, static_cast<size_t>(da_len), dpos, dwt)) {
                            break;
                        }
                    }
                    epos += static_cast<size_t>(da_len);
                } else if (!pb_skip_field(ebo, static_cast<size_t>(ebo_len), epos, ewt)) {
                    break;
                }
            }
            pos += static_cast<size_t>(ebo_len);
        } else if (!pb_skip_field(data, len, pos, wt)) {
            break;
        }
    }
    if (ids.empty()) {
        ids.push_back(0);
    }
    return ids;
}

static PJRT_Error* mock_error(const std::string& msg) {
    return reinterpret_cast<PJRT_Error*>(new MockError{msg});
}

static void mock_error_destroy(PJRT_Error_Destroy_Args* args) {
    delete reinterpret_cast<MockError*>(args->error);
}

static void mock_error_message(PJRT_Error_Message_Args* args) {
    const auto* e = reinterpret_cast<const MockError*>(args->error);
    args->message = e->message.data();
    args->message_size = e->message.size();
}

static PJRT_Error* mock_error_get_code(PJRT_Error_GetCode_Args* args) {
    args->code = PJRT_Error_Code_INTERNAL;
    return nullptr;
}

static PJRT_Error* mock_plugin_initialize(PJRT_Plugin_Initialize_Args*) { return nullptr; }
static PJRT_Error* mock_event_destroy(PJRT_Event_Destroy_Args* args) {
    delete reinterpret_cast<MockEvent*>(args->event);
    return nullptr;
}
static PJRT_Error* mock_event_await(PJRT_Event_Await_Args*) { return nullptr; }

static PJRT_Error* mock_client_create(PJRT_Client_Create_Args* args) {
    int n_devs = 1;
    MockTopologyProfile profile = MockTopologyProfile::AUTO;
    bool scramble_order = false;
    if (const char* penv = std::getenv("CLOUDTPU_MOCK_PROFILE")) {
        profile = parse_mock_topology_profile(penv);
    }
    if (args->create_options && args->num_options > 0) {
        for (size_t i = 0; i < args->num_options; ++i) {
            const PJRT_NamedValue& opt = args->create_options[i];
            if (!opt.name) continue;
            const std::string opt_name(opt.name, opt.name_size);
            if (opt_name == "mock_device_count" && opt.type == PJRT_NamedValue_kInt64) {
                n_devs = std::max(1, std::min(16, static_cast<int>(opt.int64_value)));
            } else if (opt_name == "mock_profile" && opt.type == PJRT_NamedValue_kString && opt.string_value) {
                const std::string pstr(opt.string_value, opt.value_size);
                profile = parse_mock_topology_profile(pstr.c_str());
            } else if (opt_name == "mock_scramble_order" && opt.type == PJRT_NamedValue_kBool) {
                scramble_order = opt.bool_value;
            }
        }
    }
    if (n_devs <= 1) {
        if (const char* v = std::getenv("CLOUDTPU_MOCK_DEVICES")) {
            const int parsed = std::atoi(v);
            if (parsed > 1 && parsed <= 16) {
                n_devs = parsed;
            }
        }
    }
    args->client = reinterpret_cast<PJRT_Client*>(new MockClient(n_devs, profile, scramble_order));
    return nullptr;
}
static PJRT_Error* mock_client_destroy(PJRT_Client_Destroy_Args* args) {
    delete reinterpret_cast<MockClient*>(args->client);
    return nullptr;
}
static PJRT_Error* mock_client_platform_name(PJRT_Client_PlatformName_Args* args) {
    static const char kName[] = "tpu-mock";
    args->platform_name = kName;
    args->platform_name_size = sizeof(kName) - 1;
    return nullptr;
}
static PJRT_Error* mock_client_platform_version(PJRT_Client_PlatformVersion_Args* args) {
    static const char kVer[] = "mock-pjrt-0.116";
    args->platform_version = kVer;
    args->platform_version_size = sizeof(kVer) - 1;
    return nullptr;
}
static PJRT_Error* mock_client_addressable_devices(PJRT_Client_AddressableDevices_Args* args) {
    auto* c = reinterpret_cast<MockClient*>(args->client);
    args->addressable_devices = c->device_ptrs.data();
    args->num_addressable_devices = c->device_ptrs.size();
    return nullptr;
}
static PJRT_Error* mock_device_get_description(PJRT_Device_GetDescription_Args* args) {
    auto* d = reinterpret_cast<MockDevice*>(args->device);
    args->device_description = reinterpret_cast<PJRT_DeviceDescription*>(&d->desc);
    return nullptr;
}
static PJRT_Error* mock_dev_desc_id(PJRT_DeviceDescription_Id_Args* args) {
    const auto* desc = reinterpret_cast<const MockDeviceDescription*>(args->device_description);
    args->id = desc ? desc->id : 0;
    return nullptr;
}
static PJRT_Error* mock_dev_desc_attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    const auto* desc = reinterpret_cast<const MockDeviceDescription*>(args->device_description);
    if (desc && !desc->attrs.empty()) {
        args->attributes = desc->attrs.data();
        args->num_attributes = desc->attrs.size();
    } else {
        args->attributes = nullptr;
        args->num_attributes = 0;
    }
    return nullptr;
}
static PJRT_Error* mock_dev_desc_kind(PJRT_DeviceDescription_Kind_Args* args) {
    const auto* desc = reinterpret_cast<const MockDeviceDescription*>(args->device_description);
    if (desc) {
        args->device_kind = desc->kind.data();
        args->device_kind_size = desc->kind.size();
    } else {
        static const char kKind[] = "Cloud TPU Host Mock";
        args->device_kind = kKind;
        args->device_kind_size = sizeof(kKind) - 1;
    }
    return nullptr;
}
static PJRT_Error* mock_dev_desc_to_string(PJRT_DeviceDescription_ToString_Args* args) {
    const auto* desc = reinterpret_cast<const MockDeviceDescription*>(args->device_description);
    if (desc) {
        args->to_string = desc->to_string.data();
        args->to_string_size = desc->to_string.size();
    } else {
        static const char kStr[] = "CLOUDTPU_MOCK(id=0)";
        args->to_string = kStr;
        args->to_string_size = sizeof(kStr) - 1;
    }
    return nullptr;
}
static PJRT_Error* mock_device_memory_stats(PJRT_Device_MemoryStats_Args* args) {
    auto* d = reinterpret_cast<MockDevice*>(args->device);
    args->bytes_in_use =
        (d && d->allocated_bytes) ? d->allocated_bytes->load(std::memory_order_relaxed) : 0;
    args->bytes_limit_is_set = true;
    args->bytes_limit = d ? d->hbm_limit : 16909336064LL;
    return nullptr;
}

static PJRT_Error* mock_buffer_from_host(PJRT_Client_BufferFromHostBuffer_Args* args) {
    auto* c = reinterpret_cast<MockClient*>(args->client);
    auto* d = args->device ? reinterpret_cast<MockDevice*>(args->device)
                           : (c && !c->devices.empty() ? c->devices[0].get() : nullptr);
    const PjrtDType dt = from_pjrt_buffer_type(args->type);
    if (dt == PjrtDType::INVALID) {
        return mock_error("MockPJRT: unsupported PJRT_Buffer_Type");
    }
    std::vector<int64_t> shape(args->dims, args->dims + args->num_dims);
    auto* mb = new MockBuffer();
    mb->allocated_bytes = d ? d->allocated_bytes : nullptr;
    mb->device_id = d ? d->id : 0;
    mb->tensor.allocate(dt, std::move(shape));
    if (!mb->tensor.bytes.empty() && args->data) {
        const size_t nelems = mb->tensor.num_elements();
        if (dt == PjrtDType::S4) {
            pack_s4_from_i8(static_cast<const int8_t*>(args->data), nelems, mb->tensor.bytes.data());
        } else if (dt == PjrtDType::U4) {
            pack_u4_from_u8(static_cast<const uint8_t*>(args->data), nelems, mb->tensor.bytes.data());
        } else {
            std::memcpy(mb->tensor.bytes.data(), args->data, mb->tensor.bytes.size());
        }
    }
    if (mb->allocated_bytes) {
        mb->allocated_bytes->fetch_add(static_cast<int64_t>(mb->tensor.bytes.size()), std::memory_order_relaxed);
    }
    args->done_with_host_buffer = reinterpret_cast<PJRT_Event*>(new MockEvent());
    args->buffer = reinterpret_cast<PJRT_Buffer*>(mb);
    return nullptr;
}
static PJRT_Error* mock_buffer_destroy(PJRT_Buffer_Destroy_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->buffer);
    if (mb && mb->allocated_bytes) {
        mb->allocated_bytes->fetch_sub(static_cast<int64_t>(mb->tensor.bytes.size()), std::memory_order_relaxed);
    }
    delete mb;
    return nullptr;
}
static PJRT_Error* mock_buffer_element_type(PJRT_Buffer_ElementType_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->buffer);
    args->type = to_pjrt_buffer_type(mb->tensor.dtype);
    return nullptr;
}
static PJRT_Error* mock_buffer_dimensions(PJRT_Buffer_Dimensions_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->buffer);
    args->dims = mb->tensor.dims.empty() ? nullptr : mb->tensor.dims.data();
    args->num_dims = mb->tensor.dims.size();
    return nullptr;
}
static PJRT_Error* mock_buffer_to_host(PJRT_Buffer_ToHostBuffer_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->src);
    const size_t nelems = mb->tensor.num_elements();
    const size_t host_bytes = pjrt_host_nbytes(mb->tensor.dtype, nelems);
    if (!args->dst) {
        args->dst_size = host_bytes;
        args->event = nullptr;
        return nullptr;
    }
    if (args->dst_size < host_bytes) {
        return mock_error("MockPJRT: ToHostBuffer dst_size too small");
    }
    if (nelems > 0) {
        if (mb->tensor.dtype == PjrtDType::S4 || mb->tensor.dtype == PjrtDType::U4) {
            // Mirror legacy libtpu.so behavior where S4 ToHostBuffer returns zero-extended
            // low nibbles (0x00..0x0F), exercising buffer_to_host's sign-extension path.
            unpack_u4_to_u8(mb->tensor.bytes.data(), nelems, static_cast<uint8_t*>(args->dst));
        } else {
            std::memcpy(args->dst, mb->tensor.bytes.data(), mb->tensor.bytes.size());
        }
    }
    args->event = reinterpret_cast<PJRT_Event*>(new MockEvent());
    return nullptr;
}
static PJRT_Error* mock_buffer_ready_event(PJRT_Buffer_ReadyEvent_Args* args) {
    args->event = reinterpret_cast<PJRT_Event*>(new MockEvent());
    return nullptr;
}

static PJRT_Error* mock_client_compile(PJRT_Client_Compile_Args* args) {
    try {
        if (!args->compile_options || args->compile_options_size == 0) {
            return mock_error("MockPJRT: compile_options must be non-empty (matches libtpu requirement)");
        }
        const std::string fmt(args->program->format, args->program->format_size);
        if (fmt != "mlir") {
            return mock_error("MockPJRT: expected program format 'mlir', got '" + fmt + "'");
        }
        auto* c = reinterpret_cast<MockClient*>(args->client);
        const std::string code(args->program->code, args->program->code_size);
        auto* le = new MockLoadedExecutable();
        le->exec.program = parse_stablehlo_mlir(code);
        if (c) {
            le->device_allocated_bytes.reserve(c->devices.size());
            for (const auto& d : c->devices) {
                le->device_allocated_bytes.push_back(d->allocated_bytes);
            }
        }
        le->assigned_device_ids =
            parse_assigned_device_ids_from_compile_options(args->compile_options, args->compile_options_size);
        args->executable = reinterpret_cast<PJRT_LoadedExecutable*>(le);
        return nullptr;
    } catch (const std::exception& e) {
        return mock_error(e.what());
    }
}
static PJRT_Error* mock_loaded_exec_destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    delete reinterpret_cast<MockLoadedExecutable*>(args->executable);
    return nullptr;
}
static PJRT_Error* mock_loaded_exec_get_exec(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    auto* le = reinterpret_cast<MockLoadedExecutable*>(args->loaded_executable);
    auto* ex = new MockExecutable(le->exec);
    args->executable = reinterpret_cast<PJRT_Executable*>(ex);
    return nullptr;
}
static PJRT_Error* mock_exec_destroy(PJRT_Executable_Destroy_Args* args) {
    delete reinterpret_cast<MockExecutable*>(args->executable);
    return nullptr;
}
static PJRT_Error* mock_exec_num_outputs(PJRT_Executable_NumOutputs_Args* args) {
    auto* ex = reinterpret_cast<MockExecutable*>(args->executable);
    args->num_outputs = ex->program.return_names.size();
    return nullptr;
}
static PJRT_Error* mock_loaded_exec_execute(PJRT_LoadedExecutable_Execute_Args* args) {
    try {
        auto* le = reinterpret_cast<MockLoadedExecutable*>(args->executable);
        const size_t n_devs = std::max<size_t>(1, args->num_devices);
        const int primary_assigned_id = le->assigned_device_ids.empty() ? 0 : le->assigned_device_ids[0];
        if (args->execute_device != nullptr) {
            auto* req_dev = reinterpret_cast<MockDevice*>(args->execute_device);
            if (req_dev->id != primary_assigned_id) {
                return mock_error("MockPJRT: execute_device id " + std::to_string(req_dev->id) +
                                  " does not match compiled device assignment " +
                                  std::to_string(primary_assigned_id));
            }
        }
        if (n_devs != le->assigned_device_ids.size()) {
            return mock_error("MockPJRT: num_devices " + std::to_string(n_devs) +
                              " does not match compiled replica assignment count " +
                              std::to_string(le->assigned_device_ids.size()));
        }
        const MockProgram& prog = le->exec.program;
        if (args->num_args != prog.args.size()) {
            return mock_error("MockPJRT: argument count mismatch");
        }
        std::vector<std::unordered_map<std::string, MockTensor>> envs(n_devs);
        std::vector<std::shared_ptr<std::atomic<int64_t>>> alloc_bytes(n_devs);
        for (size_t d = 0; d < n_devs; ++d) {
            const int expected_dev_id = le->assigned_device_ids[d];
            if (expected_dev_id >= 0 &&
                static_cast<size_t>(expected_dev_id) < le->device_allocated_bytes.size()) {
                alloc_bytes[d] = le->device_allocated_bytes[static_cast<size_t>(expected_dev_id)];
            }
            for (size_t i = 0; i < args->num_args; ++i) {
                auto* mb = reinterpret_cast<MockBuffer*>(args->argument_lists[d][i]);
                if (!mb) {
                    return mock_error("MockPJRT: null input buffer at arg " + std::to_string(i));
                }
                if (mb->device_id != expected_dev_id) {
                    return mock_error("MockPJRT: input buffer on device " + std::to_string(mb->device_id) +
                                      " does not match executable device " +
                                      std::to_string(expected_dev_id));
                }
                const MockTypeSpec& expected_ts = prog.args[i].second;
                if (mb->tensor.dtype != expected_ts.dtype || mb->tensor.dims != expected_ts.dims) {
                    return mock_error("MockPJRT: input arg " + std::to_string(i) + " on device " +
                                      std::to_string(expected_dev_id) +
                                      " dtype/shape does not match compiled parameter signature");
                }
                if (!alloc_bytes[d]) alloc_bytes[d] = mb->allocated_bytes;
                envs[d][prog.args[i].first] = mb->tensor;
            }
        }
        std::unordered_map<std::string, size_t> last_use;
        for (size_t idx = 0; idx < prog.ops.size(); ++idx) {
            for (const std::string& inp : prog.ops[idx].inputs) {
                last_use[inp] = idx;
            }
        }
        for (const std::string& rname : prog.return_names) {
            last_use[rname] = prog.ops.size();
        }
        for (size_t idx = 0; idx < prog.ops.size(); ++idx) {
            const MockOp& op = prog.ops[idx];
            if (is_mock_collective_op(op.kind)) {
                eval_mock_collective(op, envs);
            } else {
                for (size_t d = 0; d < n_devs; ++d) {
                    envs[d][op.out_name] = eval_mock_op(op, envs[d], static_cast<int64_t>(d));
                }
            }
            for (const std::string& inp : op.inputs) {
                auto it = last_use.find(inp);
                if (it != last_use.end() && it->second == idx) {
                    for (size_t d = 0; d < n_devs; ++d) {
                        envs[d].erase(inp);
                    }
                }
            }
        }
        for (size_t d = 0; d < n_devs; ++d) {
            const int dev_id = le->assigned_device_ids[d];
            for (size_t oi = 0; oi < prog.return_names.size(); ++oi) {
                auto* out_mb = new MockBuffer();
                out_mb->allocated_bytes = alloc_bytes[d];
                out_mb->device_id = dev_id;
                // Copy rather than move so repeated SSA values in return_names stay valid.
                out_mb->tensor = envs[d].at(prog.return_names[oi]);
                if (out_mb->allocated_bytes) {
                    out_mb->allocated_bytes->fetch_add(
                        static_cast<int64_t>(out_mb->tensor.bytes.size()), std::memory_order_relaxed);
                }
                args->output_lists[d][oi] = reinterpret_cast<PJRT_Buffer*>(out_mb);
            }
            if (args->device_complete_events) {
                args->device_complete_events[d] = reinterpret_cast<PJRT_Event*>(new MockEvent());
            }
        }
        return nullptr;
    } catch (const std::exception& e) {
        return mock_error(e.what());
    }
}

static const PJRT_Api* get_mock_pjrt_api() {
    static PJRT_Api api = []() {
        PJRT_Api a = {};
        a.struct_size = PJRT_Api_STRUCT_SIZE;
        a.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
        a.pjrt_api_version.major_version = PJRT_API_MAJOR;
        a.pjrt_api_version.minor_version = PJRT_API_MINOR;

        a.PJRT_Error_Destroy = mock_error_destroy;
        a.PJRT_Error_Message = mock_error_message;
        a.PJRT_Error_GetCode = mock_error_get_code;
        a.PJRT_Plugin_Initialize = mock_plugin_initialize;
        a.PJRT_Event_Destroy = mock_event_destroy;
        a.PJRT_Event_Await = mock_event_await;
        a.PJRT_Client_Create = mock_client_create;
        a.PJRT_Client_Destroy = mock_client_destroy;
        a.PJRT_Client_PlatformName = mock_client_platform_name;
        a.PJRT_Client_PlatformVersion = mock_client_platform_version;
        a.PJRT_Client_AddressableDevices = mock_client_addressable_devices;
        a.PJRT_Client_Compile = mock_client_compile;
        a.PJRT_Client_BufferFromHostBuffer = mock_buffer_from_host;
        a.PJRT_Device_GetDescription = mock_device_get_description;
        a.PJRT_DeviceDescription_Id = mock_dev_desc_id;
        a.PJRT_DeviceDescription_Attributes = mock_dev_desc_attributes;
        a.PJRT_DeviceDescription_Kind = mock_dev_desc_kind;
        a.PJRT_DeviceDescription_ToString = mock_dev_desc_to_string;
        a.PJRT_Device_MemoryStats = mock_device_memory_stats;
        a.PJRT_LoadedExecutable_Destroy = mock_loaded_exec_destroy;
        a.PJRT_LoadedExecutable_GetExecutable = mock_loaded_exec_get_exec;
        a.PJRT_LoadedExecutable_Execute = mock_loaded_exec_execute;
        a.PJRT_Executable_Destroy = mock_exec_destroy;
        a.PJRT_Executable_NumOutputs = mock_exec_num_outputs;
        a.PJRT_Buffer_Destroy = mock_buffer_destroy;
        a.PJRT_Buffer_ElementType = mock_buffer_element_type;
        a.PJRT_Buffer_Dimensions = mock_buffer_dimensions;
        a.PJRT_Buffer_ToHostBuffer = mock_buffer_to_host;
        a.PJRT_Buffer_ReadyEvent = mock_buffer_ready_event;
        return a;
    }();
    return &api;
}

}  // namespace

// =========================================================================
// PjrtRuntime Implementation
// =========================================================================

PjrtRuntime::~PjrtRuntime() {
    shutdown();
}

void PjrtRuntime::destroy_error(PJRT_Error* err) const {
    if (!err || !api_ || !PJRT_API_HAS(api_, PJRT_Error_Destroy)) {
        return;
    }
    PJRT_Error_Destroy_Args d_args = {};
    d_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    d_args.error = err;
    api_->PJRT_Error_Destroy(&d_args);
}

void PjrtRuntime::check_error(PJRT_Error* err, const char* where) const {
    if (!err) {
        return;
    }
    std::string msg = "(no PJRT_Error_Message)";
    if (api_ && PJRT_API_HAS(api_, PJRT_Error_Message)) {
        PJRT_Error_Message_Args m_args = {};
        m_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        m_args.error = err;
        api_->PJRT_Error_Message(&m_args);
        if (m_args.message && m_args.message_size > 0) {
            msg.assign(m_args.message, m_args.message_size);
        }
    }
    PJRT_Error_Code code = PJRT_Error_Code_UNKNOWN;
    if (api_ && PJRT_API_HAS(api_, PJRT_Error_GetCode)) {
        PJRT_Error_GetCode_Args c_args = {};
        c_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
        c_args.error = err;
        PJRT_Error* code_err = api_->PJRT_Error_GetCode(&c_args);
        if (code_err) {
            destroy_error(code_err);
        } else {
            code = c_args.code;
        }
    }
    destroy_error(err);
    throw PjrtError(code, std::string(where) + ": " + msg);
}

std::string PjrtRuntime::resolve_libtpu_path(const std::string& override_path) const {
    if (!override_path.empty()) {
        return override_path;
    }
    if (const char* env_lib = std::getenv("LIBTPU_PATH")) {
        if (env_lib[0] != '\0') return env_lib;
    }
    if (const char* env_tpu = std::getenv("TPU_LIBRARY_PATH")) {
        if (env_tpu[0] != '\0') return env_tpu;
    }
    static const char* const kCandidates[] = {
        "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so",
        "libtpu.so",
    };
    for (const char* p : kCandidates) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "libtpu.so";
}

void PjrtRuntime::init(const RuntimeConfig& cfg) {
    if (initialized_) {
        return;
    }
    try {
    cfg_ = cfg;
    is_mock_ = cfg.mock_mode;

    if (is_mock_) {
        api_ = get_mock_pjrt_api();
    } else {
#if defined(_WIN32)
        throw std::runtime_error(
            "Live libtpu.so loading is only supported on Linux/POSIX; use mock mode on Windows");
#else
        setenv("GRPC_VERBOSITY", "ERROR", 0);
        if (cfg.async_collectives) {
            const char* cur = std::getenv("LIBTPU_INIT_ARGS");
            const std::string merged = append_async_collective_libtpu_init_args(cur ? cur : "");
            setenv("LIBTPU_INIT_ARGS", merged.c_str(), 1);
        }
        const std::string so_path = resolve_libtpu_path(cfg.libtpu_path);
        dl_handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!dl_handle_) {
            const char* err = dlerror();
            throw std::runtime_error("dlopen failed for '" + so_path + "': " +
                                     (err ? err : "unknown error"));
        }
        using GetPjrtApiFn = const PJRT_Api* (*)();
        auto get_pjrt_api = reinterpret_cast<GetPjrtApiFn>(dlsym(dl_handle_, "GetPjrtApi"));
        if (!get_pjrt_api) {
            const char* err = dlerror();
            throw std::runtime_error(std::string("dlsym(GetPjrtApi) failed: ") +
                                     (err ? err : "unknown error"));
        }
        api_ = get_pjrt_api();
        if (!api_) {
            throw std::runtime_error("GetPjrtApi() returned nullptr");
        }
#endif
    }

    if (!PJRT_API_HAS(api_, PJRT_Error_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Error_Message) ||
        !PJRT_API_HAS(api_, PJRT_Event_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Event_Await) ||
        !PJRT_API_HAS(api_, PJRT_Client_Create) ||
        !PJRT_API_HAS(api_, PJRT_Client_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Client_AddressableDevices) ||
        !PJRT_API_HAS(api_, PJRT_Client_Compile) ||
        !PJRT_API_HAS(api_, PJRT_Client_BufferFromHostBuffer) ||
        !PJRT_API_HAS(api_, PJRT_Device_GetDescription) ||
        !PJRT_API_HAS(api_, PJRT_DeviceDescription_Kind) ||
        !PJRT_API_HAS(api_, PJRT_LoadedExecutable_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_LoadedExecutable_GetExecutable) ||
        !PJRT_API_HAS(api_, PJRT_LoadedExecutable_Execute) ||
        !PJRT_API_HAS(api_, PJRT_Executable_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Executable_NumOutputs) ||
        !PJRT_API_HAS(api_, PJRT_Buffer_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Buffer_ToHostBuffer)) {
        throw std::runtime_error("Required PJRT_Api function pointer missing or outside struct_size");
    }

    has_ready_event_ = PJRT_API_HAS(api_, PJRT_Buffer_ReadyEvent);

    if (PJRT_API_HAS(api_, PJRT_Plugin_Initialize)) {
        PJRT_Plugin_Initialize_Args init_args = {};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        check_error(api_->PJRT_Plugin_Initialize(&init_args), "PJRT_Plugin_Initialize");
    }

    std::vector<PJRT_NamedValue> create_opts;
    std::string mock_profile_str;
    if (is_mock_) {
        if (cfg.mock_devices > 1) {
            PJRT_NamedValue opt = {};
            opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            opt.name = "mock_device_count";
            opt.name_size = 17;
            opt.type = PJRT_NamedValue_kInt64;
            opt.int64_value = cfg.mock_devices;
            opt.value_size = 1;
            create_opts.push_back(opt);
        }
        if (cfg.mock_profile != MockTopologyProfile::AUTO) {
            switch (cfg.mock_profile) {
                case MockTopologyProfile::V5E_1: mock_profile_str = "v5e-1"; break;
                case MockTopologyProfile::V6E_8: mock_profile_str = "v6e-8"; break;
                case MockTopologyProfile::TPU7X_1T: mock_profile_str = "tpu7x-1t"; break;
                case MockTopologyProfile::TPU7X_4T: mock_profile_str = "tpu7x-4t"; break;
                default: break;
            }
            if (!mock_profile_str.empty()) {
                PJRT_NamedValue opt = {};
                opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
                opt.name = "mock_profile";
                opt.name_size = 12;
                opt.type = PJRT_NamedValue_kString;
                opt.string_value = mock_profile_str.c_str();
                opt.value_size = mock_profile_str.size();
                create_opts.push_back(opt);
            }
        }
        if (cfg.mock_scramble_device_order) {
            PJRT_NamedValue opt = {};
            opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            opt.name = "mock_scramble_order";
            opt.name_size = 19;
            opt.type = PJRT_NamedValue_kBool;
            opt.bool_value = true;
            opt.value_size = 1;
            create_opts.push_back(opt);
        }
    }

    PJRT_Client_Create_Args c_args = {};
    c_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (!create_opts.empty()) {
        c_args.create_options = create_opts.data();
        c_args.num_options = create_opts.size();
    }
    check_error(api_->PJRT_Client_Create(&c_args), "PJRT_Client_Create");
    client_ = c_args.client;

    if (PJRT_API_HAS(api_, PJRT_Client_PlatformName)) {
        PJRT_Client_PlatformName_Args p_args = {};
        p_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
        p_args.client = client_;
        check_error(api_->PJRT_Client_PlatformName(&p_args), "PJRT_Client_PlatformName");
        platform_name_.assign(p_args.platform_name, p_args.platform_name_size);
    }
    if (PJRT_API_HAS(api_, PJRT_Client_PlatformVersion)) {
        PJRT_Client_PlatformVersion_Args v_args = {};
        v_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
        v_args.client = client_;
        check_error(api_->PJRT_Client_PlatformVersion(&v_args), "PJRT_Client_PlatformVersion");
        platform_version_.assign(v_args.platform_version, v_args.platform_version_size);
    }

    PJRT_Client_AddressableDevices_Args d_args = {};
    d_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    d_args.client = client_;
    check_error(api_->PJRT_Client_AddressableDevices(&d_args), "PJRT_Client_AddressableDevices");
    if (d_args.num_addressable_devices == 0 || !d_args.addressable_devices) {
        throw std::runtime_error("No addressable Cloud TPU devices found");
    }

    devices_.clear();
    device_infos_.clear();
    for (size_t i = 0; i < d_args.num_addressable_devices; ++i) {
        PJRT_Device* dev = d_args.addressable_devices[i];
        devices_.push_back(dev);

        PjrtDeviceInfo info;
        info.index = static_cast<int>(i);

        PJRT_Device_GetDescription_Args gd = {};
        gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        gd.device = dev;
        check_error(api_->PJRT_Device_GetDescription(&gd), "PJRT_Device_GetDescription");
        PJRT_DeviceDescription* desc = gd.device_description;

        if (PJRT_API_HAS(api_, PJRT_DeviceDescription_Id)) {
            PJRT_DeviceDescription_Id_Args id_args = {};
            id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
            id_args.device_description = desc;
            check_error(api_->PJRT_DeviceDescription_Id(&id_args), "PJRT_DeviceDescription_Id");
            info.id = id_args.id;
        }
        {
            PJRT_DeviceDescription_Kind_Args k_args = {};
            k_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
            k_args.device_description = desc;
            check_error(api_->PJRT_DeviceDescription_Kind(&k_args), "PJRT_DeviceDescription_Kind");
            info.kind.assign(k_args.device_kind, k_args.device_kind_size);
        }
        if (PJRT_API_HAS(api_, PJRT_DeviceDescription_Attributes)) {
            PJRT_DeviceDescription_Attributes_Args attr_args = {};
            attr_args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
            attr_args.device_description = desc;
            PJRT_Error* attr_err = api_->PJRT_DeviceDescription_Attributes(&attr_args);
            if (!attr_err) {
                for (size_t ai = 0; ai < attr_args.num_attributes; ++ai) {
                    const PJRT_NamedValue& a = attr_args.attributes[ai];
                    if (!a.name) continue;
                    const std::string aname(a.name, a.name_size);
                    if (aname == "coords") {
                        if (a.type == PJRT_NamedValue_kInt64List && a.int64_array_value) {
                            info.coords.assign(a.int64_array_value,
                                               a.int64_array_value + a.value_size);
                        } else if (a.type == PJRT_NamedValue_kInt64) {
                            info.coords.push_back(a.int64_value);
                        }
                    } else if (aname == "core_on_chip" && a.type == PJRT_NamedValue_kInt64) {
                        info.core_on_chip = a.int64_value;
                    }
                }
            } else {
                destroy_error(attr_err);
            }
        }
        if (PJRT_API_HAS(api_, PJRT_DeviceDescription_ToString)) {
            PJRT_DeviceDescription_ToString_Args ts_args = {};
            ts_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
            ts_args.device_description = desc;
            check_error(api_->PJRT_DeviceDescription_ToString(&ts_args), "PJRT_DeviceDescription_ToString");
            info.description.assign(ts_args.to_string, ts_args.to_string_size);
        } else {
            info.description = info.kind;
        }
        if (PJRT_API_HAS(api_, PJRT_Device_MemoryStats)) {
            PJRT_Device_MemoryStats_Args ms = {};
            ms.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
            ms.device = dev;
            PJRT_Error* ms_err = api_->PJRT_Device_MemoryStats(&ms);
            if (!ms_err) {
                info.hbm_bytes_in_use = ms.bytes_in_use;
                if (ms.bytes_limit_is_set) {
                    info.hbm_bytes_limit = ms.bytes_limit;
                }
            } else {
                destroy_error(ms_err);
            }
        }
        device_infos_.push_back(std::move(info));
    }

    // Detect dual-chiplet topologies (e.g. tpu7x-1t / tpu7x-4t):
    // either two logical devices share identical chip coordinates, or core_on_chip > 0,
    // or device_kind matches Ironwood/tpu7x on an even logical count >= 2.
    bool shared_coords = false;
    for (size_t i = 0; i < device_infos_.size() && !shared_coords; ++i) {
        if (device_infos_[i].coords.empty()) continue;
        for (size_t j = i + 1; j < device_infos_.size(); ++j) {
            if (device_infos_[i].coords == device_infos_[j].coords) {
                shared_coords = true;
                break;
            }
        }
    }
    bool dual_chiplet = shared_coords;
    if (!dual_chiplet) {
        for (const auto& info : device_infos_) {
            if (info.core_on_chip > 0 ||
                (is_tpu7x_device_kind(info.kind) &&
                 device_infos_.size() >= 2 &&
                 (device_infos_.size() % 2) == 0)) {
                dual_chiplet = true;
                break;
            }
        }
    }
    for (auto& info : device_infos_) {
        info.is_dual_chiplet = dual_chiplet;
    }

    if (devices_.size() > 1) {
        const std::vector<int> order = compute_d2d_hamiltonian_order(device_infos_);
        if (order.size() == devices_.size()) {
            std::vector<PJRT_Device*> sorted_devs;
            std::vector<PjrtDeviceInfo> sorted_infos;
            sorted_devs.reserve(devices_.size());
            sorted_infos.reserve(device_infos_.size());
            for (int r = 0; r < static_cast<int>(order.size()); ++r) {
                const int src = order[static_cast<size_t>(r)];
                sorted_devs.push_back(devices_[static_cast<size_t>(src)]);
                PjrtDeviceInfo info = device_infos_[static_cast<size_t>(src)];
                info.index = r;
                if (dual_chiplet) {
                    info.chip_id = r / 2;
                    info.chiplet_id = (info.core_on_chip >= 0)
                                          ? static_cast<int32_t>(info.core_on_chip)
                                          : (r % 2);
                } else {
                    info.chip_id = r;
                    info.chiplet_id = 0;
                }
                sorted_infos.push_back(std::move(info));
            }
            devices_ = std::move(sorted_devs);
            device_infos_ = std::move(sorted_infos);
        }
    } else if (!device_infos_.empty()) {
        device_infos_[0].index = 0;
        device_infos_[0].chip_id = 0;
        device_infos_[0].chiplet_id =
            (device_infos_[0].core_on_chip >= 0)
                ? static_cast<int32_t>(device_infos_[0].core_on_chip)
                : 0;
    }
    initialized_ = true;
    } catch (...) {
        shutdown();
        throw;
    }
}

void PjrtRuntime::shutdown() {
    initialized_ = false;
    clear_exec_cache();
    if (client_ && api_ && PJRT_API_HAS(api_, PJRT_Client_Destroy)) {
        PJRT_Client_Destroy_Args args = {};
        args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        args.client = client_;
        client_ = nullptr;
        PJRT_Error* err = api_->PJRT_Client_Destroy(&args);
        if (err) {
            destroy_error(err);
        }
    }
    devices_.clear();
    device_infos_.clear();
    platform_name_.clear();
    platform_version_.clear();
    has_ready_event_ = false;
    api_ = nullptr;
    // Intentionally skip dlclose(dl_handle_) for libtpu singleton thread-safety.
}

const PjrtDeviceInfo& PjrtRuntime::device_info(int device_idx) const {
    if (device_idx < 0 || device_idx >= static_cast<int>(device_infos_.size())) {
        throw std::out_of_range("Cloud TPU device index out of range");
    }
    return device_infos_[static_cast<size_t>(device_idx)];
}

bool PjrtRuntime::is_dual_chiplet_topology() const {
    for (const auto& info : device_infos_) {
        if (info.is_dual_chiplet) {
            return true;
        }
    }
    return false;
}

bool PjrtRuntime::is_tpu7x_topology() const {
    for (const auto& info : device_infos_) {
        if (is_tpu7x_device_kind(info.kind)) {
            return true;
        }
    }
    return false;
}

bool PjrtRuntime::query_memory_stats(int device_idx, size_t* free_bytes, size_t* total_bytes) const {
    if (!is_initialized() || device_idx < 0 || device_idx >= static_cast<int>(devices_.size())) {
        return false;
    }
    if (!PJRT_API_HAS(api_, PJRT_Device_MemoryStats)) {
        return false;
    }
    PJRT_Device_MemoryStats_Args ms = {};
    ms.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    ms.device = devices_[static_cast<size_t>(device_idx)];
    PJRT_Error* err = api_->PJRT_Device_MemoryStats(&ms);
    if (err) {
        destroy_error(err);
        return false;
    }
    const int64_t limit = ms.bytes_limit_is_set
                              ? ms.bytes_limit
                              : device_infos_[static_cast<size_t>(device_idx)].hbm_bytes_limit;
    const int64_t used = std::max<int64_t>(0, ms.bytes_in_use);
    if (total_bytes) {
        *total_bytes = limit > 0 ? static_cast<size_t>(limit) : 0;
    }
    if (free_bytes) {
        *free_bytes = (limit > used) ? static_cast<size_t>(limit - used) : 0;
    }
    return true;
}

void PjrtRuntime::await_event(PJRT_Event* ev) {
    if (!ev) {
        return;
    }
    PJRT_Event_Await_Args a_args = {};
    a_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    a_args.event = ev;
    PJRT_Error* err = api_->PJRT_Event_Await(&a_args);

    PJRT_Event_Destroy_Args d_args = {};
    d_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    d_args.event = ev;
    PJRT_Error* derr = api_->PJRT_Event_Destroy(&d_args);
    if (err) {
        if (derr) destroy_error(derr);
        check_error(err, "PJRT_Event_Await");
    }
    check_error(derr, "PJRT_Event_Destroy");
}

void PjrtRuntime::await_event(PjrtEventHandle ev) {
    if (!ev) {
        return;
    }
    await_event(ev.release());
}

PjrtEventHandle PjrtRuntime::create_buffer_ready_event(PJRT_Buffer* buf) {
    if (!buf || !has_ready_event_) {
        return PjrtEventHandle(nullptr, PjrtEventDeleter{api_});
    }
    PJRT_Buffer_ReadyEvent_Args r_args = {};
    r_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    r_args.buffer = buf;
    check_error(api_->PJRT_Buffer_ReadyEvent(&r_args), "PJRT_Buffer_ReadyEvent");
    return PjrtEventHandle(r_args.event, PjrtEventDeleter{api_});
}

void PjrtRuntime::await_buffer_ready(PJRT_Buffer* buf) {
    PjrtEventHandle ev = create_buffer_ready_event(buf);
    if (ev) {
        await_event(std::move(ev));
    }
}

PjrtBufferPtr PjrtRuntime::buffer_from_host(int device_idx,
                                            const void* data,
                                            PjrtDType dtype,
                                            const std::vector<int64_t>& dims) {
    if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("buffer_from_host: device_idx out of range");
    }
    PJRT_Client_BufferFromHostBuffer_Args args = {};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = data;
    args.type = to_pjrt_buffer_type(dtype);
    args.dims = dims.empty() ? nullptr : dims.data();
    args.num_dims = dims.size();
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    args.device = devices_[static_cast<size_t>(device_idx)];
    args.memory = nullptr;
    args.device_layout = nullptr;

    check_error(api_->PJRT_Client_BufferFromHostBuffer(&args), "PJRT_Client_BufferFromHostBuffer");
    PjrtBufferHandle buf_handle(args.buffer, PjrtBufferDeleter{api_});
    if (args.done_with_host_buffer) {
        await_event(args.done_with_host_buffer);
    }

    auto out = std::make_shared<PjrtBuffer>();
    out->handle = std::move(buf_handle);
    out->dtype = dtype;
    out->dims = dims;
    return out;
}

std::vector<PjrtBufferPtr> PjrtRuntime::buffer_from_host_replicated(int num_devices,
                                                                    const void* data,
                                                                    PjrtDType dtype,
                                                                    const std::vector<int64_t>& dims) {
    if (num_devices <= 0 || num_devices > static_cast<int>(devices_.size())) {
        throw std::out_of_range("buffer_from_host_replicated: num_devices out of range");
    }
    std::vector<PjrtBufferPtr> bufs;
    bufs.reserve(static_cast<size_t>(num_devices));
    for (int d = 0; d < num_devices; ++d) {
        bufs.push_back(buffer_from_host(d, data, dtype, dims));
    }
    return bufs;
}

PjrtCompileOptions PjrtRuntime::make_spmd_compile_options(int num_chips,
                                                          int device_idx,
                                                          bool alias_passthrough) const {
    PjrtCompileOptions opts;
    opts.alias_passthrough_params = alias_passthrough;
    const int n = std::max(1, std::min(num_chips, static_cast<int>(device_infos_.size())));
    if (n <= 1) {
        opts.num_replicas = 1;
        opts.num_partitions = 1;
        opts.use_spmd_partitioning = true;
        opts.use_shardy_partitioner = true;
        if (device_infos_.size() > 1 && device_idx >= 0 && device_idx < static_cast<int>(device_infos_.size())) {
            opts.device_assignment_ids = {device_infos_[static_cast<size_t>(device_idx)].id};
        }
    } else {
        // Explicit collective replica-SPMD across n TPU chips:
        // num_replicas = n, num_partitions = 1, explicit DeviceAssignmentProto mapping [0][0..n-1].
        // Leave env_option_overrides empty: OpenXLA's CompileOptions::ApplyOption only accepts
        // xla::DebugOptions fields and rejects TPU-specific xla_tpu_* flags; opt-in async
        // collective flags are passed via LIBTPU_INIT_ARGS before libtpu.so loads.
        opts.num_replicas = n;
        opts.num_partitions = 1;
        opts.use_spmd_partitioning = false;
        opts.use_shardy_partitioner = false;
        opts.device_assignment_ids.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            opts.device_assignment_ids[static_cast<size_t>(i)] = device_infos_[static_cast<size_t>(i)].id;
        }
    }
    return opts;
}

void PjrtRuntime::buffer_to_host(PJRT_Buffer* buf, void* dst, size_t dst_size) {
    if (!buf) {
        throw std::runtime_error("buffer_to_host: null PJRT_Buffer");
    }
    // Always request explicit untiled C-order row-major host layout ({rank-1, ..., 0}).
    // Leaving host_layout = nullptr causes libtpu.so to copy out in the TPU device buffer's
    // chosen minor-to-major dimension order (transposed for non-multiple-of-8 or batched shapes).
    std::vector<int64_t> minor_to_major;
    PJRT_Buffer_MemoryLayout host_layout = {};
    bool use_host_layout = false;
    size_t nelems = 1;
    if (PJRT_API_HAS(api_, PJRT_Buffer_Dimensions)) {
        PJRT_Buffer_Dimensions_Args dim_args = {};
        dim_args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
        dim_args.buffer = buf;
        check_error(api_->PJRT_Buffer_Dimensions(&dim_args), "PJRT_Buffer_Dimensions");
        for (size_t i = 0; i < dim_args.num_dims; ++i) {
            nelems *= static_cast<size_t>(dim_args.dims[i]);
        }
        if (dim_args.num_dims > 1) {
            minor_to_major.resize(dim_args.num_dims);
            for (size_t i = 0; i < dim_args.num_dims; ++i) {
                minor_to_major[i] = static_cast<int64_t>(dim_args.num_dims - 1 - i);
            }
            host_layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
            host_layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
            host_layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
            host_layout.tiled.minor_to_major = minor_to_major.data();
            host_layout.tiled.minor_to_major_size = minor_to_major.size();
            host_layout.tiled.tile_dims = nullptr;
            host_layout.tiled.tile_dim_sizes = nullptr;
            host_layout.tiled.num_tiles = 0;
            use_host_layout = true;
        }
    }

    // Handle sub-byte S4/U4 gracefully whether PJRT ToHostBuffer writes 1 B/elem or packed 0.5 B/elem,
    // and normalize S4 1-B/elem outputs where PJRT leaves nibbles zero-extended (0x08..0x0F).
    bool is_s4_buffer = false;
    size_t s4_needed = 0;
    if (PJRT_API_HAS(api_, PJRT_Buffer_ElementType)) {
        PJRT_Buffer_ElementType_Args et_args = {};
        et_args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
        et_args.buffer = buf;
        check_error(api_->PJRT_Buffer_ElementType(&et_args), "PJRT_Buffer_ElementType");
        is_s4_buffer = (et_args.type == PJRT_Buffer_Type_S4);
        if (et_args.type == PJRT_Buffer_Type_S4 || et_args.type == PJRT_Buffer_Type_U4) {
            const PjrtDType dt = from_pjrt_buffer_type(et_args.type);
            const size_t packed_bytes = pjrt_dtype_nbytes(dt, nelems);
            PJRT_Buffer_ToHostBuffer_Args q_args = {};
            q_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
            q_args.src = buf;
            q_args.host_layout = use_host_layout ? &host_layout : nullptr;
            q_args.dst = nullptr;
            q_args.dst_size = 0;
            check_error(api_->PJRT_Buffer_ToHostBuffer(&q_args), "PJRT_Buffer_ToHostBuffer(size)");
            if (q_args.event) {
                await_event(q_args.event);
            }
            const size_t needed = q_args.dst_size;
            s4_needed = needed;
            if (needed == packed_bytes && dst_size >= nelems && nelems > packed_bytes) {
                std::vector<uint8_t> tmp(needed);
                PJRT_Buffer_ToHostBuffer_Args d_args = {};
                d_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
                d_args.src = buf;
                d_args.host_layout = use_host_layout ? &host_layout : nullptr;
                d_args.dst = tmp.data();
                d_args.dst_size = tmp.size();
                check_error(api_->PJRT_Buffer_ToHostBuffer(&d_args), "PJRT_Buffer_ToHostBuffer");
                if (d_args.event) {
                    await_event(d_args.event);
                }
                if (dt == PjrtDType::S4) {
                    unpack_s4_to_i8(tmp.data(), nelems, static_cast<int8_t*>(dst));
                } else {
                    unpack_u4_to_u8(tmp.data(), nelems, static_cast<uint8_t*>(dst));
                }
                return;
            }
            if (needed == nelems && dst_size >= packed_bytes && dst_size < nelems && nelems > packed_bytes) {
                std::vector<uint8_t> tmp(needed);
                PJRT_Buffer_ToHostBuffer_Args d_args = {};
                d_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
                d_args.src = buf;
                d_args.host_layout = use_host_layout ? &host_layout : nullptr;
                d_args.dst = tmp.data();
                d_args.dst_size = tmp.size();
                check_error(api_->PJRT_Buffer_ToHostBuffer(&d_args), "PJRT_Buffer_ToHostBuffer");
                if (d_args.event) {
                    await_event(d_args.event);
                }
                if (dt == PjrtDType::S4) {
                    pack_s4_from_i8(reinterpret_cast<const int8_t*>(tmp.data()), nelems, static_cast<uint8_t*>(dst));
                } else {
                    pack_u4_from_u8(tmp.data(), nelems, static_cast<uint8_t*>(dst));
                }
                return;
            }
        }
    }

    PJRT_Buffer_ToHostBuffer_Args args = {};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.host_layout = use_host_layout ? &host_layout : nullptr;
    args.dst = dst;
    args.dst_size = dst_size;
    check_error(api_->PJRT_Buffer_ToHostBuffer(&args), "PJRT_Buffer_ToHostBuffer");
    if (args.event) {
        await_event(args.event);
    }
    if (is_s4_buffer && s4_needed == nelems && dst_size >= nelems && nelems > 0) {
        int8_t* out = static_cast<int8_t*>(dst);
        for (size_t i = 0; i < nelems; ++i) {
            out[i] = sign_extend_nibble(static_cast<uint8_t>(out[i]));
        }
    }
}

CachedExecutablePtr PjrtRuntime::compile_or_get(const std::string& cache_key,
                                                const std::string& mlir_text,
                                                int device_idx,
                                                const PjrtCompileOptions& opts,
                                                bool* out_compiled) {
    if (out_compiled != nullptr) {
        *out_compiled = false;
    }
    if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("compile_or_get: device_idx out of range");
    }

    PjrtCompileOptions eff_opts = opts;
    const int target_dev_id = device_infos_[static_cast<size_t>(device_idx)].id;
    if (eff_opts.device_assignment_ids.empty() && eff_opts.num_replicas == 1 &&
        eff_opts.num_partitions == 1 && devices_.size() > 1) {
        eff_opts.device_assignment_ids.push_back(static_cast<int64_t>(target_dev_id));
    }

    const std::string opts_pb = encode_compile_options_proto(eff_opts);

    static const char kHex[] = "0123456789abcdef";
    std::string full_key = cache_key;
    full_key.append("|dev=");
    full_key.append(std::to_string(device_idx));
    full_key.push_back(':');
    full_key.append(std::to_string(target_dev_id));
    full_key.append("|opts=");
    full_key.reserve(full_key.size() + opts_pb.size() * 2);
    for (unsigned char b : opts_pb) {
        full_key.push_back(kHex[b >> 4]);
        full_key.push_back(kHex[b & 0x0f]);
    }

    {
        std::lock_guard<std::mutex> lock(cache_mu_);
        auto it = exec_cache_.find(full_key);
        if (it != exec_cache_.end()) {
            cache_hits_ += 1;
            lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_it);
            return it->second.exec;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const bool is_mosaic = (mlir_text.find("@tpu_custom_call") != std::string::npos);
    compile_attempts_.fetch_add(1, std::memory_order_relaxed);
    if (is_mosaic) {
        mosaic_compile_attempts_.fetch_add(1, std::memory_order_relaxed);
    }

    PJRT_Program prog = {};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = const_cast<char*>(mlir_text.data());
    prog.code_size = mlir_text.size();
    prog.format = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args c_args = {};
    c_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    c_args.client = client_;
    c_args.program = &prog;
    c_args.compile_options = opts_pb.data();
    c_args.compile_options_size = opts_pb.size();

    PJRT_Error* c_err = api_->PJRT_Client_Compile(&c_args);
    if (c_err != nullptr) {
        compile_failures_.fetch_add(1, std::memory_order_relaxed);
        check_error(c_err, "PJRT_Client_Compile");
    }
    if (is_mosaic) {
        mosaic_compile_successes_.fetch_add(1, std::memory_order_relaxed);
    }
    if (out_compiled != nullptr) {
        *out_compiled = true;
    }
    PjrtLoadedExecutableHandle loaded_handle(c_args.executable, PjrtLoadedExecutableDeleter{api_});

    PJRT_LoadedExecutable_GetExecutable_Args ge = {};
    ge.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    ge.loaded_executable = loaded_handle.get();
    PJRT_Error* ge_err = api_->PJRT_LoadedExecutable_GetExecutable(&ge);
    if (ge_err || !ge.executable) {
        if (ge_err) check_error(ge_err, "PJRT_LoadedExecutable_GetExecutable");
        throw std::runtime_error("PJRT_LoadedExecutable_GetExecutable returned null");
    }

    PJRT_Executable_NumOutputs_Args no = {};
    no.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    no.executable = ge.executable;
    PJRT_Error* no_err = api_->PJRT_Executable_NumOutputs(&no);

    PJRT_Executable_Destroy_Args ed = {};
    ed.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    ed.executable = ge.executable;
    PJRT_Error* ed_err = api_->PJRT_Executable_Destroy(&ed);

    if (no_err || ed_err) {
        if (no_err) {
            if (ed_err) destroy_error(ed_err);
            check_error(no_err, "PJRT_Executable_NumOutputs");
        }
        check_error(ed_err, "PJRT_Executable_Destroy");
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t compile_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    auto exec = std::make_shared<CachedExecutable>();
    exec->key = full_key;
    exec->handle = std::move(loaded_handle);
    exec->device_idx = device_idx;
    exec->num_replicas = std::max<int>(1, eff_opts.num_replicas);
    exec->num_outputs = no.num_outputs;

    std::lock_guard<std::mutex> lock(cache_mu_);
    // Re-check in case another thread compiled the same key concurrently.
    auto existing = exec_cache_.find(full_key);
    if (existing != exec_cache_.end()) {
        cache_hits_ += 1;
        lru_order_.splice(lru_order_.begin(), lru_order_, existing->second.lru_it);
        return existing->second.exec;
    }

    cache_misses_ += 1;
    total_compile_us_ += compile_us;

    const size_t cap = std::max<size_t>(1, cfg_.exec_cache_cap);
    while (exec_cache_.size() >= cap && !lru_order_.empty()) {
        const std::string& victim_key = lru_order_.back();
        auto vit = exec_cache_.find(victim_key);
        if (vit != exec_cache_.end()) {
            exec_cache_.erase(vit);
            cache_evictions_ += 1;
            eviction_epoch_.fetch_add(1, std::memory_order_release);
        }
        lru_order_.pop_back();
    }

    lru_order_.push_front(full_key);
    CacheEntry entry;
    entry.exec = exec;
    entry.lru_it = lru_order_.begin();
    exec_cache_.emplace(full_key, std::move(entry));
    return exec;
}

void PjrtRuntime::execute(const CachedExecutablePtr& exec,
                          int device_idx,
                          PJRT_Buffer* const* inputs,
                          size_t num_inputs,
                          std::vector<PjrtBufferPtr>& outputs,
                          const std::vector<PjrtDType>& out_dtypes,
                          const std::vector<std::vector<int64_t>>& out_dims,
                          bool sync,
                          const std::vector<int64_t>* non_donatable_input_indices) {
    if (!exec || !exec->handle) {
        throw std::runtime_error("execute: null CachedExecutable");
    }
    if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("execute: device_idx out of range");
    }
    if (exec->num_replicas != 1 || exec->device_idx != device_idx) {
        throw std::runtime_error("execute: executable compiled for device " +
                                 std::to_string(exec->device_idx) + " (replicas=" +
                                 std::to_string(exec->num_replicas) + "), cannot run on device " +
                                 std::to_string(device_idx));
    }
    if (exec->num_outputs != out_dtypes.size() || exec->num_outputs != out_dims.size()) {
        throw std::runtime_error("execute: output count mismatch vs executable");
    }

    const bool profile_time = (cfg_.verbose > 0);
    const auto t0 = profile_time ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    PJRT_Buffer* stack_raw_outputs[16];
    std::vector<PJRT_Buffer*> heap_raw_outputs;
    PJRT_Buffer** raw_out_ptr = stack_raw_outputs;
    if (exec->num_outputs > 16) {
        heap_raw_outputs.assign(exec->num_outputs, nullptr);
        raw_out_ptr = heap_raw_outputs.data();
    } else {
        for (size_t i = 0; i < exec->num_outputs; ++i) stack_raw_outputs[i] = nullptr;
    }

    PJRT_Buffer* const* arg_lists[1] = {inputs};
    PJRT_Buffer** const out_lists[1] = {raw_out_ptr};

    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    if (non_donatable_input_indices && !non_donatable_input_indices->empty()) {
        opts.non_donatable_input_indices = non_donatable_input_indices->data();
        opts.num_non_donatable_input_indices = non_donatable_input_indices->size();
    }

    PJRT_Event* dev_ev[1] = {nullptr};
    const bool use_dev_event = sync && !has_ready_event_;

    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = exec->handle.get();
    ex.options = &opts;
    ex.argument_lists = arg_lists;
    ex.num_devices = 1;
    ex.num_args = num_inputs;
    ex.output_lists = out_lists;
    ex.device_complete_events = use_dev_event ? dev_ev : nullptr;
    // PJRT TPU plugins compile single-device executables with compile-time DeviceAssignmentProto;
    // leaving execute_device = nullptr lets libtpu.so dispatch directly to the assigned device
    // (passing a non-null execute_device is only valid for portable executables).
    ex.execute_device = nullptr;

    std::vector<PjrtBufferHandle> out_handles;
    out_handles.reserve(exec->num_outputs);

    check_error(api_->PJRT_LoadedExecutable_Execute(&ex), "PJRT_LoadedExecutable_Execute");
    PjrtEventHandle dev_ev_handle(dev_ev[0], PjrtEventDeleter{api_});
    for (size_t i = 0; i < exec->num_outputs; ++i) {
        out_handles.emplace_back(raw_out_ptr[i], PjrtBufferDeleter{api_});
    }

    outputs.resize(exec->num_outputs);
    for (size_t i = 0; i < exec->num_outputs; ++i) {
        auto buf = std::make_shared<PjrtBuffer>();
        buf->handle = std::move(out_handles[i]);
        buf->dtype = out_dtypes[i];
        buf->dims = out_dims[i];
        outputs[i] = std::move(buf);
    }

    if (sync) {
        if (use_dev_event && dev_ev_handle) {
            await_event(std::move(dev_ev_handle));
        } else if (has_ready_event_) {
            for (const auto& out_buf : outputs) {
                await_buffer_ready(out_buf->handle.get());
            }
        }
    }

    execute_calls_.fetch_add(1, std::memory_order_relaxed);
    if (profile_time) {
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t exec_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        total_execute_us_.fetch_add(exec_us, std::memory_order_relaxed);
    }
}

void PjrtRuntime::execute_spmd(const CachedExecutablePtr& exec,
                               int num_devices,
                               PJRT_Buffer* const* const* per_device_inputs,
                               size_t num_inputs,
                               std::vector<std::vector<PjrtBufferPtr>>& per_device_outputs,
                               const std::vector<PjrtDType>& out_dtypes,
                               const std::vector<std::vector<int64_t>>& out_dims,
                               bool sync,
                               const std::vector<int64_t>* non_donatable_input_indices) {
    if (!exec || !exec->handle) {
        throw std::runtime_error("execute_spmd: null CachedExecutable");
    }
    if (num_devices <= 0 || num_devices > static_cast<int>(devices_.size())) {
        throw std::out_of_range("execute_spmd: num_devices out of range");
    }
    if (exec->num_replicas != num_devices) {
        throw std::runtime_error("execute_spmd: executable compiled for " +
                                 std::to_string(exec->num_replicas) + " replicas, cannot run on " +
                                 std::to_string(num_devices) + " devices");
    }
    if (exec->num_outputs != out_dtypes.size() || exec->num_outputs != out_dims.size()) {
        throw std::runtime_error("execute_spmd: output count mismatch vs executable");
    }

    const bool profile_time = (cfg_.verbose > 0);
    const auto t0 = profile_time ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const size_t n_devs = static_cast<size_t>(num_devices);
    std::vector<std::vector<PJRT_Buffer*>> raw_outs(n_devs, std::vector<PJRT_Buffer*>(exec->num_outputs, nullptr));
    std::vector<PJRT_Buffer**> out_lists(n_devs);
    for (size_t d = 0; d < n_devs; ++d) {
        out_lists[d] = raw_outs[d].data();
    }

    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    const uint32_t raw_lid = next_launch_id_.fetch_add(1, std::memory_order_relaxed);
    const int32_t safe_lid = static_cast<int32_t>(raw_lid & 0x7FFFFFFFU);
    opts.launch_id = (safe_lid > 0) ? safe_lid : 1;
    if (non_donatable_input_indices && !non_donatable_input_indices->empty()) {
        opts.non_donatable_input_indices = non_donatable_input_indices->data();
        opts.num_non_donatable_input_indices = non_donatable_input_indices->size();
    }

    const bool use_dev_event = sync && !has_ready_event_;
    std::vector<PJRT_Event*> dev_events(use_dev_event ? n_devs : 0, nullptr);

    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = exec->handle.get();
    ex.options = &opts;
    ex.argument_lists = per_device_inputs;
    ex.num_devices = n_devs;
    ex.num_args = num_inputs;
    ex.output_lists = out_lists.data();
    ex.device_complete_events = use_dev_event ? dev_events.data() : nullptr;
    ex.execute_device = nullptr;

    std::vector<PjrtEventHandle> dev_ev_handles;
    dev_ev_handles.reserve(dev_events.size());
    std::vector<std::vector<PjrtBufferHandle>> out_handles(n_devs);
    for (size_t d = 0; d < n_devs; ++d) {
        out_handles[d].reserve(exec->num_outputs);
    }

    check_error(api_->PJRT_LoadedExecutable_Execute(&ex), "PJRT_LoadedExecutable_Execute(SPMD)");
    for (PJRT_Event* ev : dev_events) {
        dev_ev_handles.emplace_back(ev, PjrtEventDeleter{api_});
    }
    for (size_t d = 0; d < n_devs; ++d) {
        for (size_t i = 0; i < exec->num_outputs; ++i) {
            out_handles[d].emplace_back(raw_outs[d][i], PjrtBufferDeleter{api_});
        }
    }

    per_device_outputs.resize(n_devs);
    for (size_t d = 0; d < n_devs; ++d) {
        per_device_outputs[d].resize(exec->num_outputs);
        for (size_t i = 0; i < exec->num_outputs; ++i) {
            auto buf = std::make_shared<PjrtBuffer>();
            buf->handle = std::move(out_handles[d][i]);
            buf->dtype = out_dtypes[i];
            buf->dims = out_dims[i];
            per_device_outputs[d][i] = std::move(buf);
        }
    }

    if (sync) {
        if (use_dev_event) {
            for (auto& ev : dev_ev_handles) {
                if (ev) await_event(std::move(ev));
            }
        } else if (has_ready_event_) {
            for (size_t d = 0; d < n_devs; ++d) {
                for (const auto& out_buf : per_device_outputs[d]) {
                    await_buffer_ready(out_buf->handle.get());
                }
            }
        }
    }

    execute_calls_.fetch_add(1, std::memory_order_relaxed);
    if (profile_time) {
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t exec_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        total_execute_us_.fetch_add(exec_us, std::memory_order_relaxed);
    }
}

void PjrtRuntime::clear_exec_cache() {
    std::lock_guard<std::mutex> lock(cache_mu_);
    exec_cache_.clear();
    lru_order_.clear();
    eviction_epoch_.fetch_add(1, std::memory_order_release);
}

CachedExecutablePtr PjrtRuntime::get_cached_exec(const std::string& full_key) {
    std::lock_guard<std::mutex> lock(cache_mu_);
    auto it = exec_cache_.find(full_key);
    if (it == exec_cache_.end()) {
        return nullptr;
    }
    cache_hits_ += 1;
    lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_it);
    return it->second.exec;
}

bool PjrtRuntime::has_cached_exec(const std::string& full_key) const {
    std::lock_guard<std::mutex> lock(cache_mu_);
    return exec_cache_.find(full_key) != exec_cache_.end();
}

CacheStats PjrtRuntime::cache_stats() const {
    std::lock_guard<std::mutex> lock(cache_mu_);
    CacheStats st;
    st.entries = exec_cache_.size();
    st.hits = cache_hits_;
    st.misses = cache_misses_;
    st.evictions = cache_evictions_;
    st.compile_us = total_compile_us_;
    st.execute_us = total_execute_us_.load(std::memory_order_relaxed);
    st.compile_attempts = compile_attempts_.load(std::memory_order_relaxed);
    st.compile_failures = compile_failures_.load(std::memory_order_relaxed);
    st.mosaic_compile_attempts = mosaic_compile_attempts_.load(std::memory_order_relaxed);
    st.mosaic_compile_successes = mosaic_compile_successes_.load(std::memory_order_relaxed);
    st.execute_calls = execute_calls_.load(std::memory_order_relaxed);
    return st;
}

PjrtRuntime& shared_pjrt_runtime(const RuntimeConfig* override_cfg) {
    static std::mutex mu;
    static PjrtRuntime rt;
    std::lock_guard<std::mutex> lock(mu);
    if (!rt.is_initialized() && override_cfg != nullptr) {
        rt.init(*override_cfg);
    }
    return rt;
}

}  // namespace cloudtpu
