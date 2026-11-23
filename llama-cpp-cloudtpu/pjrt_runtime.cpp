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
#include <sstream>
#include <stdexcept>
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
        case PjrtDType::S8: return PJRT_Buffer_Type_S8;
        case PjrtDType::U8: return PJRT_Buffer_Type_U8;
        case PjrtDType::S16: return PJRT_Buffer_Type_S16;
        case PjrtDType::S32: return PJRT_Buffer_Type_S32;
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
        case PJRT_Buffer_Type_S8: return PjrtDType::S8;
        case PJRT_Buffer_Type_U8: return PjrtDType::U8;
        case PJRT_Buffer_Type_S16: return PjrtDType::S16;
        case PJRT_Buffer_Type_S32: return PjrtDType::S32;
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
        case PjrtDType::S8: return "i8";
        case PjrtDType::U8: return "ui8";
        case PjrtDType::S16: return "i16";
        case PjrtDType::S32: return "i32";
        case PjrtDType::S64: return "i64";
        default: return "invalid";
    }
}

size_t pjrt_dtype_size(PjrtDType dt) {
    switch (dt) {
        case PjrtDType::S64: return 8;
        case PjrtDType::F32:
        case PjrtDType::S32: return 4;
        case PjrtDType::F16:
        case PjrtDType::BF16:
        case PjrtDType::S16: return 2;
        case PjrtDType::PRED:
        case PjrtDType::S8:
        case PjrtDType::U8: return 1;
        default: return 0;
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

}  // namespace

std::string encode_compile_options_proto(const PjrtCompileOptions& opts) {
    // xla::ExecutableBuildOptionsProto (openxla/xla/xla/pjrt/proto/compile_options.proto)
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
    if (opts.use_shardy_partitioner) {
        append_tag_varint(ebo, 19, 1);
    }

    // xla::CompileOptionsProto { executable_build_options (field 3, length-delimited) }
    std::string out;
    append_tag_bytes(out, 3, ebo);
    return out;
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
    return num_elements() * pjrt_dtype_size(dtype);
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
        bytes.assign(num_elements() * pjrt_dtype_size(dt), 0);
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
            case PjrtDType::S8:
                return static_cast<float>(reinterpret_cast<const int8_t*>(bytes.data())[idx]);
            case PjrtDType::U8:
                return static_cast<float>(bytes[idx]);
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
            default:
                write_f32(idx, static_cast<float>(v));
                break;
        }
    }
};

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
    if (tok == "i8" || tok == "si8") return PjrtDType::S8;
    if (tok == "ui8") return PjrtDType::U8;
    if (tok == "i16" || tok == "si16") return PjrtDType::S16;
    if (tok == "i32" || tok == "si32") return PjrtDType::S32;
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
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        MAXIMUM,
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
    std::vector<int64_t> dims_attr;
    std::vector<int64_t> lhs_batch;
    std::vector<int64_t> rhs_batch;
    std::vector<int64_t> lhs_contract;
    std::vector<int64_t> rhs_contract;
    std::vector<int64_t> slice_starts;
    std::vector<int64_t> slice_limits;
    std::vector<int64_t> slice_strides;
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

        // Accumulate multi-line ops (e.g. "stablehlo.reduce", "stablehlo.gather", "stablehlo.scatter")
        std::string stmt = line;
        if (stmt.find("\"stablehlo.reduce\"") != std::string::npos ||
            stmt.find("\"stablehlo.gather\"") != std::string::npos ||
            stmt.find("\"stablehlo.scatter\"") != std::string::npos) {
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
            const size_t d1 = rhs.rfind('>');
            const std::string lit = trim_copy(rhs.substr(d0 + 6, d1 - (d0 + 6)));
            const bool is_int = (op.out_type.dtype == PjrtDType::S8 || op.out_type.dtype == PjrtDType::U8 ||
                                 op.out_type.dtype == PjrtDType::S16 || op.out_type.dtype == PjrtDType::S32 ||
                                 op.out_type.dtype == PjrtDType::S64);
            auto parse_mlir_float = [](const std::string& s) -> double {
                if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
                    const uint32_t bits = static_cast<uint32_t>(std::stoul(s, nullptr, 16));
                    float f = 0.0f;
                    std::memcpy(&f, &bits, sizeof(f));
                    return static_cast<double>(f);
                }
                return std::stod(s);
            };
            if (!lit.empty() && lit.front() == '[') {
                op.is_splat = false;
                std::string inner = lit.substr(1, lit.size() - 2);
                std::istringstream vs(inner);
                std::string tok;
                while (std::getline(vs, tok, ',')) {
                    tok = trim_copy(tok);
                    if (tok.empty()) continue;
                    if (is_int) op.array_i.push_back(std::stoll(tok));
                    else op.array_f.push_back(parse_mlir_float(tok));
                }
            } else {
                op.is_splat = true;
                if (is_int) op.scalar_i = std::stoll(lit);
                else op.scalar_f = parse_mlir_float(lit);
            }
        } else if (rhs.rfind("stablehlo.iota ", 0) == 0) {
            op.kind = MockOp::Kind::IOTA;
            const size_t dp = rhs.find("dim = ");
            op.iota_dim = std::stoll(rhs.substr(dp + 6));
        } else if (rhs.rfind("stablehlo.convert ", 0) == 0) {
            op.kind = MockOp::Kind::CONVERT;
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
        } else {
            throw std::runtime_error("MockPJRT: unsupported StableHLO op: " + rhs);
        }

        prog.ops.push_back(std::move(op));
    }

    return prog;
}

static MockTensor eval_mock_op(const MockOp& op, const std::unordered_map<std::string, MockTensor>& env) {
    MockTensor out;
    out.allocate(op.out_type.dtype, op.out_type.dims);
    const size_t n_out = out.num_elements();

    const bool is_int = (out.dtype == PjrtDType::S8 || out.dtype == PjrtDType::U8 ||
                         out.dtype == PjrtDType::S16 || out.dtype == PjrtDType::S32 ||
                         out.dtype == PjrtDType::S64);

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
            const bool src_int = (a.dtype == PjrtDType::S8 || a.dtype == PjrtDType::U8 ||
                                  a.dtype == PjrtDType::S16 || a.dtype == PjrtDType::S32 ||
                                  a.dtype == PjrtDType::S64);
            for (size_t i = 0; i < n_out; ++i) {
                if (is_int && src_int) out.write_i64(i, a.read_i64(i));
                else if (is_int) out.write_i64(i, static_cast<int64_t>(a.read_f32(i)));
                else if (src_int) out.write_f32(i, static_cast<float>(a.read_i64(i)));
                else out.write_f32(i, a.read_f32(i));
            }
            break;
        }
        case MockOp::Kind::ADD:
        case MockOp::Kind::SUBTRACT:
        case MockOp::Kind::MULTIPLY:
        case MockOp::Kind::DIVIDE:
        case MockOp::Kind::MAXIMUM:
        case MockOp::Kind::BIT_AND:
        case MockOp::Kind::SHR_LOGICAL: {
            const MockTensor& a = env.at(op.inputs[0]);
            const MockTensor& b = env.at(op.inputs[1]);
            for (size_t i = 0; i < n_out; ++i) {
                if (is_int) {
                    const int64_t va = a.read_i64(i);
                    const int64_t vb = b.read_i64(i);
                    int64_t vr = 0;
                    if (op.kind == MockOp::Kind::ADD) vr = va + vb;
                    else if (op.kind == MockOp::Kind::SUBTRACT) vr = va - vb;
                    else if (op.kind == MockOp::Kind::MULTIPLY) vr = va * vb;
                    else if (op.kind == MockOp::Kind::DIVIDE) vr = (vb != 0) ? (va / vb) : 0;
                    else if (op.kind == MockOp::Kind::MAXIMUM) vr = std::max(va, vb);
                    else if (op.kind == MockOp::Kind::BIT_AND) vr = (va & vb);
                    else if (op.kind == MockOp::Kind::SHR_LOGICAL) {
                        vr = static_cast<int64_t>(static_cast<uint64_t>(va) >> static_cast<unsigned>(vb));
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
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
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
                std::memcpy(out.bytes.data() + i * elem_sz, a.bytes.data() + src_idx * elem_sz, elem_sz);
            }
            break;
        }
        case MockOp::Kind::TRANSPOSE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
            for (size_t i = 0; i < n_out; ++i) {
                size_t rem = i;
                size_t src_idx = 0;
                for (size_t d = 0; d < out.dims.size(); ++d) {
                    const size_t c = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_idx += c * in_strides[static_cast<size_t>(op.dims_attr[d])];
                }
                std::memcpy(out.bytes.data() + i * elem_sz, a.bytes.data() + src_idx * elem_sz, elem_sz);
            }
            break;
        }
        case MockOp::Kind::SLICE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const auto out_strides = compute_row_major_strides(out.dims);
            const auto in_strides = compute_row_major_strides(a.dims);
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
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
                std::memcpy(out.bytes.data() + i * elem_sz, a.bytes.data() + src_idx * elem_sz, elem_sz);
            }
            break;
        }
        case MockOp::Kind::CONCATENATE: {
            const auto out_strides = compute_row_major_strides(out.dims);
            const size_t cdim = static_cast<size_t>(op.concat_dim);
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
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
                    std::memcpy(out.bytes.data() + dst_idx * elem_sz, a.bytes.data() + i * elem_sz, elem_sz);
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
            const size_t elem_sz = pjrt_dtype_size(out.dtype);

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
                std::memcpy(out.bytes.data() + oi * elem_sz, operand.bytes.data() + src_idx * elem_sz, elem_sz);
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
                        std::memcpy(out.bytes.data() + dst_row * elem_sz,
                                    updates.bytes.data() + src_row * elem_sz,
                                    static_cast<size_t>(D) * elem_sz);
                    }
                }
            }
            break;
        }
        case MockOp::Kind::COMPARE: {
            const MockTensor& a = env.at(op.inputs[0]);
            const MockTensor& b = env.at(op.inputs[1]);
            const bool a_int = (a.dtype == PjrtDType::PRED || a.dtype == PjrtDType::S8 ||
                                a.dtype == PjrtDType::U8 || a.dtype == PjrtDType::S16 ||
                                a.dtype == PjrtDType::S32 || a.dtype == PjrtDType::S64);
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
            const size_t elem_sz = pjrt_dtype_size(out.dtype);
            const bool scalar_pred = (p.num_elements() == 1);
            for (size_t i = 0; i < n_out; ++i) {
                const bool take_t = (p.read_i64(scalar_pred ? 0 : i) != 0);
                const uint8_t* src_ptr = (take_t ? t.bytes.data() : f.bytes.data()) + i * elem_sz;
                std::memcpy(out.bytes.data() + i * elem_sz, src_ptr, elem_sz);
            }
            break;
        }
    }
    return out;
}

// Mock PJRT C API opaque structs
struct MockError {
    std::string message;
};
struct MockEvent {};
struct MockClient;
struct MockDeviceDescription {
    int id = 0;
    std::string to_str;
};
struct MockDevice {
    MockClient* client = nullptr;
    int id = 0;
    MockDeviceDescription desc;
};
struct MockClient {
    std::vector<std::unique_ptr<MockDevice>> devices;
    std::vector<PJRT_Device*> device_ptrs;
    std::shared_ptr<std::atomic<int64_t>> allocated_bytes = std::make_shared<std::atomic<int64_t>>(0);

    explicit MockClient(int num_devices) {
        const int count = std::max(1, num_devices);
        devices.reserve(static_cast<size_t>(count));
        device_ptrs.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            auto d = std::make_unique<MockDevice>();
            d->client = this;
            d->id = i;
            d->desc.id = i;
            d->desc.to_str = "CLOUDTPU_MOCK(id=" + std::to_string(i) + ")";
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
    std::shared_ptr<std::atomic<int64_t>> allocated_bytes;
    int assigned_device_id = 0;
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

static int parse_assigned_device_id_from_compile_options(const char* buf, size_t len) {
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
                                    uint64_t dev_id = 0;
                                    if (pb_read_varint(cd + cpos, static_cast<size_t>(ids_len), ipos, dev_id)) {
                                        return static_cast<int>(dev_id);
                                    }
                                    cpos += static_cast<size_t>(ids_len);
                                } else if (cfield == 1 && cwt == 0) {
                                    uint64_t dev_id = 0;
                                    if (pb_read_varint(cd, static_cast<size_t>(cd_len), cpos, dev_id)) {
                                        return static_cast<int>(dev_id);
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
    return 0;
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
    int num_devices = 1;
    if (args->create_options && args->num_options > 0) {
        for (size_t i = 0; i < args->num_options; ++i) {
            const PJRT_NamedValue& opt = args->create_options[i];
            if (opt.name && opt.name_size == 17 &&
                std::memcmp(opt.name, "mock_device_count", 17) == 0 &&
                opt.type == PJRT_NamedValue_kInt64) {
                num_devices = std::max(1, static_cast<int>(opt.int64_value));
            }
        }
    }
    args->client = reinterpret_cast<PJRT_Client*>(new MockClient(num_devices));
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
    auto* dd = reinterpret_cast<MockDeviceDescription*>(args->device_description);
    args->id = dd ? dd->id : 0;
    return nullptr;
}
static PJRT_Error* mock_dev_desc_kind(PJRT_DeviceDescription_Kind_Args* args) {
    static const char kKind[] = "Cloud TPU Host Mock (v5e-1)";
    args->device_kind = kKind;
    args->device_kind_size = sizeof(kKind) - 1;
    return nullptr;
}
static PJRT_Error* mock_dev_desc_to_string(PJRT_DeviceDescription_ToString_Args* args) {
    auto* dd = reinterpret_cast<MockDeviceDescription*>(args->device_description);
    if (dd) {
        args->to_string = dd->to_str.c_str();
        args->to_string_size = dd->to_str.size();
    } else {
        static const char kStr[] = "CLOUDTPU_MOCK(id=0)";
        args->to_string = kStr;
        args->to_string_size = sizeof(kStr) - 1;
    }
    return nullptr;
}
static PJRT_Error* mock_device_memory_stats(PJRT_Device_MemoryStats_Args* args) {
    auto* d = reinterpret_cast<MockDevice*>(args->device);
    auto* c = d->client;
    args->bytes_in_use = c && c->allocated_bytes
                             ? c->allocated_bytes->load(std::memory_order_relaxed)
                             : 0;
    args->bytes_limit_is_set = true;
    args->bytes_limit = 16909336064LL;  // 15.748 GiB (v5e-1 HBM2e nominal)
    return nullptr;
}

static PJRT_Error* mock_buffer_from_host(PJRT_Client_BufferFromHostBuffer_Args* args) {
    auto* c = reinterpret_cast<MockClient*>(args->client);
    auto* dev = reinterpret_cast<MockDevice*>(args->device);
    const PjrtDType dt = from_pjrt_buffer_type(args->type);
    if (dt == PjrtDType::INVALID) {
        return mock_error("MockPJRT: unsupported PJRT_Buffer_Type");
    }
    std::vector<int64_t> shape(args->dims, args->dims + args->num_dims);
    auto* mb = new MockBuffer();
    mb->allocated_bytes = c ? c->allocated_bytes : nullptr;
    mb->device_id = dev ? dev->id : 0;
    mb->tensor.allocate(dt, std::move(shape));
    if (!mb->tensor.bytes.empty() && args->data) {
        std::memcpy(mb->tensor.bytes.data(), args->data, mb->tensor.bytes.size());
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
static PJRT_Error* mock_buffer_dimensions(PJRT_Buffer_Dimensions_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->buffer);
    args->dims = mb->tensor.dims.empty() ? nullptr : mb->tensor.dims.data();
    args->num_dims = mb->tensor.dims.size();
    return nullptr;
}
static PJRT_Error* mock_buffer_to_host(PJRT_Buffer_ToHostBuffer_Args* args) {
    auto* mb = reinterpret_cast<MockBuffer*>(args->src);
    if (!args->dst) {
        args->dst_size = mb->tensor.bytes.size();
        args->event = nullptr;
        return nullptr;
    }
    if (args->dst_size < mb->tensor.bytes.size()) {
        return mock_error("MockPJRT: ToHostBuffer dst_size too small");
    }
    if (!mb->tensor.bytes.empty()) {
        std::memcpy(args->dst, mb->tensor.bytes.data(), mb->tensor.bytes.size());
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
        le->allocated_bytes = c ? c->allocated_bytes : nullptr;
        le->assigned_device_id =
            parse_assigned_device_id_from_compile_options(args->compile_options, args->compile_options_size);
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
        if (args->execute_device != nullptr) {
            auto* req_dev = reinterpret_cast<MockDevice*>(args->execute_device);
            if (req_dev->id != le->assigned_device_id) {
                return mock_error("MockPJRT: execute_device id " + std::to_string(req_dev->id) +
                                  " does not match compiled device assignment " +
                                  std::to_string(le->assigned_device_id));
            }
        }
        const MockProgram& prog = le->exec.program;
        if (args->num_args != prog.args.size()) {
            return mock_error("MockPJRT: argument count mismatch");
        }
        std::unordered_map<std::string, MockTensor> env;
        std::shared_ptr<std::atomic<int64_t>> allocated_bytes = le->allocated_bytes;
        for (size_t i = 0; i < args->num_args; ++i) {
            auto* mb = reinterpret_cast<MockBuffer*>(args->argument_lists[0][i]);
            if (mb->device_id != le->assigned_device_id) {
                return mock_error("MockPJRT: input buffer on device " + std::to_string(mb->device_id) +
                                  " does not match executable device " +
                                  std::to_string(le->assigned_device_id));
            }
            if (!allocated_bytes) allocated_bytes = mb->allocated_bytes;
            env[prog.args[i].first] = mb->tensor;
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
            MockTensor res = eval_mock_op(op, env);
            for (const std::string& inp : op.inputs) {
                auto it = last_use.find(inp);
                if (it != last_use.end() && it->second == idx) {
                    env.erase(inp);
                }
            }
            env[op.out_name] = std::move(res);
        }
        for (size_t oi = 0; oi < prog.return_names.size(); ++oi) {
            auto* out_mb = new MockBuffer();
            out_mb->allocated_bytes = allocated_bytes;
            out_mb->device_id = le->assigned_device_id;
            // Copy rather than move so repeated SSA values in return_names stay valid.
            out_mb->tensor = env.at(prog.return_names[oi]);
            if (allocated_bytes) {
                allocated_bytes->fetch_add(static_cast<int64_t>(out_mb->tensor.bytes.size()), std::memory_order_relaxed);
            }
            args->output_lists[0][oi] = reinterpret_cast<PJRT_Buffer*>(out_mb);
        }
        if (args->device_complete_events) {
            args->device_complete_events[0] = reinterpret_cast<PJRT_Event*>(new MockEvent());
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
        a.PJRT_DeviceDescription_Kind = mock_dev_desc_kind;
        a.PJRT_DeviceDescription_ToString = mock_dev_desc_to_string;
        a.PJRT_Device_MemoryStats = mock_device_memory_stats;
        a.PJRT_LoadedExecutable_Destroy = mock_loaded_exec_destroy;
        a.PJRT_LoadedExecutable_GetExecutable = mock_loaded_exec_get_exec;
        a.PJRT_LoadedExecutable_Execute = mock_loaded_exec_execute;
        a.PJRT_Executable_Destroy = mock_exec_destroy;
        a.PJRT_Executable_NumOutputs = mock_exec_num_outputs;
        a.PJRT_Buffer_Destroy = mock_buffer_destroy;
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

    PJRT_NamedValue mock_dev_opt = {};
    mock_dev_opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    mock_dev_opt.name = "mock_device_count";
    mock_dev_opt.name_size = 17;
    mock_dev_opt.type = PJRT_NamedValue_kInt64;
    mock_dev_opt.int64_value = std::max(1, cfg.mock_device_count);

    PJRT_Client_Create_Args c_args = {};
    c_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (is_mock_ && cfg.mock_device_count > 1) {
        c_args.create_options = &mock_dev_opt;
        c_args.num_options = 1;
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
    if (PJRT_API_HAS(api_, PJRT_Buffer_Dimensions)) {
        PJRT_Buffer_Dimensions_Args dim_args = {};
        dim_args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
        dim_args.buffer = buf;
        check_error(api_->PJRT_Buffer_Dimensions(&dim_args), "PJRT_Buffer_Dimensions");
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
}

CachedExecutablePtr PjrtRuntime::compile_or_get(const std::string& cache_key,
                                                const std::string& mlir_text,
                                                int device_idx,
                                                const PjrtCompileOptions& opts) {
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

    check_error(api_->PJRT_Client_Compile(&c_args), "PJRT_Client_Compile");
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
                          bool sync) {
    if (!exec || !exec->handle) {
        throw std::runtime_error("execute: null CachedExecutable");
    }
    if (device_idx < 0 || device_idx >= static_cast<int>(devices_.size())) {
        throw std::out_of_range("execute: device_idx out of range");
    }
    if (exec->device_idx != device_idx) {
        throw std::runtime_error("execute: executable compiled for device " +
                                 std::to_string(exec->device_idx) + ", cannot run on device " +
                                 std::to_string(device_idx));
    }
    if (exec->num_outputs != out_dtypes.size() || exec->num_outputs != out_dims.size()) {
        throw std::runtime_error("execute: output count mismatch vs executable");
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<PJRT_Buffer*> raw_outputs(exec->num_outputs, nullptr);
    PJRT_Buffer* const* arg_lists[1] = {inputs};
    PJRT_Buffer** const out_lists[1] = {raw_outputs.data()};

    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

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
    // Executable is statically assigned to the target chip via DeviceAssignmentProto at compile time.
    ex.execute_device = nullptr;

    std::vector<PjrtBufferHandle> out_handles;
    out_handles.reserve(exec->num_outputs);

    check_error(api_->PJRT_LoadedExecutable_Execute(&ex), "PJRT_LoadedExecutable_Execute");
    PjrtEventHandle dev_ev_handle(dev_ev[0], PjrtEventDeleter{api_});
    for (size_t i = 0; i < exec->num_outputs; ++i) {
        out_handles.emplace_back(raw_outputs[i], PjrtBufferDeleter{api_});
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

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t exec_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    std::lock_guard<std::mutex> lock(cache_mu_);
    total_execute_us_ += exec_us;
}

void PjrtRuntime::clear_exec_cache() {
    std::lock_guard<std::mutex> lock(cache_mu_);
    exec_cache_.clear();
    lru_order_.clear();
}

CacheStats PjrtRuntime::cache_stats() const {
    std::lock_guard<std::mutex> lock(cache_mu_);
    CacheStats st;
    st.entries = exec_cache_.size();
    st.hits = cache_hits_;
    st.misses = cache_misses_;
    st.evictions = cache_evictions_;
    st.compile_us = total_compile_us_;
    st.execute_us = total_execute_us_;
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
