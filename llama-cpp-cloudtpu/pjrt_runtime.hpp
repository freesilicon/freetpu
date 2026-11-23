#pragma once

#include <pjrt_c_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cloudtpu {

enum class PjrtDType {
    INVALID = 0,
    PRED,
    F32,
    F16,
    BF16,
    S8,
    U8,
    S16,
    S32,
    U32,
    S64,
};

class PjrtError : public std::runtime_error {
public:
    PjrtError(PJRT_Error_Code code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}

    PJRT_Error_Code code() const noexcept { return code_; }

    bool is_resource_exhausted() const noexcept {
        return code_ == PJRT_Error_Code_RESOURCE_EXHAUSTED;
    }

private:
    PJRT_Error_Code code_;
};

PJRT_Buffer_Type to_pjrt_buffer_type(PjrtDType dt);
PjrtDType from_pjrt_buffer_type(PJRT_Buffer_Type dt);
const char* pjrt_dtype_mlir_name(PjrtDType dt);
size_t pjrt_dtype_size(PjrtDType dt);

// Pure C++17 half/bfloat16 conversion helpers (zero external dependencies).
uint16_t f32_to_bf16(float f);
float bf16_to_f32(uint16_t b);
uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);

// Counts "libpython" mappings in /proc/self/maps (0 on non-Linux or pure C++ processes).
size_t count_libpython_mappings();

struct PjrtCompileOptions {
    int32_t num_replicas = 1;
    int32_t num_partitions = 1;
    bool use_spmd_partitioning = true;
    bool use_shardy_partitioner = true;
    // Optional explicit DeviceAssignmentProto mapping [partition][replica] -> device_id.
    // Length must equal num_replicas * num_partitions when non-empty.
    std::vector<int64_t> device_assignment_ids;
};

// Encodes xla::CompileOptionsProto in pure C++ protobuf wire format (zero libprotobuf dep).
std::string encode_compile_options_proto(const PjrtCompileOptions& opts);

struct PjrtDeviceInfo {
    int index = 0;
    int id = 0;
    std::string kind;
    std::string description;
    int64_t hbm_bytes_in_use = 0;
    int64_t hbm_bytes_limit = 0;
};

struct RuntimeConfig {
    bool mock_mode = false;
    int mock_devices = 1;  // Number of virtual TPU chips in Host Mock PJRT (e.g. 4=tpu7x, 8=v6e-8)
    int verbose = 0;
    uint32_t exec_cache_cap = 4096;
    std::string libtpu_path;
};

struct CacheStats {
    size_t entries = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t compile_us = 0;
    uint64_t execute_us = 0;
};

class PjrtRuntime;

struct PjrtBufferDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Buffer* buf) const noexcept;
};

using PjrtBufferHandle = std::unique_ptr<PJRT_Buffer, PjrtBufferDeleter>;

struct PjrtEventDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Event* ev) const noexcept;
};

using PjrtEventHandle = std::unique_ptr<PJRT_Event, PjrtEventDeleter>;

struct PjrtBuffer {
    PjrtBufferHandle handle;
    PjrtDType dtype = PjrtDType::INVALID;
    std::vector<int64_t> dims;

    size_t num_elements() const;
    size_t num_bytes() const;
};

using PjrtBufferPtr = std::shared_ptr<PjrtBuffer>;

struct PjrtLoadedExecutableDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_LoadedExecutable* exec) const noexcept;
};

using PjrtLoadedExecutableHandle = std::unique_ptr<PJRT_LoadedExecutable, PjrtLoadedExecutableDeleter>;

struct CachedExecutable {
    std::string key;
    PjrtLoadedExecutableHandle handle;
    int device_idx = 0;
    int num_replicas = 1;
    size_t num_outputs = 0;
};

using CachedExecutablePtr = std::shared_ptr<const CachedExecutable>;

class PjrtRuntime {
public:
    PjrtRuntime() = default;
    ~PjrtRuntime();

    PjrtRuntime(const PjrtRuntime&) = delete;
    PjrtRuntime& operator=(const PjrtRuntime&) = delete;

    // Initialize either live libtpu.so via dlopen() -> GetPjrtApi(), or in-process Host Mock PJRT.
    void init(const RuntimeConfig& cfg);
    void shutdown();

    bool is_initialized() const { return initialized_; }
    bool is_mock() const { return is_mock_; }
    bool has_ready_event() const { return has_ready_event_; }

    const std::string& platform_name() const { return platform_name_; }
    const std::string& platform_version() const { return platform_version_; }
    int device_count() const { return static_cast<int>(devices_.size()); }
    const PjrtDeviceInfo& device_info(int device_idx) const;

    bool query_memory_stats(int device_idx, size_t* free_bytes, size_t* total_bytes) const;

    PjrtBufferPtr buffer_from_host(int device_idx,
                                   const void* data,
                                   PjrtDType dtype,
                                   const std::vector<int64_t>& dims);

    std::vector<PjrtBufferPtr> buffer_from_host_replicated(int num_devices,
                                                           const void* data,
                                                           PjrtDType dtype,
                                                           const std::vector<int64_t>& dims);

    void buffer_to_host(PJRT_Buffer* buf, void* dst, size_t dst_size);
    PjrtEventHandle create_buffer_ready_event(PJRT_Buffer* buf);
    void await_buffer_ready(PJRT_Buffer* buf);
    void await_event(PJRT_Event* ev);
    void await_event(PjrtEventHandle ev);

    // Builds compile options (including DeviceAssignmentProto for multi-chip SPMD or
    // explicit single-device targeting on multi-chip hosts) for a 1D chip mesh.
    PjrtCompileOptions make_spmd_compile_options(int num_chips, int device_idx = 0) const;

    // Compile StableHLO MLIR ("mlir" format) or return an owning reference to a cached executable.
    // Thread-safe LRU cache keyed by canonical structural program key and target device assignment.
    CachedExecutablePtr compile_or_get(const std::string& cache_key,
                                       const std::string& mlir_text,
                                       int device_idx = 0,
                                       const PjrtCompileOptions& opts = {});

    void execute(const CachedExecutablePtr& exec,
                 int device_idx,
                 PJRT_Buffer* const* inputs,
                 size_t num_inputs,
                 std::vector<PjrtBufferPtr>& outputs,
                 const std::vector<PjrtDType>& out_dtypes,
                 const std::vector<std::vector<int64_t>>& out_dims,
                 bool sync);

    void execute_spmd(const CachedExecutablePtr& exec,
                      int num_devices,
                      PJRT_Buffer* const* const* per_device_inputs,
                      size_t num_inputs,
                      std::vector<std::vector<PjrtBufferPtr>>& per_device_outputs,
                      const std::vector<PjrtDType>& out_dtypes,
                      const std::vector<std::vector<int64_t>>& out_dims,
                      bool sync);

    void clear_exec_cache();
    CacheStats cache_stats() const;
    uint64_t eviction_epoch() const { return eviction_epoch_.load(std::memory_order_acquire); }

    // Exposes raw PJRT API pointers for diagnostics/tests.
    const PJRT_Api* pjrt_api() const { return api_; }

private:
    void destroy_error(PJRT_Error* err) const;
    void check_error(PJRT_Error* err, const char* where) const;
    std::string resolve_libtpu_path(const std::string& override_path) const;

    RuntimeConfig cfg_;
    bool initialized_ = false;
    bool is_mock_ = false;
    bool has_ready_event_ = false;
    void* dl_handle_ = nullptr;
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
    std::string platform_name_;
    std::string platform_version_;
    std::vector<PJRT_Device*> devices_;
    std::vector<PjrtDeviceInfo> device_infos_;

    mutable std::mutex cache_mu_;
    using LruList = std::list<std::string>;
    struct CacheEntry {
        CachedExecutablePtr exec;
        LruList::iterator lru_it;
    };
    LruList lru_order_;
    std::unordered_map<std::string, CacheEntry> exec_cache_;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t cache_evictions_ = 0;
    uint64_t total_compile_us_ = 0;
    std::atomic<uint64_t> total_execute_us_{0};
    std::atomic<uint64_t> eviction_epoch_{0};
};

// Shared process-wide PJRT runtime singleton (libtpu.so owns exclusive /dev/vfio/0 per process).
PjrtRuntime& shared_pjrt_runtime(const RuntimeConfig* override_cfg = nullptr);

}  // namespace cloudtpu
