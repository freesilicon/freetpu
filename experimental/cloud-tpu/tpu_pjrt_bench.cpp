// Zero-Python C++17 PJRT TPU Runtime Prototype & GEMM Benchmark
// Targets Colab TPU v5e-1 via direct dlopen(libtpu.so) -> GetPjrtApi().
// Compile on Linux x86_64:
//   g++ -O3 -std=c++17 -Wall -Wextra -I"$HOME/.cache/freetpu-pjrt/ir" \
//     experimental/cloud-tpu/tpu_pjrt_bench.cpp -ldl -pthread -o "$HOME/.cache/freetpu-pjrt/tpu_pjrt_bench"

#include <pjrt_c_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define PJRT_API_HAS(api, field)                                            \
  ((api) != nullptr &&                                                      \
   (offsetof(PJRT_Api, field) + sizeof((api)->field) <= (api)->struct_size) \
   && ((api)->field != nullptr))

inline uint16_t f32_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((u >> 16) | 0x0040U);
  }
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7fffU + lsb;
  return static_cast<uint16_t>(u >> 16);
}

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

std::string read_file_bytes(const std::string& path, bool* ok = nullptr) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    if (ok) *ok = false;
    return {};
  }
  std::streamsize sz = ifs.tellg();
  if (sz < 0) {
    if (ok) *ok = false;
    return {};
  }
  std::string buf(static_cast<size_t>(sz), '\0');
  ifs.seekg(0, std::ios::beg);
  if (sz > 0 && !ifs.read(&buf[0], sz)) {
    if (ok) *ok = false;
    return {};
  }
  if (ok) *ok = true;
  return buf;
}

size_t count_libpython_in_maps(std::vector<std::string>* hits = nullptr) {
  std::ifstream ifs("/proc/self/maps");
  if (!ifs.is_open()) {
    // Non-Linux host check or unreadable /proc/self/maps.
    return 0;
  }
  std::string line;
  size_t count = 0;
  while (std::getline(ifs, line)) {
    if (line.find("libpython") != std::string::npos) {
      ++count;
      if (hits) hits->push_back(line);
    }
  }
  return count;
}

double percentile_linear(const std::vector<double>& sorted, double pct) {
  if (sorted.empty()) return 0.0;
  double pos = (pct / 100.0) * static_cast<double>(sorted.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(pos));
  size_t hi = static_cast<size_t>(std::ceil(pos));
  double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

struct Stats {
  double median_us = 0.0;
  double min_us = 0.0;
  double p99_us = 0.0;
};

Stats compute_stats(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Stats s;
  s.min_us = samples.front();
  s.median_us = percentile_linear(samples, 50.0);
  s.p99_us = percentile_linear(samples, 99.0);
  return s;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

struct TpuExecutable {
  PJRT_LoadedExecutable* handle = nullptr;
  size_t num_outputs = 1;
};

class TpuRuntime {
 public:
  TpuRuntime() = default;
  ~TpuRuntime() { tpuShutdown(); }

  TpuRuntime(const TpuRuntime&) = delete;
  TpuRuntime& operator=(const TpuRuntime&) = delete;

  void tpuInit(const std::string& libtpu_path) {
    setenv("GRPC_VERBOSITY", "ERROR", 0);
    dl_handle_ = dlopen(libtpu_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!dl_handle_) {
      const char* err = dlerror();
      throw std::runtime_error("dlopen failed for " + libtpu_path + ": " +
                               (err ? err : "unknown error"));
    }

    using GetPjrtApiFn = const PJRT_Api* (*)();
    auto get_pjrt_api =
        reinterpret_cast<GetPjrtApiFn>(dlsym(dl_handle_, "GetPjrtApi"));
    if (!get_pjrt_api) {
      const char* err = dlerror();
      throw std::runtime_error(std::string("dlsym(GetPjrtApi) failed: ") +
                               (err ? err : "unknown error"));
    }

    api_ = get_pjrt_api();
    if (!api_) {
      throw std::runtime_error("GetPjrtApi() returned nullptr");
    }

    std::cout << "CPP_PJRT_API: {\"runtime_struct_size\": "
              << api_->struct_size
              << ", \"runtime_version\": \""
              << api_->pjrt_api_version.major_version << "."
              << api_->pjrt_api_version.minor_version << "\""
              << ", \"header_struct_size\": " << PJRT_Api_STRUCT_SIZE
              << ", \"header_version\": \"" << PJRT_API_MAJOR << "."
              << PJRT_API_MINOR << "\"}" << std::endl;

    // Verify all core PJRT symbols sit within runtime api_->struct_size.
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
        !PJRT_API_HAS(api_, PJRT_LoadedExecutable_Execute) ||
        !PJRT_API_HAS(api_, PJRT_Buffer_Destroy) ||
        !PJRT_API_HAS(api_, PJRT_Buffer_ToHostBuffer)) {
      throw std::runtime_error("Required PJRT_Api function pointer missing or "
                               "outside runtime struct_size");
    }

    if (PJRT_API_HAS(api_, PJRT_Plugin_Initialize)) {
      PJRT_Plugin_Initialize_Args init_args = {};
      init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
      checkError(api_->PJRT_Plugin_Initialize(&init_args),
                 "PJRT_Plugin_Initialize");
    }

    PJRT_Client_Create_Args c_args = {};
    c_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    checkError(api_->PJRT_Client_Create(&c_args), "PJRT_Client_Create");
    client_ = c_args.client;

    std::string platform_name = "unknown";
    std::string platform_version = "unknown";
    if (PJRT_API_HAS(api_, PJRT_Client_PlatformName)) {
      PJRT_Client_PlatformName_Args p_args = {};
      p_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
      p_args.client = client_;
      checkError(api_->PJRT_Client_PlatformName(&p_args),
                 "PJRT_Client_PlatformName");
      platform_name.assign(p_args.platform_name, p_args.platform_name_size);
    }
    if (PJRT_API_HAS(api_, PJRT_Client_PlatformVersion)) {
      PJRT_Client_PlatformVersion_Args v_args = {};
      v_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
      v_args.client = client_;
      checkError(api_->PJRT_Client_PlatformVersion(&v_args),
                 "PJRT_Client_PlatformVersion");
      platform_version.assign(v_args.platform_version,
                              v_args.platform_version_size);
    }

    PJRT_Client_AddressableDevices_Args d_args = {};
    d_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    d_args.client = client_;
    checkError(api_->PJRT_Client_AddressableDevices(&d_args),
               "PJRT_Client_AddressableDevices");
    if (d_args.num_addressable_devices == 0 || !d_args.addressable_devices) {
      throw std::runtime_error("No addressable TPU devices found");
    }
    device_ = d_args.addressable_devices[0];

    PJRT_Device_GetDescription_Args gd_args = {};
    gd_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    gd_args.device = device_;
    checkError(api_->PJRT_Device_GetDescription(&gd_args),
               "PJRT_Device_GetDescription");
    PJRT_DeviceDescription* desc = gd_args.device_description;

    int dev_id = -1;
    if (PJRT_API_HAS(api_, PJRT_DeviceDescription_Id)) {
      PJRT_DeviceDescription_Id_Args id_args = {};
      id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
      id_args.device_description = desc;
      checkError(api_->PJRT_DeviceDescription_Id(&id_args),
                 "PJRT_DeviceDescription_Id");
      dev_id = id_args.id;
    }

    PJRT_DeviceDescription_Kind_Args k_args = {};
    k_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    k_args.device_description = desc;
    checkError(api_->PJRT_DeviceDescription_Kind(&k_args),
               "PJRT_DeviceDescription_Kind");
    std::string kind(k_args.device_kind, k_args.device_kind_size);

    std::string dev_str;
    if (PJRT_API_HAS(api_, PJRT_DeviceDescription_ToString)) {
      PJRT_DeviceDescription_ToString_Args ts_args = {};
      ts_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
      ts_args.device_description = desc;
      checkError(api_->PJRT_DeviceDescription_ToString(&ts_args),
                 "PJRT_DeviceDescription_ToString");
      dev_str.assign(ts_args.to_string, ts_args.to_string_size);
    }

    int64_t hbm_limit = -1;
    int64_t hbm_in_use = -1;
    if (PJRT_API_HAS(api_, PJRT_Device_MemoryStats)) {
      PJRT_Device_MemoryStats_Args ms_args = {};
      ms_args.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
      ms_args.device = device_;
      PJRT_Error* ms_err = api_->PJRT_Device_MemoryStats(&ms_args);
      if (!ms_err) {
        hbm_in_use = ms_args.bytes_in_use;
        if (ms_args.bytes_limit_is_set) hbm_limit = ms_args.bytes_limit;
      } else {
        destroyError(ms_err);
      }
    }

    std::cout << "CPP_PJRT_DEVICE: {\"platform\": \""
              << json_escape(platform_name) << "\", \"platform_version\": \""
              << json_escape(platform_version)
              << "\", \"num_addressable_devices\": "
              << d_args.num_addressable_devices << ", \"device_id\": " << dev_id
              << ", \"device_kind\": \"" << json_escape(kind)
              << "\", \"device_str\": \"" << json_escape(dev_str)
              << "\", \"hbm_bytes_in_use\": " << hbm_in_use
              << ", \"hbm_bytes_limit\": " << hbm_limit
              << ", \"has_ready_event\": "
              << (PJRT_API_HAS(api_, PJRT_Buffer_ReadyEvent) ? "true"
                                                             : "false")
              << "}" << std::endl;
  }

  PJRT_Buffer* tpuCopyH2D(const void* data, PJRT_Buffer_Type dtype,
                          const std::vector<int64_t>& dims) {
    PJRT_Client_BufferFromHostBuffer_Args args = {};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client_;
    args.data = data;
    args.type = dtype;
    args.dims = dims.data();
    args.num_dims = dims.size();
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    args.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    args.device = device_;
    args.memory = nullptr;
    args.device_layout = nullptr;
    checkError(api_->PJRT_Client_BufferFromHostBuffer(&args),
               "PJRT_Client_BufferFromHostBuffer");
    if (args.done_with_host_buffer) {
      tpuSyncEvent(args.done_with_host_buffer);
    }
    return args.buffer;
  }

  size_t queryHostBufferBytes(PJRT_Buffer* buf) {
    PJRT_Buffer_ToHostBuffer_Args args = {};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.host_layout = nullptr;
    args.dst = nullptr;
    args.dst_size = 0;
    checkError(api_->PJRT_Buffer_ToHostBuffer(&args),
               "PJRT_Buffer_ToHostBuffer(query_size)");
    if (args.event) {
      tpuSyncEvent(args.event);
    }
    return args.dst_size;
  }

  void tpuCopyD2H(PJRT_Buffer* buf, void* dst, size_t dst_size) {
    PJRT_Buffer_ToHostBuffer_Args args = {};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.host_layout = nullptr;
    args.dst = dst;
    args.dst_size = dst_size;
    checkError(api_->PJRT_Buffer_ToHostBuffer(&args),
               "PJRT_Buffer_ToHostBuffer");
    if (args.event) {
      tpuSyncEvent(args.event);
    }
  }

  TpuExecutable tpuCompile(const std::string& program_bytes,
                           const std::string& format,
                           const std::string& compile_options_pb) {
    PJRT_Program prog = {};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = const_cast<char*>(program_bytes.data());
    prog.code_size = program_bytes.size();
    prog.format = format.data();
    prog.format_size = format.size();

    PJRT_Client_Compile_Args c_args = {};
    c_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    c_args.client = client_;
    c_args.program = &prog;
    c_args.compile_options =
        compile_options_pb.empty() ? nullptr : compile_options_pb.data();
    c_args.compile_options_size = compile_options_pb.size();

    checkError(api_->PJRT_Client_Compile(&c_args), "PJRT_Client_Compile");

    TpuExecutable exec;
    exec.handle = c_args.executable;
    exec.num_outputs = 1;

    if (PJRT_API_HAS(api_, PJRT_LoadedExecutable_GetExecutable) &&
        PJRT_API_HAS(api_, PJRT_Executable_Destroy)) {
      PJRT_LoadedExecutable_GetExecutable_Args ge = {};
      ge.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
      ge.loaded_executable = exec.handle;
      PJRT_Error* ge_err = api_->PJRT_LoadedExecutable_GetExecutable(&ge);
      if (!ge_err && ge.executable) {
        if (PJRT_API_HAS(api_, PJRT_Executable_NumOutputs)) {
          PJRT_Executable_NumOutputs_Args no = {};
          no.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
          no.executable = ge.executable;
          PJRT_Error* no_err = api_->PJRT_Executable_NumOutputs(&no);
          if (!no_err && no.num_outputs > 0) {
            exec.num_outputs = no.num_outputs;
          } else if (no_err) {
            destroyError(no_err);
          }
        }
        PJRT_Executable_Destroy_Args ed = {};
        ed.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        ed.executable = ge.executable;
        PJRT_Error* ed_err = api_->PJRT_Executable_Destroy(&ed);
        if (ed_err) destroyError(ed_err);
      } else if (ge_err) {
        destroyError(ge_err);
      }
    }
    return exec;
  }

  void tpuLaunch(const TpuExecutable& exec, PJRT_Buffer* const* inputs,
                 size_t num_inputs, PJRT_Buffer** outputs,
                 PJRT_Event** device_complete_events = nullptr) {
    PJRT_ExecuteOptions opts = {};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    opts.context = nullptr;

    PJRT_Buffer* const* arg_lists[1] = {inputs};
    PJRT_Buffer** const out_lists[1] = {outputs};

    PJRT_LoadedExecutable_Execute_Args ex = {};
    ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ex.executable = exec.handle;
    ex.options = &opts;
    ex.argument_lists = arg_lists;
    ex.num_devices = 1;
    ex.num_args = num_inputs;
    ex.output_lists = out_lists;
    ex.device_complete_events = device_complete_events;
    ex.execute_device = nullptr;

    checkError(api_->PJRT_LoadedExecutable_Execute(&ex),
               "PJRT_LoadedExecutable_Execute");
  }

  void tpuSyncBuffer(PJRT_Buffer* buf) {
    if (!PJRT_API_HAS(api_, PJRT_Buffer_ReadyEvent)) {
      throw std::runtime_error("PJRT_Buffer_ReadyEvent unavailable");
    }
    PJRT_Buffer_ReadyEvent_Args r_args = {};
    r_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    r_args.buffer = buf;
    checkError(api_->PJRT_Buffer_ReadyEvent(&r_args), "PJRT_Buffer_ReadyEvent");
    if (r_args.event) {
      tpuSyncEvent(r_args.event);
    }
  }

  void tpuSyncEvent(PJRT_Event* ev) {
    if (!ev) return;
    PJRT_Event_Await_Args a_args = {};
    a_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    a_args.event = ev;
    PJRT_Error* err = api_->PJRT_Event_Await(&a_args);

    PJRT_Event_Destroy_Args d_args = {};
    d_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    d_args.event = ev;
    PJRT_Error* derr = api_->PJRT_Event_Destroy(&d_args);
    if (err) {
      if (derr) destroyError(derr);
      checkError(err, "PJRT_Event_Await");
    }
    checkError(derr, "PJRT_Event_Destroy");
  }

  void tpuFree(PJRT_Buffer* buf) {
    if (!buf) return;
    PJRT_Buffer_Destroy_Args args = {};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buf;
    checkError(api_->PJRT_Buffer_Destroy(&args), "PJRT_Buffer_Destroy");
  }

  void tpuFreeExec(TpuExecutable& exec) {
    if (!exec.handle) return;
    PJRT_LoadedExecutable_Destroy_Args args = {};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = exec.handle;
    exec.handle = nullptr;
    checkError(api_->PJRT_LoadedExecutable_Destroy(&args),
               "PJRT_LoadedExecutable_Destroy");
  }

  void tpuShutdown() {
    if (client_ && api_ && PJRT_API_HAS(api_, PJRT_Client_Destroy)) {
      PJRT_Client_Destroy_Args args = {};
      args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      args.client = client_;
      client_ = nullptr;
      PJRT_Error* err = api_->PJRT_Client_Destroy(&args);
      if (err) destroyError(err);
    }
    // Note: intentionally skip dlclose(dl_handle_) as standard for libtpu/C++ singletons.
    api_ = nullptr;
  }

  bool hasReadyEvent() const {
    return PJRT_API_HAS(api_, PJRT_Buffer_ReadyEvent);
  }

 private:
  void destroyError(PJRT_Error* err) {
    if (!err || !api_ || !PJRT_API_HAS(api_, PJRT_Error_Destroy)) return;
    PJRT_Error_Destroy_Args d_args = {};
    d_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    d_args.error = err;
    api_->PJRT_Error_Destroy(&d_args);
  }

  void checkError(PJRT_Error* err, const char* where) {
    if (!err) return;
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
    destroyError(err);
    throw std::runtime_error(std::string(where) + ": " + msg);
  }

  void* dl_handle_ = nullptr;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
  PJRT_Device* device_ = nullptr;
};

struct CompileOptionCandidate {
  std::string label;
  std::string bytes;
};

std::vector<CompileOptionCandidate> load_compile_option_candidates(
    const std::string& ir_dir) {
  std::vector<CompileOptionCandidate> out;
  bool ok = false;
  std::string full = read_file_bytes(ir_dir + "/compile_options.pb", &ok);
  if (ok && !full.empty()) {
    out.push_back({"compile_options.pb", std::move(full)});
  }
  std::string minimal =
      read_file_bytes(ir_dir + "/compile_options_minimal.pb", &ok);
  if (ok && !minimal.empty()) {
    out.push_back({"compile_options_minimal.pb", std::move(minimal)});
  }
  // Built-in fallback CompileOptionsProto:
  // executable_build_options {
  //   num_replicas: 1, num_partitions: 1,
  //   use_spmd_partitioning: true, use_shardy_partitioner: true
  // }
  static const uint8_t kBuiltinProto[] = {
      0x1a, 0x09, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01, 0x98, 0x01, 0x01};
  out.push_back({"builtin_1x1_spmd_shardy",
                 std::string(reinterpret_cast<const char*>(kBuiltinProto),
                             sizeof(kBuiltinProto))});
  static const uint8_t kBuiltinNoShardyProto[] = {
      0x1a, 0x06, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01};
  out.push_back({"builtin_1x1_spmd",
                 std::string(reinterpret_cast<const char*>(kBuiltinNoShardyProto),
                             sizeof(kBuiltinNoShardyProto))});
  return out;
}

TpuExecutable compile_shape_program(
    TpuRuntime& rt, const std::string& ir_dir, const std::string& tag,
    const std::vector<CompileOptionCandidate>& opts_list,
    std::string* used_ir_out, std::string* used_opts_out) {
  struct IrCandidate {
    std::string suffix;
    std::string label;
  };
  const IrCandidate irs[] = {
      {".mlirbc", "mlirbc"},
      {".mlir", "mlir_text"},
  };

  std::string last_err;
  for (const auto& ir : irs) {
    bool ok = false;
    std::string code =
        read_file_bytes(ir_dir + "/gemm_" + tag + ir.suffix, &ok);
    if (!ok || code.empty()) continue;

    for (const auto& opt : opts_list) {
      try {
        TpuExecutable exec = rt.tpuCompile(code, "mlir", opt.bytes);
        if (used_ir_out) *used_ir_out = ir.label;
        if (used_opts_out) *used_opts_out = opt.label;
        return exec;
      } catch (const std::exception& e) {
        last_err = ir.label + "+" + opt.label + ": " + e.what();
      }
    }
  }
  throw std::runtime_error("Failed to compile gemm_" + tag + ": " + last_err);
}

void fill_bf16_pattern(std::vector<uint16_t>& buf, uint64_t seed) {
  uint16_t lut[4096];
  for (size_t i = 0; i < 4096; ++i) {
    float v = static_cast<float>((static_cast<int>((i + seed) % 65) - 32)) *
              0.0078125f;  // exact 2^-7 multiples in [-0.25, +0.25]
    lut[i] = f32_to_bf16(v);
  }
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = lut[(i * 17ULL + seed) & 4095ULL];
  }
}

void print_bench_json(const std::string& shape, const std::string& mode,
                      const std::string& dtype, int warmup, int iters,
                      const Stats& st, double flops,
                      const std::string& ir_fmt, const std::string& opt_fmt) {
  double med_tflops = (st.median_us > 0.0) ? (flops / (st.median_us * 1e6)) : 0.0;
  double peak_tflops = (st.min_us > 0.0) ? (flops / (st.min_us * 1e6)) : 0.0;
  std::cout << std::fixed << std::setprecision(2)
            << "CPP_PJRT_BENCH: {\"shape\": \"" << shape
            << "\", \"mode\": \"" << mode
            << "\", \"dtype\": \"" << dtype
            << "\", \"warmup\": " << warmup
            << ", \"iters\": " << iters
            << ", \"median_us\": " << st.median_us
            << ", \"min_us\": " << st.min_us
            << ", \"p99_us\": " << st.p99_us
            << ", \"median_tflops\": " << med_tflops
            << ", \"peak_tflops\": " << peak_tflops
            << ", \"ir\": \"" << ir_fmt
            << "\", \"compile_opts\": \"" << opt_fmt << "\"}" << std::endl;
}

void verify_and_bench_small(
    TpuRuntime& rt, const std::string& ir_dir,
    const std::vector<CompileOptionCandidate>& opts_list) {
  constexpr int64_t M = 1, K = 64, N = 32;
  const std::string tag = "1x64x32_bf16_f32";
  const std::string dtype = "bfloat16->float32";
  const double flops = 2.0 * M * K * N;

  std::vector<uint16_t> host_a(M * K);
  std::vector<uint16_t> host_b(K * N);
  for (int64_t k = 0; k < K; ++k) {
    float va = static_cast<float>((k % 17) - 8) * 0.125f;
    host_a[k] = f32_to_bf16(va);
    for (int64_t n = 0; n < N; ++n) {
      float vb = static_cast<float>(((k * 7 + n * 3) % 13) - 6) * 0.0625f;
      host_b[k * N + n] = f32_to_bf16(vb);
    }
  }

  std::vector<float> ref_c(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += bf16_to_f32(host_a[m * K + k]) * bf16_to_f32(host_b[k * N + n]);
      }
      ref_c[m * N + n] = acc;
    }
  }

  std::string ir_fmt, opt_fmt;
  TpuExecutable exec =
      compile_shape_program(rt, ir_dir, tag, opts_list, &ir_fmt, &opt_fmt);
  PJRT_Buffer* buf_a = rt.tpuCopyH2D(host_a.data(), PJRT_Buffer_Type_BF16, {M, K});
  PJRT_Buffer* buf_b = rt.tpuCopyH2D(host_b.data(), PJRT_Buffer_Type_BF16, {K, N});
  PJRT_Buffer* inputs[2] = {buf_a, buf_b};
  std::vector<PJRT_Buffer*> outputs(exec.num_outputs, nullptr);

  // 1) Numeric parity check via PJRT_Buffer_ToHostBuffer
  rt.tpuLaunch(exec, inputs, 2, outputs.data(), nullptr);
  if (rt.hasReadyEvent()) {
    rt.tpuSyncBuffer(outputs[0]);
  }
  size_t queried_bytes = rt.queryHostBufferBytes(outputs[0]);
  std::vector<float> host_c(M * N, 0.0f);
  if (queried_bytes != host_c.size() * sizeof(float)) {
    for (auto*& b : outputs) {
      rt.tpuFree(b);
      b = nullptr;
    }
    rt.tpuFree(buf_a);
    rt.tpuFree(buf_b);
    rt.tpuFreeExec(exec);
    throw std::runtime_error("Unexpected output host size for 1x64x32: " +
                             std::to_string(queried_bytes));
  }
  rt.tpuCopyD2H(outputs[0], host_c.data(), queried_bytes);
  for (auto*& b : outputs) {
    rt.tpuFree(b);
    b = nullptr;
  }

  double max_abs_err = 0.0;
  for (size_t i = 0; i < host_c.size(); ++i) {
    if (!std::isfinite(host_c[i])) {
      rt.tpuFree(buf_a);
      rt.tpuFree(buf_b);
      rt.tpuFreeExec(exec);
      throw std::runtime_error("Non-finite output in 1x64x32 parity check");
    }
    double err = std::abs(static_cast<double>(host_c[i]) -
                          static_cast<double>(ref_c[i]));
    if (err > max_abs_err) max_abs_err = err;
  }
  bool pass = (max_abs_err <= 1e-4);
  std::cout << std::scientific << std::setprecision(6)
            << "CPP_PJRT_PARITY: {\"shape\": \"1x64x32\", \"dtype\": \""
            << dtype << "\", \"max_abs_err\": " << max_abs_err
            << ", \"tol\": 1e-4, \"pass\": " << (pass ? "true" : "false")
            << ", \"ir\": \"" << ir_fmt << "\", \"compile_opts\": \""
            << opt_fmt << "\"}" << std::endl;
  if (!pass) {
    rt.tpuFree(buf_a);
    rt.tpuFree(buf_b);
    rt.tpuFreeExec(exec);
    throw std::runtime_error("1x64x32 numeric parity failed");
  }

  constexpr int kWarmup = 10;
  constexpr int kIters = 100;
  using Clock = std::chrono::steady_clock;

  // 2) Benchmark: per-step PJRT_Buffer_ReadyEvent sync (matches block_until_ready)
  if (rt.hasReadyEvent()) {
    for (int i = 0; i < kWarmup; ++i) {
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), nullptr);
      rt.tpuSyncBuffer(outputs[0]);
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    std::vector<double> lat_us;
    lat_us.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
      auto t0 = Clock::now();
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), nullptr);
      rt.tpuSyncBuffer(outputs[0]);
      auto t1 = Clock::now();
      lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    print_bench_json("1x64x32", "sync_ready_event", dtype, kWarmup, kIters,
                     compute_stats(lat_us), flops, ir_fmt, opt_fmt);
  }

  // 3) Benchmark: per-step Execute device_complete_events sync
  {
    PJRT_Event* dev_events[1] = {nullptr};
    for (int i = 0; i < kWarmup; ++i) {
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), dev_events);
      rt.tpuSyncEvent(dev_events[0]);
      dev_events[0] = nullptr;
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    std::vector<double> lat_us;
    lat_us.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
      auto t0 = Clock::now();
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), dev_events);
      rt.tpuSyncEvent(dev_events[0]);
      auto t1 = Clock::now();
      dev_events[0] = nullptr;
      lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    print_bench_json("1x64x32", "sync_device_complete", dtype, kWarmup, kIters,
                     compute_stats(lat_us), flops, ir_fmt, opt_fmt);
  }

  // 4) Benchmark: per-step ToHostBuffer + Event_Await (full H2D-ready dispatch + D2H)
  {
    for (int i = 0; i < kWarmup; ++i) {
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), nullptr);
      rt.tpuCopyD2H(outputs[0], host_c.data(), host_c.size() * sizeof(float));
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    std::vector<double> lat_us;
    lat_us.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
      auto t0 = Clock::now();
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), nullptr);
      rt.tpuCopyD2H(outputs[0], host_c.data(), host_c.size() * sizeof(float));
      auto t1 = Clock::now();
      lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }
    print_bench_json("1x64x32", "sync_to_host", dtype, kWarmup, kIters,
                     compute_stats(lat_us), flops, ir_fmt, opt_fmt);
  }

  // 5) Benchmark: async dispatch queue (enqueue 100 launches, then single final sync)
  {
    std::vector<PJRT_Buffer*> queued(kIters * exec.num_outputs, nullptr);
    for (int i = 0; i < kWarmup; ++i) {
      PJRT_Event* dev_ev[1] = {nullptr};
      bool need_dev = (!rt.hasReadyEvent() && i == kWarmup - 1);
      rt.tpuLaunch(exec, inputs, 2, outputs.data(), need_dev ? dev_ev : nullptr);
      if (i == kWarmup - 1) {
        if (rt.hasReadyEvent()) rt.tpuSyncBuffer(outputs[0]);
        else rt.tpuSyncEvent(dev_ev[0]);
      }
      for (auto*& b : outputs) {
        rt.tpuFree(b);
        b = nullptr;
      }
    }

    std::vector<double> enqueue_us;
    enqueue_us.reserve(kIters);
    PJRT_Event* last_dev_ev[1] = {nullptr};
    auto wall_t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
      bool need_dev = (!rt.hasReadyEvent() && i == kIters - 1);
      auto t0 = Clock::now();
      rt.tpuLaunch(exec, inputs, 2, &queued[i * exec.num_outputs],
                   need_dev ? last_dev_ev : nullptr);
      auto t1 = Clock::now();
      enqueue_us.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    auto enqueue_end = Clock::now();
    if (rt.hasReadyEvent()) {
      rt.tpuSyncBuffer(queued[(kIters - 1) * exec.num_outputs]);
    } else {
      rt.tpuSyncEvent(last_dev_ev[0]);
    }
    auto wall_t1 = Clock::now();

    for (auto*& b : queued) {
      rt.tpuFree(b);
      b = nullptr;
    }

    print_bench_json("1x64x32", "async_enqueue_only", dtype, kWarmup, kIters,
                     compute_stats(enqueue_us), flops, ir_fmt, opt_fmt);

    double enqueue_total_us =
        std::chrono::duration<double, std::micro>(enqueue_end - wall_t0).count();
    double pipelined_total_us =
        std::chrono::duration<double, std::micro>(wall_t1 - wall_t0).count();
    std::cout << std::fixed << std::setprecision(2)
              << "CPP_PJRT_PIPELINED: {\"shape\": \"1x64x32\", \"iters\": "
              << kIters << ", \"enqueue_total_us\": " << enqueue_total_us
              << ", \"pipelined_total_us\": " << pipelined_total_us
              << ", \"enqueue_per_iter_us\": " << (enqueue_total_us / kIters)
              << ", \"pipelined_amortized_us\": "
              << (pipelined_total_us / kIters) << "}" << std::endl;
  }

  rt.tpuFree(buf_a);
  rt.tpuFree(buf_b);
  rt.tpuFreeExec(exec);
}

void bench_large_gemm(TpuRuntime& rt, const std::string& ir_dir, int64_t M,
                      int64_t K, int64_t N, const std::string& tag,
                      const std::string& dtype,
                      const std::vector<CompileOptionCandidate>& opts_list) {
  std::vector<uint16_t> host_a(static_cast<size_t>(M) * K);
  std::vector<uint16_t> host_b(static_cast<size_t>(K) * N);
  fill_bf16_pattern(host_a, 11);
  fill_bf16_pattern(host_b, 37);

  std::string ir_fmt, opt_fmt;
  TpuExecutable exec =
      compile_shape_program(rt, ir_dir, tag, opts_list, &ir_fmt, &opt_fmt);
  PJRT_Buffer* buf_a = rt.tpuCopyH2D(host_a.data(), PJRT_Buffer_Type_BF16, {M, K});
  PJRT_Buffer* buf_b = rt.tpuCopyH2D(host_b.data(), PJRT_Buffer_Type_BF16, {K, N});
  // Release host vectors promptly before timed loop.
  {
    std::vector<uint16_t>().swap(host_a);
    std::vector<uint16_t>().swap(host_b);
  }

  PJRT_Buffer* inputs[2] = {buf_a, buf_b};
  std::vector<PJRT_Buffer*> outputs(exec.num_outputs, nullptr);

  constexpr int kWarmup = 5;
  constexpr int kIters = 20;
  using Clock = std::chrono::steady_clock;

  auto sync_fn = [&](PJRT_Buffer* out_buf, PJRT_Event* dev_ev) {
    if (rt.hasReadyEvent()) {
      rt.tpuSyncBuffer(out_buf);
    } else {
      rt.tpuSyncEvent(dev_ev);
    }
  };

  for (int i = 0; i < kWarmup; ++i) {
    PJRT_Event* dev_ev[1] = {nullptr};
    rt.tpuLaunch(exec, inputs, 2, outputs.data(),
                 rt.hasReadyEvent() ? nullptr : dev_ev);
    sync_fn(outputs[0], dev_ev[0]);
    for (auto*& b : outputs) {
      rt.tpuFree(b);
      b = nullptr;
    }
  }

  std::vector<double> lat_us;
  lat_us.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    PJRT_Event* dev_ev[1] = {nullptr};
    auto t0 = Clock::now();
    rt.tpuLaunch(exec, inputs, 2, outputs.data(),
                 rt.hasReadyEvent() ? nullptr : dev_ev);
    sync_fn(outputs[0], dev_ev[0]);
    auto t1 = Clock::now();
    lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    for (auto*& b : outputs) {
      rt.tpuFree(b);
      b = nullptr;
    }
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) *
                 static_cast<double>(N);
  std::string shape_str =
      std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
  print_bench_json(shape_str, "sync_ready_event", dtype, kWarmup, kIters,
                   compute_stats(lat_us), flops, ir_fmt, opt_fmt);

  rt.tpuFree(buf_a);
  rt.tpuFree(buf_b);
  rt.tpuFreeExec(exec);
}

std::string resolve_libtpu_path() {
  const char* env_lib = std::getenv("LIBTPU_PATH");
  if (env_lib != nullptr && env_lib[0] != '\0') {
    return env_lib;
  }
#ifdef FREETPU_DEFAULT_LIBTPU_PATH
  return FREETPU_DEFAULT_LIBTPU_PATH;
#else
  return "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so";
#endif
}

std::string resolve_default_ir_dir() {
  const char* env_ir = std::getenv("TPU_PJRT_IR_DIR");
  if (env_ir != nullptr && env_ir[0] != '\0') {
    return env_ir;
  }
  const char* home = std::getenv("HOME");
  if (home != nullptr && home[0] != '\0') {
    return std::string(home) + "/.cache/freetpu-pjrt/ir";
  }
  throw std::runtime_error("Set TPU_PJRT_IR_DIR or HOME to locate PJRT IR artifacts");
}

}  // namespace

int main() {
  try {
    const std::string ir_dir = resolve_default_ir_dir();
    const std::string libtpu_path = resolve_libtpu_path();

    TpuRuntime rt;
    rt.tpuInit(libtpu_path);

    std::vector<CompileOptionCandidate> opts_list =
        load_compile_option_candidates(ir_dir);

    verify_and_bench_small(rt, ir_dir, opts_list);
    bench_large_gemm(rt, ir_dir, 1024, 1024, 1024, "1024x1024x1024_bf16_f32",
                     "bfloat16->float32", opts_list);
    bench_large_gemm(rt, ir_dir, 4096, 4096, 4096, "4096x4096x4096_bf16_bf16",
                     "bfloat16->bfloat16", opts_list);
    bench_large_gemm(rt, ir_dir, 8192, 8192, 8192, "8192x8192x8192_bf16_bf16",
                     "bfloat16->bfloat16", opts_list);

    std::vector<std::string> py_hits;
    size_t libpython_count = count_libpython_in_maps(&py_hits);
    std::cout << "MAPS_LIBPYTHON=" << libpython_count << std::endl;
    for (const auto& h : py_hits) {
      std::cout << "MAPS_LIBPYTHON_HIT: " << h << std::endl;
    }
    if (libpython_count != 0) {
      throw std::runtime_error("Expected 0 libpython mappings in /proc/self/maps");
    }

    rt.tpuShutdown();
    std::cout << "=== CPP_PJRT_COMPLETE ===" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CPP_PJRT_ERROR: " << e.what() << std::endl;
    return 1;
  }
}
