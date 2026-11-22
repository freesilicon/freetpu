// SPDX-License-Identifier: Apache-2.0
// Clean-room PJRT C API v0.116 provider exported from libopentpu.so (GetPjrtApi).
// Zero Google libtpu.so dependencies.

#include "opentpu.h"
#include "pjrt_c_api.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct PJRT_Event {
  bool ready = true;
};

struct OtPjrtMemory {
  PJRT_Memory base{};
  int id = 0;
  std::string kind = "device";
  std::string debug_str = "TPU_HBM:0";
  PJRT_Device* device = nullptr;
};

struct PJRT_DeviceDescription {
  int id = 0;
  int process_index = 0;
  std::string kind = "TPU v5 lite";
  std::string debug_str = "OpenTPU:0";
};

struct PJRT_Device {
  PJRT_DeviceDescription desc{};
  OtPjrtMemory mem{};
  ot_context_t* ctx = nullptr;
  PJRT_Memory* mem_ptrs[1] = {&mem.base};
};

struct PJRT_Client {
  ot_context_t* ctx = nullptr;
  PJRT_Device device{};
  PJRT_Device* dev_ptrs[1] = {&device};
  PJRT_Memory* mem_ptrs[1] = {&device.mem.base};
  std::string platform_name = "tpu";
  std::string platform_version = "OpenTPU 0.1.0";
};

struct PJRT_Buffer {
  ot_context_t* ctx = nullptr;
  PJRT_Device* device = nullptr;
  ot_buffer_t* ot_buf = nullptr;
  PJRT_Buffer_Type dtype = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  std::vector<int64_t> minor_to_major;
  bool deleted = false;
};

struct PJRT_Executable {
  std::string name = "opentpu_matmul";
  uint32_t m_dim = 0;
  uint32_t k_dim = 0;
  uint32_t n_dim = 0;
  std::vector<int64_t> out_dims;
  PJRT_Buffer_Type out_type = PJRT_Buffer_Type_BF16;
  const char* out_mem_kind = "device";
  size_t out_mem_kind_size = 6;
  size_t out_dim_size = 2;
};

struct PJRT_LoadedExecutable {
  PJRT_Client* client = nullptr;
  PJRT_Executable exec{};
  bool deleted = false;
};

namespace {

struct OtPjrtError {
  PJRT_Error base{};
  PJRT_Error_Code code = PJRT_Error_Code_INTERNAL;
  std::string msg;
};

void err_vtable_destroy(PJRT_Error* e) {
  delete reinterpret_cast<OtPjrtError*>(e);
}

void err_vtable_message(const PJRT_Error* e, const char** msg, size_t* len) {
  const auto* oe = reinterpret_cast<const OtPjrtError*>(e);
  *msg = oe->msg.c_str();
  *len = oe->msg.size();
}

PJRT_Error_Code err_vtable_code(const PJRT_Error* e) {
  return reinterpret_cast<const OtPjrtError*>(e)->code;
}

void err_vtable_payload(const PJRT_Error*, PJRT_Error_PayloadVisitor, void*) {}

const PJRT_Error_FunctionTable kErrVtable = {
    PJRT_Error_FunctionTable_STRUCT_SIZE,
    sizeof(OtPjrtError),
    nullptr,
    err_vtable_destroy,
    err_vtable_message,
    err_vtable_code,
    err_vtable_payload,
};

PJRT_Error* make_err(PJRT_Error_Code code, const std::string& msg) {
  auto* e = new (std::nothrow) OtPjrtError();
  if (!e) return nullptr;
  e->base.vtable = &kErrVtable;
  e->code = code;
  e->msg = msg;
  return &e->base;
}

size_t dtype_bytes(PJRT_Buffer_Type t) {
  switch (t) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
    case PJRT_Buffer_Type_F8E5M2:
    case PJRT_Buffer_Type_F8E4M3FN:
      return 1;
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_S64:
    case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64:
      return 8;
    default:
      return 0;
  }
}

// Parse simple textual StableHLO dot_general:
//   %arg0: tensor<MxKxbf16>, %arg1: tensor<KxNxbf16> ... dot_general ...
bool parse_stablehlo_matmul(const char* code, size_t len, uint32_t* m_out,
                            uint32_t* k_out, uint32_t* n_out) {
  if (!code || len == 0) return false;
  std::string s(code, len);
  if (s.find("dot_general") == std::string::npos && s.find("dot") == std::string::npos) {
    return false;
  }
  // Extract first two tensor<AxBxbf16> shapes
  std::vector<std::pair<uint32_t, uint32_t>> shapes;
  size_t pos = 0;
  while (shapes.size() < 2) {
    size_t t = s.find("tensor<", pos);
    if (t == std::string::npos) break;
    size_t gt = s.find('>', t + 7);
    if (gt == std::string::npos) break;
    std::string inner = s.substr(t + 7, gt - (t + 7));
    pos = gt + 1;
    if (inner.find("bf16") == std::string::npos) continue;
    size_t x1 = inner.find('x');
    if (x1 == std::string::npos) continue;
    size_t x2 = inner.find('x', x1 + 1);
    if (x2 == std::string::npos) continue;
    uint32_t d0 = static_cast<uint32_t>(std::strtoul(inner.substr(0, x1).c_str(), nullptr, 10));
    uint32_t d1 = static_cast<uint32_t>(
        std::strtoul(inner.substr(x1 + 1, x2 - x1 - 1).c_str(), nullptr, 10));
    if (d0 > 0 && d1 > 0) shapes.push_back({d0, d1});
  }
  if (shapes.size() < 2) return false;
  if (shapes[0].second != shapes[1].first) return false;
  *m_out = shapes[0].first;
  *k_out = shapes[0].second;
  *n_out = shapes[1].second;
  return true;
}

void ot_pjrt_error_destroy(PJRT_Error_Destroy_Args* args) {
  if (args && args->error && args->error->vtable && args->error->vtable->destroy) {
    args->error->vtable->destroy(args->error);
  }
}

void ot_pjrt_error_message(PJRT_Error_Message_Args* args) {
  if (!args || !args->error) return;
  args->error->vtable->message(args->error, &args->message, &args->message_size);
}

PJRT_Error* ot_pjrt_error_get_code(PJRT_Error_GetCode_Args* args) {
  if (!args || !args->error) return nullptr;
  args->code = args->error->vtable->get_code(args->error);
  return nullptr;
}

PJRT_Error* ot_pjrt_plugin_initialize(PJRT_Plugin_Initialize_Args*) { return nullptr; }

PJRT_Error* ot_pjrt_plugin_attributes(PJRT_Plugin_Attributes_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_event_destroy(PJRT_Event_Destroy_Args* args) {
  if (args) delete args->event;
  return nullptr;
}

PJRT_Error* ot_pjrt_event_is_ready(PJRT_Event_IsReady_Args* args) {
  if (!args || !args->event) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
  args->is_ready = args->event->ready;
  return nullptr;
}

PJRT_Error* ot_pjrt_event_error(PJRT_Event_Error_Args*) { return nullptr; }
PJRT_Error* ot_pjrt_event_await(PJRT_Event_Await_Args*) { return nullptr; }

PJRT_Error* ot_pjrt_event_on_ready(PJRT_Event_OnReady_Args* args) {
  if (!args || !args->callback) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null callback");
  args->callback(nullptr, args->user_arg);
  return nullptr;
}

PJRT_Error* ot_pjrt_client_create(PJRT_Client_Create_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
  ot_config_t cfg = ot_config_default();
  const char* mock_env = std::getenv("OPENTPU_MOCK");
  if (mock_env && std::strcmp(mock_env, "1") == 0) cfg.mock_mode = 1;

  ot_context_t* ctx = nullptr;
  ot_status_t st = ot_context_create(&cfg, &ctx);
  if (st != OT_OK) {
    return make_err(PJRT_Error_Code_INTERNAL,
                    std::string("ot_context_create failed: ") + ot_status_string(st));
  }

  auto* c = new (std::nothrow) PJRT_Client();
  if (!c) {
    ot_context_destroy(ctx);
    return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  }
  c->ctx = ctx;
  c->device.ctx = ctx;
  c->device.mem.device = &c->device;
  c->device.mem_ptrs[0] = &c->device.mem.base;
  c->mem_ptrs[0] = &c->device.mem.base;
  ot_device_info_t info = {};
  if (ot_device_get_info(ctx, &info) == OT_OK) {
    if (info.chip_gen == OT_CHIP_V6E) c->device.desc.kind = "TPU v6 lite";
    else c->device.desc.kind = "TPU v5 lite";
  }
  args->client = c;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_destroy(PJRT_Client_Destroy_Args* args) {
  if (!args || !args->client) return nullptr;
  ot_context_destroy(args->client->ctx);
  delete args->client;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_platform_name(PJRT_Client_PlatformName_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->platform_name = args->client->platform_name.c_str();
  args->platform_name_size = args->client->platform_name.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_client_process_index(PJRT_Client_ProcessIndex_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
  args->process_index = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_platform_version(PJRT_Client_PlatformVersion_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->platform_version = args->client->platform_version.c_str();
  args->platform_version_size = args->client->platform_version.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_client_devices(PJRT_Client_Devices_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->devices = args->client->dev_ptrs;
  args->num_devices = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_addressable_devices(PJRT_Client_AddressableDevices_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->addressable_devices = args->client->dev_ptrs;
  args->num_addressable_devices = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_lookup_device(PJRT_Client_LookupDevice_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  if (args->id != 0) return make_err(PJRT_Error_Code_NOT_FOUND, "unknown device id");
  args->device = &args->client->device;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_lookup_addr_device(PJRT_Client_LookupAddressableDevice_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  if (args->local_hardware_id != 0) {
    return make_err(PJRT_Error_Code_NOT_FOUND, "unknown local hardware id");
  }
  args->addressable_device = &args->client->device;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_addressable_memories(PJRT_Client_AddressableMemories_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->addressable_memories = args->client->mem_ptrs;
  args->num_addressable_memories = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_default_device_assignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  if (!args || !args->default_assignment || args->default_assignment_size < 1) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "invalid assignment buffer");
  }
  if (args->num_replicas != 1 || args->num_partitions != 1) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED, "single-chip only");
  }
  args->default_assignment[0] = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_compile(PJRT_Client_Compile_Args* args) {
  if (!args || !args->client || !args->program) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null compile args");
  }
  uint32_t m = 0, k = 0, n = 0;
  if (!parse_stablehlo_matmul(args->program->code, args->program->code_size, &m, &k, &n)) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "OpenTPU currently lowers textual StableHLO bf16 dot_general");
  }
  auto* le = new (std::nothrow) PJRT_LoadedExecutable();
  if (!le) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  le->client = args->client;
  le->exec.m_dim = m;
  le->exec.k_dim = k;
  le->exec.n_dim = n;
  le->exec.out_dims = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
  args->executable = le;
  return nullptr;
}

PJRT_Error* ot_pjrt_buffer_from_host_buffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
  size_t elem_sz = dtype_bytes(args->type);
  if (elem_sz == 0) return make_err(PJRT_Error_Code_UNIMPLEMENTED, "unsupported buffer dtype");
  size_t elems = 1;
  for (size_t i = 0; i < args->num_dims; ++i) {
    if (args->dims[i] < 0) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "negative dim");
    elems *= static_cast<size_t>(args->dims[i]);
  }
  size_t nbytes = elems * elem_sz;
  if (nbytes == 0) nbytes = 1;

  ot_buffer_t* ot_buf = nullptr;
  ot_status_t st = ot_buffer_alloc(args->client->ctx, nbytes, &ot_buf);
  if (st != OT_OK) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, ot_status_string(st));

  if (args->data && elems > 0) {
    st = ot_buffer_h2d(args->client->ctx, ot_buf, args->data, elems * elem_sz);
    if (st != OT_OK) {
      ot_buffer_free(ot_buf);
      return make_err(PJRT_Error_Code_INTERNAL, ot_status_string(st));
    }
  }

  auto* buf = new (std::nothrow) PJRT_Buffer();
  if (!buf) {
    ot_buffer_free(ot_buf);
    return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  }
  buf->ctx = args->client->ctx;
  buf->device = args->device ? args->device : &args->client->device;
  buf->ot_buf = ot_buf;
  buf->dtype = args->type;
  buf->dims.assign(args->dims, args->dims + args->num_dims);
  for (size_t i = args->num_dims; i > 0; --i) {
    buf->minor_to_major.push_back(static_cast<int64_t>(i - 1));
  }

  args->buffer = buf;
  args->done_with_host_buffer = new (std::nothrow) PJRT_Event{true};
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_id(PJRT_DeviceDescription_Id_Args* args) {
  if (!args || !args->device_description) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->id = args->device_description->id;
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_process_index(PJRT_DeviceDescription_ProcessIndex_Args* args) {
  if (!args || !args->device_description) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->process_index = args->device_description->process_index;
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_attributes(PJRT_DeviceDescription_Attributes_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_kind(PJRT_DeviceDescription_Kind_Args* args) {
  if (!args || !args->device_description) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->device_kind = args->device_description->kind.c_str();
  args->device_kind_size = args->device_description->kind.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_debug_string(PJRT_DeviceDescription_DebugString_Args* args) {
  if (!args || !args->device_description) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->debug_string = args->device_description->debug_str.c_str();
  args->debug_string_size = args->device_description->debug_str.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_devdesc_to_string(PJRT_DeviceDescription_ToString_Args* args) {
  if (!args || !args->device_description) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->to_string = args->device_description->debug_str.c_str();
  args->to_string_size = args->device_description->debug_str.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_device_get_description(PJRT_Device_GetDescription_Args* args) {
  if (!args || !args->device) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->device_description = &args->device->desc;
  return nullptr;
}

PJRT_Error* ot_pjrt_device_is_addressable(PJRT_Device_IsAddressable_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->is_addressable = true;
  return nullptr;
}

PJRT_Error* ot_pjrt_device_local_hw_id(PJRT_Device_LocalHardwareId_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->local_hardware_id = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_device_addr_memories(PJRT_Device_AddressableMemories_Args* args) {
  if (!args || !args->device) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->memories = args->device->mem_ptrs;
  args->num_memories = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_device_default_memory(PJRT_Device_DefaultMemory_Args* args) {
  if (!args || !args->device) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->memory = &args->device->mem.base;
  return nullptr;
}

PJRT_Error* ot_pjrt_device_memory_stats(PJRT_Device_MemoryStats_Args* args) {
  if (!args || !args->device) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  ot_device_info_t info = {};
  ot_device_get_info(args->device->ctx, &info);
  args->bytes_in_use = 0;
  args->bytes_limit = static_cast<int64_t>(info.hbm_bytes_total);
  args->bytes_limit_is_set = true;
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_id(PJRT_Memory_Id_Args* args) {
  if (!args || !args->memory) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->id = reinterpret_cast<const OtPjrtMemory*>(args->memory)->id;
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_kind(PJRT_Memory_Kind_Args* args) {
  if (!args || !args->memory) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  const auto* m = reinterpret_cast<const OtPjrtMemory*>(args->memory);
  args->kind = m->kind.c_str();
  args->kind_size = m->kind.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_kind_id(PJRT_Memory_Kind_Id_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->kind_id = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_debug_string(PJRT_Memory_DebugString_Args* args) {
  if (!args || !args->memory) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  const auto* m = reinterpret_cast<const OtPjrtMemory*>(args->memory);
  args->debug_string = m->debug_str.c_str();
  args->debug_string_size = m->debug_str.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_to_string(PJRT_Memory_ToString_Args* args) {
  if (!args || !args->memory) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  const auto* m = reinterpret_cast<const OtPjrtMemory*>(args->memory);
  args->to_string = m->debug_str.c_str();
  args->to_string_size = m->debug_str.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_memory_addr_devices(PJRT_Memory_AddressableByDevices_Args* args) {
  if (!args || !args->memory) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  auto* m = reinterpret_cast<OtPjrtMemory*>(args->memory);
  args->devices = &m->device;
  args->num_devices = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_destroy(PJRT_Executable_Destroy_Args*) { return nullptr; }

PJRT_Error* ot_pjrt_exec_name(PJRT_Executable_Name_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->executable_name = args->executable->name.c_str();
  args->executable_name_size = args->executable->name.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_num_replicas(PJRT_Executable_NumReplicas_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_replicas = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_num_partitions(PJRT_Executable_NumPartitions_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_partitions = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_num_outputs(PJRT_Executable_NumOutputs_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_outputs = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_code_bytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->size_in_bytes = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_cost_analysis(PJRT_Executable_GetCostAnalysis_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_properties = 0;
  args->properties = nullptr;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_output_mem_kinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_outputs = 1;
  args->memory_kinds = &args->executable->out_mem_kind;
  args->memory_kind_sizes = &args->executable->out_mem_kind_size;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_output_types(PJRT_Executable_OutputElementTypes_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->output_types = &args->executable->out_type;
  args->num_output_types = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_output_dims(PJRT_Executable_OutputDimensions_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->num_outputs = 1;
  args->dims = args->executable->out_dims.data();
  args->dim_sizes = &args->executable->out_dim_size;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
  if (args) delete args->executable;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_get_exec(PJRT_LoadedExecutable_GetExecutable_Args* args) {
  if (!args || !args->loaded_executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->executable = &args->loaded_executable->exec;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_addr_devices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->addressable_devices = args->executable->client->dev_ptrs;
  args->num_addressable_devices = 1;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_delete(PJRT_LoadedExecutable_Delete_Args* args) {
  if (args && args->executable) args->executable->deleted = true;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->is_deleted = args->executable->deleted;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_execute(PJRT_LoadedExecutable_Execute_Args* args) {
  if (!args || !args->executable || args->num_devices != 1 || args->num_args != 2) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "expected 1 device and 2 matmul args");
  }
  PJRT_LoadedExecutable* le = args->executable;
  PJRT_Buffer* lhs = args->argument_lists[0][0];
  PJRT_Buffer* rhs = args->argument_lists[0][1];
  if (!lhs || !rhs) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null input buffer");

  size_t out_bytes = static_cast<size_t>(le->exec.m_dim) * le->exec.n_dim * sizeof(uint16_t);
  ot_buffer_t* ot_out = nullptr;
  ot_status_t st = ot_buffer_alloc(le->client->ctx, out_bytes, &ot_out);
  if (st != OT_OK) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, ot_status_string(st));

  st = ot_matmul_bf16(le->client->ctx, lhs->ot_buf, rhs->ot_buf, ot_out,
                      le->exec.m_dim, le->exec.k_dim, le->exec.n_dim);
  if (st != OT_OK) {
    ot_buffer_free(ot_out);
    return make_err(PJRT_Error_Code_INTERNAL, ot_status_string(st));
  }

  auto* out_buf = new (std::nothrow) PJRT_Buffer();
  if (!out_buf) {
    ot_buffer_free(ot_out);
    return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  }
  out_buf->ctx = le->client->ctx;
  out_buf->device = &le->client->device;
  out_buf->ot_buf = ot_out;
  out_buf->dtype = PJRT_Buffer_Type_BF16;
  out_buf->dims = le->exec.out_dims;
  out_buf->minor_to_major = {1, 0};

  args->output_lists[0][0] = out_buf;
  if (args->device_complete_events) {
    args->device_complete_events[0] = new (std::nothrow) PJRT_Event{true};
  }
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_destroy(PJRT_Buffer_Destroy_Args* args) {
  if (!args || !args->buffer) return nullptr;
  ot_buffer_free(args->buffer->ot_buf);
  delete args->buffer;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_elem_type(PJRT_Buffer_ElementType_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->type = args->buffer->dtype;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_dims(PJRT_Buffer_Dimensions_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_unpadded_dims(PJRT_Buffer_UnpaddedDimensions_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->unpadded_dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_dyn_dims(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_mem_layout(PJRT_Buffer_GetMemoryLayout_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  std::memset(&args->layout, 0, sizeof(args->layout));
  args->layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  args->layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  args->layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  args->layout.tiled.minor_to_major = args->buffer->minor_to_major.data();
  args->layout.tiled.minor_to_major_size = args->buffer->minor_to_major.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_size(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->on_device_size_in_bytes = ot_buffer_size(args->buffer->ot_buf);
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_device(PJRT_Buffer_Device_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->device = args->buffer->device;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_memory(PJRT_Buffer_Memory_Args* args) {
  if (!args || !args->buffer || !args->buffer->device) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  }
  args->memory = &args->buffer->device->mem.base;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_delete(PJRT_Buffer_Delete_Args* args) {
  if (args && args->buffer) args->buffer->deleted = true;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_is_deleted(PJRT_Buffer_IsDeleted_Args* args) {
  if (!args || !args->buffer) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_copy_to_device(PJRT_Buffer_CopyToDevice_Args*) {
  return make_err(PJRT_Error_Code_UNIMPLEMENTED, "single device only");
}

PJRT_Error* ot_pjrt_buf_to_host(PJRT_Buffer_ToHostBuffer_Args* args) {
  if (!args || !args->src) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
  size_t need = ot_buffer_size(args->src->ot_buf);
  if (!args->dst) {
    args->dst_size = need;
    args->event = nullptr;
    return nullptr;
  }
  if (args->dst_size < need) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "dst_size smaller than buffer size");
  }
  ot_status_t st = ot_buffer_d2h(args->src->ctx, args->src->ot_buf, args->dst, need);
  if (st != OT_OK) return make_err(PJRT_Error_Code_INTERNAL, ot_status_string(st));
  args->event = new (std::nothrow) PJRT_Event{true};
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_is_on_cpu(PJRT_Buffer_IsOnCpu_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->is_on_cpu = false;
  return nullptr;
}

PJRT_Error* ot_pjrt_buf_ready_event(PJRT_Buffer_ReadyEvent_Args* args) {
  if (!args) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->event = new (std::nothrow) PJRT_Event{true};
  return nullptr;
}

PJRT_Api build_pjrt_api() {
  PJRT_Api api{};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.extension_start = nullptr;
  api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;

  api.PJRT_Error_Destroy = ot_pjrt_error_destroy;
  api.PJRT_Error_Message = ot_pjrt_error_message;
  api.PJRT_Error_GetCode = ot_pjrt_error_get_code;

  api.PJRT_Plugin_Initialize = ot_pjrt_plugin_initialize;
  api.PJRT_Plugin_Attributes = ot_pjrt_plugin_attributes;

  api.PJRT_Event_Destroy = ot_pjrt_event_destroy;
  api.PJRT_Event_IsReady = ot_pjrt_event_is_ready;
  api.PJRT_Event_Error = ot_pjrt_event_error;
  api.PJRT_Event_Await = ot_pjrt_event_await;
  api.PJRT_Event_OnReady = ot_pjrt_event_on_ready;

  api.PJRT_Client_Create = ot_pjrt_client_create;
  api.PJRT_Client_Destroy = ot_pjrt_client_destroy;
  api.PJRT_Client_PlatformName = ot_pjrt_client_platform_name;
  api.PJRT_Client_ProcessIndex = ot_pjrt_client_process_index;
  api.PJRT_Client_PlatformVersion = ot_pjrt_client_platform_version;
  api.PJRT_Client_Devices = ot_pjrt_client_devices;
  api.PJRT_Client_AddressableDevices = ot_pjrt_client_addressable_devices;
  api.PJRT_Client_LookupDevice = ot_pjrt_client_lookup_device;
  api.PJRT_Client_LookupAddressableDevice = ot_pjrt_client_lookup_addr_device;
  api.PJRT_Client_AddressableMemories = ot_pjrt_client_addressable_memories;
  api.PJRT_Client_Compile = ot_pjrt_client_compile;
  api.PJRT_Client_DefaultDeviceAssignment = ot_pjrt_client_default_device_assignment;
  api.PJRT_Client_BufferFromHostBuffer = ot_pjrt_buffer_from_host_buffer;

  api.PJRT_DeviceDescription_Id = ot_pjrt_devdesc_id;
  api.PJRT_DeviceDescription_ProcessIndex = ot_pjrt_devdesc_process_index;
  api.PJRT_DeviceDescription_Attributes = ot_pjrt_devdesc_attributes;
  api.PJRT_DeviceDescription_Kind = ot_pjrt_devdesc_kind;
  api.PJRT_DeviceDescription_DebugString = ot_pjrt_devdesc_debug_string;
  api.PJRT_DeviceDescription_ToString = ot_pjrt_devdesc_to_string;

  api.PJRT_Device_GetDescription = ot_pjrt_device_get_description;
  api.PJRT_Device_IsAddressable = ot_pjrt_device_is_addressable;
  api.PJRT_Device_LocalHardwareId = ot_pjrt_device_local_hw_id;
  api.PJRT_Device_AddressableMemories = ot_pjrt_device_addr_memories;
  api.PJRT_Device_DefaultMemory = ot_pjrt_device_default_memory;
  api.PJRT_Device_MemoryStats = ot_pjrt_device_memory_stats;

  api.PJRT_Memory_Id = ot_pjrt_memory_id;
  api.PJRT_Memory_Kind = ot_pjrt_memory_kind;
  api.PJRT_Memory_Kind_Id = ot_pjrt_memory_kind_id;
  api.PJRT_Memory_DebugString = ot_pjrt_memory_debug_string;
  api.PJRT_Memory_ToString = ot_pjrt_memory_to_string;
  api.PJRT_Memory_AddressableByDevices = ot_pjrt_memory_addr_devices;

  api.PJRT_Executable_Destroy = ot_pjrt_exec_destroy;
  api.PJRT_Executable_Name = ot_pjrt_exec_name;
  api.PJRT_Executable_NumReplicas = ot_pjrt_exec_num_replicas;
  api.PJRT_Executable_NumPartitions = ot_pjrt_exec_num_partitions;
  api.PJRT_Executable_NumOutputs = ot_pjrt_exec_num_outputs;
  api.PJRT_Executable_SizeOfGeneratedCodeInBytes = ot_pjrt_exec_code_bytes;
  api.PJRT_Executable_GetCostAnalysis = ot_pjrt_exec_cost_analysis;
  api.PJRT_Executable_OutputMemoryKinds = ot_pjrt_exec_output_mem_kinds;
  api.PJRT_Executable_OutputElementTypes = ot_pjrt_exec_output_types;
  api.PJRT_Executable_OutputDimensions = ot_pjrt_exec_output_dims;

  api.PJRT_LoadedExecutable_Destroy = ot_pjrt_lexec_destroy;
  api.PJRT_LoadedExecutable_GetExecutable = ot_pjrt_lexec_get_exec;
  api.PJRT_LoadedExecutable_AddressableDevices = ot_pjrt_lexec_addr_devices;
  api.PJRT_LoadedExecutable_Delete = ot_pjrt_lexec_delete;
  api.PJRT_LoadedExecutable_IsDeleted = ot_pjrt_lexec_is_deleted;
  api.PJRT_LoadedExecutable_Execute = ot_pjrt_lexec_execute;

  api.PJRT_Buffer_Destroy = ot_pjrt_buf_destroy;
  api.PJRT_Buffer_ElementType = ot_pjrt_buf_elem_type;
  api.PJRT_Buffer_Dimensions = ot_pjrt_buf_dims;
  api.PJRT_Buffer_UnpaddedDimensions = ot_pjrt_buf_unpadded_dims;
  api.PJRT_Buffer_DynamicDimensionIndices = ot_pjrt_buf_dyn_dims;
  api.PJRT_Buffer_GetMemoryLayout = ot_pjrt_buf_mem_layout;
  api.PJRT_Buffer_OnDeviceSizeInBytes = ot_pjrt_buf_size;
  api.PJRT_Buffer_Device = ot_pjrt_buf_device;
  api.PJRT_Buffer_Memory = ot_pjrt_buf_memory;
  api.PJRT_Buffer_Delete = ot_pjrt_buf_delete;
  api.PJRT_Buffer_IsDeleted = ot_pjrt_buf_is_deleted;
  api.PJRT_Buffer_CopyToDevice = ot_pjrt_buf_copy_to_device;
  api.PJRT_Buffer_ToHostBuffer = ot_pjrt_buf_to_host;
  api.PJRT_Buffer_IsOnCpu = ot_pjrt_buf_is_on_cpu;
  api.PJRT_Buffer_ReadyEvent = ot_pjrt_buf_ready_event;
  return api;
}

const PJRT_Api kPjrtApi = build_pjrt_api();

}  // namespace

extern "C" {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const PJRT_Api* GetPjrtApi(void) {
  return &kPjrtApi;
}

}  // extern "C"
