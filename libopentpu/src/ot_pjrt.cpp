// SPDX-License-Identifier: Apache-2.0
// Clean-room PJRT C API v0.116 provider exported from libopentpu.so (GetPjrtApi).
// Zero Google libtpu.so dependencies.

#include "opentpu.h"
#include "ot_internal.h"
#include "pjrt_c_api.h"

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
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

struct PJRT_TopologyDescription {
  std::string platform_name = "tpu";
  std::string platform_version = "OpenTPU 0.1.0";
  PJRT_DeviceDescription* desc_ptrs[1] = {nullptr};
  bool owned_by_client = false;
};

struct PJRT_SerializedTopology {
  std::string data;
};

struct PJRT_DeviceAssignmentSerialized {
  std::string data;
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
  PJRT_TopologyDescription topology{};
  std::string platform_name = "tpu";
  std::string platform_version = "OpenTPU 0.1.0";
};

struct PJRT_Buffer {
  ot_context_t* ctx = nullptr;
  PJRT_Device* device = nullptr;
  ot_buffer_t* ot_buf = nullptr;
  size_t logical_bytes = 0;
  PJRT_Buffer_Type dtype = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  std::vector<int64_t> minor_to_major;
  bool deleted = false;
};

struct PJRT_Executable {
  std::string name = "opentpu_matmul";
  std::string fingerprint;
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

std::string trim_ws(const std::string& in) {
  size_t b = 0;
  while (b < in.size() && std::isspace(static_cast<unsigned char>(in[b]))) ++b;
  size_t e = in.size();
  while (e > b && std::isspace(static_cast<unsigned char>(in[e - 1]))) --e;
  return in.substr(b, e - b);
}

bool parse_strict_u32(const std::string& s, uint32_t* out) {
  if (s.empty() || !out) return false;
  uint64_t v = 0;
  for (char ch : s) {
    if (ch < '0' || ch > '9') return false;
    v = v * 10u + static_cast<uint64_t>(ch - '0');
    if (v > std::numeric_limits<uint32_t>::max()) return false;
  }
  if (v == 0) return false;
  *out = static_cast<uint32_t>(v);
  return true;
}

bool parse_bf16_2d_tensor_type(const std::string& raw, uint32_t* d0, uint32_t* d1) {
  std::string s = trim_ws(raw);
  if (s.size() < 12 || s.compare(0, 7, "tensor<") != 0 || s.back() != '>') return false;
  std::string inner = s.substr(7, s.size() - 8);
  size_t x1 = inner.find('x');
  if (x1 == std::string::npos) return false;
  size_t x2 = inner.find('x', x1 + 1);
  if (x2 == std::string::npos) return false;
  if (inner.substr(x2 + 1) != "bf16") return false;
  return parse_strict_u32(inner.substr(0, x1), d0) &&
         parse_strict_u32(inner.substr(x1 + 1, x2 - x1 - 1), d1);
}

bool is_valid_ssa_name(const std::string& s) {
  if (s.size() < 2 || s[0] != '%') return false;
  for (size_t i = 1; i < s.size(); ++i) {
    unsigned char ch = static_cast<unsigned char>(s[i]);
    if (!std::isalnum(ch) && ch != '_' && ch != '.' && ch != '$') return false;
  }
  return true;
}

std::string strip_line_comments(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '/') {
      while (i < s.size() && s[i] != '\n') ++i;
      continue;
    }
    out.push_back(s[i++]);
  }
  return out;
}

// Strictly parse a single-op textual StableHLO BF16 2D matmul program:
//   module ... {
//     func.func [public] @main(%a: tensor<MxKxbf16>, %b: tensor<KxNxbf16>) -> tensor<MxNxbf16> {
//       %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<MxKxbf16>, tensor<KxNxbf16>) -> tensor<MxNxbf16>
//       return %0 : tensor<MxNxbf16>
//     }
//   }
// Rejects any extra operations, post-ops (e.g. negate/add), batching dims,
// mismatched operand order/shapes, or unsupported attributes.
bool parse_stablehlo_matmul(const char* code, size_t len, uint32_t* m_out,
                            uint32_t* k_out, uint32_t* n_out) {
  if (!code || len == 0 || !m_out || !k_out || !n_out) return false;
  std::string cleaned = strip_line_comments(std::string(code, len));

  // Locate the single func.func definition and ensure there isn't a second function.
  size_t fpos = cleaned.find("func.func");
  if (fpos == std::string::npos) return false;
  if (cleaned.find("func.func", fpos + 9) != std::string::npos) return false;

  size_t lparen = cleaned.find('(', fpos);
  if (lparen == std::string::npos) return false;
  size_t rparen = cleaned.find(')', lparen + 1);
  if (rparen == std::string::npos) return false;

  // Parse exactly two arguments: `%arg0: tensor<MxKxbf16>, %arg1: tensor<KxNxbf16>`.
  std::string arg_list = trim_ws(cleaned.substr(lparen + 1, rparen - lparen - 1));
  size_t comma = arg_list.find(',');
  if (comma == std::string::npos || arg_list.find(',', comma + 1) != std::string::npos) {
    return false;
  }
  auto parse_arg = [](const std::string& raw, std::string* ssa, uint32_t* d0, uint32_t* d1) -> bool {
    size_t colon = raw.find(':');
    if (colon == std::string::npos) return false;
    *ssa = trim_ws(raw.substr(0, colon));
    if (!is_valid_ssa_name(*ssa)) return false;
    return parse_bf16_2d_tensor_type(raw.substr(colon + 1), d0, d1);
  };
  std::string arg0_ssa, arg1_ssa;
  uint32_t m0 = 0, k0 = 0, k1 = 0, n1 = 0;
  if (!parse_arg(arg_list.substr(0, comma), &arg0_ssa, &m0, &k0)) return false;
  if (!parse_arg(arg_list.substr(comma + 1), &arg1_ssa, &k1, &n1)) return false;
  if (arg0_ssa == arg1_ssa || k0 != k1) return false;

  // Parse return type before `{`.
  size_t lbrace = cleaned.find('{', rparen + 1);
  if (lbrace == std::string::npos) return false;
  std::string ret_sig = trim_ws(cleaned.substr(rparen + 1, lbrace - rparen - 1));
  if (ret_sig.compare(0, 2, "->") != 0) return false;
  std::string ret_type_str = trim_ws(ret_sig.substr(2));
  if (!ret_type_str.empty() && ret_type_str.front() == '(' && ret_type_str.back() == ')') {
    ret_type_str = trim_ws(ret_type_str.substr(1, ret_type_str.size() - 2));
  }
  uint32_t rm = 0, rn = 0;
  if (!parse_bf16_2d_tensor_type(ret_type_str, &rm, &rn)) return false;
  if (rm != m0 || rn != n1) return false;

  // Extract matching function body `{ ... }`.
  int depth = 1;
  size_t rbrace = lbrace + 1;
  while (rbrace < cleaned.size() && depth > 0) {
    if (cleaned[rbrace] == '{') ++depth;
    else if (cleaned[rbrace] == '}') --depth;
    if (depth > 0) ++rbrace;
  }
  if (depth != 0) return false;

  // Collect non-empty body lines; must be exactly 2 statements (dot + return).
  std::vector<std::string> stmts;
  size_t cur = lbrace + 1;
  while (cur < rbrace) {
    size_t nl = cleaned.find('\n', cur);
    if (nl == std::string::npos || nl > rbrace) nl = rbrace;
    std::string line = trim_ws(cleaned.substr(cur, nl - cur));
    if (!line.empty()) stmts.push_back(line);
    cur = nl + 1;
  }
  if (stmts.size() != 2) return false;

  // Statement 1: `%res = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (T0, T1) -> T2`
  const std::string& s0 = stmts[0];
  size_t eq = s0.find('=');
  if (eq == std::string::npos) return false;
  std::string res_ssa = trim_ws(s0.substr(0, eq));
  if (!is_valid_ssa_name(res_ssa)) return false;

  std::string rhs = trim_ws(s0.substr(eq + 1));
  bool is_dot_general = false;
  size_t op_len = 0;
  if (rhs.compare(0, 21, "stablehlo.dot_general") == 0 &&
      rhs.size() > 21 && std::isspace(static_cast<unsigned char>(rhs[21]))) {
    is_dot_general = true;
    op_len = 21;
  } else if (rhs.compare(0, 13, "stablehlo.dot") == 0 &&
             rhs.size() > 13 && std::isspace(static_cast<unsigned char>(rhs[13]))) {
    is_dot_general = false;
    op_len = 13;
  } else {
    return false;
  }

  size_t type_colon = rhs.find(':', op_len);
  if (type_colon == std::string::npos) return false;
  std::string body_part = trim_ws(rhs.substr(op_len, type_colon - op_len));
  std::string type_part = trim_ws(rhs.substr(type_colon + 1));

  // Parse operands: `%a, %b` (followed by `, ...` for dot_general).
  size_t c1 = body_part.find(',');
  if (c1 == std::string::npos) return false;
  std::string op_lhs = trim_ws(body_part.substr(0, c1));
  std::string after_lhs = trim_ws(body_part.substr(c1 + 1));
  size_t c2 = after_lhs.find(',');
  std::string op_rhs = trim_ws(c2 == std::string::npos ? after_lhs : after_lhs.substr(0, c2));
  std::string attrs = c2 == std::string::npos ? std::string() : trim_ws(after_lhs.substr(c2 + 1));
  if (op_lhs != arg0_ssa || op_rhs != arg1_ssa) return false;

  if (is_dot_general) {
    // Reject batching_dims or unexpected attributes; require contracting_dims = [1] x [0].
    std::string compact_attrs;
    for (char ch : attrs) {
      if (!std::isspace(static_cast<unsigned char>(ch))) compact_attrs.push_back(ch);
    }
    if (compact_attrs != "contracting_dims=[1]x[0]" &&
        compact_attrs != "batching_dims=[]x[],contracting_dims=[1]x[0]" &&
        compact_attrs != "contracting_dims=[1]x[0],batching_dims=[]x[]") {
      return false;
    }
  } else {
    if (!attrs.empty()) return false;
  }

  // Validate op type signature: `(tensor<MxKxbf16>, tensor<KxNxbf16>) -> tensor<MxNxbf16>`.
  size_t arrow = type_part.find("->");
  if (arrow == std::string::npos) return false;
  std::string in_types = trim_ws(type_part.substr(0, arrow));
  std::string out_type = trim_ws(type_part.substr(arrow + 2));
  if (in_types.size() < 2 || in_types.front() != '(' || in_types.back() != ')') return false;
  in_types = trim_ws(in_types.substr(1, in_types.size() - 2));
  size_t tc = in_types.find(',');
  if (tc == std::string::npos || in_types.find(',', tc + 1) != std::string::npos) return false;
  uint32_t om0 = 0, ok0 = 0, ok1 = 0, on1 = 0, omm = 0, omn = 0;
  if (!parse_bf16_2d_tensor_type(in_types.substr(0, tc), &om0, &ok0)) return false;
  if (!parse_bf16_2d_tensor_type(in_types.substr(tc + 1), &ok1, &on1)) return false;
  if (!parse_bf16_2d_tensor_type(out_type, &omm, &omn)) return false;
  if (om0 != m0 || ok0 != k0 || ok1 != k0 || on1 != n1 || omm != m0 || omn != n1) return false;

  // Statement 2: `return %res : tensor<MxNxbf16>` or `func.return %res : tensor<MxNxbf16>`.
  const std::string& s1 = stmts[1];
  std::string ret_rest;
  if (s1.compare(0, 7, "return ") == 0) {
    ret_rest = trim_ws(s1.substr(7));
  } else if (s1.compare(0, 12, "func.return ") == 0) {
    ret_rest = trim_ws(s1.substr(12));
  } else {
    return false;
  }
  size_t rcolon = ret_rest.find(':');
  if (rcolon == std::string::npos) return false;
  std::string ret_ssa = trim_ws(ret_rest.substr(0, rcolon));
  if (ret_ssa != res_ssa) return false;
  uint32_t fm = 0, fn = 0;
  if (!parse_bf16_2d_tensor_type(ret_rest.substr(rcolon + 1), &fm, &fn)) return false;
  if (fm != m0 || fn != n1) return false;

  // Guard against size_t overflow in BF16 buffer byte calculations.
  auto fits_bf16 = [](uint32_t a, uint32_t b) -> bool {
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return prod <= std::numeric_limits<size_t>::max() / sizeof(uint16_t);
  };
  if (!fits_bf16(m0, k0) || !fits_bf16(k0, n1) || !fits_bf16(m0, n1)) return false;

  *m_out = m0;
  *k_out = k0;
  *n_out = n1;
  return true;
}

bool is_dense_row_major_strides(const int64_t* dims, size_t num_dims,
                                const int64_t* byte_strides, size_t elem_sz) {
  int64_t expected = static_cast<int64_t>(elem_sz);
  for (size_t i = num_dims; i > 0; --i) {
    size_t d = i - 1;
    if (dims[d] > 1 && byte_strides[d] != expected) return false;
    if (dims[d] > 0) {
      if (expected > std::numeric_limits<int64_t>::max() / dims[d]) return false;
      expected *= dims[d];
    }
  }
  return true;
}

bool is_supported_dense_layout(const PJRT_Buffer_MemoryLayout* layout,
                               const int64_t* dims, size_t num_dims,
                               size_t elem_sz) {
  if (!layout) return true;
  if (layout->type == PJRT_Buffer_MemoryLayout_Type_Tiled) {
    if (layout->tiled.num_tiles != 0) return false;
    if (layout->tiled.minor_to_major_size != num_dims) return false;
    for (size_t i = 0; i < num_dims; ++i) {
      int64_t expected_dim = static_cast<int64_t>(num_dims - 1 - i);
      if (layout->tiled.minor_to_major[i] != expected_dim) return false;
    }
    return true;
  }
  if (layout->type == PJRT_Buffer_MemoryLayout_Type_Strides) {
    if (layout->strides.num_byte_strides != num_dims) return false;
    if (num_dims > 0 && !layout->strides.byte_strides) return false;
    return is_dense_row_major_strides(dims, num_dims, layout->strides.byte_strides, elem_sz);
  }
  return false;
}

bool gather_strided_host_buffer(const void* data, const int64_t* dims, size_t num_dims,
                                const int64_t* byte_strides, size_t elem_sz,
                                size_t total_elems, std::vector<uint8_t>* out) {
  out->assign(total_elems * elem_sz, 0);
  if (total_elems == 0) return true;
  const auto* base = static_cast<const uint8_t*>(data);
  if (num_dims == 0) {
    std::memcpy(out->data(), base, elem_sz);
    return true;
  }
  std::vector<int64_t> idx(num_dims, 0);
  for (size_t linear = 0; linear < total_elems; ++linear) {
    int64_t byte_off = 0;
    for (size_t d = 0; d < num_dims; ++d) {
      byte_off += idx[d] * byte_strides[d];
    }
    std::memcpy(out->data() + linear * elem_sz, base + byte_off, elem_sz);
    for (size_t i = num_dims; i > 0; --i) {
      size_t d = i - 1;
      if (++idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }
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
  c->topology.platform_name = c->platform_name;
  c->topology.platform_version = c->platform_version;
  c->topology.desc_ptrs[0] = &c->device.desc;
  c->topology.owned_by_client = true;
  args->client = c;
  return nullptr;
}

PJRT_Error* ot_pjrt_client_topology_description(PJRT_Client_TopologyDescription_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
  args->topology = &args->client->topology;
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_destroy(PJRT_TopologyDescription_Destroy_Args* args) {
  if (args && args->topology && !args->topology->owned_by_client) {
    delete args->topology;
  }
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_platform_name(PJRT_TopologyDescription_PlatformName_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  args->platform_name = args->topology->platform_name.c_str();
  args->platform_name_size = args->topology->platform_name.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_platform_version(PJRT_TopologyDescription_PlatformVersion_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  args->platform_version = args->topology->platform_version.c_str();
  args->platform_version_size = args->topology->platform_version.size();
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_get_dev_descs(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  args->descriptions = args->topology->desc_ptrs;
  args->num_descriptions = args->topology->desc_ptrs[0] ? 1 : 0;
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_attributes(PJRT_TopologyDescription_Attributes_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

void ot_pjrt_serialized_topo_deleter(PJRT_SerializedTopology* s) {
  delete s;
}

PJRT_Error* ot_pjrt_topo_serialize(PJRT_TopologyDescription_Serialize_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  auto* st = new (std::nothrow) PJRT_SerializedTopology();
  if (!st) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  const char* kind =
      args->topology->desc_ptrs[0] ? args->topology->desc_ptrs[0]->kind.c_str() : "unknown";
  st->data = args->topology->platform_name + "|" + args->topology->platform_version + "|" + kind;
  args->serialized_bytes = st->data.c_str();
  args->serialized_bytes_size = st->data.size();
  args->serialized_topology = st;
  args->serialized_topology_deleter = ot_pjrt_serialized_topo_deleter;
  return nullptr;
}

PJRT_Error* ot_pjrt_topo_fingerprint(PJRT_TopologyDescription_Fingerprint_Args* args) {
  if (!args || !args->topology) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null topology");
  const char* kind =
      args->topology->desc_ptrs[0] ? args->topology->desc_ptrs[0]->kind.c_str() : "unknown";
  std::string key =
      args->topology->platform_name + "|" + args->topology->platform_version + "|" + kind;
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : key) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ULL;
  }
  args->fingerprint = h;
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
  const char* fmt = args->program->format;
  size_t fmt_len = args->program->format_size;
  if (!fmt || fmt_len != 4 || std::memcmp(fmt, "mlir", 4) != 0) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "OpenTPU M4 supports program format 'mlir' (textual StableHLO) only");
  }
  if (args->program->code && args->program->code_size >= 4 &&
      static_cast<unsigned char>(args->program->code[0]) == 'M' &&
      static_cast<unsigned char>(args->program->code[1]) == 'L' &&
      static_cast<unsigned char>(args->program->code[2]) == 0xefu &&
      static_cast<unsigned char>(args->program->code[3]) == 'R') {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "OpenTPU M4 is zero-dependency and does not yet decode binary MLIR bytecode "
                    "(portable StableHLO artifacts); pass textual StableHLO MLIR");
  }
  uint32_t m = 0, k = 0, n = 0;
  if (!parse_stablehlo_matmul(args->program->code, args->program->code_size, &m, &k, &n)) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "OpenTPU M4 lowers single-op textual StableHLO bf16 dot_general only");
  }
  auto* le = new (std::nothrow) PJRT_LoadedExecutable();
  if (!le) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  le->client = args->client;
  le->exec.m_dim = m;
  le->exec.k_dim = k;
  le->exec.n_dim = n;
  le->exec.out_dims = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
  le->exec.fingerprint = "opentpu_bf16_matmul_" + std::to_string(m) + "x" +
                         std::to_string(k) + "x" + std::to_string(n);
  args->executable = le;
  return nullptr;
}

PJRT_Error* ot_pjrt_buffer_from_host_buffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  if (!args || !args->client) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
  if (args->num_dims > 0 && !args->dims) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null dims");
  }
  size_t elem_sz = dtype_bytes(args->type);
  if (elem_sz == 0) return make_err(PJRT_Error_Code_UNIMPLEMENTED, "unsupported buffer dtype");
  size_t elems = 1;
  for (size_t i = 0; i < args->num_dims; ++i) {
    if (args->dims[i] < 0) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "negative dim");
    size_t d = static_cast<size_t>(args->dims[i]);
    if (d > 0 && elems > std::numeric_limits<size_t>::max() / d) {
      return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "element count overflow");
    }
    elems *= d;
  }
  if (elems > std::numeric_limits<size_t>::max() / elem_sz) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "byte size overflow");
  }
  size_t logical_bytes = elems * elem_sz;

  if (args->num_byte_strides != 0) {
    if (!args->byte_strides || args->num_byte_strides != args->num_dims) {
      return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "invalid byte_strides");
    }
  }
  if (args->device_layout &&
      !is_supported_dense_layout(args->device_layout, args->dims, args->num_dims, elem_sz)) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "unsupported device_layout (dense row-major only)");
  }

  const void* upload_ptr = args->data;
  std::vector<uint8_t> packed_host;
  if (elems > 0) {
    if (!args->data) {
      return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null host data for non-empty buffer");
    }
    if (args->num_byte_strides != 0 &&
        !is_dense_row_major_strides(args->dims, args->num_dims, args->byte_strides, elem_sz)) {
      gather_strided_host_buffer(args->data, args->dims, args->num_dims, args->byte_strides,
                                 elem_sz, elems, &packed_host);
      upload_ptr = packed_host.data();
    }
  }

  size_t alloc_bytes = logical_bytes == 0 ? 1 : logical_bytes;
  ot_buffer_t* ot_buf = nullptr;
  ot_status_t st = ot_buffer_alloc(args->client->ctx, alloc_bytes, &ot_buf);
  if (st != OT_OK) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, ot_status_string(st));

  if (upload_ptr && logical_bytes > 0) {
    st = ot_buffer_h2d(args->client->ctx, ot_buf, upload_ptr, logical_bytes);
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
  buf->logical_bytes = logical_bytes;
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
  if (!args || !args->device || !args->device->ctx) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  }
  uint64_t used = 0;
  uint64_t limit = 0;
  {
    std::lock_guard<std::mutex> lk(args->device->ctx->mu);
    used = args->device->ctx->transport.hbm_alloc.used_bytes();
    limit = args->device->ctx->transport.hbm_alloc.capacity_bytes();
  }
  args->bytes_in_use = static_cast<int64_t>(used);
  args->peak_bytes_in_use = 0;
  args->peak_bytes_in_use_is_set = false;
  args->num_allocs = 0;
  args->num_allocs_is_set = false;
  args->largest_alloc_size = 0;
  args->largest_alloc_size_is_set = false;
  args->bytes_limit = static_cast<int64_t>(limit);
  args->bytes_limit_is_set = true;
  args->bytes_reserved = 0;
  args->bytes_reserved_is_set = false;
  args->peak_bytes_reserved = 0;
  args->peak_bytes_reserved_is_set = false;
  args->bytes_reservable_limit = 0;
  args->bytes_reservable_limit_is_set = false;
  args->largest_free_block_bytes = 0;
  args->largest_free_block_bytes_is_set = false;
  args->pool_bytes = 0;
  args->pool_bytes_is_set = false;
  args->peak_pool_bytes = 0;
  args->peak_pool_bytes_is_set = false;
  if (args->struct_size >=
      offsetof(PJRT_Device_MemoryStats_Args, peak_allocated_bytes_is_set) +
          sizeof(args->peak_allocated_bytes_is_set)) {
    args->peak_allocated_bytes = 0;
    args->peak_allocated_bytes_is_set = false;
  }
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

PJRT_Error* ot_pjrt_exec_destroy(PJRT_Executable_Destroy_Args* args) {
  if (args) delete args->executable;
  return nullptr;
}

PJRT_Error* ot_pjrt_exec_fingerprint(PJRT_Executable_Fingerprint_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->executable_fingerprint = args->executable->fingerprint.c_str();
  args->executable_fingerprint_size = args->executable->fingerprint.size();
  return nullptr;
}

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
  auto* copy = new (std::nothrow) PJRT_Executable(args->loaded_executable->exec);
  if (!copy) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  args->executable = copy;
  return nullptr;
}

void ot_pjrt_device_assignment_deleter(PJRT_DeviceAssignmentSerialized* da) {
  delete da;
}

PJRT_Error* ot_pjrt_lexec_get_device_assignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  auto* da = new (std::nothrow) PJRT_DeviceAssignmentSerialized();
  if (!da) return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "out of memory");
  // Valid xla::DeviceAssignmentProto wire encoding for 1 replica x 1 computation -> device 0:
  //   field 1 (replica_count = 1): 0x08 0x01
  //   field 2 (computation_count = 1): 0x10 0x01
  //   field 3 (computation_devices[0] = { field 1 (replica_device_ids = [0]) }): 0x1a 0x02 0x08 0x00
  static constexpr char kSingleDeviceAssignmentProto[8] = {
      0x08, 0x01, 0x10, 0x01, 0x1a, 0x02, 0x08, 0x00};
  da->data.assign(kSingleDeviceAssignmentProto, sizeof(kSingleDeviceAssignmentProto));
  args->serialized_bytes = da->data.c_str();
  args->serialized_bytes_size = da->data.size();
  args->serialized_device_assignment = da;
  args->serialized_device_assignment_deleter = ot_pjrt_device_assignment_deleter;
  return nullptr;
}

PJRT_Error* ot_pjrt_lexec_fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
  if (!args || !args->executable) return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null");
  args->executable_fingerprint = args->executable->exec.fingerprint.c_str();
  args->executable_fingerprint_size = args->executable->exec.fingerprint.size();
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
  if (le->deleted) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "executable has been deleted");
  }
  PJRT_Buffer* lhs = args->argument_lists[0][0];
  PJRT_Buffer* rhs = args->argument_lists[0][1];
  if (!lhs || !rhs || lhs->deleted || rhs->deleted || !lhs->ot_buf || !rhs->ot_buf) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "null or deleted input buffer");
  }
  if (lhs->ctx != le->client->ctx || rhs->ctx != le->client->ctx) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "input buffer belongs to another client");
  }
  if (lhs->dtype != PJRT_Buffer_Type_BF16 || rhs->dtype != PJRT_Buffer_Type_BF16 ||
      lhs->dims.size() != 2 || rhs->dims.size() != 2 ||
      lhs->dims[0] != static_cast<int64_t>(le->exec.m_dim) ||
      lhs->dims[1] != static_cast<int64_t>(le->exec.k_dim) ||
      rhs->dims[0] != static_cast<int64_t>(le->exec.k_dim) ||
      rhs->dims[1] != static_cast<int64_t>(le->exec.n_dim)) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "input buffer dtype or shape mismatch");
  }

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
  out_buf->logical_bytes = out_bytes;
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
  if (args->buffer->ot_buf) {
    ot_buffer_free(args->buffer->ot_buf);
    args->buffer->ot_buf = nullptr;
  }
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
  if (args->buffer->deleted || !args->buffer->ot_buf) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "buffer has been deleted");
  }
  args->on_device_size_in_bytes = args->buffer->logical_bytes;
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
  if (args && args->buffer) {
    if (args->buffer->ot_buf) {
      ot_buffer_free(args->buffer->ot_buf);
      args->buffer->ot_buf = nullptr;
    }
    args->buffer->deleted = true;
  }
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
  if (args->src->deleted || !args->src->ot_buf) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "buffer has been deleted");
  }
  size_t elem_sz = dtype_bytes(args->src->dtype);
  if (args->host_layout &&
      !is_supported_dense_layout(args->host_layout, args->src->dims.data(), args->src->dims.size(),
                                 elem_sz)) {
    return make_err(PJRT_Error_Code_UNIMPLEMENTED,
                    "unsupported host_layout (dense row-major only)");
  }
  size_t need = args->src->logical_bytes;
  if (!args->dst) {
    args->dst_size = need;
    args->event = nullptr;
    return nullptr;
  }
  if (args->dst_size < need) {
    return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "dst_size smaller than buffer size");
  }
  if (need > 0) {
    ot_status_t st = ot_buffer_d2h(args->src->ctx, args->src->ot_buf, args->dst, need);
    if (st != OT_OK) return make_err(PJRT_Error_Code_INTERNAL, ot_status_string(st));
  }
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

#define OT_PJRT_UNIMPL(fn_name, ArgsT, msg)                                       \
  PJRT_Error* fn_name(ArgsT*) {                                                   \
    return make_err(PJRT_Error_Code_UNIMPLEMENTED, msg);                          \
  }

OT_PJRT_UNIMPL(ot_pjrt_exec_optimized_program, PJRT_Executable_OptimizedProgram_Args,
               "PJRT_Executable_OptimizedProgram not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_exec_serialize, PJRT_Executable_Serialize_Args,
               "PJRT_Executable_Serialize not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_exec_deserialize_and_load, PJRT_Executable_DeserializeAndLoad_Args,
               "PJRT_Executable_DeserializeAndLoad not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_exec_compiled_mem_stats, PJRT_Executable_GetCompiledMemoryStats_Args,
               "PJRT_Executable_GetCompiledMemoryStats not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_exec_compile_options, PJRT_Executable_GetCompileOptions_Args,
               "PJRT_Executable_GetCompileOptions not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_copy_raw_to_host, PJRT_Buffer_CopyRawToHost_Args,
               "PJRT_Buffer_CopyRawToHost not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_copy_raw_to_host_future, PJRT_Buffer_CopyRawToHostFuture_Args,
               "PJRT_Buffer_CopyRawToHostFuture not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_copy_to_memory, PJRT_Buffer_CopyToMemory_Args,
               "PJRT_Buffer_CopyToMemory not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_unsafe_ptr, PJRT_Buffer_UnsafePointer_Args,
               "PJRT_Buffer_UnsafePointer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_inc_ext_ref, PJRT_Buffer_IncreaseExternalReferenceCount_Args,
               "PJRT_Buffer_IncreaseExternalReferenceCount not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_dec_ext_ref, PJRT_Buffer_DecreaseExternalReferenceCount_Args,
               "PJRT_Buffer_DecreaseExternalReferenceCount not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_opaque_ptr, PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
               "PJRT_Buffer_OpaqueDeviceMemoryDataPointer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_donate_ctrl, PJRT_Buffer_DonateWithControlDependency_Args,
               "PJRT_Buffer_DonateWithControlDependency not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_buf_bitcast, PJRT_Buffer_Bitcast_Args,
               "PJRT_Buffer_Bitcast not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_create_view, PJRT_Client_CreateViewOfDeviceBuffer_Args,
               "PJRT_Client_CreateViewOfDeviceBuffer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_create_async_h2d, PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
               "PJRT_Client_CreateBuffersForAsyncHostToDevice not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_dma_map, PJRT_Client_DmaMap_Args,
               "PJRT_Client_DmaMap not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_dma_unmap, PJRT_Client_DmaUnmap_Args,
               "PJRT_Client_DmaUnmap not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_create_uninit_buf, PJRT_Client_CreateUninitializedBuffer_Args,
               "PJRT_Client_CreateUninitializedBuffer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_update_global_proc, PJRT_Client_UpdateGlobalProcessInfo_Args,
               "PJRT_Client_UpdateGlobalProcessInfo not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_create_alias_buf, PJRT_Client_CreateAliasBuffer_Args,
               "PJRT_Client_CreateAliasBuffer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_fulfill_alias_buf, PJRT_Client_FulfillAliasBuffer_Args,
               "PJRT_Client_FulfillAliasBuffer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_create_error_buf, PJRT_Client_CreateErrorBuffer_Args,
               "PJRT_Client_CreateErrorBuffer not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_client_load, PJRT_Client_Load_Args,
               "PJRT_Client_Load not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_compile_standalone, PJRT_Compile_Args,
               "PJRT_Compile not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_topo_deserialize, PJRT_TopologyDescription_Deserialize_Args,
               "PJRT_TopologyDescription_Deserialize not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_device_poison_exec, PJRT_Device_PoisonExecution_Args,
               "PJRT_Device_PoisonExecution not implemented in libopentpu")
OT_PJRT_UNIMPL(ot_pjrt_device_clear_mem_stats, PJRT_Device_ClearMemoryStats_Args,
               "PJRT_Device_ClearMemoryStats not implemented in libopentpu")

#undef OT_PJRT_UNIMPL

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
  api.PJRT_Client_TopologyDescription = ot_pjrt_client_topology_description;
  api.PJRT_Client_CreateViewOfDeviceBuffer = ot_pjrt_client_create_view;
  api.PJRT_Client_CreateBuffersForAsyncHostToDevice = ot_pjrt_client_create_async_h2d;
  api.PJRT_Client_DmaMap = ot_pjrt_client_dma_map;
  api.PJRT_Client_DmaUnmap = ot_pjrt_client_dma_unmap;
  api.PJRT_Client_CreateUninitializedBuffer = ot_pjrt_client_create_uninit_buf;
  api.PJRT_Client_UpdateGlobalProcessInfo = ot_pjrt_client_update_global_proc;
  api.PJRT_Client_CreateAliasBuffer = ot_pjrt_client_create_alias_buf;
  api.PJRT_Client_FulfillAliasBuffer = ot_pjrt_client_fulfill_alias_buf;
  api.PJRT_Client_CreateErrorBuffer = ot_pjrt_client_create_error_buf;
  api.PJRT_Client_Load = ot_pjrt_client_load;
  api.PJRT_Compile = ot_pjrt_compile_standalone;

  api.PJRT_TopologyDescription_Destroy = ot_pjrt_topo_destroy;
  api.PJRT_TopologyDescription_PlatformName = ot_pjrt_topo_platform_name;
  api.PJRT_TopologyDescription_PlatformVersion = ot_pjrt_topo_platform_version;
  api.PJRT_TopologyDescription_GetDeviceDescriptions = ot_pjrt_topo_get_dev_descs;
  api.PJRT_TopologyDescription_Serialize = ot_pjrt_topo_serialize;
  api.PJRT_TopologyDescription_Deserialize = ot_pjrt_topo_deserialize;
  api.PJRT_TopologyDescription_Attributes = ot_pjrt_topo_attributes;
  api.PJRT_TopologyDescription_Fingerprint = ot_pjrt_topo_fingerprint;

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
  api.PJRT_Device_PoisonExecution = ot_pjrt_device_poison_exec;
  api.PJRT_Device_ClearMemoryStats = ot_pjrt_device_clear_mem_stats;

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
  api.PJRT_Executable_OptimizedProgram = ot_pjrt_exec_optimized_program;
  api.PJRT_Executable_Serialize = ot_pjrt_exec_serialize;
  api.PJRT_Executable_DeserializeAndLoad = ot_pjrt_exec_deserialize_and_load;
  api.PJRT_Executable_GetCompiledMemoryStats = ot_pjrt_exec_compiled_mem_stats;
  api.PJRT_Executable_GetCompileOptions = ot_pjrt_exec_compile_options;
  api.PJRT_Executable_Fingerprint = ot_pjrt_exec_fingerprint;

  api.PJRT_LoadedExecutable_Destroy = ot_pjrt_lexec_destroy;
  api.PJRT_LoadedExecutable_GetExecutable = ot_pjrt_lexec_get_exec;
  api.PJRT_LoadedExecutable_AddressableDevices = ot_pjrt_lexec_addr_devices;
  api.PJRT_LoadedExecutable_Delete = ot_pjrt_lexec_delete;
  api.PJRT_LoadedExecutable_IsDeleted = ot_pjrt_lexec_is_deleted;
  api.PJRT_LoadedExecutable_Execute = ot_pjrt_lexec_execute;
  api.PJRT_LoadedExecutable_GetDeviceAssignment = ot_pjrt_lexec_get_device_assignment;
  api.PJRT_LoadedExecutable_Fingerprint = ot_pjrt_lexec_fingerprint;

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
  api.PJRT_Buffer_CopyToMemory = ot_pjrt_buf_copy_to_memory;
  api.PJRT_Buffer_ToHostBuffer = ot_pjrt_buf_to_host;
  api.PJRT_Buffer_CopyRawToHost = ot_pjrt_buf_copy_raw_to_host;
  api.PJRT_Buffer_CopyRawToHostFuture = ot_pjrt_buf_copy_raw_to_host_future;
  api.PJRT_Buffer_IsOnCpu = ot_pjrt_buf_is_on_cpu;
  api.PJRT_Buffer_ReadyEvent = ot_pjrt_buf_ready_event;
  api.PJRT_Buffer_UnsafePointer = ot_pjrt_buf_unsafe_ptr;
  api.PJRT_Buffer_IncreaseExternalReferenceCount = ot_pjrt_buf_inc_ext_ref;
  api.PJRT_Buffer_DecreaseExternalReferenceCount = ot_pjrt_buf_dec_ext_ref;
  api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = ot_pjrt_buf_opaque_ptr;
  api.PJRT_Buffer_DonateWithControlDependency = ot_pjrt_buf_donate_ctrl;
  api.PJRT_Buffer_Bitcast = ot_pjrt_buf_bitcast;
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
