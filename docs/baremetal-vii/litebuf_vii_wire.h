/*
 * Clean-room Litebuf VII wire protocol definitions and inline builders/parsers
 * for Google Pixel 10 Pro (rango / Tensor G5) DarwiNN / EdgeTPU /dev/edgetpu-soc.
 *
 * Derived from static RE of libtachyon_core.so / buenos.ko and verified
 * byte-for-byte against single-process LD_PRELOAD wire captures (fc, add).
 * Compatible with C11 and C++17.
 */
#ifndef BAREMETAL_VII_LITEBUF_VII_WIRE_H_
#define BAREMETAL_VII_LITEBUF_VII_WIRE_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
#define LBVII_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#else
#define LBVII_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LBVII_FN static inline __attribute__((unused))
#else
#define LBVII_FN static inline
#endif

/* =========================================================================
 * Litebuf Container Header Constants
 * ========================================================================= */

#define LITEBUF_HEADER_SIZE            16u
#define LITEBUF_ALIGN_BYTES            16u
#define LITEBUF_LOG2_ALIGN_16          4u

/* Outer mailbox schema: silicon.darwinn.control.driver.mailbox.runtime_messages */
#define LITEBUF_SCHEMA_RUNTIME_MESSAGES 0x0008u

/* Inner DIVE schema: silicon.dive.api.runtime_commands */
#define LITEBUF_SCHEMA_DIVE_COMMANDS    0x00feu

/* Minimum DMA space reserved ahead of inner DIVE payloads in vii_buffer */
#define LITEBUF_VII_CMD_BUFFER_OFFSET   0x200u

/* Threshold in buenos.ko for inline runtime_command vs iremap large command */
#define LITEBUF_VII_INLINE_MAX_BYTES    96u

/* Response payload copied by EDGETPU_VII_LITEBUF_RESPONSE (0x30 bytes) */
#define LITEBUF_VII_RESPONSE_BYTES      48u

/* Observed default Buenos A0 / rango operational parameters */
#define LITEBUF_DEFAULT_CORE_POWER_STATE   0x0bu /* Active TPU core OPP */
#define LITEBUF_DEFAULT_MEM_POWER_STATE    0x03u /* Active memory OPP */
#define LITEBUF_DEFAULT_QOS_CLASS          0x01u
#define LITEBUF_DEFAULT_CLUSTER_ID         0x02u /* ctz(1u << 2) on rango */
#define LITEBUF_CORE_SESSION_NONE          0xffffffffu

/* =========================================================================
 * Union Discriminators (Command & Response Tags)
 * ========================================================================= */

enum litebuf_vii_command_type {
  LITEBUF_VII_CMD_NONE             = 0u,
  LITEBUF_VII_CMD_REGISTER_MODEL   = 1u,
  LITEBUF_VII_CMD_UNREGISTER_MODEL = 2u,
  LITEBUF_VII_CMD_EXECUTE          = 3u,
  LITEBUF_VII_CMD_ACQUIRE_WAKELOCK = 4u,
  LITEBUF_VII_CMD_RELEASE_WAKELOCK = 5u,
  LITEBUF_VII_CMD_CANCEL           = 6u
};

enum litebuf_vii_response_type {
  LITEBUF_VII_RESP_VOID_OK        = 0u,
  LITEBUF_VII_RESP_EXECUTE        = 1u,
  LITEBUF_VII_RESP_REGISTER_MODEL = 2u,
  LITEBUF_VII_RESP_ERROR          = 3u,
  LITEBUF_VII_RESP_CANCEL         = 4u,
  LITEBUF_VII_RESP_REPRIORITIZE   = 5u
};

enum litebuf_vii_response_error_code {
  LITEBUF_VII_ERR_NONE                    = 0u,
  LITEBUF_VII_ERR_INVALID_RUNTIME_COMMAND = 1u,
  LITEBUF_VII_ERR_INVALID_VII_SEQ_ID      = 2u,
  LITEBUF_VII_ERR_INVALID_PRIORITY        = 3u
};

enum litebuf_dive_command_type {
  LITEBUF_DIVE_CMD_NONE                       = 0u,
  LITEBUF_DIVE_CMD_REGISTER_MODEL             = 1u,
  LITEBUF_DIVE_CMD_EXECUTE                    = 2u,
  LITEBUF_DIVE_CMD_REGISTER_INTERPRETER_MODEL = 3u
};

/* =========================================================================
 * Litebuf Common 16-byte Container Header
 * ========================================================================= */

struct litebuf_header {
  uint16_t schema_id;      /* 0x0008 (outer) or 0x00fe (inner DIVE) */
  uint8_t  log2_alignment; /* 4 (16-byte alignment) */
  uint8_t  reserved0;      /* 0 */
  uint32_t reserved1;      /* 0 */
  uint64_t root_offset;    /* Offset to root table from container base (0x10) */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_header) == 16, "litebuf_header must be 16B");

/* =========================================================================
 * Outer Mailbox Layer: runtime_messages::Command & Payloads
 * OffsetType = int16_t, SizeType = uint16_t
 * ========================================================================= */

struct litebuf_vii_command_root {
  uint16_t table_size;   /* Logical table size: 7 */
  int16_t  union_offset; /* Signed byte offset from Command to payload */
  uint16_t union_type;   /* enum litebuf_vii_command_type */
  uint8_t  priority;     /* Queue priority (0 observed on Execute/Reg/Unreg) */
  uint8_t  pad0;
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_command_root) == 8,
                    "litebuf_vii_command_root must be 8B");

struct litebuf_vii_register_model {
  uint16_t table_size;             /* +0x00: Logical table size: 99 (0x63) */
  uint16_t pad0[3];                /* +0x02..+0x07 */
  uint64_t dive_cmd_address;       /* +0x08: IOVA of inner DIVE Register cmd */
  uint32_t dive_cmd_size_bytes;    /* +0x10: Byte size of inner DIVE cmd */
  uint32_t pad1;                   /* +0x14 */
  uint64_t dive_image_address;     /* +0x18: IOVA of RV64 ELF interpreter (4K aligned) */
  uint32_t dive_image_size_bytes;  /* +0x20: Exact byte size of ext[1] image */
  uint32_t pad2;                   /* +0x24 */
  uint64_t dive_scratch_address;   /* +0x28: IOVA of SCRATCHPAD tensor DMA buffer */
  uint32_t dive_scratch_size_bytes;/* +0x30: Byte size of SCRATCHPAD tensor */
  uint32_t pad3;                   /* +0x34 */
  int32_t  dive_heap_size;         /* +0x38: Executable.id_28 (e.g. 32768) */
  int32_t  exec_field_29;          /* +0x3c: Executable.id_29 (e.g. 81920) */
  uint32_t core_session_handle;    /* +0x40: 0 observed on rango single-session */
  uint8_t  cluster_id;             /* +0x44: ctz(cluster_bitset), 2 on rango */
  uint8_t  exec_field_31;          /* +0x45: Executable.id_31 (0 when absent) */
  int16_t  model_id_str_offset;    /* +0x46: Relative offset to model_id C-string */
  int16_t  model_id_str_size;      /* +0x48: Byte size incl. NUL (e.g. 8 for "Unknown") */
  uint16_t pad4;                   /* +0x4a */
  uint32_t exec_field_14;          /* +0x4c: Executable.id_14 (model-specific constant) */
  uint32_t flags0;                 /* +0x50: 0 */
  uint32_t pad5;                   /* +0x54 */
  uint64_t compiler_cl;            /* +0x58: Package.compiler_cl (or 0) */
  uint8_t  flag_60;                /* +0x60: 0 */
  uint8_t  subgraph_flag_18;       /* +0x61: SubGraph.id_18 (0 or 1) */
  uint8_t  exec_field_34;          /* +0x62: Executable.id_34 (0 when absent) */
  uint8_t  pad6[5];                /* +0x63..+0x67 */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_register_model) == 104,
                    "litebuf_vii_register_model must be 104B");

struct litebuf_vii_unregister_model {
  uint16_t table_size; /* +0x00: Logical table size: 8 */
  uint16_t pad0;       /* +0x02 */
  uint32_t model_id;   /* +0x04: Handle returned by RegisterModelResponse */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_unregister_model) == 8,
                    "litebuf_vii_unregister_model must be 8B");

struct litebuf_vii_execute {
  uint16_t table_size;            /* +0x00: Logical table size: 44 (0x2c) */
  uint16_t pad0[3];               /* +0x02..+0x07 */
  uint64_t dive_cmd_address;      /* +0x08: IOVA of inner DIVE Execute cmd */
  uint32_t dive_cmd_size_bytes;   /* +0x10: Byte size of inner DIVE Execute cmd */
  uint32_t pad1;                  /* +0x14 */
  uint32_t model_id;              /* +0x18: Handle returned by RegisterModelResponse */
  uint8_t  tpu_core_power_state;  /* +0x1c: 0x0b observed */
  uint8_t  memory_power_state;    /* +0x1d: 0x03 observed */
  uint8_t  qos_class;             /* +0x1e: 0x01 observed */
  uint8_t  perf_metrics_flag;     /* +0x1f: 0x01 observed */
  uint8_t  cluster_id;            /* +0x20: 0x02 observed on rango */
  uint8_t  pad2[7];               /* +0x21..+0x27 */
  uint32_t core_session_handle;   /* +0x28: 0xffffffff when unused */
  uint32_t pad3;                  /* +0x2c */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_execute) == 48,
                    "litebuf_vii_execute must be 48B");

struct litebuf_vii_wakelock_cmd {
  uint16_t table_size;           /* +0x00: Logical table size: 5 */
  uint8_t  tpu_core_power_state; /* +0x02 */
  uint8_t  memory_power_state;   /* +0x03 */
  uint8_t  cluster_ids_bitset;   /* +0x04 */
  uint8_t  pad0;                 /* +0x05 */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_wakelock_cmd) == 6,
                    "litebuf_vii_wakelock_cmd must be 6B");

/* =========================================================================
 * Outer Mailbox Layer: runtime_messages::ViiResponse (48B ioctl output)
 * ========================================================================= */

struct litebuf_vii_response_root {
  uint16_t table_size;   /* +0x00: 12 observed (must be >= 6) */
  int16_t  union_offset; /* +0x02: Signed byte offset to response union table */
  uint16_t union_type;   /* +0x04: enum litebuf_vii_response_type */
  uint16_t pad0;         /* +0x06 */
  uint32_t fw_cycles_or_seq; /* +0x08: Firmware-populated value (non-critical) */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_vii_response_root) == 12,
                    "litebuf_vii_response_root must be 12B");

/* Parsed high-level view of 48B ViiResponse */
struct litebuf_vii_parsed_response {
  uint16_t response_type;   /* enum litebuf_vii_response_type */
  uint32_t handle;          /* RegisterModelResponse: graph/model_id handle */
  uint64_t execution_us;    /* ExecuteResponse: hardware execution time (us) */
  uint32_t error_code;      /* ErrorResponse: enum litebuf_vii_response_error_code */
  uint32_t fw_status_word;  /* Root word +0x08 */
  int      valid;           /* 1 if structurally valid Litebuf response */
};

/* =========================================================================
 * Inner DIVE Layer: silicon::dive::api::runtime_commands
 * OffsetType = int32_t, SizeType = uint32_t
 * ========================================================================= */

struct litebuf_dive_command_root {
  uint32_t table_size;   /* +0x00: 12 (0x0c) */
  int32_t  union_offset; /* +0x04: Signed byte offset from Command to member */
  uint32_t union_type;   /* +0x08: enum litebuf_dive_command_type */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_dive_command_root) == 12,
                    "litebuf_dive_command_root must be 12B");

struct litebuf_dive_buffer_slice {
  uint64_t address;    /* +0x00: Device IOVA */
  uint64_t size_bytes; /* +0x08: Slice size in bytes */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_dive_buffer_slice) == 16,
                    "litebuf_dive_buffer_slice must be 16B");

struct litebuf_dive_register_interpreter_model {
  uint32_t table_size;               /* +0x00: 28 (0x1c) */
  uint32_t pad0;                     /* +0x04 */
  uint64_t dive_vm_ir_address;       /* +0x08: ext[0] DIVE VM IR IOVA (64B aligned) */
  uint64_t dive_vm_ir_size_bytes;    /* +0x10: ext[0] exact byte size */
  int32_t  registration_info_offset; /* +0x18: Offset from this table to RegistrationInfo */
  uint32_t pad1;                     /* +0x1c */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_dive_register_interpreter_model) == 32,
                    "litebuf_dive_register_interpreter_model must be 32B");

struct litebuf_dive_registration_info {
  uint32_t table_size;                     /* +0x00: 64 (0x40) */
  uint32_t pad0;                           /* +0x04 */
  uint64_t model_info_scratch_address;     /* +0x08: Optional ModelInfoScratch IOVA (0) */
  uint64_t model_info_scratch_size_bytes;  /* +0x10: Optional size (0) */
  uint64_t pad1;                           /* +0x18 */
  int32_t  parameters_offset;              /* +0x20: Offset from this table to param slices */
  uint32_t parameters_count;               /* +0x24: Number of parameter buffer slices (0 or 1) */
  uint64_t executable_address;             /* +0x28: Optional legacy exec addr (0) */
  uint64_t model_info_address;             /* +0x30: Optional LitebufModelInfo IOVA (0) */
  uint64_t model_info_size_bytes;          /* +0x38: Optional size (0) */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_dive_registration_info) == 64,
                    "litebuf_dive_registration_info must be 64B");

struct litebuf_dive_execute {
  uint32_t table_size;        /* +0x00: 80 (0x50) */
  int32_t  inputs_offset;     /* +0x04: Offset to inputs buffer slice array */
  uint32_t inputs_count;      /* +0x08 */
  int32_t  outputs_offset;    /* +0x0c: Offset to outputs buffer slice array */
  uint32_t outputs_count;     /* +0x10 */
  int32_t  empty0_offset;     /* +0x14 */
  uint32_t empty0_count;      /* +0x18 */
  int32_t  shape_info_offset; /* +0x1c */
  uint32_t shape_info_count;  /* +0x20 */
  int32_t  empty1_offset;     /* +0x24 */
  uint32_t empty1_count;      /* +0x28 */
  int32_t  empty2_offset;     /* +0x2c */
  uint32_t empty2_count;      /* +0x30 */
  uint32_t reserved0[5];      /* +0x34..+0x47 */
  int32_t  parameters_offset; /* +0x48: Offset to parameter buffer slice array */
  uint32_t parameters_count;  /* +0x4c: Number of parameter buffer slices (0 or 1) */
};
LBVII_STATIC_ASSERT(sizeof(struct litebuf_dive_execute) == 80,
                    "litebuf_dive_execute must be 80B");

/* =========================================================================
 * Align Helper & Inline Wire Builders / Response Parser
 * ========================================================================= */

LBVII_FN size_t litebuf_align16(size_t n) {
  return (n + 15u) & ~((size_t)15u);
}

/*
 * Builds inner DIVE RegisterInterpreterModel command in `dst` (up to `cap`).
 * Returns byte length (128 with 0 params, 144 with 1 param), or 0 on overflow.
 */
LBVII_FN size_t litebuf_build_dive_register_interpreter(
    void* dst, size_t cap,
    uint64_t dive_vm_ir_iova, uint64_t dive_vm_ir_size,
    const struct litebuf_dive_buffer_slice* params, uint32_t num_params,
    uint64_t model_info_scratch_iova, uint64_t model_info_scratch_size) {
  const size_t reg_info_off = 0x40u;
  const size_t params_off = 0x80u;
  const size_t raw_size = params_off + (size_t)num_params * sizeof(struct litebuf_dive_buffer_slice);
  const size_t total = litebuf_align16(raw_size);
  if (!dst || cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_DIVE_COMMANDS;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_dive_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 12u;
  root.union_offset = 16; /* 0x20 - 0x10 */
  root.union_type = LITEBUF_DIVE_CMD_REGISTER_INTERPRETER_MODEL;
  memcpy(p + 0x10u, &root, sizeof(root));

  struct litebuf_dive_register_interpreter_model rim;
  memset(&rim, 0, sizeof(rim));
  rim.table_size = 28u;
  rim.dive_vm_ir_address = dive_vm_ir_iova;
  rim.dive_vm_ir_size_bytes = dive_vm_ir_size;
  rim.registration_info_offset = (int32_t)(reg_info_off - 0x20u); /* 0x20 */
  memcpy(p + 0x20u, &rim, sizeof(rim));

  struct litebuf_dive_registration_info ri;
  memset(&ri, 0, sizeof(ri));
  ri.table_size = 64u;
  ri.model_info_scratch_address = model_info_scratch_iova;
  ri.model_info_scratch_size_bytes = model_info_scratch_size;
  ri.parameters_offset = (int32_t)(params_off - reg_info_off); /* 0x40 */
  ri.parameters_count = num_params;
  memcpy(p + reg_info_off, &ri, sizeof(ri));

  if (num_params > 0u && params != NULL) {
    memcpy(p + params_off, params, (size_t)num_params * sizeof(struct litebuf_dive_buffer_slice));
  }
  return total;
}

/*
 * Builds inner DIVE Execute command in `dst` (up to `cap`).
 * Returns 16B-aligned byte length (>=144), or 0 on overflow.
 */
LBVII_FN size_t litebuf_build_dive_execute(
    void* dst, size_t cap,
    const struct litebuf_dive_buffer_slice* inputs, uint32_t num_inputs,
    const struct litebuf_dive_buffer_slice* outputs, uint32_t num_outputs,
    const struct litebuf_dive_buffer_slice* params, uint32_t num_params) {
  const size_t exec_off = 0x20u;
  const size_t in_off = 0x70u;
  const size_t out_off = in_off + (size_t)num_inputs * sizeof(struct litebuf_dive_buffer_slice);
  const size_t tail_off = out_off + (size_t)num_outputs * sizeof(struct litebuf_dive_buffer_slice);
  const size_t total = litebuf_align16(
      tail_off + (size_t)num_params * sizeof(struct litebuf_dive_buffer_slice));
  if (!dst || cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_DIVE_COMMANDS;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_dive_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 12u;
  root.union_offset = 16; /* 0x20 - 0x10 */
  root.union_type = LITEBUF_DIVE_CMD_EXECUTE;
  memcpy(p + 0x10u, &root, sizeof(root));

  const int32_t tail_rel = (int32_t)(tail_off - exec_off);
  struct litebuf_dive_execute ex;
  memset(&ex, 0, sizeof(ex));
  ex.table_size = 80u;
  ex.inputs_offset = (int32_t)(in_off - exec_off);
  ex.inputs_count = num_inputs;
  ex.outputs_offset = (int32_t)(out_off - exec_off);
  ex.outputs_count = num_outputs;
  ex.empty0_offset = tail_rel;
  ex.shape_info_offset = tail_rel;
  ex.empty1_offset = tail_rel;
  ex.empty2_offset = tail_rel;
  ex.parameters_offset = tail_rel;
  ex.parameters_count = num_params;
  memcpy(p + exec_off, &ex, sizeof(ex));

  if (num_inputs > 0u && inputs != NULL) {
    memcpy(p + in_off, inputs, (size_t)num_inputs * sizeof(struct litebuf_dive_buffer_slice));
  }
  if (num_outputs > 0u && outputs != NULL) {
    memcpy(p + out_off, outputs, (size_t)num_outputs * sizeof(struct litebuf_dive_buffer_slice));
  }
  if (num_params > 0u && params != NULL) {
    memcpy(p + tail_off, params, (size_t)num_params * sizeof(struct litebuf_dive_buffer_slice));
  }
  return total;
}

/*
 * Builds outer mailbox RegisterModel command (aligned size, e.g. 144B for "Unknown").
 */
struct litebuf_register_model_params {
  uint64_t dive_cmd_address;
  uint32_t dive_cmd_size_bytes;
  uint64_t dive_image_address;
  uint32_t dive_image_size_bytes;
  uint64_t dive_scratch_address;
  uint32_t dive_scratch_size_bytes;
  int32_t  dive_heap_size;
  int32_t  exec_field_29;
  uint32_t core_session_handle;
  uint8_t  cluster_id;
  uint8_t  exec_field_31;
  uint32_t exec_field_14;
  uint64_t compiler_cl;
  uint8_t  subgraph_flag_18;
  uint8_t  exec_field_34;
  const char* model_id; /* NULL -> "Unknown" */
};

LBVII_FN size_t litebuf_build_vii_register_model(
    void* dst, size_t cap, const struct litebuf_register_model_params* in) {
  if (!dst || !in) return 0u;
  const char* mid = (in->model_id && in->model_id[0]) ? in->model_id : "Unknown";
  const size_t str_bytes = strlen(mid) + 1u;
  const size_t reg_off = 0x18u;
  const size_t str_off = reg_off + sizeof(struct litebuf_vii_register_model); /* 0x80 */
  const size_t total = litebuf_align16(str_off + str_bytes);
  if (cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_RUNTIME_MESSAGES;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_vii_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 7u;
  root.union_offset = 8;
  root.union_type = LITEBUF_VII_CMD_REGISTER_MODEL;
  memcpy(p + 0x10u, &root, sizeof(root));

  struct litebuf_vii_register_model rm;
  memset(&rm, 0, sizeof(rm));
  rm.table_size = 99u;
  rm.dive_cmd_address = in->dive_cmd_address;
  rm.dive_cmd_size_bytes = in->dive_cmd_size_bytes;
  rm.dive_image_address = in->dive_image_address;
  rm.dive_image_size_bytes = in->dive_image_size_bytes;
  rm.dive_scratch_address = in->dive_scratch_address;
  rm.dive_scratch_size_bytes = in->dive_scratch_size_bytes;
  rm.dive_heap_size = in->dive_heap_size;
  rm.exec_field_29 = in->exec_field_29;
  rm.core_session_handle = in->core_session_handle;
  rm.cluster_id = in->cluster_id;
  rm.exec_field_31 = in->exec_field_31;
  rm.model_id_str_offset = (int16_t)(str_off - reg_off); /* 0x68 */
  rm.model_id_str_size = (int16_t)str_bytes;
  rm.exec_field_14 = in->exec_field_14;
  rm.compiler_cl = in->compiler_cl;
  rm.subgraph_flag_18 = in->subgraph_flag_18;
  rm.exec_field_34 = in->exec_field_34;
  memcpy(p + reg_off, &rm, sizeof(rm));

  memcpy(p + str_off, mid, str_bytes);
  return total;
}

/*
 * Builds outer mailbox Execute command (always 80B inline <= 96B limit).
 */
LBVII_FN size_t litebuf_build_vii_execute(
    void* dst, size_t cap,
    uint64_t dive_cmd_iova, uint32_t dive_cmd_size, uint32_t model_id,
    uint8_t core_power_state, uint8_t mem_power_state, uint8_t qos_class,
    uint8_t perf_metrics_flag, uint8_t cluster_id, uint32_t core_session_handle,
    uint8_t priority) {
  const size_t total = 80u; /* 16 (hdr) + 8 (root) + 48 (Execute) = 72 -> 80 */
  if (!dst || cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_RUNTIME_MESSAGES;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_vii_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 7u;
  root.union_offset = 8;
  root.union_type = LITEBUF_VII_CMD_EXECUTE;
  root.priority = priority;
  memcpy(p + 0x10u, &root, sizeof(root));

  struct litebuf_vii_execute ex;
  memset(&ex, 0, sizeof(ex));
  ex.table_size = 44u;
  ex.dive_cmd_address = dive_cmd_iova;
  ex.dive_cmd_size_bytes = dive_cmd_size;
  ex.model_id = model_id;
  ex.tpu_core_power_state = core_power_state;
  ex.memory_power_state = mem_power_state;
  ex.qos_class = qos_class;
  ex.perf_metrics_flag = perf_metrics_flag;
  ex.cluster_id = cluster_id;
  ex.core_session_handle = core_session_handle;
  memcpy(p + 0x18u, &ex, sizeof(ex));
  return total;
}

/*
 * Builds outer mailbox UnregisterModel command (always 32B inline <= 96B limit).
 */
LBVII_FN size_t litebuf_build_vii_unregister_model(
    void* dst, size_t cap, uint32_t model_id) {
  const size_t total = 32u;
  if (!dst || cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_RUNTIME_MESSAGES;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_vii_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 7u;
  root.union_offset = 8;
  root.union_type = LITEBUF_VII_CMD_UNREGISTER_MODEL;
  memcpy(p + 0x10u, &root, sizeof(root));

  struct litebuf_vii_unregister_model um;
  memset(&um, 0, sizeof(um));
  um.table_size = 8u;
  um.model_id = model_id;
  memcpy(p + 0x18u, &um, sizeof(um));
  return total;
}

/*
 * Builds outer mailbox AcquireWakeLock (type=4) or ReleaseWakeLock (type=5) command (32B inline).
 */
LBVII_FN size_t litebuf_build_vii_wakelock(
    void* dst, size_t cap, int acquire,
    uint8_t core_power_state, uint8_t mem_power_state, uint8_t cluster_ids_bitset) {
  const size_t total = 32u;
  if (!dst || cap < total) return 0u;

  uint8_t* p = (uint8_t*)dst;
  memset(p, 0, total);

  struct litebuf_header hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.schema_id = LITEBUF_SCHEMA_RUNTIME_MESSAGES;
  hdr.log2_alignment = LITEBUF_LOG2_ALIGN_16;
  hdr.root_offset = LITEBUF_HEADER_SIZE;
  memcpy(p + 0x00u, &hdr, sizeof(hdr));

  struct litebuf_vii_command_root root;
  memset(&root, 0, sizeof(root));
  root.table_size = 7u;
  root.union_offset = 8;
  root.union_type = acquire ? LITEBUF_VII_CMD_ACQUIRE_WAKELOCK : LITEBUF_VII_CMD_RELEASE_WAKELOCK;
  memcpy(p + 0x10u, &root, sizeof(root));

  struct litebuf_vii_wakelock_cmd wc;
  memset(&wc, 0, sizeof(wc));
  wc.table_size = 5u;
  wc.tpu_core_power_state = core_power_state;
  wc.memory_power_state = mem_power_state;
  wc.cluster_ids_bitset = cluster_ids_bitset;
  memcpy(p + 0x18u, &wc, sizeof(wc));
  return total;
}

/*
 * Decodes 48-byte ViiResponse payload matching vii_litebuf_response_parser.cc (0xf147c).
 */
LBVII_FN struct litebuf_vii_parsed_response litebuf_parse_vii_response(
    const void* buf, size_t len) {
  struct litebuf_vii_parsed_response out;
  memset(&out, 0, sizeof(out));
  if (!buf || len < LITEBUF_HEADER_SIZE + 8u) return out;

  const uint8_t* p = (const uint8_t*)buf;
  struct litebuf_header hdr;
  memcpy(&hdr, p, sizeof(hdr));
  if (hdr.root_offset + 6u > len) return out;

  uint16_t tbl_sz = 0;
  int16_t u_off = 0;
  uint16_t u_type = 0;
  memcpy(&tbl_sz, p + hdr.root_offset + 0u, 2u);
  memcpy(&u_off, p + hdr.root_offset + 2u, 2u);
  memcpy(&u_type, p + hdr.root_offset + 4u, 2u);
  if (tbl_sz < 6u || u_type > LITEBUF_VII_RESP_REPRIORITIZE) return out;

  out.response_type = u_type;
  if (tbl_sz >= 12u && hdr.root_offset + 12u <= len) {
    memcpy(&out.fw_status_word, p + hdr.root_offset + 8u, 4u);
  }
  out.valid = 1;

  if (u_type == LITEBUF_VII_RESP_VOID_OK) {
    return out;
  }

  const int64_t mem_pos_s = (int64_t)hdr.root_offset + (int64_t)u_off;
  if (mem_pos_s < 0 || (size_t)mem_pos_s + 4u > len) {
    out.valid = 0;
    return out;
  }
  const size_t mem_pos = (size_t)mem_pos_s;
  uint16_t m_sz = 0;
  memcpy(&m_sz, p + mem_pos, 2u);

  if (u_type == LITEBUF_VII_RESP_EXECUTE) {
    uint64_t lo = 0, hi = 0;
    if (m_sz >= 8u && mem_pos + 8u <= len) {
      uint32_t v = 0;
      memcpy(&v, p + mem_pos + 4u, 4u);
      lo = v;
    }
    if (m_sz >= 12u && mem_pos + 12u <= len) {
      uint32_t v = 0;
      memcpy(&v, p + mem_pos + 8u, 4u);
      hi = v;
    }
    out.execution_us = lo | (hi << 32);
  } else if (u_type == LITEBUF_VII_RESP_REGISTER_MODEL) {
    if (m_sz >= 8u && mem_pos + 8u <= len) {
      memcpy(&out.handle, p + mem_pos + 4u, 4u);
    }
  } else if (u_type == LITEBUF_VII_RESP_ERROR) {
    if (m_sz >= 8u && mem_pos + 8u <= len) {
      memcpy(&out.error_code, p + mem_pos + 4u, 4u);
    }
  }
  return out;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BAREMETAL_VII_LITEBUF_VII_WIRE_H_ */
