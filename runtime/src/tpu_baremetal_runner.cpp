// Standalone zero-Google-.so DarwiNN / EdgeTPU Litebuf VII bare-metal runner.
// Drives /dev/edgetpu-soc (0xED) and /dev/dma_heap/system directly using
// clean-room Litebuf VII headers and native C++ .dgc0 FlatBuffer parsing.

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "edgetpu_buenos_uapi.h"
#include "json_util.hpp"
#include "litebuf_vii_wire.h"

namespace {

using namespace tpu_runner;

// ---------------------------------------------------------------------------
// Native Bounds-Checked FlatBuffer Reader (.dgc0 -> DWS0 -> SubGraph/Executable)
// ---------------------------------------------------------------------------

struct FbView {
  const uint8_t* data = nullptr;
  size_t size = 0;

  bool in_bounds(size_t off, size_t n) const {
    return off <= size && n <= size - off;
  }

  template <typename T>
  bool read(size_t off, T* out) const {
    if (!in_bounds(off, sizeof(T))) return false;
    std::memcpy(out, data + off, sizeof(T));
    return true;
  }

  bool follow_uoffset(size_t field_pos, size_t* target) const {
    uint32_t rel = 0;
    if (!read(field_pos, &rel) || rel == 0) return false;
    if (field_pos > size - rel) return false;
    *target = field_pos + rel;
    return *target <= size;
  }

  bool table_field(size_t table_pos, uint16_t slot, size_t* field_pos) const {
    int32_t vtoff_rel = 0;
    if (!read(table_pos, &vtoff_rel)) return false;
    int64_t vtable_s = static_cast<int64_t>(table_pos) - static_cast<int64_t>(vtoff_rel);
    if (vtable_s < 0) return false;
    size_t vtable = static_cast<size_t>(vtable_s);
    uint16_t vsz = 0, osz = 0;
    if (!read(vtable, &vsz) || !read(vtable + 2, &osz)) return false;
    size_t slot_off = 4u + static_cast<size_t>(slot) * 2u;
    if (slot_off + 2u > vsz) return false;
    uint16_t foff = 0;
    if (!read(vtable + slot_off, &foff) || foff == 0 || foff >= osz) return false;
    if (table_pos + foff > size) return false;
    *field_pos = table_pos + foff;
    return true;
  }

  uint64_t field_u64(size_t tbl, uint16_t slot, uint64_t def = 0) const {
    size_t fp = 0;
    uint64_t v = def;
    if (table_field(tbl, slot, &fp)) read(fp, &v);
    return v;
  }

  uint32_t field_u32(size_t tbl, uint16_t slot, uint32_t def = 0) const {
    size_t fp = 0;
    uint32_t v = def;
    if (table_field(tbl, slot, &fp)) read(fp, &v);
    return v;
  }

  int64_t field_i64(size_t tbl, uint16_t slot, int64_t def = 0) const {
    size_t fp = 0;
    int64_t v = def;
    if (table_field(tbl, slot, &fp)) read(fp, &v);
    return v;
  }

  int32_t field_i32(size_t tbl, uint16_t slot, int32_t def = 0) const {
    size_t fp = 0;
    int32_t v = def;
    if (table_field(tbl, slot, &fp)) read(fp, &v);
    return v;
  }

  uint8_t field_u8(size_t tbl, uint16_t slot, uint8_t def = 0) const {
    size_t fp = 0;
    uint8_t v = def;
    if (table_field(tbl, slot, &fp)) read(fp, &v);
    return v;
  }

  bool byte_vector_field(size_t tbl, uint16_t slot, FbView* sub) const {
    size_t fp = 0, vp = 0;
    if (!table_field(tbl, slot, &fp) || !follow_uoffset(fp, &vp)) return false;
    uint32_t len = 0;
    if (!read(vp, &len) || !in_bounds(vp + 4, len)) return false;
    sub->data = data + vp + 4;
    sub->size = len;
    return true;
  }

  bool uoffset_vector_field(size_t tbl, uint16_t slot, std::vector<size_t>* elems) const {
    elems->clear();
    size_t fp = 0, vp = 0;
    if (!table_field(tbl, slot, &fp) || !follow_uoffset(fp, &vp)) return false;
    uint32_t len = 0;
    if (!read(vp, &len) || len > 1000000u || !in_bounds(vp + 4, static_cast<size_t>(len) * 4u)) {
      return false;
    }
    for (uint32_t i = 0; i < len; ++i) {
      size_t elem = 0;
      if (!follow_uoffset(vp + 4u + static_cast<size_t>(i) * 4u, &elem)) return false;
      elems->push_back(elem);
    }
    return true;
  }

  bool i32_vector_field(size_t tbl, uint16_t slot, std::vector<int32_t>* vals) const {
    vals->clear();
    size_t fp = 0, vp = 0;
    if (!table_field(tbl, slot, &fp) || !follow_uoffset(fp, &vp)) return false;
    uint32_t len = 0;
    if (!read(vp, &len) || len > 1000000u || !in_bounds(vp + 4, static_cast<size_t>(len) * 4u)) {
      return false;
    }
    for (uint32_t i = 0; i < len; ++i) {
      int32_t v = 0;
      read(vp + 4u + static_cast<size_t>(i) * 4u, &v);
      vals->push_back(v);
    }
    return true;
  }
};

struct Dgc0Slice {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct ParsedDgc0Model {
  std::vector<uint64_t> input_sizes;
  std::vector<uint64_t> output_sizes;
  uint64_t dive_scratch_size = 0;
  uint64_t compiler_cl = 0;
  int32_t dive_heap_size = 0;
  int32_t exec_field_29 = 0;
  uint32_t exec_field_14 = 0;
  uint8_t cluster_id = LITEBUF_DEFAULT_CLUSTER_ID;
  uint8_t exec_field_31 = 0;
  uint8_t exec_field_34 = 0;
  uint8_t subgraph_flag_18 = 0;
  Dgc0Slice ext_dive_vm_ir{};
  Dgc0Slice ext_dive_image{};
  bool has_params = false;
  Dgc0Slice ext_params{};
};

size_t darwinn_dtype_bytes(uint8_t dtype) {
  switch (dtype) {
    case 1:  // INT8
    case 2:  // UINT8 / RAW_BYTES
    case 8:  // BOOL
      return 1;
    case 3:  // INT16
    case 5:  // FLOAT16
      return 2;
    case 4:  // INT32
    case 6:  // FLOAT32
      return 4;
    case 7:  // INT64
      return 8;
    default:
      return 1;
  }
}

bool parse_dgc0_native(const std::vector<uint8_t>& blob, ParsedDgc0Model* out, std::string* err) {
  if (blob.size() < 16 || std::memcmp(blob.data() + 4, "DGC0", 4) != 0) {
    *err = "File is not a valid DGC0 container";
    return false;
  }

  FbView dgc{blob.data(), blob.size()};
  uint32_t dgc_root = 0;
  if (!dgc.read(0, &dgc_root) || dgc_root >= blob.size()) {
    *err = "Invalid DGC0 root offset";
    return false;
  }

  // External buffer spans live in DGC0 slot 7
  std::vector<size_t> ext_tables;
  if (!dgc.uoffset_vector_field(dgc_root, 7, &ext_tables) || ext_tables.size() < 2) {
    *err = "Missing DGC0 external_buffers (< 2 entries)";
    return false;
  }
  auto read_ext = [&](size_t idx, Dgc0Slice* s) -> bool {
    s->offset = dgc.field_u64(ext_tables[idx], 0, 0);
    s->size = dgc.field_u64(ext_tables[idx], 1, 0);
    return s->size > 0 && s->offset <= blob.size() && s->size <= blob.size() - s->offset;
  };
  if (!read_ext(0, &out->ext_dive_vm_ir) || !read_ext(1, &out->ext_dive_image)) {
    *err = "Out-of-bounds DGC0 external buffer ext[0]/ext[1]";
    return false;
  }
  if (ext_tables.size() >= 3) {
    if (!read_ext(2, &out->ext_params)) {
      *err = "Out-of-bounds DGC0 external buffer ext[2]";
      return false;
    }
    out->has_params = true;
  }

  // Package (DWS0) lives in DGC0 slot 0
  FbView dws{};
  if (!dgc.byte_vector_field(dgc_root, 0, &dws) || dws.size < 16 ||
      std::memcmp(dws.data + 4, "DWS0", 4) != 0) {
    *err = "Missing or invalid nested Package DWS0 in DGC0 slot 0";
    return false;
  }

  uint32_t pkg_root = 0;
  if (!dws.read(0, &pkg_root) || pkg_root >= dws.size) {
    *err = "Invalid DWS0 root offset";
    return false;
  }
  out->compiler_cl = dws.field_u32(pkg_root, 10, 0);
  if (out->compiler_cl == 0) {
    out->compiler_cl = dgc.field_u64(dgc_root, 5, 0);
  }

  // SubGraph[0]
  std::vector<size_t> subgraphs;
  if (!dws.uoffset_vector_field(pkg_root, 3, &subgraphs) || subgraphs.empty()) {
    *err = "No SubGraphs in DWS0";
    return false;
  }
  size_t sg = subgraphs[0];
  out->subgraph_flag_18 = dws.field_u8(sg, 18, 0);

  // Extract SCRATCHPAD tensor byte size from SubGraph.slot_0 (tensors)
  std::vector<size_t> tensors;
  if (dws.uoffset_vector_field(sg, 0, &tensors)) {
    for (size_t tp : tensors) {
      uint8_t storage = dws.field_u8(tp, 5, 0);
      if (storage == 3) {  // SCRATCHPAD
        std::vector<int32_t> dims;
        dws.i32_vector_field(tp, 0, &dims);
        uint8_t dt = dws.field_u8(tp, 1, 2);
        uint64_t bytes = darwinn_dtype_bytes(dt);
        for (int32_t d : dims) {
          if (d > 0) bytes *= static_cast<uint64_t>(d);
        }
        if (bytes > out->dive_scratch_size) out->dive_scratch_size = bytes;
      }
    }
  }

  // Operation with OpExecutableCall (detail_type == 2) -> MultiExecutable -> Executable[0]
  std::vector<size_t> ops;
  if (!dws.uoffset_vector_field(sg, 3, &ops) || ops.empty()) {
    *err = "No operations in SubGraph[0]";
    return false;
  }
  size_t op_call = 0;
  bool found_call = false;
  for (size_t op : ops) {
    if (dws.field_u8(op, 0, 0) == 2) {  // Operation.detail_type (slot 0) == 2 (OpExecutableCall)
      size_t fp = 0;
      if (dws.table_field(op, 1, &fp) && dws.follow_uoffset(fp, &op_call)) {
        found_call = true;
        break;
      }
    }
  }
  if (!found_call) {
    *err = "No OpExecutableCall (detail_type=2) in SubGraph[0]";
    return false;
  }

  FbView mexec{};
  if (!dws.byte_vector_field(op_call, 0, &mexec) || mexec.size < 8) {
    *err = "Missing nested MultiExecutable in OpExecutableCall";
    return false;
  }
  uint32_t me_root = 0;
  if (!mexec.read(0, &me_root) || me_root >= mexec.size) {
    *err = "Invalid MultiExecutable root offset";
    return false;
  }
  std::vector<size_t> execs;
  if (!mexec.uoffset_vector_field(me_root, 1, &execs) || execs.empty()) {
    *err = "No Executables in MultiExecutable";
    return false;
  }
  size_t ex = execs[0];

  auto read_io_sizes = [&](uint16_t slot, std::vector<uint64_t>* sizes) {
    sizes->clear();
    std::vector<size_t> vec;
    if (!mexec.uoffset_vector_field(ex, slot, &vec)) return;
    for (size_t tp : vec) {
      uint64_t sz64 = mexec.field_u64(tp, 3, 0);
      uint32_t sz32 = mexec.field_u32(tp, 1, 0);
      sizes->push_back(sz64 != 0 ? sz64 : static_cast<uint64_t>(sz32));
    }
  };
  read_io_sizes(3, &out->input_sizes);   // Executable.inputs (slot 3)
  read_io_sizes(4, &out->output_sizes);  // Executable.outputs (slot 4)

  if (out->input_sizes.empty() || out->output_sizes.empty()) {
    *err = "Executable missing input or output descriptor";
    return false;
  }

  out->exec_field_14 = mexec.field_u32(ex, 14, 0);
  out->dive_heap_size = static_cast<int32_t>(mexec.field_i64(ex, 28, 0));
  out->exec_field_29 = static_cast<int32_t>(mexec.field_i64(ex, 29, 0));
  out->exec_field_31 = mexec.field_u8(ex, 31, 0);
  out->exec_field_34 = mexec.field_u8(ex, 34, 0);

  uint32_t hw_cluster = mexec.field_u32(ex, 8, LITEBUF_DEFAULT_CLUSTER_ID);
  out->cluster_id = static_cast<uint8_t>(hw_cluster & 0xffu);
  return true;
}

// ---------------------------------------------------------------------------
// RAII DMA-BUF Allocation, Mapping, and Cache Sync
// ---------------------------------------------------------------------------

struct DmaBufRegion {
  int tpu_fd = -1;
  int dmabuf_fd = -1;
  uint8_t* mapped = nullptr;
  size_t alloc_size = 0;
  size_t payload_size = 0;
  uint64_t iova = 0;
  bool iova_mapped = false;
  uint32_t map_flags = 0;
  uint32_t die_index = 0xffffffffu;

  DmaBufRegion() = default;
  DmaBufRegion(const DmaBufRegion&) = delete;
  DmaBufRegion& operator=(const DmaBufRegion&) = delete;
  DmaBufRegion(DmaBufRegion&& o) noexcept { *this = std::move(o); }
  DmaBufRegion& operator=(DmaBufRegion&& o) noexcept {
    if (this != &o) {
      cleanup();
      tpu_fd = o.tpu_fd;
      dmabuf_fd = o.dmabuf_fd;
      mapped = o.mapped;
      alloc_size = o.alloc_size;
      payload_size = o.payload_size;
      iova = o.iova;
      iova_mapped = o.iova_mapped;
      map_flags = o.map_flags;
      die_index = o.die_index;
      o.tpu_fd = -1;
      o.dmabuf_fd = -1;
      o.mapped = nullptr;
      o.iova_mapped = false;
    }
    return *this;
  }
  ~DmaBufRegion() { cleanup(); }

  void cleanup() {
    if (iova_mapped && tpu_fd >= 0) {
      struct edgetpu_map_dmabuf_ioctl u{};
      u.offset = 0;
      u.size = alloc_size;
      u.device_address = iova;
      u.dmabuf_fd = dmabuf_fd;
      u.flags = map_flags;
      u.die_index = die_index;
      ::ioctl(tpu_fd, EDGETPU_UNMAP_DMABUF, &u);
      iova_mapped = false;
    }
    if (mapped && mapped != MAP_FAILED) {
      ::munmap(mapped, alloc_size);
      mapped = nullptr;
    }
    if (dmabuf_fd >= 0) {
      ::close(dmabuf_fd);
      dmabuf_fd = -1;
    }
  }

  bool init(int dev_fd, size_t payload_bytes, uint32_t flags, std::string* err) {
    tpu_fd = dev_fd;
    payload_size = payload_bytes;
    alloc_size = std::max<size_t>(4096u, (payload_bytes + 4095u) & ~static_cast<size_t>(4095u));
    map_flags = flags;

    std::string derr;
    if (alloc_dma_heap_buffer(alloc_size, &dmabuf_fd, derr) != 0) {
      *err = "DMA-BUF alloc failed (" + std::to_string(alloc_size) + "B): " + derr;
      return false;
    }

    void* ptr = ::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    if (ptr == MAP_FAILED) {
      *err = "mmap(dmabuf) failed errno=" + std::to_string(errno);
      cleanup();
      return false;
    }
    mapped = static_cast<uint8_t*>(ptr);

    struct edgetpu_map_dmabuf_ioctl m{};
    m.offset = 0;
    m.size = alloc_size;
    m.dmabuf_fd = dmabuf_fd;
    m.flags = flags;
    m.die_index = die_index;
    if (::ioctl(tpu_fd, EDGETPU_MAP_DMABUF, &m) != 0) {
      *err = "EDGETPU_MAP_DMABUF failed errno=" + std::to_string(errno);
      cleanup();
      return false;
    }
    iova = m.device_address;
    iova_mapped = true;
    return true;
  }

  bool upload(const uint8_t* src, size_t len, std::string* err) {
    if (len > alloc_size) {
      *err = "Upload size exceeds DMA-BUF capacity";
      return false;
    }
    if (sync_dmabuf(dmabuf_fd, LOCAL_DMA_BUF_SYNC_START | LOCAL_DMA_BUF_SYNC_WRITE) != 0) {
      *err = "DMA_BUF_IOCTL_SYNC START|WRITE failed errno=" + std::to_string(errno);
      return false;
    }
    if (src && len > 0) std::memcpy(mapped, src, len);
    if (len < alloc_size) std::memset(mapped + len, 0, alloc_size - len);
    if (sync_dmabuf(dmabuf_fd, LOCAL_DMA_BUF_SYNC_END | LOCAL_DMA_BUF_SYNC_WRITE) != 0) {
      *err = "DMA_BUF_IOCTL_SYNC END|WRITE failed errno=" + std::to_string(errno);
      return false;
    }
    return true;
  }

  bool zero_fill(std::string* err) { return upload(nullptr, 0, err); }

  bool download(uint8_t* dst, size_t len, std::string* err) const {
    if (len > alloc_size) {
      *err = "Download size exceeds DMA-BUF capacity";
      return false;
    }
    if (sync_dmabuf(dmabuf_fd, LOCAL_DMA_BUF_SYNC_START | LOCAL_DMA_BUF_SYNC_READ) != 0) {
      *err = "DMA_BUF_IOCTL_SYNC START|READ failed errno=" + std::to_string(errno);
      return false;
    }
    std::memcpy(dst, mapped, len);
    if (sync_dmabuf(dmabuf_fd, LOCAL_DMA_BUF_SYNC_END | LOCAL_DMA_BUF_SYNC_READ) != 0) {
      *err = "DMA_BUF_IOCTL_SYNC END|READ failed errno=" + std::to_string(errno);
      return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Direct /dev/edgetpu-soc Session & Litebuf VII Submit/Wait
// ---------------------------------------------------------------------------

struct BaremetalSession {
  int fd = -1;
  int resp_efd = -1;
  int err_efd = -1;
  bool wake_locked = false;
  uint64_t credit_size = 0;
  struct edgetpu_fw_version fw_ver{};
  uint64_t next_seq = 0;

  ~BaremetalSession() {
    if (wake_locked && fd >= 0) {
      ::ioctl(fd, EDGETPU_RELEASE_WAKE_LOCK);
      wake_locked = false;
    }
    if (err_efd >= 0) ::close(err_efd);
    if (resp_efd >= 0) ::close(resp_efd);
    if (fd >= 0) ::close(fd);
  }

  bool open_and_init(std::string* err) {
    fd = ::open("/dev/edgetpu-soc", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      *err = "open(/dev/edgetpu-soc) failed errno=" + std::to_string(errno);
      return false;
    }
    ::ioctl(fd, EDGETPU_GET_CREDIT_SIZE, &credit_size);

    struct edgetpu_mailbox_attr attr{};
    attr.cmd_queue_size = 24;
    attr.resp_queue_size = 12;
    attr.sizeof_cmd = 48;
    attr.sizeof_resp = 24;
    attr.priority = 8;
    attr.cmdq_tail_doorbell = 1;
    attr.partition_type = 0;
    attr.client_priv = 1;
    if (::ioctl(fd, EDGETPU_CREATE_GROUP, &attr) != 0) {
      *err = "EDGETPU_CREATE_GROUP failed errno=" + std::to_string(errno);
      return false;
    }

    resp_efd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    err_efd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (resp_efd < 0 || err_efd < 0) {
      *err = "eventfd creation failed errno=" + std::to_string(errno);
      return false;
    }

    struct edgetpu_event_register ev0{EDGETPU_EVENT_RESPDATA, static_cast<__u32>(resp_efd)};
    if (::ioctl(fd, EDGETPU_SET_EVENTFD, &ev0) != 0) {
      *err = "EDGETPU_SET_EVENTFD(RESPDATA) failed errno=" + std::to_string(errno);
      return false;
    }
    struct edgetpu_event_register ev1{EDGETPU_EVENT_FATAL_ERROR, static_cast<__u32>(err_efd)};
    ::ioctl(fd, EDGETPU_SET_EVENTFD, &ev1);

    if (::ioctl(fd, EDGETPU_ACQUIRE_WAKE_LOCK) != 0) {
      *err = "EDGETPU_ACQUIRE_WAKE_LOCK failed errno=" + std::to_string(errno);
      return false;
    }
    wake_locked = true;

    ::ioctl(fd, EDGETPU_FIRMWARE_VERSION, &fw_ver);
    return true;
  }

  bool submit_and_wait(const void* outer_buf, size_t outer_len,
                       struct litebuf_vii_parsed_response* parsed_out,
                       uint16_t* resp_code_out, std::string* err) {
    uint64_t seq = next_seq++;
    struct edgetpu_vii_litebuf_command_ioctl cmd{};
    cmd.litebuf_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(outer_buf));
    cmd.litebuf_size = static_cast<__u32>(outer_len);
    cmd.seq = seq;

    int rc = -1;
    while (true) {
      rc = ::ioctl(fd, EDGETPU_VII_LITEBUF_COMMAND, &cmd);
      if (rc == 0) break;
      if (errno == EINTR || errno == EAGAIN) continue;
      *err = "EDGETPU_VII_LITEBUF_COMMAND(seq=" + std::to_string(seq) +
             ") failed errno=" + std::to_string(errno);
      return false;
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    uint8_t resp_buf[LITEBUF_VII_RESPONSE_BYTES]{};

    while (true) {
      struct edgetpu_vii_litebuf_response_ioctl r{};
      r.litebuf_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resp_buf));
      std::memset(resp_buf, 0, sizeof(resp_buf));

      if (::ioctl(fd, EDGETPU_VII_LITEBUF_RESPONSE, &r) == 0) {
        if (r.seq == seq) {
          if (resp_code_out) *resp_code_out = r.code;
          *parsed_out = litebuf_parse_vii_response(resp_buf, sizeof(resp_buf));
          if (r.code != 0) {
            *err = "Firmware returned non-zero VII response code=" + std::to_string(r.code);
            return false;
          }
          if (!parsed_out->valid) {
            *err = "Invalid ViiResponse Litebuf payload";
            return false;
          }
          if (parsed_out->response_type == LITEBUF_VII_RESP_ERROR) {
            *err = "ViiResponse ERROR code=" + std::to_string(parsed_out->error_code);
            return false;
          }
          return true;
        }
        continue;
      }

      if (errno != ENOENT && errno != EAGAIN && errno != EINTR) {
        *err = "EDGETPU_VII_LITEBUF_RESPONSE failed errno=" + std::to_string(errno);
        return false;
      }

      auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        *err = "Timed out waiting for VII response seq=" + std::to_string(seq);
        return false;
      }
      int rem_ms = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
      if (rem_ms < 1) rem_ms = 1;

      struct pollfd pfds[2] = {{resp_efd, POLLIN, 0}, {err_efd, POLLIN, 0}};
      int prc = ::poll(pfds, 2, rem_ms);
      if (prc < 0 && errno == EINTR) continue;
      if (prc <= 0) {
        *err = "poll(resp_efd) timed out/failed waiting seq=" + std::to_string(seq);
        return false;
      }
      if (pfds[1].revents & POLLIN) {
        uint64_t val = 0;
        if (::read(err_efd, &val, sizeof(val)) > 0) {
          *err = "EdgeTPU fatal error eventfd signaled";
          return false;
        }
      }
      if (pfds[0].revents & POLLIN) {
        uint64_t val = 0;
        ::read(resp_efd, &val, sizeof(val));
      }
    }
  }
};

// ---------------------------------------------------------------------------
// /proc/self/maps Verification (Zero Google TPU Libraries Proof)
// ---------------------------------------------------------------------------

struct ProcMapsAudit {
  bool maps_has_google_tpu_libs = false;
  std::vector<std::string> mapped_libs;
};

ProcMapsAudit audit_proc_maps() {
  ProcMapsAudit audit;
  std::ifstream ifs("/proc/self/maps");
  std::set<std::string> libs;
  std::string line;
  while (std::getline(ifs, line)) {
    size_t sp = line.find('/');
    if (sp == std::string::npos) continue;
    std::string path = line.substr(sp);
    if (path.find(".so") == std::string::npos) continue;
    libs.insert(path);
    if (path.find("tachyon") != std::string::npos ||
        path.find("edgetpu") != std::string::npos ||
        path.find("darwinn") != std::string::npos ||
        path.find("libgxp") != std::string::npos ||
        path.find("tflite") != std::string::npos ||
        path.find("tensorflow") != std::string::npos ||
        path.find("/vendor/") != std::string::npos) {
      audit.maps_has_google_tpu_libs = true;
    }
  }
  audit.mapped_libs.assign(libs.begin(), libs.end());
  return audit;
}

// ---------------------------------------------------------------------------
// CLI & Runner Driver
// ---------------------------------------------------------------------------

struct RunnerArgs {
  std::string dgc0_path;
  std::string in_path;
  std::string in2_path;
  std::string ref_path;
  std::string out_dump_path;
  int iters = 1;
  bool json_only = false;
};

void print_usage(const char* argv0) {
  std::fprintf(stderr,
      "Usage:\n"
      "  %s --dgc0 <file.dgc0> --in <in.bin> [--in2 <in2.bin>] "
      "[--ref <ref.bin>] [--out-dump <out.bin>] [--iters <N>] [--json]\n",
      argv0);
}

bool prepare_inputs(const ParsedDgc0Model& model, const RunnerArgs& args,
                    std::vector<std::vector<uint8_t>>* input_blobs, std::string* err) {
  std::vector<uint8_t> in1;
  std::string rerr;
  if (!read_binary_file(args.in_path, in1, rerr)) {
    *err = rerr;
    return false;
  }

  size_t num_in = model.input_sizes.size();
  input_blobs->resize(num_in);

  if (num_in == 1) {
    if (in1.size() != model.input_sizes[0]) {
      *err = "Input size mismatch: got " + std::to_string(in1.size()) +
             "B expected " + std::to_string(model.input_sizes[0]) + "B";
      return false;
    }
    (*input_blobs)[0] = std::move(in1);
    return true;
  }

  if (num_in == 2 && !args.in2_path.empty()) {
    std::vector<uint8_t> in2;
    if (!read_binary_file(args.in2_path, in2, rerr)) {
      *err = rerr;
      return false;
    }
    if (in1.size() != model.input_sizes[0] || in2.size() != model.input_sizes[1]) {
      *err = "Dual-input size mismatch";
      return false;
    }
    (*input_blobs)[0] = std::move(in1);
    (*input_blobs)[1] = std::move(in2);
    return true;
  }

  // Concatenated multi-input file (e.g. int8_add_1x16x16x16.in.bin = 8192B = 4096 + 4096)
  uint64_t sum = 0;
  for (uint64_t sz : model.input_sizes) sum += sz;
  if (in1.size() != sum) {
    *err = "Multi-input size mismatch: file has " + std::to_string(in1.size()) +
           "B expected " + std::to_string(sum) + "B (or pass --in2)";
    return false;
  }
  size_t cursor = 0;
  for (size_t i = 0; i < num_in; ++i) {
    size_t sz = static_cast<size_t>(model.input_sizes[i]);
    (*input_blobs)[i].assign(in1.begin() + cursor, in1.begin() + cursor + sz);
    cursor += sz;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  RunnerArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (a == "--dgc0" && i + 1 < argc) {
      args.dgc0_path = argv[++i];
    } else if (a == "--in" && i + 1 < argc) {
      args.in_path = argv[++i];
    } else if (a == "--in2" && i + 1 < argc) {
      args.in2_path = argv[++i];
    } else if (a == "--ref" && i + 1 < argc) {
      args.ref_path = argv[++i];
    } else if (a == "--out-dump" && i + 1 < argc) {
      args.out_dump_path = argv[++i];
    } else if (a == "--iters" && i + 1 < argc) {
      args.iters = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--json") {
      args.json_only = true;
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", a.c_str());
      print_usage(argv[0]);
      return 2;
    }
  }

  if (args.dgc0_path.empty() || args.in_path.empty()) {
    std::fprintf(stderr, "--dgc0 and --in are required\n");
    print_usage(argv[0]);
    return 2;
  }

  std::string err;
  std::vector<uint8_t> dgc0_blob;
  if (!read_binary_file(args.dgc0_path, dgc0_blob, err)) {
    std::fprintf(stderr, "Failed reading dgc0: %s\n", err.c_str());
    return 1;
  }

  ParsedDgc0Model model;
  if (!parse_dgc0_native(dgc0_blob, &model, &err)) {
    std::fprintf(stderr, "Failed parsing dgc0: %s\n", err.c_str());
    return 1;
  }

  std::vector<std::vector<uint8_t>> input_blobs;
  if (!prepare_inputs(model, args, &input_blobs, &err)) {
    std::fprintf(stderr, "Input prep failed: %s\n", err.c_str());
    return 1;
  }

  std::vector<uint8_t> ref_blob;
  bool has_ref = !args.ref_path.empty();
  if (has_ref && !read_binary_file(args.ref_path, ref_blob, err)) {
    std::fprintf(stderr, "Failed reading ref: %s\n", err.c_str());
    return 1;
  }

  std::vector<ThermalZoneReading> therm_before = read_thermal_zones();
  BaremetalSession sess;
  bool exec_ok = false;
  std::string stage = "open";
  uint32_t model_handle = 0;
  double mean_ms = 0.0;
  std::vector<double> latencies_ms;
  std::vector<uint64_t> hw_us_list;
  uint64_t last_hw_us = 0;
  std::vector<uint8_t> out0_bytes(static_cast<size_t>(model.output_sizes[0]), 0);
  bool out_dump_ok = false;
  std::string out_dump_err;

  do {
    if (!sess.open_and_init(&err)) break;

    const uint32_t map_flags = EDGETPU_MAP_DMA_BIDIRECTIONAL | EDGETPU_MAP_36BIT_IOVA;
    stage = "alloc_model_bufs";

    DmaBufRegion scratch_buf, image_buf, vm_ir_buf, params_buf, reg_cmd_buf, exec_cmd_buf;
    size_t scratch_sz = model.dive_scratch_size > 0 ? static_cast<size_t>(model.dive_scratch_size) : 4096u;
    if (!scratch_buf.init(sess.fd, scratch_sz, map_flags, &err) ||
        !scratch_buf.zero_fill(&err)) break;

    if (!image_buf.init(sess.fd, static_cast<size_t>(model.ext_dive_image.size), map_flags, &err) ||
        !image_buf.upload(dgc0_blob.data() + model.ext_dive_image.offset,
                          static_cast<size_t>(model.ext_dive_image.size), &err)) break;

    if (!vm_ir_buf.init(sess.fd, static_cast<size_t>(model.ext_dive_vm_ir.size), map_flags, &err) ||
        !vm_ir_buf.upload(dgc0_blob.data() + model.ext_dive_vm_ir.offset,
                          static_cast<size_t>(model.ext_dive_vm_ir.size), &err)) break;

    struct litebuf_dive_buffer_slice param_slice{};
    uint32_t param_count = 0;
    if (model.has_params) {
      if (!params_buf.init(sess.fd, static_cast<size_t>(model.ext_params.size), map_flags, &err) ||
          !params_buf.upload(dgc0_blob.data() + model.ext_params.offset,
                             static_cast<size_t>(model.ext_params.size), &err)) break;
      param_slice.address = params_buf.iova;
      param_slice.size_bytes = model.ext_params.size;
      param_count = 1;
    }

    // Build & submit RegisterModel
    stage = "register";
    uint8_t inner_reg[256]{};
    size_t inner_reg_len = litebuf_build_dive_register_interpreter(
        inner_reg, sizeof(inner_reg), vm_ir_buf.iova, model.ext_dive_vm_ir.size,
        model.has_params ? &param_slice : nullptr, param_count, 0, 0);
    if (inner_reg_len == 0) { err = "Failed building inner RegisterInterpreterModel"; break; }

    if (!reg_cmd_buf.init(sess.fd, inner_reg_len, map_flags, &err) ||
        !reg_cmd_buf.upload(inner_reg, inner_reg_len, &err)) break;

    struct litebuf_register_model_params rp{};
    rp.dive_cmd_address = reg_cmd_buf.iova;
    rp.dive_cmd_size_bytes = static_cast<uint32_t>(inner_reg_len);
    rp.dive_image_address = image_buf.iova;
    rp.dive_image_size_bytes = static_cast<uint32_t>(model.ext_dive_image.size);
    rp.dive_scratch_address = scratch_buf.iova;
    rp.dive_scratch_size_bytes = static_cast<uint32_t>(model.dive_scratch_size);
    rp.dive_heap_size = model.dive_heap_size;
    rp.exec_field_29 = model.exec_field_29;
    rp.core_session_handle = 0;
    rp.cluster_id = model.cluster_id;
    rp.exec_field_31 = model.exec_field_31;
    rp.exec_field_14 = model.exec_field_14;
    rp.compiler_cl = model.compiler_cl;
    rp.subgraph_flag_18 = model.subgraph_flag_18;
    rp.exec_field_34 = model.exec_field_34;
    rp.model_id = "Unknown";

    uint8_t outer_reg[256]{};
    size_t outer_reg_len = litebuf_build_vii_register_model(outer_reg, sizeof(outer_reg), &rp);
    if (outer_reg_len == 0) { err = "Failed building outer RegisterModel"; break; }

    struct litebuf_vii_parsed_response reg_resp{};
    uint16_t resp_code = 0;
    if (!sess.submit_and_wait(outer_reg, outer_reg_len, &reg_resp, &resp_code, &err)) break;
    if (reg_resp.response_type != LITEBUF_VII_RESP_REGISTER_MODEL) {
      err = "Expected REGISTER_MODEL response type=2, got " + std::to_string(reg_resp.response_type);
      break;
    }
    model_handle = reg_resp.handle;

    // Allocate & map I/O buffers + inner Execute command DMA buffer
    stage = "alloc_io_bufs";
    std::vector<DmaBufRegion> in_bufs(input_blobs.size());
    std::vector<DmaBufRegion> out_bufs(model.output_sizes.size());
    std::vector<struct litebuf_dive_buffer_slice> in_slices(input_blobs.size());
    std::vector<struct litebuf_dive_buffer_slice> out_slices(model.output_sizes.size());
    bool io_ok = true;
    for (size_t i = 0; i < input_blobs.size(); ++i) {
      if (!in_bufs[i].init(sess.fd, input_blobs[i].size(), map_flags, &err) ||
          !in_bufs[i].upload(input_blobs[i].data(), input_blobs[i].size(), &err)) {
        io_ok = false;
        break;
      }
      in_slices[i] = {in_bufs[i].iova, static_cast<uint64_t>(input_blobs[i].size())};
    }
    if (!io_ok) break;
    for (size_t i = 0; i < model.output_sizes.size(); ++i) {
      if (!out_bufs[i].init(sess.fd, static_cast<size_t>(model.output_sizes[i]), map_flags, &err) ||
          !out_bufs[i].zero_fill(&err)) {
        io_ok = false;
        break;
      }
      out_slices[i] = {out_bufs[i].iova, model.output_sizes[i]};
    }
    if (!io_ok) break;

    uint8_t inner_exec[512]{};
    size_t inner_exec_len = litebuf_build_dive_execute(
        inner_exec, sizeof(inner_exec),
        in_slices.data(), static_cast<uint32_t>(in_slices.size()),
        out_slices.data(), static_cast<uint32_t>(out_slices.size()),
        model.has_params ? &param_slice : nullptr, param_count);
    if (inner_exec_len == 0) { err = "Failed building inner DIVE Execute"; break; }
    if (!exec_cmd_buf.init(sess.fd, inner_exec_len, map_flags, &err) ||
        !exec_cmd_buf.upload(inner_exec, inner_exec_len, &err)) break;

    uint8_t outer_exec[128]{};
    size_t outer_exec_len = litebuf_build_vii_execute(
        outer_exec, sizeof(outer_exec),
        exec_cmd_buf.iova, static_cast<uint32_t>(inner_exec_len), model_handle,
        LITEBUF_DEFAULT_CORE_POWER_STATE, LITEBUF_DEFAULT_MEM_POWER_STATE,
        LITEBUF_DEFAULT_QOS_CLASS, 1u, model.cluster_id, LITEBUF_CORE_SESSION_NONE, 0u);
    if (outer_exec_len == 0) { err = "Failed building outer Execute"; break; }

    // Execute N iterations
    stage = "execute";
    double total_ms = 0.0;
    for (int it = 0; it < args.iters; ++it) {
      struct timespec t0{}, t1{};
      ::clock_gettime(CLOCK_MONOTONIC, &t0);
      struct litebuf_vii_parsed_response exec_resp{};
      if (!sess.submit_and_wait(outer_exec, outer_exec_len, &exec_resp, &resp_code, &err)) {
        io_ok = false;
        break;
      }
      ::clock_gettime(CLOCK_MONOTONIC, &t1);
      double ms = monotonic_ms_since(t0, t1);
      if (exec_resp.response_type != LITEBUF_VII_RESP_EXECUTE) {
        err = "Expected EXECUTE response type=1, got " + std::to_string(exec_resp.response_type);
        io_ok = false;
        break;
      }
      latencies_ms.push_back(ms);
      total_ms += ms;
      last_hw_us = exec_resp.execution_us;
      hw_us_list.push_back(last_hw_us);
    }
    if (!io_ok) break;
    mean_ms = total_ms / static_cast<double>(args.iters);

    stage = "download";
    if (!out_bufs[0].download(out0_bytes.data(), out0_bytes.size(), &err)) break;

    if (!args.out_dump_path.empty()) {
      out_dump_ok = write_binary_file(args.out_dump_path, out0_bytes.data(), out0_bytes.size(), out_dump_err);
    }

    // Unregister model cleanly
    stage = "unregister";
    uint8_t outer_unreg[64]{};
    size_t outer_unreg_len = litebuf_build_vii_unregister_model(outer_unreg, sizeof(outer_unreg), model_handle);
    struct litebuf_vii_parsed_response unreg_resp{};
    if (!sess.submit_and_wait(outer_unreg, outer_unreg_len, &unreg_resp, &resp_code, &err)) break;

    stage = "done";
    exec_ok = true;
  } while (false);

  std::vector<ThermalZoneReading> therm_after = read_thermal_zones();
  ProcMapsAudit maps_audit = audit_proc_maps();

  bool verify_ok = false;
  int max_abs_err = -1;
  size_t exact_matches = 0;
  size_t total_elems = out0_bytes.size();
  if (exec_ok && has_ref) {
    if (ref_blob.size() == out0_bytes.size()) {
      max_abs_err = 0;
      for (size_t i = 0; i < ref_blob.size(); ++i) {
        int a = static_cast<int8_t>(out0_bytes[i]);
        int b = static_cast<int8_t>(ref_blob[i]);
        int diff = a > b ? (a - b) : (b - a);
        if (diff > max_abs_err) max_abs_err = diff;
        if (diff == 0) ++exact_matches;
      }
      verify_ok = (max_abs_err == 0);
    } else {
      err = "Ref size mismatch: out=" + std::to_string(out0_bytes.size()) +
            " ref=" + std::to_string(ref_blob.size());
    }
  }

  std::ostringstream js;
  js << "{\n"
     << "  \"mode\": \"baremetal-vii\",\n"
     << "  \"dgc0\": " << json_str(args.dgc0_path) << ",\n"
     << "  \"dgc0_bytes\": " << dgc0_blob.size() << ",\n"
     << "  \"input_sizes\": [";
  for (size_t i = 0; i < model.input_sizes.size(); ++i) {
    js << (i ? "," : "") << model.input_sizes[i];
  }
  js << "],\n  \"output_sizes\": [";
  for (size_t i = 0; i < model.output_sizes.size(); ++i) {
    js << (i ? "," : "") << model.output_sizes[i];
  }
  js << "],\n"
     << "  \"dive_scratch_bytes\": " << model.dive_scratch_size << ",\n"
     << "  \"has_params\": " << (model.has_params ? "true" : "false") << ",\n"
     << "  \"fw_version\": {\"major\": " << sess.fw_ver.major_version
     << ", \"minor\": " << sess.fw_ver.minor_version
     << ", \"vii\": " << sess.fw_ver.vii_version
     << ", \"kci\": " << sess.fw_ver.kci_version << "},\n"
     << "  \"model_handle\": " << model_handle << ",\n"
     << "  \"execute\": {\n"
     << "    \"ok\": " << (exec_ok ? "true" : "false") << ",\n"
     << "    \"stage\": " << json_str(stage) << ",\n"
     << "    \"error\": " << json_str(err) << ",\n"
     << "    \"iters\": " << args.iters << ",\n"
     << "    \"mean_ms\": " << mean_ms << ",\n"
     << "    \"hw_execution_time_us\": " << last_hw_us << ",\n"
     << "    \"hw_us_per_iter\": [";
  for (size_t i = 0; i < hw_us_list.size(); ++i) {
    js << (i ? "," : "") << hw_us_list[i];
  }
  js << "],\n    \"latencies_ms\": [";
  for (size_t i = 0; i < latencies_ms.size(); ++i) {
    js << (i ? "," : "") << latencies_ms[i];
  }
  js << "]\n  },\n";

  if (!args.out_dump_path.empty()) {
    js << "  \"out_dump\": {\"path\": " << json_str(args.out_dump_path)
       << ", \"ok\": " << (out_dump_ok ? "true" : "false")
       << ", \"error\": " << json_str(out_dump_err) << "},\n";
  }

  js << "  \"verify\": {\n"
     << "    \"checked\": " << (has_ref ? "true" : "false") << ",\n"
     << "    \"ok\": " << (verify_ok ? "true" : "false") << ",\n"
     << "    \"max_abs_err\": " << max_abs_err << ",\n"
     << "    \"exact_matches\": " << exact_matches << ",\n"
     << "    \"total_elements\": " << total_elems << "\n"
     << "  },\n"
     << "  \"maps_has_google_tpu_libs\": " << (maps_audit.maps_has_google_tpu_libs ? "true" : "false") << ",\n"
     << "  \"mapped_libs\": [";
  for (size_t i = 0; i < maps_audit.mapped_libs.size(); ++i) {
    js << (i ? "," : "") << json_str(maps_audit.mapped_libs[i]);
  }
  js << "],\n"
     << "  \"thermal_deltas\": [";
  bool first_tz = true;
  for (const auto& after : therm_after) {
    for (const auto& before : therm_before) {
      if (after.name == before.name && after.temp_ok && before.temp_ok) {
        if (!first_tz) js << ",";
        first_tz = false;
        js << "{\"zone\":" << json_str(after.name)
           << ",\"type\":" << json_str(after.type)
           << ",\"before_mc\":" << before.temp_millic
           << ",\"after_mc\":" << after.temp_millic
           << ",\"delta_mc\":" << (after.temp_millic - before.temp_millic) << "}";
        break;
      }
    }
  }
  js << "]\n}\n";

  if (!args.json_only) {
    std::fprintf(stderr,
        "tpu_baremetal_runner: ok=%d stage=%s handle=0x%x mean_ms=%.3f hw_us=%llu "
        "verify=%d/%zu max_abs_err=%d maps_google=%d err=%s\n",
        exec_ok ? 1 : 0, stage.c_str(), model_handle, mean_ms,
        static_cast<unsigned long long>(last_hw_us),
        static_cast<int>(exact_matches), total_elems, max_abs_err,
        maps_audit.maps_has_google_tpu_libs ? 1 : 0, err.c_str());
  }
  std::printf("%s", js.str().c_str());

  bool all_ok = exec_ok && !maps_audit.maps_has_google_tpu_libs && (!has_ref || verify_ok);
  return all_ok ? 0 : 1;
}
