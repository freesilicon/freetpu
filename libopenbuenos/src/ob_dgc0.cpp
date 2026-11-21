#include "ob_internal.h"

#include <cstring>

namespace ob {

namespace {

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
    if (!read(vp, &len) || len > 1000000u ||
        !in_bounds(vp + 4, static_cast<size_t>(len) * 4u)) {
      return false;
    }
    elems->reserve(len);
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
    if (!read(vp, &len) || len > 1000000u ||
        !in_bounds(vp + 4, static_cast<size_t>(len) * 4u)) {
      return false;
    }
    vals->reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
      int32_t v = 0;
      read(vp + 4u + static_cast<size_t>(i) * 4u, &v);
      vals->push_back(v);
    }
    return true;
  }
};

size_t darwinn_dtype_bytes(uint8_t dtype) {
  switch (dtype) {
    case 1:
    case 2:
    case 8:
      return 1;
    case 3:
    case 5:
      return 2;
    case 4:
    case 6:
      return 4;
    case 7:
      return 8;
    default:
      return 1;
  }
}

}  // namespace

ob_status_t parse_dgc0_buffer(const uint8_t* data, size_t size,
                              ParsedDgc0* out, std::string* err) {
  if (!data || !out || size < 16u || std::memcmp(data + 4, "DGC0", 4) != 0) {
    if (err) *err = "Buffer is not a valid DGC0 container";
    return OB_ERR_BAD_DGC0;
  }

  *out = ParsedDgc0{};
  FbView dgc{data, size};
  uint32_t dgc_root = 0;
  if (!dgc.read(0, &dgc_root) || dgc_root >= size) {
    if (err) *err = "Invalid DGC0 root offset";
    return OB_ERR_BAD_DGC0;
  }

  std::vector<size_t> ext_tables;
  if (!dgc.uoffset_vector_field(dgc_root, 7, &ext_tables) || ext_tables.size() < 2u) {
    if (err) *err = "Missing DGC0 external_buffers (< 2 entries)";
    return OB_ERR_BAD_DGC0;
  }

  auto read_ext = [&](size_t idx, Dgc0Slice* s) -> bool {
    s->offset = dgc.field_u64(ext_tables[idx], 0, 0);
    s->size = dgc.field_u64(ext_tables[idx], 1, 0);
    return s->size > 0 && s->offset <= size && s->size <= size - s->offset;
  };
  if (!read_ext(0, &out->ext_dive_vm_ir) || !read_ext(1, &out->ext_dive_image)) {
    if (err) *err = "Out-of-bounds DGC0 external buffer ext[0]/ext[1]";
    return OB_ERR_BAD_DGC0;
  }
  if (ext_tables.size() >= 3u) {
    if (!read_ext(2, &out->ext_params)) {
      if (err) *err = "Out-of-bounds DGC0 external buffer ext[2]";
      return OB_ERR_BAD_DGC0;
    }
    out->has_params = true;
  }

  FbView dws{};
  if (!dgc.byte_vector_field(dgc_root, 0, &dws) || dws.size < 16u ||
      std::memcmp(dws.data + 4, "DWS0", 4) != 0) {
    if (err) *err = "Missing or invalid nested Package DWS0 in DGC0 slot 0";
    return OB_ERR_BAD_DGC0;
  }

  uint32_t pkg_root = 0;
  if (!dws.read(0, &pkg_root) || pkg_root >= dws.size) {
    if (err) *err = "Invalid DWS0 root offset";
    return OB_ERR_BAD_DGC0;
  }

  out->compiler_cl = dws.field_u32(pkg_root, 10, 0);
  if (out->compiler_cl == 0) {
    out->compiler_cl = dgc.field_u64(dgc_root, 5, 0);
  }

  std::vector<size_t> subgraphs;
  if (!dws.uoffset_vector_field(pkg_root, 3, &subgraphs) || subgraphs.empty()) {
    if (err) *err = "No SubGraphs in DWS0";
    return OB_ERR_BAD_DGC0;
  }
  size_t sg = subgraphs[0];
  out->subgraph_flag_18 = dws.field_u8(sg, 18, 0);

  std::vector<size_t> tensors;
  if (dws.uoffset_vector_field(sg, 0, &tensors)) {
    for (size_t tp : tensors) {
      uint8_t storage = dws.field_u8(tp, 5, 0);
      if (storage == 3u) {  // SCRATCHPAD
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

  std::vector<size_t> ops;
  if (!dws.uoffset_vector_field(sg, 3, &ops) || ops.empty()) {
    if (err) *err = "No operations in SubGraph[0]";
    return OB_ERR_BAD_DGC0;
  }

  size_t op_call = 0;
  bool found_call = false;
  for (size_t op : ops) {
    if (dws.field_u8(op, 0, 0) == 2u) {  // OpExecutableCall
      size_t fp = 0;
      if (dws.table_field(op, 1, &fp) && dws.follow_uoffset(fp, &op_call)) {
        found_call = true;
        break;
      }
    }
  }
  if (!found_call) {
    if (err) *err = "No OpExecutableCall (detail_type=2) in SubGraph[0]";
    return OB_ERR_BAD_DGC0;
  }

  FbView mexec{};
  if (!dws.byte_vector_field(op_call, 0, &mexec) || mexec.size < 8u) {
    if (err) *err = "Missing nested MultiExecutable in OpExecutableCall";
    return OB_ERR_BAD_DGC0;
  }
  uint32_t me_root = 0;
  if (!mexec.read(0, &me_root) || me_root >= mexec.size) {
    if (err) *err = "Invalid MultiExecutable root offset";
    return OB_ERR_BAD_DGC0;
  }
  std::vector<size_t> execs;
  if (!mexec.uoffset_vector_field(me_root, 1, &execs) || execs.empty()) {
    if (err) *err = "No Executables in MultiExecutable";
    return OB_ERR_BAD_DGC0;
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
  read_io_sizes(3, &out->input_sizes);
  read_io_sizes(4, &out->output_sizes);

  if (out->input_sizes.empty() || out->output_sizes.empty()) {
    if (err) *err = "Executable missing input or output descriptor";
    return OB_ERR_BAD_DGC0;
  }
  if (out->input_sizes.size() > OB_MAX_GRAPH_IO ||
      out->output_sizes.size() > OB_MAX_GRAPH_IO) {
    if (err) *err = "Executable exceeds OB_MAX_GRAPH_IO";
    return OB_ERR_BAD_DGC0;
  }

  out->exec_field_14 = mexec.field_u32(ex, 14, 0);
  out->dive_heap_size = static_cast<int32_t>(mexec.field_i64(ex, 28, 0));
  out->exec_field_29 = static_cast<int32_t>(mexec.field_i64(ex, 29, 0));
  out->exec_field_31 = mexec.field_u8(ex, 31, 0);
  out->exec_field_34 = mexec.field_u8(ex, 34, 0);

  uint32_t hw_cluster = mexec.field_u32(ex, 8, LITEBUF_DEFAULT_CLUSTER_ID);
  out->cluster_id = static_cast<uint8_t>(hw_cluster & 0xffu);
  return OB_OK;
}

void fill_dgc0_info(const ParsedDgc0& parsed, uint32_t model_handle,
                    bool cache_hit, ob_dgc0_info_t* out) {
  if (!out) return;
  std::memset(out, 0, sizeof(*out));
  out->num_inputs = static_cast<uint32_t>(parsed.input_sizes.size());
  out->num_outputs = static_cast<uint32_t>(parsed.output_sizes.size());
  for (uint32_t i = 0; i < out->num_inputs && i < OB_MAX_GRAPH_IO; ++i) {
    out->input_sizes[i] = parsed.input_sizes[i];
  }
  for (uint32_t i = 0; i < out->num_outputs && i < OB_MAX_GRAPH_IO; ++i) {
    out->output_sizes[i] = parsed.output_sizes[i];
  }
  out->dive_scratch_size = parsed.dive_scratch_size;
  out->ext_vm_ir_size = parsed.ext_dive_vm_ir.size;
  out->ext_image_size = parsed.ext_dive_image.size;
  out->ext_params_size = parsed.has_params ? parsed.ext_params.size : 0u;
  out->compiler_cl = parsed.compiler_cl;
  out->model_handle = model_handle;
  out->cluster_id = parsed.cluster_id;
  out->has_params = parsed.has_params ? 1u : 0u;
  out->cache_hit = cache_hit ? 1u : 0u;
}

}  // namespace ob

extern "C" {

ob_status_t ob_dgc0_inspect(const void* dgc0_bytes,
                            size_t dgc0_size,
                            ob_dgc0_info_t* out_info) {
  if (!dgc0_bytes || dgc0_size == 0 || !out_info) return OB_ERR_INVALID_ARG;
  ob::ParsedDgc0 parsed;
  std::string err;
  ob_status_t st = ob::parse_dgc0_buffer(
      static_cast<const uint8_t*>(dgc0_bytes), dgc0_size, &parsed, &err);
  if (st != OB_OK) return st;
  ob::fill_dgc0_info(parsed, 0u, false, out_info);
  return OB_OK;
}

}  // extern "C"
