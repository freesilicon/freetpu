#include "ob_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "edgetpu_buenos_uapi.h"
#endif

namespace ob {

namespace {

// Pre-allocate one 4 KiB inner-DIVE Execute command DMA-BUF per credit slot.
ob_status_t init_credit_slots(ob_context* ctx) {
  for (uint32_t i = 0; i < ctx->credit_size; ++i) {
    ctx->credits[i] = CreditSlot{};
    ob_status_t st = ob_buffer_alloc(ctx, 4096u,
                                     OB_BUF_BIDIRECTIONAL | OB_BUF_NO_POOL | OB_BUF_ZERO_INIT,
                                     &ctx->credits[i].exec_cmd_buf);
    if (st != OB_OK) {
      for (uint32_t j = 0; j < i; ++j) {
        ob_buffer_free(ctx->credits[j].exec_cmd_buf);
        ctx->credits[j].exec_cmd_buf = nullptr;
      }
      return st;
    }
  }
  return OB_OK;
}

void destroy_credit_slots(ob_context* ctx) {
  for (uint32_t i = 0; i < OB_MAX_CREDITS; ++i) {
    if (ctx->credits[i].exec_cmd_buf) {
      ob_buffer_free(ctx->credits[i].exec_cmd_buf);
      ctx->credits[i].exec_cmd_buf = nullptr;
    }
    ctx->credits[i].state = CreditSlot::FREE;
    ctx->credits[i].handle = 0;
    ctx->credits[i].graph_entry = nullptr;
  }
  ctx->inflight_count = 0;
}

uint32_t alloc_seq_locked(ob_context* ctx) {
  // buenos.ko requires seq >> 32 == 0 (32-bit sequence number).
  uint32_t s = ctx->next_seq;
  ctx->next_seq = (ctx->next_seq + 1u) & 0x7fffffffu;
  return s;
}

#if defined(__linux__)
// Pop one VII response from kernel and either deliver to `target_seq` or stash in
// the matching in-flight CreditSlot (handles out-of-order completions safely).
// Returns:
//   1  = matched target_seq (written to target_resp / target_code)
//   0  = popped and stashed another slot's response (caller should keep draining)
//  -1  = queue empty (ENOENT / EAGAIN / EINTR)
//  -2  = fatal ioctl error (out_err populated)
int pop_and_demux_one_locked(ob_context* ctx,
                             bool has_target,
                             uint32_t target_seq,
                             struct litebuf_vii_parsed_response* target_resp,
                             uint16_t* target_code,
                             std::string* out_err) {
  uint8_t resp_buf[LITEBUF_VII_RESPONSE_BYTES]{};
  struct edgetpu_vii_litebuf_response_ioctl r {};
  r.litebuf_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resp_buf));

  if (::ioctl(ctx->tpu_fd, EDGETPU_VII_LITEBUF_RESPONSE, &r) != 0) {
    int e = errno;
    if (e == ENOENT || e == EAGAIN || e == EINTR) return -1;
    if (out_err) {
      *out_err = "EDGETPU_VII_LITEBUF_RESPONSE failed errno=" + std::to_string(e);
    }
    return -2;
  }

  struct litebuf_vii_parsed_response parsed =
      litebuf_parse_vii_response(resp_buf, sizeof(resp_buf));
  uint64_t now_ns = monotonic_now_ns();

  if (has_target && r.seq == static_cast<uint64_t>(target_seq)) {
    if (target_resp) *target_resp = parsed;
    if (target_code) *target_code = r.code;
    return 1;
  }

  // Demux into matching in-flight async credit slot
  for (uint32_t i = 0; i < ctx->credit_size; ++i) {
    CreditSlot& slot = ctx->credits[i];
    if (slot.state == CreditSlot::IN_FLIGHT &&
        static_cast<uint64_t>(slot.seq) == r.seq) {
      slot.complete_time_ns = now_ns;
      slot.vii_resp_code = r.code;
      if (r.code != 0) {
        slot.state = CreditSlot::FAILED;
        slot.error_status = OB_ERR_FW_ERROR;
        slot.error_msg = "Firmware returned non-zero VII code=" + std::to_string(r.code);
      } else if (!parsed.valid) {
        slot.state = CreditSlot::FAILED;
        slot.error_status = OB_ERR_FW_ERROR;
        slot.error_msg = "Invalid ViiResponse Litebuf payload";
      } else if (parsed.response_type == LITEBUF_VII_RESP_ERROR) {
        slot.state = CreditSlot::FAILED;
        slot.error_status = OB_ERR_FW_ERROR;
        slot.error_msg = "ViiResponse ERROR code=" + std::to_string(parsed.error_code);
      } else if (parsed.response_type != LITEBUF_VII_RESP_EXECUTE) {
        slot.state = CreditSlot::FAILED;
        slot.error_status = OB_ERR_FW_ERROR;
        slot.error_msg = "Unexpected ViiResponse type=" + std::to_string(parsed.response_type);
      } else {
        slot.state = CreditSlot::COMPLETED;
        slot.hw_execution_us = parsed.execution_us;
        slot.error_status = OB_OK;
      }
      return 0;
    }
  }
  return 0;
}

ob_status_t poll_eventfds_locked(ob_context* ctx, uint64_t deadline_ns,
                                 std::string* err) {
  uint64_t now_ns = monotonic_now_ns();
  if (now_ns >= deadline_ns) {
    if (err) *err = "Timed out waiting for VII response";
    return OB_ERR_TIMEOUT;
  }
  int rem_ms = static_cast<int>((deadline_ns - now_ns) / 1000000ULL);
  if (rem_ms < 1) rem_ms = 1;

  struct pollfd pfds[2] = {{ctx->resp_efd, POLLIN, 0}, {ctx->err_efd, POLLIN, 0}};
  int prc = ::poll(pfds, 2, rem_ms);
  if (prc < 0) {
    if (errno == EINTR) return OB_OK;
    if (err) *err = "poll(resp_efd) failed errno=" + std::to_string(errno);
    return OB_ERR_IOCTL;
  }
  if (prc == 0) {
    if (err) *err = "poll(resp_efd) timed out";
    return OB_ERR_TIMEOUT;
  }
  if (pfds[1].revents & POLLIN) {
    uint64_t val = 0;
    if (::read(ctx->err_efd, &val, sizeof(val)) > 0) {
      if (err) *err = "EdgeTPU fatal error eventfd signaled";
      return OB_ERR_DEVICE;
    }
  }
  if (pfds[0].revents & POLLIN) {
    uint64_t val = 0;
    ssize_t unused = ::read(ctx->resp_efd, &val, sizeof(val));
    (void)unused;
  }
  return OB_OK;
}
#endif

}  // namespace

ob_status_t device_open(ob_context* ctx) {
  if (ctx->mock_mode) {
    ctx->credit_size =
        (ctx->cfg.max_inflight > 0 && ctx->cfg.max_inflight <= OB_MAX_CREDITS)
            ? ctx->cfg.max_inflight
            : OB_MAX_CREDITS;
    ctx->fw_major = 1u;
    ctx->fw_minor = 1u;
    ctx->fw_vii = 3u;
    ctx->fw_kci = 3u;
    return init_credit_slots(ctx);
  }

#if defined(__linux__)
  ctx->tpu_fd = ::open(ctx->device_path.c_str(), O_RDWR | O_CLOEXEC);
  if (ctx->tpu_fd < 0) {
    set_last_error(ctx, "open(" + ctx->device_path + ") failed errno=" +
                            std::to_string(errno));
    return OB_ERR_DEVICE;
  }
  ctx->dma_heap_fd = ::open(ctx->dma_heap_path.c_str(), O_RDONLY | O_CLOEXEC);

  uint64_t dev_credits = OB_MAX_CREDITS;
  if (::ioctl(ctx->tpu_fd, EDGETPU_GET_CREDIT_SIZE, &dev_credits) != 0) {
    dev_credits = 8u;  // AOSP fallback if ENOTTY
  }
  if (dev_credits == 0u || dev_credits > OB_MAX_CREDITS) {
    dev_credits = OB_MAX_CREDITS;
  }
  uint32_t req_credits =
      (ctx->cfg.max_inflight > 0) ? ctx->cfg.max_inflight : static_cast<uint32_t>(dev_credits);
  ctx->credit_size = std::min<uint32_t>(req_credits, static_cast<uint32_t>(dev_credits));
  if (ctx->credit_size < 1u) ctx->credit_size = 1u;

  struct edgetpu_mailbox_attr attr {};
  attr.cmd_queue_size = 24;
  attr.resp_queue_size = 12;
  attr.sizeof_cmd = 48;
  attr.sizeof_resp = 24;
  attr.priority = 8;
  attr.cmdq_tail_doorbell = 1;
  attr.partition_type = 0;
  attr.client_priv = 1;
  if (::ioctl(ctx->tpu_fd, EDGETPU_CREATE_GROUP, &attr) != 0) {
    set_last_error(ctx, "EDGETPU_CREATE_GROUP failed errno=" + std::to_string(errno));
    device_close(ctx);
    return OB_ERR_IOCTL;
  }

  ctx->resp_efd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ctx->err_efd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (ctx->resp_efd < 0 || ctx->err_efd < 0) {
    set_last_error(ctx, "eventfd creation failed errno=" + std::to_string(errno));
    device_close(ctx);
    return OB_ERR_DEVICE;
  }

  struct edgetpu_event_register ev0 {
      EDGETPU_EVENT_RESPDATA, static_cast<__u32>(ctx->resp_efd)};
  if (::ioctl(ctx->tpu_fd, EDGETPU_SET_EVENTFD, &ev0) != 0) {
    set_last_error(ctx, "EDGETPU_SET_EVENTFD(RESPDATA) failed errno=" +
                            std::to_string(errno));
    device_close(ctx);
    return OB_ERR_IOCTL;
  }
  struct edgetpu_event_register ev1 {
      EDGETPU_EVENT_FATAL_ERROR, static_cast<__u32>(ctx->err_efd)};
  ::ioctl(ctx->tpu_fd, EDGETPU_SET_EVENTFD, &ev1);

  if (::ioctl(ctx->tpu_fd, EDGETPU_ACQUIRE_WAKE_LOCK) != 0) {
    set_last_error(ctx, "EDGETPU_ACQUIRE_WAKE_LOCK failed errno=" +
                            std::to_string(errno));
    device_close(ctx);
    return OB_ERR_IOCTL;
  }
  ctx->wake_locked = true;

  struct edgetpu_fw_version fw {};
  if (::ioctl(ctx->tpu_fd, EDGETPU_FIRMWARE_VERSION, &fw) == 0) {
    ctx->fw_major = fw.major_version;
    ctx->fw_minor = fw.minor_version;
    ctx->fw_vii = fw.vii_version;
    ctx->fw_kci = fw.kci_version;
  }

  ob_status_t st = init_credit_slots(ctx);
  if (st != OB_OK) {
    device_close(ctx);
    return st;
  }
  return OB_OK;
#else
  set_last_error(ctx, "Hardware /dev/edgetpu-soc requires Linux/Android (use mock_mode=1 on host)");
  return OB_ERR_UNSUPPORTED;
#endif
}

void device_release_fw_wakelock(ob_context* ctx) {
  if (!ctx || !ctx->fw_wakelock_held) return;
  uint8_t outer_wl[64]{};
  size_t wl_len = litebuf_build_vii_wakelock(
      outer_wl, sizeof(outer_wl), 0,
      ctx->fw_wakelock_core, ctx->fw_wakelock_mem, ctx->fw_wakelock_bitset);
  if (wl_len > 0u) {
    struct litebuf_vii_parsed_response resp {};
    uint16_t code = 0;
    device_submit_control(ctx, outer_wl, wl_len, &resp, &code, ctx->cfg.timeout_ms);
  }
  ctx->fw_wakelock_held = false;
  ctx->fw_wakelock_core = 0u;
  ctx->fw_wakelock_mem = 0u;
  ctx->fw_wakelock_bitset = 0u;
}

void device_sync_fw_wakelock(ob_context* ctx) {
  if (!ctx) return;
  const bool want_wl = ctx->fw_wakelock_enabled && (ctx->cfg.perf_mode >= 1u);
  const uint8_t cl =
      (ctx->cfg.cluster_id != 0u) ? ctx->cfg.cluster_id : LITEBUF_DEFAULT_CLUSTER_ID;
  const uint8_t bitset = static_cast<uint8_t>(1u << (cl & 7u));
  const uint8_t core_ps = ctx->cfg.core_power_state;
  const uint8_t mem_ps = ctx->cfg.mem_power_state;

  if (ctx->fw_wakelock_held && want_wl &&
      ctx->fw_wakelock_core == core_ps &&
      ctx->fw_wakelock_mem == mem_ps &&
      ctx->fw_wakelock_bitset == bitset) {
    return;
  }
  if (ctx->fw_wakelock_held) {
    device_release_fw_wakelock(ctx);
  }
  if (!want_wl) return;

  uint8_t outer_wl[64]{};
  size_t wl_len = litebuf_build_vii_wakelock(
      outer_wl, sizeof(outer_wl), 1, core_ps, mem_ps, bitset);
  if (wl_len == 0u) return;

  struct litebuf_vii_parsed_response resp {};
  uint16_t code = 0;
  ob_status_t st = device_submit_control(
      ctx, outer_wl, wl_len, &resp, &code, ctx->cfg.timeout_ms);
  if (st == OB_OK) {
    ctx->fw_wakelock_held = true;
    ctx->fw_wakelock_core = core_ps;
    ctx->fw_wakelock_mem = mem_ps;
    ctx->fw_wakelock_bitset = bitset;
  }
}

void device_close(ob_context* ctx) {
  if (!ctx) return;
  device_release_fw_wakelock(ctx);
  destroy_credit_slots(ctx);
#if defined(__linux__)
  if (!ctx->mock_mode) {
    if (ctx->wake_locked && ctx->tpu_fd >= 0) {
      ::ioctl(ctx->tpu_fd, EDGETPU_RELEASE_WAKE_LOCK);
      ctx->wake_locked = false;
    }
    if (ctx->err_efd >= 0) {
      ::close(ctx->err_efd);
      ctx->err_efd = -1;
    }
    if (ctx->resp_efd >= 0) {
      ::close(ctx->resp_efd);
      ctx->resp_efd = -1;
    }
    if (ctx->dma_heap_fd >= 0) {
      ::close(ctx->dma_heap_fd);
      ctx->dma_heap_fd = -1;
    }
    if (ctx->tpu_fd >= 0) {
      ::close(ctx->tpu_fd);
      ctx->tpu_fd = -1;
    }
  }
#endif
}

ob_status_t device_submit_control(ob_context* ctx,
                                  const void* outer_buf,
                                  size_t outer_len,
                                  struct litebuf_vii_parsed_response* parsed_out,
                                  uint16_t* resp_code_out,
                                  uint32_t timeout_ms) {
  if (!ctx || !outer_buf || outer_len < LITEBUF_HEADER_SIZE + sizeof(struct litebuf_vii_command_root)) {
    return OB_ERR_INVALID_ARG;
  }
  if (parsed_out) std::memset(parsed_out, 0, sizeof(*parsed_out));
  if (resp_code_out) *resp_code_out = 0;

  std::lock_guard<std::mutex> lk(ctx->vii_mu);
  const uint32_t seq = alloc_seq_locked(ctx);

  if (ctx->mock_mode) {
    // Validate outer Litebuf header & command root structurally
    const uint8_t* p = static_cast<const uint8_t*>(outer_buf);
    struct litebuf_header hdr {};
    struct litebuf_vii_command_root root {};
    std::memcpy(&hdr, p, sizeof(hdr));
    if (hdr.schema_id != LITEBUF_SCHEMA_RUNTIME_MESSAGES ||
        hdr.root_offset + sizeof(root) > outer_len) {
      set_last_error(ctx, "Mock control submit received malformed outer Litebuf header");
      return OB_ERR_INVALID_ARG;
    }
    std::memcpy(&root, p + hdr.root_offset, sizeof(root));
    if (parsed_out) {
      parsed_out->valid = 1;
      parsed_out->fw_status_word = seq;
      if (root.union_type == LITEBUF_VII_CMD_REGISTER_MODEL) {
        parsed_out->response_type = LITEBUF_VII_RESP_REGISTER_MODEL;
        parsed_out->handle = ctx->next_mock_model_handle++;
      } else if (root.union_type == LITEBUF_VII_CMD_UNREGISTER_MODEL) {
        parsed_out->response_type = LITEBUF_VII_RESP_VOID_OK;
      } else if (root.union_type == LITEBUF_VII_CMD_EXECUTE) {
        parsed_out->response_type = LITEBUF_VII_RESP_EXECUTE;
        parsed_out->execution_us = 100u;
      } else {
        parsed_out->response_type = LITEBUF_VII_RESP_VOID_OK;
      }
    }
    return OB_OK;
  }

#if defined(__linux__)
  struct edgetpu_vii_litebuf_command_ioctl cmd {};
  cmd.litebuf_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(outer_buf));
  cmd.litebuf_size = static_cast<__u32>(outer_len);
  cmd.seq = static_cast<uint64_t>(seq);

  while (true) {
    if (::ioctl(ctx->tpu_fd, EDGETPU_VII_LITEBUF_COMMAND, &cmd) == 0) break;
    if (errno == EINTR || errno == EAGAIN) continue;
    set_last_error(ctx, "EDGETPU_VII_LITEBUF_COMMAND(seq=" + std::to_string(seq) +
                            ") failed errno=" + std::to_string(errno));
    return OB_ERR_IOCTL;
  }

  uint32_t wait_ms = (timeout_ms > 0) ? timeout_ms : ctx->cfg.timeout_ms;
  uint64_t deadline_ns = monotonic_now_ns() + static_cast<uint64_t>(wait_ms) * 1000000ULL;

  while (true) {
    struct litebuf_vii_parsed_response parsed {};
    uint16_t code = 0;
    std::string err;
    int rc = pop_and_demux_one_locked(ctx, true, seq, &parsed, &code, &err);
    if (rc == 1) {
      if (resp_code_out) *resp_code_out = code;
      if (parsed_out) *parsed_out = parsed;
      if (code != 0) {
        set_last_error(ctx, "Firmware returned non-zero VII response code=" +
                                std::to_string(code));
        return OB_ERR_FW_ERROR;
      }
      if (!parsed.valid) {
        set_last_error(ctx, "Invalid ViiResponse Litebuf payload");
        return OB_ERR_FW_ERROR;
      }
      if (parsed.response_type == LITEBUF_VII_RESP_ERROR) {
        set_last_error(ctx, "ViiResponse ERROR code=" +
                                std::to_string(parsed.error_code));
        return OB_ERR_FW_ERROR;
      }
      return OB_OK;
    }
    if (rc == 0) {
      continue;  // Stashed an unrelated async slot response; keep draining
    }
    if (rc == -2) {
      set_last_error(ctx, err);
      return OB_ERR_IOCTL;
    }

    ob_status_t pst = poll_eventfds_locked(ctx, deadline_ns, &err);
    if (pst != OB_OK) {
      set_last_error(ctx, err);
      return pst;
    }
  }
#else
  (void)timeout_ms;
  return OB_ERR_UNSUPPORTED;
#endif
}

ob_status_t device_submit_execute_async(ob_context* ctx,
                                        CachedGraph* entry,
                                        ob_buffer* const* inputs,
                                        uint32_t num_inputs,
                                        ob_buffer* const* outputs,
                                        uint32_t num_outputs,
                                        ob_exec_handle_t* out_handle,
                                        const ob_buffer* override_params) {
  if (!ctx || !entry || !out_handle) return OB_ERR_INVALID_ARG;
  *out_handle = 0;

  if (num_inputs != entry->parsed.input_sizes.size() ||
      num_outputs != entry->parsed.output_sizes.size()) {
    set_last_error(ctx, "Input/output tensor count mismatch against .dgc0 executable");
    return OB_ERR_INVALID_ARG;
  }

  struct litebuf_dive_buffer_slice in_slices[OB_MAX_GRAPH_IO]{};
  struct litebuf_dive_buffer_slice out_slices[OB_MAX_GRAPH_IO]{};

  for (uint32_t i = 0; i < num_inputs; ++i) {
    if (!inputs || !inputs[i] || inputs[i]->ctx != ctx) return OB_ERR_INVALID_ARG;
    if (inputs[i]->requested_size < entry->parsed.input_sizes[i]) {
      set_last_error(ctx, "Input buffer[" + std::to_string(i) + "] smaller than .dgc0 requirement");
      return OB_ERR_INVALID_ARG;
    }
    in_slices[i].address = inputs[i]->slab.iova;
    in_slices[i].size_bytes = entry->parsed.input_sizes[i];
  }
  for (uint32_t i = 0; i < num_outputs; ++i) {
    if (!outputs || !outputs[i] || outputs[i]->ctx != ctx) return OB_ERR_INVALID_ARG;
    if (outputs[i]->requested_size < entry->parsed.output_sizes[i]) {
      set_last_error(ctx, "Output buffer[" + std::to_string(i) + "] smaller than .dgc0 requirement");
      return OB_ERR_INVALID_ARG;
    }
    out_slices[i].address = outputs[i]->slab.iova;
    out_slices[i].size_bytes = entry->parsed.output_sizes[i];
  }

  struct litebuf_dive_buffer_slice param_slice {};
  uint32_t param_count = 0;
  const ob_buffer* pbuf = override_params ? override_params : entry->params_buf;
  if (entry->parsed.has_params && pbuf) {
    param_slice.address = pbuf->slab.iova;
    param_slice.size_bytes = entry->parsed.ext_params.size;
    param_count = 1u;
  }

  uint8_t inner_exec[512]{};
  size_t inner_exec_len = litebuf_build_dive_execute(
      inner_exec, sizeof(inner_exec),
      in_slices, num_inputs,
      out_slices, num_outputs,
      param_count ? &param_slice : nullptr, param_count);
  if (inner_exec_len == 0u) {
    set_last_error(ctx, "Failed to serialize inner DIVE Execute Litebuf");
    return OB_ERR_OVERFLOW;
  }

  std::lock_guard<std::mutex> lk(ctx->vii_mu);
  if (ctx->inflight_count >= ctx->credit_size) {
    set_last_error(ctx, "All VII execution credits occupied (max_inflight=" +
                            std::to_string(ctx->credit_size) + ")");
    return OB_ERR_BUSY;
  }

  // Find a FREE credit slot
  CreditSlot* slot = nullptr;
  for (uint32_t i = 0; i < ctx->credit_size; ++i) {
    if (ctx->credits[i].state == CreditSlot::FREE) {
      slot = &ctx->credits[i];
      break;
    }
  }
  if (!slot || !slot->exec_cmd_buf) {
    return OB_ERR_BUSY;
  }

  ob_status_t wst = ob_buffer_write(slot->exec_cmd_buf, inner_exec, inner_exec_len, 0u);
  if (wst != OB_OK) return wst;

  uint8_t cluster = (ctx->cfg.cluster_id != 0u) ? ctx->cfg.cluster_id : entry->parsed.cluster_id;
  uint8_t outer_exec[128]{};
  // Positional wire bytes +0x1c..+0x1f (per docs/baremetal-vii/dvfs_power_states.md):
  //   +0x1c = fw_priority, +0x1d = tpu_core_power_state,
  //   +0x1e = memory_power_state, +0x1f = qos_class
  size_t outer_exec_len = litebuf_build_vii_execute(
      outer_exec, sizeof(outer_exec),
      slot->exec_cmd_buf->slab.iova, static_cast<uint32_t>(inner_exec_len),
      entry->model_handle,
      ctx->fw_priority, ctx->cfg.core_power_state,
      ctx->cfg.mem_power_state, ctx->cfg.qos_class, cluster,
      LITEBUF_CORE_SESSION_NONE, ctx->cfg.priority);
  if (outer_exec_len == 0u) {
    set_last_error(ctx, "Failed to serialize outer VII Execute Litebuf");
    return OB_ERR_OVERFLOW;
  }

  const uint32_t seq = alloc_seq_locked(ctx);
  const uint64_t now_ns = monotonic_now_ns();

  if (!ctx->mock_mode) {
#if defined(__linux__)
    struct edgetpu_vii_litebuf_command_ioctl cmd {};
    cmd.litebuf_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(outer_exec));
    cmd.litebuf_size = static_cast<__u32>(outer_exec_len);
    cmd.seq = static_cast<uint64_t>(seq);

    while (true) {
      if (::ioctl(ctx->tpu_fd, EDGETPU_VII_LITEBUF_COMMAND, &cmd) == 0) break;
      if (errno == EINTR) continue;
      if (errno == EAGAIN) {
        set_last_error(ctx, "EDGETPU_VII_LITEBUF_COMMAND returned EAGAIN (queue full)");
        return OB_ERR_BUSY;
      }
      set_last_error(ctx, "EDGETPU_VII_LITEBUF_COMMAND(seq=" + std::to_string(seq) +
                              ") failed errno=" + std::to_string(errno));
      return OB_ERR_IOCTL;
    }
#else
    return OB_ERR_UNSUPPORTED;
#endif
  }

  const ob_exec_handle_t handle = ctx->next_exec_handle++;
  slot->state = CreditSlot::IN_FLIGHT;
  slot->handle = handle;
  slot->seq = seq;
  slot->graph_entry = entry;
  slot->submit_time_ns = now_ns;
  slot->complete_time_ns = 0;
  slot->hw_execution_us = ctx->mock_mode ? 380u : 0u;
  slot->vii_resp_code = 0;
  slot->error_status = OB_OK;
  slot->error_msg.clear();
  ctx->inflight_count += 1u;

  *out_handle = handle;
  return OB_OK;
}

ob_status_t device_wait_handle(ob_context* ctx,
                               ob_exec_handle_t handle,
                               uint32_t timeout_ms,
                               ob_invoke_stats_t* out_stats) {
  if (!ctx || handle == 0u) return OB_ERR_INVALID_ARG;

  CachedGraph* completed_entry = nullptr;
  ob_status_t result = OB_OK;
  ob_invoke_stats_t stats {};

  {
    std::lock_guard<std::mutex> lk(ctx->vii_mu);
    CreditSlot* slot = nullptr;
    for (uint32_t i = 0; i < ctx->credit_size; ++i) {
      if (ctx->credits[i].state != CreditSlot::FREE &&
          ctx->credits[i].handle == handle) {
        slot = &ctx->credits[i];
        break;
      }
    }
    if (!slot) {
      set_last_error(ctx, "Unknown or already-completed ob_exec_handle_t");
      return OB_ERR_NOT_FOUND;
    }

    if (ctx->mock_mode) {
      slot->complete_time_ns = monotonic_now_ns();
      slot->state = CreditSlot::COMPLETED;
      slot->vii_resp_code = 0;
      slot->error_status = OB_OK;
    } else {
#if defined(__linux__)
      uint32_t wait_ms = (timeout_ms > 0) ? timeout_ms : ctx->cfg.timeout_ms;
      const uint64_t wait_start_ns = monotonic_now_ns();
      const uint64_t deadline_ns =
          wait_start_ns + static_cast<uint64_t>(wait_ms) * 1000000ULL;

      // Adaptive bounded user-space spin before poll(resp_efd):
      // eliminates 150-350 us kernel wake/schedule jitter when TPU finishes in 40-400 us
      // (covers both SmolLM2 ~266 us and Qwen2.5 SwiGLU ~215-350 us chunks),
      // while sleeping in poll() immediately for long ops (ema_hw_us >= 600 us) or spin_us == 0.
      const bool allow_spin = (ctx->spin_us > 0u) &&
                              (ctx->ema_hw_us == 0u || ctx->ema_hw_us < 600u);
      const uint64_t spin_deadline_ns = allow_spin
          ? std::min<uint64_t>(deadline_ns,
                               wait_start_ns + static_cast<uint64_t>(ctx->spin_us) * 1000ULL)
          : 0u;
      bool used_poll = false;
      uint32_t spin_iters = 0u;

      while (slot->state == CreditSlot::IN_FLIGHT) {
        std::string err;
        int rc = pop_and_demux_one_locked(ctx, false, 0u, nullptr, nullptr, &err);
        if (rc == 0) {
          continue;  // Keep draining any ready responses
        }
        if (rc == -2) {
          set_last_error(ctx, err);
          return OB_ERR_IOCTL;
        }
        if (slot->state != CreditSlot::IN_FLIGHT) break;

        if (allow_spin && !used_poll) {
          uint64_t now_ns = monotonic_now_ns();
          if (now_ns < spin_deadline_ns) {
            ++spin_iters;
            // Pace non-blocking ioctl probes every ~4 us to avoid kernel lock contention
            // with the IRQ handler while catching completion within <= 4 us.
            const uint64_t next_probe_ns = std::min<uint64_t>(spin_deadline_ns, now_ns + 4000ULL);
            while (monotonic_now_ns() < next_probe_ns) {
#if defined(__aarch64__)
              asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
              asm volatile("pause" ::: "memory");
#endif
            }
            continue;
          }
        }

        used_poll = true;
        ctx->prof.poll_waits += 1u;
        ob_status_t pst = poll_eventfds_locked(ctx, deadline_ns, &err);
        if (pst != OB_OK) {
          set_last_error(ctx, err);
          return pst;
        }
      }
      if (!used_poll && spin_iters > 0u) {
        ctx->prof.spin_hits += 1u;
      }
      if (slot->state == CreditSlot::COMPLETED && slot->hw_execution_us > 0u) {
        if (ctx->ema_hw_us == 0u) {
          ctx->ema_hw_us = slot->hw_execution_us;
        } else {
          ctx->ema_hw_us = (ctx->ema_hw_us * 3u + slot->hw_execution_us) >> 2u;
        }
      }
#else
      (void)timeout_ms;
      return OB_ERR_UNSUPPORTED;
#endif
    }

    // Harvest slot & release credit
    completed_entry = slot->graph_entry;
    stats.hw_execution_us = slot->hw_execution_us;
    uint64_t dt_ns = (slot->complete_time_ns >= slot->submit_time_ns)
                         ? (slot->complete_time_ns - slot->submit_time_ns)
                         : 0u;
    stats.wall_time_us = dt_ns / 1000ULL;
    stats.seq = slot->seq;
    stats.model_handle = completed_entry ? completed_entry->model_handle : 0u;
    stats.vii_resp_code = slot->vii_resp_code;
    stats.flags = ctx->mock_mode ? static_cast<uint16_t>(OB_STAT_FLAG_MOCK) : static_cast<uint16_t>(0u);
    stats.tiles_dispatched = 1u;

    result = (slot->state == CreditSlot::COMPLETED) ? OB_OK : slot->error_status;
    if (result != OB_OK && !slot->error_msg.empty()) {
      set_last_error(ctx, slot->error_msg);
    }

    slot->state = CreditSlot::FREE;
    slot->handle = 0;
    slot->seq = 0;
    slot->graph_entry = nullptr;
    slot->error_msg.clear();
    if (ctx->inflight_count > 0u) ctx->inflight_count -= 1u;
  }

  // Release graph ref outside vii_mu to respect cache_mu -> vii_mu lock order
  if (completed_entry) {
    std::lock_guard<std::mutex> clk(ctx->cache_mu);
    graph_entry_unref_locked(ctx, completed_entry);
  }

  if (out_stats) *out_stats = stats;
  return result;
}

ob_status_t device_wait_all(ob_context* ctx, uint32_t timeout_ms) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  ob_status_t first_err = OB_OK;
  while (true) {
    ob_exec_handle_t next_handle = 0;
    {
      std::lock_guard<std::mutex> lk(ctx->vii_mu);
      for (uint32_t i = 0; i < ctx->credit_size; ++i) {
        if (ctx->credits[i].state != CreditSlot::FREE) {
          next_handle = ctx->credits[i].handle;
          break;
        }
      }
    }
    if (next_handle == 0u) break;
    ob_status_t st = device_wait_handle(ctx, next_handle, timeout_ms, nullptr);
    if (st != OB_OK && first_err == OB_OK) first_err = st;
  }
  return first_err;
}

}  // namespace ob

namespace {

bool read_env_u8(const char* name, uint8_t* out) {
  const char* s = std::getenv(name);
  if (!s || !s[0]) return false;
  char* end = nullptr;
  unsigned long v = std::strtoul(s, &end, 0);
  if (end == s || v > 255ul) return false;
  *out = static_cast<uint8_t>(v);
  return true;
}

bool read_env_u32(const char* name, uint32_t* out) {
  const char* s = std::getenv(name);
  if (!s || !s[0]) return false;
  char* end = nullptr;
  unsigned long v = std::strtoul(s, &end, 0);
  if (end == s || v > 0xfffffffful) return false;
  *out = static_cast<uint32_t>(v);
  return true;
}

bool read_env_flag(const char* name, bool default_val) {
  const char* s = std::getenv(name);
  if (!s || !s[0]) return default_val;
  return (s[0] != '0');
}

struct ObPowerProfile {
  uint8_t fw_priority;           // Execute +0x1c (default 0x0b; 0x00..0x01 foreground peak)
  uint8_t tpu_core_power_state;  // Execute +0x1d & WakeLock +0x02 (0x03..0x19; >=0x1a rejected)
  uint8_t memory_power_state;    // Execute +0x1e & WakeLock +0x03 (0x01..0x15; >=0x16 rejected)
  uint8_t qos_class;             // Execute +0x1f ({0,1,2,4,5}; 3 invalid)
};

// Silicon-verified Tensor G5 (buenos_a0 / rango) power profiles (docs/baremetal-vii/dvfs_power_states.md):
//   0: MIN      (  38.4 MHz, sleep OPP, no FW WakeLock)
//   1: LOW      (  mid OPP,  + FW WakeLock)
//   2: BALANCED (1113.6 MHz, + FW WakeLock)
//   3: HIGH     (1113-1286 MHz, + FW WakeLock)
//   4: MAX      (1286.4 MHz, mem=0x0d, 983.6 INT8 GOPS on 64x1536x1536)
//   5: PEAK     (1286.4 MHz, mem=0x15, 27.59 GB/s, 152 us / 0.482 ms sync wall)
static constexpr ObPowerProfile kObPowerProfiles[6] = {
    {0x0bu, 0x03u, 0x01u, 0x01u},
    {0x0bu, 0x0cu, 0x09u, 0x01u},
    {0x0bu, 0x16u, 0x0du, 0x01u},
    {0x0bu, 0x16u, 0x15u, 0x01u},
    {0x00u, 0x19u, 0x0du, 0x01u},
    {0x01u, 0x19u, 0x15u, 0x02u},
};

void apply_perf_mode_to_cfg(ob_config_t* cfg, uint32_t perf_mode, uint8_t* out_fw_prio) {
  if (!cfg) return;
  if (perf_mode > 5u) perf_mode = 5u;
  cfg->perf_mode = static_cast<uint8_t>(perf_mode);
  const ObPowerProfile& p = kObPowerProfiles[perf_mode];
  cfg->core_power_state = p.tpu_core_power_state;
  cfg->mem_power_state = p.memory_power_state;
  cfg->qos_class = p.qos_class;
  if (out_fw_prio) *out_fw_prio = p.fw_priority;
}

}  // namespace

extern "C" {

ob_config_t ob_config_default(void) {
  ob_config_t cfg {};
  cfg.struct_size = static_cast<uint32_t>(sizeof(ob_config_t));
  cfg.mock_mode = 0;
  cfg.device_path = "/dev/edgetpu-soc";
  cfg.dma_heap_path = "/dev/dma_heap/system";
  cfg.max_inflight = OB_MAX_CREDITS;
  cfg.graph_cache_capacity = ob::kDefaultGraphCacheCapacity;
  cfg.buffer_pool_max_bytes = static_cast<uint32_t>(ob::kDefaultBufferPoolMaxBytes);
  cfg.timeout_ms = ob::kDefaultTimeoutMs;
  cfg.cluster_id = 0u;  // 0 -> use .dgc0 cluster_id
  cfg.priority = 0u;
  apply_perf_mode_to_cfg(&cfg, 5u, nullptr);  // Default: 1286.4 MHz peak OPP
  return cfg;
}

ob_status_t ob_context_create(const ob_config_t* config, ob_context_t** out_ctx) {
  if (!out_ctx) return OB_ERR_INVALID_ARG;
  *out_ctx = nullptr;

  const ob_config_t def = ob_config_default();
  ob_config_t eff = def;
  uint8_t fw_prio = kObPowerProfiles[5].fw_priority;
  if (config) {
    eff.mock_mode = config->mock_mode;
    if (config->device_path && config->device_path[0]) eff.device_path = config->device_path;
    if (config->dma_heap_path && config->dma_heap_path[0]) eff.dma_heap_path = config->dma_heap_path;
    if (config->max_inflight > 0u) eff.max_inflight = config->max_inflight;
    if (config->graph_cache_capacity > 0u) eff.graph_cache_capacity = config->graph_cache_capacity;
    if (config->buffer_pool_max_bytes > 0u) eff.buffer_pool_max_bytes = config->buffer_pool_max_bytes;
    if (config->timeout_ms > 0u) eff.timeout_ms = config->timeout_ms;
    const bool from_default_struct = (config->struct_size == sizeof(ob_config_t));
    if (config->perf_mode != 0u) {
      apply_perf_mode_to_cfg(&eff, config->perf_mode, &fw_prio);
    } else if (from_default_struct) {
      apply_perf_mode_to_cfg(&eff, 0u, &fw_prio);
    }
    if (config->core_power_state != 0u && config->core_power_state != def.core_power_state) {
      eff.core_power_state = config->core_power_state;
    }
    if (config->mem_power_state != 0u && config->mem_power_state != def.mem_power_state) {
      eff.mem_power_state = config->mem_power_state;
    }
    if (config->qos_class != 0u && config->qos_class != def.qos_class) {
      eff.qos_class = config->qos_class;
    }
    eff.cluster_id = config->cluster_id;
    eff.priority = config->priority;
  }
  // Environment overrides take precedence over struct defaults for A/B benchmarking
  uint8_t env_pm = 0u;
  if (read_env_u8("OPENBUENOS_PERF_MODE", &env_pm)) {
    apply_perf_mode_to_cfg(&eff, env_pm, &fw_prio);
  }
  read_env_u8("OPENBUENOS_FW_PRIORITY", &fw_prio);
  read_env_u8("OPENBUENOS_CORE_POWER_STATE", &eff.core_power_state);
  read_env_u8("OPENBUENOS_MEM_POWER_STATE", &eff.mem_power_state);
  read_env_u8("OPENBUENOS_QOS_CLASS", &eff.qos_class);
  read_env_u8("OPENBUENOS_CLUSTER_ID", &eff.cluster_id);
  read_env_u8("OPENBUENOS_PRIORITY", &eff.priority);

  bool wl_enabled = true;
  uint8_t env_wl = 1u;
  if (read_env_u8("OPENBUENOS_WAKELOCK", &env_wl)) {
    wl_enabled = (env_wl != 0u);
  }

  ob_context* ctx = new (std::nothrow) ob_context();
  if (!ctx) return OB_ERR_NO_MEMORY;

  ctx->cfg = eff;
  ctx->fw_priority = fw_prio;
  ctx->fw_wakelock_enabled = wl_enabled;
  ctx->profile_enabled = read_env_flag("OPENBUENOS_PROFILE", false);
  ctx->verbose_enabled = read_env_flag("OPENBUENOS_VERBOSE", false);
  ctx->strict_weight_hash = read_env_flag("OPENBUENOS_STRICT_WEIGHT_HASH", false);
  ctx->large_tiles_enabled = !read_env_flag("OPENBUENOS_DISABLE_LARGE_TILES", false);
  ctx->batched_large_tiles_enabled =
      ctx->large_tiles_enabled && !read_env_flag("OPENBUENOS_DISABLE_BATCHED_LARGE_TILES", false);
  ctx->horiz_fusion_enabled = !read_env_flag("OPENBUENOS_DISABLE_HORIZ_FUSION", false);
  ctx->swiglu_chain_enabled = !read_env_flag("OPENBUENOS_DISABLE_SWIGLU_CHAIN", false);
  ctx->swiglu_prefill_enabled = read_env_flag("OPENBUENOS_SWIGLU_PREFILL", false);
  if (const char* s = std::getenv("OPENBUENOS_SWIGLU_MAX_CREST"); s && s[0]) {
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (end != s && v >= 0.0f) {
      ctx->swiglu_max_crest = v;
    }
  }
  uint32_t env_swap_slots = 1u;
  if (read_env_u32("OPENBUENOS_PARAM_SWAP_SLOTS", &env_swap_slots)) {
    if (env_swap_slots < 1u) env_swap_slots = 1u;
    if (env_swap_slots > OB_MAX_CREDITS) env_swap_slots = OB_MAX_CREDITS;
  }
  ctx->param_swap_slots = env_swap_slots;
  uint32_t env_spin_us = 450u;
  if (read_env_u32("OPENBUENOS_SPIN_US", &env_spin_us)) {
    if (env_spin_us > 10000u) env_spin_us = 10000u;
  }
  ctx->spin_us = env_spin_us;
  ctx->device_path = eff.device_path;
  ctx->dma_heap_path = eff.dma_heap_path;
  ctx->mock_mode = (eff.mock_mode != 0);
  ctx->cache_capacity = eff.graph_cache_capacity;
  ctx->pool_max_bytes = static_cast<size_t>(eff.buffer_pool_max_bytes);

  ob_status_t st = ob::device_open(ctx);
  if (st != OB_OK) {
    ob::buffer_pool_clear(ctx);
    delete ctx;
    return st;
  }
  ob::device_sync_fw_wakelock(ctx);

  *out_ctx = ctx;
  return OB_OK;
}

void ob_context_destroy(ob_context_t* ctx) {
  if (!ctx) return;
  ob::device_wait_all(ctx, ctx->cfg.timeout_ms);
  if (ctx->profile_enabled || (ctx->verbose_enabled && (ctx->prof.calls > 0u || ctx->prof.swiglu_calls > 0u))) {
    ob_profile_stats_t ps {};
    if (ob_context_get_profile(ctx, &ps) == OB_OK) {
      std::fprintf(
          stderr,
          "libopenbuenos: profile calls=%llu tiles=%llu horiz_fusions=%llu horiz_saved=%llu "
          "swiglu_calls=%llu swiglu_chained=%llu swiglu_fb=%llu "
          "hw_us=%llu wall_us=%llu quant_us=%llu hash_us=%llu pack_us=%llu dma_sync_us=%llu "
          "submit_us=%llu wait_us=%llu dequant_us=%llu sat_retries=%llu spin_hits=%llu "
          "poll_waits=%llu pool_hits=%llu pool_misses=%llu w_hits=%llu w_misses=%llu "
          "w_evict=%llu w_bytes=%llu w_peak_bytes=%llu w_budget_bytes=%llu\n",
          static_cast<unsigned long long>(ps.calls),
          static_cast<unsigned long long>(ps.tiles_dispatched),
          static_cast<unsigned long long>(ps.horiz_fusions),
          static_cast<unsigned long long>(ps.horiz_jobs_saved),
          static_cast<unsigned long long>(ps.swiglu_calls),
          static_cast<unsigned long long>(ps.swiglu_chunks_chained),
          static_cast<unsigned long long>(ps.swiglu_fallbacks),
          static_cast<unsigned long long>(ps.hw_us),
          static_cast<unsigned long long>(ps.total_wall_us),
          static_cast<unsigned long long>(ps.quant_us),
          static_cast<unsigned long long>(ps.hash_us),
          static_cast<unsigned long long>(ps.weight_pack_us),
          static_cast<unsigned long long>(ps.dma_sync_us),
          static_cast<unsigned long long>(ps.submit_us),
          static_cast<unsigned long long>(ps.wait_us),
          static_cast<unsigned long long>(ps.dequant_us),
          static_cast<unsigned long long>(ps.sat_retries),
          static_cast<unsigned long long>(ps.spin_hits),
          static_cast<unsigned long long>(ps.poll_waits),
          static_cast<unsigned long long>(ps.pool_hits),
          static_cast<unsigned long long>(ps.pool_misses),
          static_cast<unsigned long long>(ps.weight_cache_hits),
          static_cast<unsigned long long>(ps.weight_cache_misses),
          static_cast<unsigned long long>(ps.weight_cache_evictions),
          static_cast<unsigned long long>(ps.weight_cache_bytes),
          static_cast<unsigned long long>(ps.weight_cache_peak_bytes),
          static_cast<unsigned long long>(ps.weight_cache_budget_bytes));
    }
  }
  ob::gemm_release_pipeline_buffers(ctx);
  ob::graph_cache_clear(ctx);
  ob::shared_image_clear(ctx);
  ob::device_close(ctx);
  ob::buffer_pool_clear(ctx);
  delete ctx;
}

const char* ob_status_string(ob_status_t status) {
  switch (status) {
    case OB_OK:              return "OB_OK";
    case OB_ERR_INVALID_ARG: return "OB_ERR_INVALID_ARG";
    case OB_ERR_NO_MEMORY:   return "OB_ERR_NO_MEMORY";
    case OB_ERR_DEVICE:      return "OB_ERR_DEVICE";
    case OB_ERR_IOCTL:       return "OB_ERR_IOCTL";
    case OB_ERR_TIMEOUT:     return "OB_ERR_TIMEOUT";
    case OB_ERR_BAD_DGC0:    return "OB_ERR_BAD_DGC0";
    case OB_ERR_BUSY:        return "OB_ERR_BUSY";
    case OB_ERR_NOT_FOUND:   return "OB_ERR_NOT_FOUND";
    case OB_ERR_UNSUPPORTED: return "OB_ERR_UNSUPPORTED";
    case OB_ERR_FW_ERROR:    return "OB_ERR_FW_ERROR";
    case OB_ERR_OVERFLOW:    return "OB_ERR_OVERFLOW";
    case OB_ERR_IO:          return "OB_ERR_IO";
    default:                 return "OB_ERR_UNKNOWN";
  }
}

const char* ob_context_last_error(const ob_context_t* ctx) {
  if (!ctx) return "";
  std::lock_guard<std::mutex> lk(ctx->err_mu);
  return ctx->last_error.c_str();
}

uint32_t ob_context_credit_size(const ob_context_t* ctx) {
  return ctx ? ctx->credit_size : 0u;
}

int ob_context_is_mock(const ob_context_t* ctx) {
  return (ctx && ctx->mock_mode) ? 1 : 0;
}

ob_status_t ob_context_set_perf_mode(ob_context_t* ctx, uint32_t perf_mode) {
  if (!ctx || perf_mode > 5u) return OB_ERR_INVALID_ARG;
  {
    std::lock_guard<std::mutex> lk(ctx->vii_mu);
    apply_perf_mode_to_cfg(&ctx->cfg, perf_mode, &ctx->fw_priority);
  }
  ob::device_sync_fw_wakelock(ctx);
  return OB_OK;
}

ob_status_t ob_context_set_profile(ob_context_t* ctx, int enable) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  std::lock_guard<std::mutex> lk(ctx->gemm_mu);
  ctx->profile_enabled = (enable != 0);
  return OB_OK;
}

ob_status_t ob_context_get_profile(const ob_context_t* ctx, ob_profile_stats_t* out_profile) {
  if (!ctx || !out_profile) return OB_ERR_INVALID_ARG;
  std::memset(out_profile, 0, sizeof(*out_profile));
  ob_context* mctx = const_cast<ob_context*>(ctx);
  {
    std::lock_guard<std::mutex> glk(mctx->gemm_mu);
    out_profile->quant_us = mctx->prof.quant_ns / 1000ULL;
    out_profile->hash_us = mctx->prof.hash_ns / 1000ULL;
    out_profile->weight_pack_us = mctx->prof.weight_pack_ns / 1000ULL;
    out_profile->dma_sync_us = mctx->prof.dma_sync_ns / 1000ULL;
    out_profile->submit_us = mctx->prof.submit_ns / 1000ULL;
    out_profile->wait_us = mctx->prof.wait_ns / 1000ULL;
    out_profile->dequant_us = mctx->prof.dequant_ns / 1000ULL;
    out_profile->hw_us = mctx->prof.hw_us;
    out_profile->total_wall_us = mctx->prof.total_wall_ns / 1000ULL;
    out_profile->calls = mctx->prof.calls;
    out_profile->tiles_dispatched = mctx->prof.tiles_dispatched;
    out_profile->horiz_fusions = mctx->prof.horiz_fusions;
    out_profile->horiz_jobs_saved = mctx->prof.horiz_jobs_saved;
    out_profile->sat_retries = mctx->prof.sat_retries;
    out_profile->weight_cache_hits = mctx->weight_cache_hits;
    out_profile->weight_cache_misses = mctx->weight_cache_misses;
    out_profile->weight_cache_bytes = mctx->weight_cache_bytes;
    out_profile->weight_cache_peak_bytes = mctx->weight_cache_peak_bytes;
    out_profile->weight_cache_budget_bytes = ob::compute_weight_cache_budget_bytes(mctx);
    out_profile->weight_cache_evictions = mctx->weight_cache_evictions;
    out_profile->swiglu_calls = mctx->prof.swiglu_calls;
    out_profile->swiglu_chunks_chained = mctx->prof.swiglu_chunks_chained;
    out_profile->swiglu_fallbacks = mctx->prof.swiglu_fallbacks;
    {
      std::lock_guard<std::mutex> vlk(mctx->vii_mu);
      out_profile->spin_hits = mctx->prof.spin_hits;
      out_profile->poll_waits = mctx->prof.poll_waits;
    }
    {
      std::lock_guard<std::mutex> plk(mctx->pool_mu);
      out_profile->pool_hits = mctx->pool_hits;
      out_profile->pool_misses = mctx->pool_misses;
    }
  }
  return OB_OK;
}

ob_status_t ob_context_reset_profile(ob_context_t* ctx) {
  if (!ctx) return OB_ERR_INVALID_ARG;
  std::lock_guard<std::mutex> glk(ctx->gemm_mu);
  std::lock_guard<std::mutex> vlk(ctx->vii_mu);
  ctx->prof = ob::ProfileCounters{};
  return OB_OK;
}

}  // extern "C"
