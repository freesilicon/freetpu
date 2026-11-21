#include "ob_internal.h"

#include <algorithm>
#include <cerrno>
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

void device_close(ob_context* ctx) {
  if (!ctx) return;
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
  size_t outer_exec_len = litebuf_build_vii_execute(
      outer_exec, sizeof(outer_exec),
      slot->exec_cmd_buf->slab.iova, static_cast<uint32_t>(inner_exec_len),
      entry->model_handle,
      ctx->cfg.core_power_state, ctx->cfg.mem_power_state,
      ctx->cfg.qos_class, 1u, cluster,
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
      uint64_t deadline_ns =
          monotonic_now_ns() + static_cast<uint64_t>(wait_ms) * 1000000ULL;

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
    stats.flags = ctx->mock_mode ? OB_STAT_FLAG_MOCK : 0u;
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
  cfg.core_power_state = LITEBUF_DEFAULT_CORE_POWER_STATE;
  cfg.mem_power_state = LITEBUF_DEFAULT_MEM_POWER_STATE;
  cfg.qos_class = LITEBUF_DEFAULT_QOS_CLASS;
  cfg.cluster_id = 0u;  // 0 -> use .dgc0 cluster_id
  cfg.priority = 0u;
  return cfg;
}

ob_status_t ob_context_create(const ob_config_t* config, ob_context_t** out_ctx) {
  if (!out_ctx) return OB_ERR_INVALID_ARG;
  *out_ctx = nullptr;

  ob_config_t eff = ob_config_default();
  if (config) {
    eff.mock_mode = config->mock_mode;
    if (config->device_path && config->device_path[0]) eff.device_path = config->device_path;
    if (config->dma_heap_path && config->dma_heap_path[0]) eff.dma_heap_path = config->dma_heap_path;
    if (config->max_inflight > 0u) eff.max_inflight = config->max_inflight;
    if (config->graph_cache_capacity > 0u) eff.graph_cache_capacity = config->graph_cache_capacity;
    if (config->buffer_pool_max_bytes > 0u) eff.buffer_pool_max_bytes = config->buffer_pool_max_bytes;
    if (config->timeout_ms > 0u) eff.timeout_ms = config->timeout_ms;
    if (config->core_power_state != 0u) eff.core_power_state = config->core_power_state;
    if (config->mem_power_state != 0u) eff.mem_power_state = config->mem_power_state;
    if (config->qos_class != 0u) eff.qos_class = config->qos_class;
    eff.cluster_id = config->cluster_id;
    eff.priority = config->priority;
  }

  ob_context* ctx = new (std::nothrow) ob_context();
  if (!ctx) return OB_ERR_NO_MEMORY;

  ctx->cfg = eff;
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

  *out_ctx = ctx;
  return OB_OK;
}

void ob_context_destroy(ob_context_t* ctx) {
  if (!ctx) return;
  ob::device_wait_all(ctx, ctx->cfg.timeout_ms);
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

}  // extern "C"
