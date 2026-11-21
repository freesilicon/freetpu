# Bare-Metal Litebuf VII Wire Protocol (`rango` / Buenos A0 `/dev/edgetpu-soc`)

Clean-room specification of the DarwiNN / EdgeTPU Litebuf VII mailbox wire protocol on Pixel 10 Pro (`rango`, Buenos A0), allowing zero-Google-`.so` execution directly over `/dev/edgetpu-soc` (`0xED`) and `/dev/dma_heap/system`.

Companion header: [`litebuf_vii_wire.h`](litebuf_vii_wire.h).

---

## 1. Two-Layer Protocol Architecture & Kernel Boundary

Every graph registration and inference uses a **two-layer** Litebuf payload:

1. **Outer Mailbox Layer (`silicon.darwinn.control.driver.mailbox.runtime_messages`)**:
   - Schema ID `0x0008`, alignment `16` bytes, `int16_t` relative offsets, `uint16_t` sizes.
   - Passed by userspace pointer/size in `struct edgetpu_vii_litebuf_command_ioctl` (`EDGETPU_VII_LITEBUF_COMMAND`, `0xc038ed25`).
   - Kernel handling in `buenos.ko.decomp.c`:
     - `litebuf_size == 0`: emits 128B mailbox command `type = 3` (`EDGETPU_VII_LITEBUF_NULL_COMMAND`).
     - `1 <= litebuf_size <= 96`: copies inline into `runtime_command`0..96)`, mailbox `type = 0` (`EDGETPU_VII_LITEBUF_RUNTIME_COMMAND`). Applies to `Execute` (`80 B`), `UnregisterModel` (`32 B`), `AcquireWakeLock` (`32 B`), `ReleaseWakeLock` (`32 B`).
     - `litebuf_size > 96`: kernel allocates DMA memory (`edgetpu_iremap_alloc`), copies userspace `litebuf_address` into it, writes `{(__u32)dma_addr, (__u32)size}` into `large_runtime_command`, mailbox `type = 1` (`EDGETPU_VII_LITEBUF_LARGE_RUNTIME_COMMAND`). Applies to `RegisterModel` (`144 B` with `"Unknown"` model id).
2. **Inner DIVE Runtime Layer (`silicon.dive.api.runtime_commands`)**:
   - Schema ID `0x00fe`, alignment `16` bytes, `int32_t` relative offsets, `uint32_t` sizes.
   - Resides inside a `EDGETPU_MAP_DMABUF`-mapped DMA-BUF (`vii_buffer`), passed to firmware via `(dive_cmd_address, dive_cmd_size_bytes)` inside outer `RegisterModel` or `Execute`.
   - `libtachyon_core.so` places inner payloads at `+0x200` within a DMA buffer (`kViiCommandBufferSize = 0x200`), though standalone DMA-BUFs mapped at `+0x000` (`iova`) work identically (confirmed in [`fc_wire_dump.log`).

---

## 2. Devicebringup & Session Sequence (`Observed`)

Exact ioctl sequence observed in `fc_wire_dump.log` and `add_wire_dump.log`:

1. `open("/dev/edgetpu-soc", O_RDWR | O_CLOEXEC)` -> `tpu_fd`.
2. `EDGETPU_GET_CREDIT_SIZE` (`0x8008ed27`) -> returns `16`.
3. `EDGETPU_CREATE_GROUP` (`0x4014ed06`) with `edgetpu_mailbox_attr`:
   - Confirmed working: `cmd_queue_size=24, resp_queue_size=12, sizeof_cmd=48, sizeof_resp=24, priority=8, cmdq_tail_doorbell=1, partition_type=0, client_priv=1` (`tpu_direct_runner --direct-ioctl` also works with `cmdq=16, respq=16, sizeof_cmd=128, sizeof_resp=64, prio=8, cmdq_tail_doorbell=1`).
4. `EDGETPU_SET_EVENTFD` (`0x4008ed05`) with `{EDGETPU_EVENT_RESPDATA (0), resp_efd}` (and optionally `EDGETPU_EVENT_FATAL_ERROR (1)`).
5. `EDGETPU_ACQUIRE_WAKE_LOCK` (`0x0000ed1a`) -> powers up TPU block for the group.
   - Note: `libedgetpu_util.so` does **not** issue `AcquireWakeLock` (`type=4`) or `ReleaseWakeLock` (`type=5`) Litebuf VII packets alongside ioctl `EDGETPU_ACQUIRE_WAKE_LOCK`; kernel/power sequence handles it via ioctl.
6. Optional `EDGETPU_FIRMWARE_VERSION` (`0x8010ed1b`) -> observed `major=1, minor=1, vii=3, kci=3`.
7. Allocate DMA-BUFs from `/dev/dma_heap/system` (`DMA_HEAP_IOCTL_ALLOC` `0xc0184800`), CPU-sync (`DMA_BUF_IOCTL_SYNC` `0x40086200`), and map to TPU via `EDGETPU_MAP_DMABUF` (`0xc028ed11`, `flags = EDGETPU_MAP_DMA_BIDIRECTIONAL | EDGETPU_MAP_36BIT_IOVA = 0x8`, `die_index = 0xffffffff`).
   - DMA-BUF map sizes should be page-aligned (`(sz + 4095) & ~4095`), while Litebuf slice `size_bytes` fields pass the **exact payload size** (`638977` for RV64 interpreter, `8512` for `ext[0]`, `16`/`4096` for tensors).
8. Submit `RegisterModel` (VII `seq = 0`) -> poll `resp_efd` -> `EDGETPU_VII_LITEBUF_RESPONSE` (`0xc018ed26`) -> decode `model_id` handle (`0x2ff` observed).
9. Submit `Execute` (`seq = 1..N`) -> poll `resp_efd` -> `EDGETPU_VII_LITEBUF_RESPONSE` -> `DMA_BUF_IOCTL_SYNC` read output buffer.
10. Submit `UnregisterModel` (`model_id`) -> poll `resp_efd` -> `EDGETPU_VII_LITEBUF_RESPONSE` -> `EDGETPU_UNMAP_DMABUF` -> `EDGETPU_RELEASE_WAKE_LOCK` (`0x0000ed19`) -> close fds.

---

## 3. Binary Layouts

### 3.1 Common 16-byte Litebuf Header (`litebuf::Builder::Finish`, `0xa91ac`)

| Offset | Type | Field | Value |
|---|---|---|---|
| `0x00` | `uint16_t` | `schema_id` | `0x0008` (outer `runtime_messages`), `0x00fe` (inner `dive::runtime_commands`) |
| `0x02` | `uint8_t` | `log2_alignment` | `4` (`ctz(16)`) |
| `0x03` | `uint8_t` | `reserved0` | `0` |
| `0x04` | `uint32_t` | `reserved1` | `0` |
| `0x08` | `uint64_t` | `root_offset` | `0x10` (offset from container base to root table) |

### 3.2 Outer `runtime_messages::Command` (`0x10..0x17`, `FUN_001d45c8`)

| Rel Off | Type | Field | Description |
|---|---|---|---|
| `+0x00` | `uint16_t` | `table_size` | `7` |
| `+0x02` | `int16_t` | `union_offset` | Signed offset from `Command` to active member (`8` -> `0x18`) |
| `+0x04` | `uint16_t` | `union_type` | `1=RegisterModel`, `2=UnregisterModel`, `3=Execute`, `4=AcquireWakeLock`, `5=ReleaseWakeLock`, `6=Cancel` |
| `+0x06` | `uint8_t` | `priority` | `0` |
| `+0x07` | `uint8_t` | `pad` | `0` |

### 3.3 Outer `RegisterModel` (`union_type = 1`, `104 B`, `FUN_001d2348` + `0xd46c4`)

| Rel Off | Type | Field | Observed / `.dgc0` Source |
|---|---|---|---|
| `+0x00` | `uint16_t` | `table_size` | `99` (`0x63`) |
| `+0x08` | `uint64_t` | `dive_cmd_address` | IOVA of inner DIVE `RegisterInterpreterModel` |
| `+0x10` | `uint32_t` | `dive_cmd_size_bytes` | `128` (no params) or `144` (1 param slice) |
| `+0x18` | `uint64_t` | `dive_image_address` | 4K-aligned IOVA of `ext[1]` RV64 ELF interpreter |
| `+0x20` | `uint32_t` | `dive_image_size_bytes` | Exact `ext[1]` size (`638977`) |
| `+0x28` | `uint64_t` | `dive_scratch_address` | IOVA of `SCRATCHPAD` tensor DMA buffer |
| `+0x30` | `uint32_t` | `dive_scratch_size_bytes` | Size from `SCRATCHPAD` tensor (`192512` fc / `204800` add) |
| `+0x38` | `int32_t` | `dive_heap_size` | `Executable` vtable `id_28` (`32768`) |
| `+0x3c` | `int32_t` | `exec_field_29` | `Executable` vtable `id_29` (`81920`) |
| `+0x40` | `uint32_t` | `core_session_handle` | `0` observed |
| `+0x44` | `uint8_t` | `cluster_id` | `2` (`ctz(1u << 2)` from `Executable.hardware_cluster_id`) |
| `+0x45` | `uint8_t` | `exec_field_31` | `Executable.id_31` (`0` when absent) |
| `+0x46` | `int16_t` | `model_id_str_offset` | `0x68` (points to `"Unknown\0"` after table) |
| `+0x48` | `int16_t` | `model_id_str_size` | `8` (includes NUL) |
| `+0x4c` | `uint32_t` | `exec_field_14` | `Executable.id_14` (`2560` fc, `11584` add, `9024` p4_k512_n64) |
| `+0x58` | `uint64_t` | `compiler_cl` | `Package.compiler_cl` (`925724637` observed) |
| `+0x61` | `uint8_t` | `subgraph_flag_18` | `SubGraph.id_18` (`0` when absent) |
| `+0x62` | `uint8_t` | `exec_field_34` | `Executable.id_34` (`0` when absent) |

### 3.4 Outer `Execute` (`union_type = 3`, `48 B`, `FUN_001d4128` + `0xd5018`)

Total outer `Execute` Litebuf container is `80 B` (`<= 96 B` inline threshold).

| Rel Off | Type | Field | Observed (`fc` & `add`) |
|---|---|---|---|
| `+0x00` | `uint16_t` | `table_size` | `44` (`0x2c`) |
| `+0x08` | `uint64_t` | `dive_cmd_address` | IOVA of inner DIVE `Execute` command |
| `+0x10` | `uint32_t` | `dive_cmd_size_bytes` | `160` (`144` for 1-in/1-out/0-param, etc.) |
| `+0x18` | `uint32_t` | `model_id` | Handle returned by `RegisterModelResponse` (`0x2ff`) |
| `+0x1c` | `uint8_t` | `tpu_core_power_state` | `0x0b` |
| `+0x1d` | `uint8_t` | `memory_power_state` | `0x03` |
| `+0x1e` | `uint8_t` | `qos_class` | `0x01` |
| `+0x1f` | `uint8_t` | `perf_metrics_flag` | `0x01` |
| `+0x20` | `uint8_t` | `cluster_id` | `0x02` |
| `+0x28` | `uint32_t` | `core_session_handle` | `0xffffffff` (`LITEBUF_CORE_SESSION_NONE`) |

### 3.5 Outer `UnregisterModel` (`union_type = 2`, `32 B` total, `FUN_001d2958`)

- `+0x00` `u16 table_size = 8`, `+0x04` `u32 model_id`.

### 3.6 Inner DIVE `runtime_commands::Command` (`schema_id = 0x00fe`)

Root table at `+0x10` (`12 B`): `{ u32 table_size=12; i32 union_offset=16; u32 union_type; }`.

- **`RegisterInterpreterModel` (`union_type = 3`, at `+0x20`, `32 B`)**:
  - `+0x00`: `u32 table_size = 28`
  - `+0x08`: `u64 dive_vm_ir_address` (`ext[0]` IOVA, 64B aligned)
  - `+0x10`: `u64 dive_vm_ir_size_bytes` (`ext[0]` exact size)
  - `+0x18`: `i32 registration_info_offset` (`0x20` -> `+0x40`)
  - **`RegistrationInfo` (`at +0x40`, `64 B`)**:
    - `+0x00`: `u32 table_size = 64`
    - `+0x08`: `u64 model_info_scratch_address` (`0` for standard Buenos `.dgc0`s)
    - `+0x10`: `u64 model_info_scratch_size_bytes` (`0`)
    - `+0x20`: `i32 parameters_offset` (`0x40` -> `+0x80`)
    - `+0x24`: `u32 parameters_count` (`0` or `1`)
    - `+0x28..+0x3f`: `executable_address` (`0`), `model_info_address` (`0`), `model_info_size` (`0`)
  - At `+0x80`: array of `parameters_count` `struct litebuf_dive_buffer_slice { u64 address; u64 size_bytes; }`.

- **`Execute` (`union_type = 2`, at `+0x20`, `80 B`, `FUN_001a4eb0`)**:
  - `+0x00`: `u32 table_size = 80`
  - `+0x04`: `i32 inputs_offset` (`0x50` -> `+0x70`), `+0x08`: `u32 inputs_count`
  - `+0x0c`: `i32 outputs_offset`, `+0x10`: `u32 outputs_count`
  - `+0x14, +0x1c, +0x24, +0x2c`: offsets of optional vectors (count `0`, offset points to `tail_off - 0x20`)
  - `+0x48`: `i32 parameters_offset` (`tail_off - 0x20`), `+0x4c`: `u32 parameters_count` (`0` or `1`, duplicating `ext[2]` params as observed in `fc_wire_dump.log`!)
  - Slice arrays at `+0x70`: inputs, then outputs, then parameters (`{u64 address, u64 size_bytes}` each).

### 3.7 `ViiResponse` (`48 B`, `vii_litebuf_response_parser.cc` `0xf147c`)

Root table at `+hdr.root_offset` (`0x10`):
- `u16 table_size` (`>= 6`, `12` observed)
- `i16 union_offset` (`12` -> `0x1c`)
- `u16 union_type`:
  - `0` (`VOID_OK`): e.g., `UnregisterModel` completion.
  - `1` (`EXECUTE`): member table `+0x04` (`u32`, plus upper `u32` at `+0x08` if `m_sz >= 12`) = `hw_execution_time_us` (`587` observed on `fc`, `660` observed on `add`).
  - `2` (`REGISTER_MODEL`): member table `+0x04` (`u32`) = `model_id` handle (`0x2ff` observed).
  - `3` (`ERROR`): member table `+0x04` (`u32`) = `1 INVALID_RUNTIME_COMMAND`, `2 INVALID_VII_SEQ_ID`, `3 INVALID_PRIORITY`.
  - `4` (`CANCEL`), `5` (`REPRIORITIZE`).

---

## 4. Mapping `.dgc0` Containers to IOVAs (`fc`, `add`, `custom_gemm_microkernel`, Phase 4 Multi-Tile)

Observed across `fc.dgc0`, `add.dgc0`, `conv2d.dgc0`, `custom_gemm_microkernel.dgc0`, `p4_k512_n64.dgc0`, `p4_3d_6x48x48_relu6.dgc0`:

1. Verify root magic at `+4 == "DGC0"`.
2. `DGC0` root table:
   - `slot 0`: byte vector containing `Package` (`DWS0`) FlatBuffer.
   - `slot 5`: `compiler_cl` (`uint32_t`).
   - `slot 7`: `external_buffers` table vector (`ext[i].slot_0 = offset`, `ext[i].slot_1 = size_bytes`):
     - `ext[0]`: DIVE VM IR (`DiveVmIr`), mapped 4K-aligned, exact size passed to inner `RegisterInterpreterModel`.
     - `ext[1]`: RV64 SiFive X280 interpreter (`DiveImage`), mapped 4K-aligned (`flags=0x8`), exact size (`638977`) passed to outer `RegisterModel`.
     - `ext[2]` (optional; absent in `add.dgc0`): packed weights/parameters (`Params`), mapped via DMA-BUF `flags=0x8` and passed in both inner `RegistrationInfo` and inner `Execute` `parameters` vectors.
3. Inside `Package` (`DWS0`):
   - `SubGraph[0]`: read `input_tensor_indices`, `output_tensor_indices`, and `tensors[]`:
     - Any tensor with `storage_type == 3` (`SCRATCHPAD`): compute `byte_size = product(dims) * dtype_bytes(data_type)` -> allocate DMA-BUF -> `dive_scratch_{address, size_bytes}`.
     - Note `SubGraph` `slot 18` -> `RegisterModel.subgraph_flag_18`.
   - `Operation[0].detail (OpExecutableCall) -> MultiExecutable -> Executable[0]`:
     - `Executable.inputs[k]` (`slot 1`/`3`) & `Executable.outputs[k]` (`slot 1`/`3`) give exact byte sizes of caller-supplied tensors (`input_0`, `input_1`, `output_0`).
     - Scalars mapped to `RegisterModel`: `id_28` (`dive_heap_size`), `id_29`, `id_14`, `id_31`, `id_34`, and `cluster_id = 2` (on `rango`).

---

## 5. Evidence & Confidence Ledger

- **Observed (`libtachyon_core.so` Ghidra decompilation + `buenos.ko.decomp.c`)**:
  - Container header builder `FUN_001a91ac` (`0xa91ac`), table constructors `FUN_001d45c8..0x1d5018`, outer builders `FUN_001d2348` (`RegisterModel`), `FUN_001d2958` (`UnregisterModel`), `FUN_001d2cfc` (`AcquireWakeLock`), `FUN_001d2fbc` (`ReleaseWakeLock`), `FUN_001d327c` (`Cancel`), `FUN_001d4128` (`Execute`), inner builders `FUN_001d3a28`/`0xd3f98` (`RegisterInterpreterModel`), `FUN_001a4eb0` (`Execute`), response parser `FUN_001f147c` (`0xf147c`), `.dgc0` field extractor `FUN_001d7c2c` (`0xd7c2c`), ioctl layer `FUN_001f0c30`/`0xf10fc` (`ikv_queue.cc`).
- **Observed (live single-process `LD_PRELOAD` wire capture on `rango`)**:
  - `work/litebuf-vii/fc_wire_dump.log` and `work/litebuf-vii/add_wire_dump.log` (`tpu_direct_runner --run-dgc0 ... --use-tachyon 0`, zero system daemons touched).
  - Byte-for-byte match verified by `work/litebuf-vii/test_wire_match.cpp` (`memcmp == 0` on all 7 serialized commands + all parsed responses), plus `-fsyntax-only -Wall -Wextra -Werror` C11/C++17 builds on both NDK AArch64 and host clang.
- **Observed (live silicon `tpu_baremetal_runner` verification on `rango`)**:
  - `runtime/build/tpu_baremetal_runner` (`-std=c++17 -O2 -Wall -Wextra -Werror -fPIE -pie -static-libstdc++ -Wl,--as-needed`) shows `DT_NEEDED` = `[libdl.so, libc.so]` only (`libdl.so` only for `dl_iterate_phdr` inside static libunwind; zero `liblog`, zero Google/vendor `.so`, zero `dlopen`/`dlsym` symbols).
  - `/proc/self/maps` audit (`46/46` runs): `maps_has_google_tpu_libs = false`, mapping only Bionic `libc.so`/`libdl.so`/`libm.so` + `/system/lib64/{libc++.so,libnetd_client.so}`.
  - `fc.dgc0` (`5 iters`): `execute.ok=true`, `verify.ok=true`, `16/16` (`max_abs_err=0`), `hw_execution_time_us=422`, `mean_ms=1.666`, TPU `thermal_zone24` `-274000 -> 27000 mC` (`delta_mc=+301000`).
  - `add.dgc0` (2-input, `5 iters`): `execute.ok=true`, `3059/4096` (`max_abs_err=1`, identical to Google `tpu_direct_runner` in `compile_run_add.json`), `hw_execution_time_us=518`, `mean_ms=1.424`.
  - `conv2d.dgc0` (`5 iters`): `execute.ok=true`, `4084/4096` (`max_abs_err=1`, identical to Google `tpu_direct_runner` in `compile_run_conv2d.json`), `hw_execution_time_us=526`, `mean_ms=1.786`.
  - `custom_gemm_microkernel.dgc0` (Phase 3 custom microkernel, `5 iters`): `execute.ok=true`, `verify.ok=true`, `32/32` (`max_abs_err=0`), byte-identical (`direct_byte_equal=true`), `hw_execution_time_us=387`, `mean_ms=1.748`.
  - Phase 4 reference sweep (`42/42` `.dgc0`s): `execute.ok=42/42`, `verify.ok=42/42` (`max_abs_err=0`, `direct_byte_equal=42/42`), `hw_us=366..639`.
  - Phase 4 novel clean-room custom multi-tile `.dgc0` microkernels (`7/7` emitted with zero TFLite compiler involvement, `phase4_custom_multitile_live/summary.json`): `baremetal=7/7` (`custom_4x32x32_nonpot005_relu`, `custom_8x64x64_nonpot003_relu6`, `custom_16x64x64_nonpot007_none`, `custom_6x48x48_nonpot007_relu6`, `custom_3x32x24_nonpot005_relu`, `custom_32x32x32_nonpot005_none`, `custom_10x32x32_unseen_relu6`), `max_abs_err=0`, `bm_vs_direct_byte_identical=7/7`, `bm_vs_ref_byte_identical=7/7`, `hw_us=420..504`, `mean_ms=1.283..2.465`, `maps_has_google_tpu_libs=false` (`7/7`), TPU `thermal_zone24` `-274000 -> 30000 mC` (`+304000`).
  - Phase 5 `0x640` LLM spline probes (`3/3`, `spline_probes/summary.json`): `silu_probe` (`461us`), `gelu_probe` (`473us`), `exp_probe` (`600us`) all `4096/4096` exact (`max_abs_err=0`, `bm_vs_direct_byte_identical=true`, `maps_has_google_tpu_libs=false`).
  - Evidence artifacts: `work/litebuf-vii/live_verify/summary.json`, `work/device-runner/phase4_custom_multitile_live/summary.json`, `work/device-runner/phase5_live/spline_probes/summary.json`.
- **Unknowns / Scope Bounds**:
  - Multi-subgraph `DMG0` container support (out of scope for single-subgraph `.dgc0` models; all 56 tested single-subgraph `.dgc0` models executed end-to-end).
