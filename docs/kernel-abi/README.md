# Tensor G5 (`laguna` / `buenos` + `gxp`) Kernel-User ABI & Verification Matrix

Offline clean-room kernel UAPI/ABI documentation for `rango` (`CP3A.261005.005`, Linux `6.6.127-android15-8-gee206ffe8752-ab16254410`).

## Deliverables

- [gcip_uapi.h](gcip_uapi.h) — Shared IIF (`/dev/iif` & shared fence table), GCIP KCI wire structs/codes (`6, 28, 29, 30` extensions), firmware info, telemetry ring headers, and `edgetpu_ext_driver_cmd` constants.
- [edgetpu_buenos_uapi.h](edgetpu_buenos_uapi.h) — `/dev/edgetpu-soc` (`/dev/edgetpu` symlink) & `/dev/edgetpu-limited` UAPI (`buenos.ko`, interface `1.4`, `0xED` ioctls `0..44`, `mmap` `0x1b00000..0x2300000`, Litebuf VII & IKV additional info).
- [gxp_laguna_uapi.h](gxp_laguna_uapi.h) — `/dev/gxp` UAPI (`gxp.ko`, interface `1.33`, `0xEE` ioctls `0..43`, `core_count = 3`, UCI `mid_in_fences`/`mid_out_fences` extension & 64B `gxp_uci_additional_info_root`).

---

## 1. Input Artifacts & Provenance

| Artifact | Path | Size / SHA-256 or Git HEAD |
|---|---|---|
| `buenos.ko` | `extracted/vendor_dlkm/lib/modules/buenos.ko` | `798000` B / `461a8525f310ae3e4047786621b1923c15d57efc37f0db6b20ff214186ae557a` |
| `gxp.ko` | `extracted/vendor_dlkm/lib/modules/gxp.ko` | `752344` B / `92f071a5b6580855886b314efc461fcdc3c00411256c5d7fc2f647239199e34b` |
| `edgetpu-rio-android16` | `aosp-kernel/edgetpu-rio-android16/` | `ff69a09bd9854786c1704c4369f2298dae4e4c72` |
| `gxp-zuma-android16` | `aosp-kernel/gxp-zuma-android16/` | `41a585e40a021252468c31e3e4ee5f8c0bb02b28` |

Modinfo & device-tree aliases observed from `buenos.ko` / `gxp.ko` / `modules.dep` / `modules.alias`:
- **`buenos.ko`**: `name=buenos`, `scmversion=gcc524543248c`, `firmware=google/edgetpu-buenos.fw`, `depends=iif,gsa,google-mem-qos,pixel_trim`, `alias=of:N*T*Cgoogle,edgetpu-laguna`, `alias=of:N*T*Cgoogle,edgetpu-lagunaC*`.
- **`gxp.ko`**: `name=gxp`, `scmversion=gcc97177d49ea`, `depends=buenos,iif,gsa,google_pm_qos`, `alias=of:N*T*Cgoogle,gxp-lga`, `alias=of:N*T*Cgoogle,gxp-lgaC*`, `alias=of:N*T*Cgoogle,gxp`, `alias=of:N*T*Cgoogle,gxpC*`.
- **`modules.dep`**: `/vendor/lib/modules/gxp.ko: /vendor/lib/modules/buenos.ko`.

---

## 2. Cross-Module Coupling (`gxp.ko` -> `buenos.ko`)

Observed directly in `gxp.ko` imports/relocations and `buenos.ko` `.ksymtab` + disassembly:
1. **`gxp.ko` imports `edgetpu_ext_driver_cmd`** (undefined external symbol in `gxp.ko`, `EXPORT_SYMBOL` in `buenos.ko` at `0x5880`).
2. During `gxp_platform_probe` (`0x10d490`), `gxp.ko` parses DT properties `"tpu-device"` and `"gxp-tpu-mbx-offset"` (storing `tpu_dev` at `gxp + 0x480` and TPU mailbox base at `gxp + 0x488`), and parses `"gsa-device"` (`gxp + 0x4b8`). It also sets `ganymede_platform_parse_dt` for SoC DT parsing.
3. **`edgetpu_ext_driver_cmd(etdev, flags, cmd, in, out)`** dispatch observed in `buenos.ko`:
   - `cmd = 0`: Acquire external mailbox (`0x27` when `flags == 0`, `0x28` when `flags == 1`; `edgetpu_mailbox_enable_ext`), returning mailbox register/offset info in `out`.
   - `cmd = 1`: Disable external mailbox (`edgetpu_mailbox_disable_ext`).
   - `cmd = 2`: Query TPU client ID / PASID from `[group->etdomain + 0x8]` (if `extra_high` flag set and group matches `etdev` high group, ORs `0x40000000`). Called by `gxp_mcu_after_map_tpu_mbx_queue` (`0x25d74`), which then calls `gxp_kci_link_unlink_offload_vmbox` (KCI `0x11` link / `0x12` unlink with `offload_chip_type = 0`).
   - `cmd = 3`: Acquire `struct iif_manager *` via `iif_manager_get()`.

---

## 3. EdgeTPU (`/dev/edgetpu-soc` & `/dev/edgetpu-limited`) Dispatch Matrix (`buenos.ko`)

Base magic: `EDGETPU_IOCTL_BASE = 0xED`. Jump tables in `buenos.ko` cover `0x4008ed08..0x4008ed29` and `0x4004ed0e..0x4004ed2c` plus explicit checks.

| nr | Observed Hex | Macro / Encoding | `_IOC_SIZE` | Handler (`buenos.ko`) | AOSP `edgetpu/rio` Status & Delta | Userspace (`vendor/lib64`) |
|---:|---|---|---:|---|---|---|
| 0 | `0xc020ed00` | `_IOWR(0xed, 0, struct edgetpu_map_ioctl)` | 32 | `edgetpu_ioctl_map_buffer(..., limited=0)` | Match; bit 9 forces bit 4; bit 3 = 36-bit IOVA | `libtachyon_core`, `libedgetpu_litert` |
| 4 | `0x4020ed04` | `_IOW(0xed, 4, struct edgetpu_map_ioctl)` | 32 | `edgetpu_ioctl_unmap_buffer(..., limited=0)` | Match | `libtachyon_core` |
| 5 | `0x4008ed05` | `_IOW(0xed, 5, struct edgetpu_event_register)` | 8 | `edgetpu_ioctl_set_eventfd` | Match (`0` respdata, `1` fatal) | `libtachyon_core` |
| 6 | `0x4014ed06` | `_IOW(0xed, 6, struct edgetpu_mailbox_attr)` | 20 | `edgetpu_ioctl_create_group` | Match + finalizes (`status = 1`), byte `+0x10` bits 5/6/7 | `libtachyon_core` |
| 7 | `—` | `(EDGETPU_JOIN_GROUP)` | — | `-ENOTTY` (`-25`) | **Removed** vs AOSP | — |
| 8 | `0x0000ed08` | `_IO(0xed, 8)` | 0 | No-op returning `0` | `EDGETPU_FINALIZE_GROUP` no-op (`CREATE_GROUP` finalizes) | `libtachyon_core` |
| 9 | `0x4008ed09` | `_IOW(0xed, 9, struct edgetpu_event_register)` | 8 | `edgetpu_ioctl_set_perdie_eventfd` | Match (`0x1000` logs, `0x1001` traces) | — |
| 14 | `0x4004ed0e` | `_IOW(0xed, 14, __u32)` | 4 | `edgetpu_group_unset_eventfd` | Arg passed by value (`__u32 event_id`) | `libtachyon_core` |
| 15 | `0x4004ed0f` | `_IOW(0xed, 15, __u32)` | 4 | `edgetpu_ioctl_unset_perdie_eventfd` | Arg passed by value (`__u32 event_id`) | — |
| 16 | `0x4020ed10` | `_IOW(0xed, 16, struct edgetpu_sync_ioctl)` | 32 | `edgetpu_ioctl_sync_buffer` | Match | `libtachyon_core` |
| 17 | `0xc028ed11` | `_IOWR(0xed, 17, struct edgetpu_map_dmabuf_ioctl)` | 40 | `edgetpu_ioctl_map_dmabuf(..., limited=0)` | Match | `libtachyon_core` |
| 18 | `0x4028ed12` | `_IOW(0xed, 18, struct edgetpu_map_dmabuf_ioctl)` | 40 | `edgetpu_ioctl_unmap_dmabuf(..., limited=0)` | Match | `libtachyon_core` |
| 19 | `—` | `(EDGETPU_ALLOCATE_DEVICE_BUFFER)` | — | `-ENOTTY` | **Removed** vs AOSP | — |
| 20 | `0xc088ed14` | `_IOWR(0xed, 20, struct edgetpu_create_sync_fence_data)` | 136 | `edgetpu_ioctl_sync_fence_create` | Match | `libtachyon_core` |
| 21 | `0x4008ed15` | `_IOW(0xed, 21, struct edgetpu_signal_sync_fence_data)` | 8 | `edgetpu_ioctl_sync_fence_signal` | Match | `libtachyon_core` |
| 22–23 | `—` | `(MAP/UNMAP_BULK_DMABUF)` | — | `-ENOTTY` | **Removed** vs AOSP | — |
| 24 | `0xc008ed18` | `_IOWR(0xed, 24, struct edgetpu_sync_fence_status)` | 8 | `edgetpu_ioctl_sync_fence_status` | Match | `libtachyon_core` |
| 25 | `0x0000ed19` | `_IO(0xed, 25)` | 0 | `edgetpu_ioctl_release_wakelock` | Match | `libtachyon_core` |
| 26 | `0x0000ed1a` | `_IO(0xed, 26)` | 0 | `edgetpu_ioctl_acquire_wakelock(client, 0)` | Match | `libtachyon_core` |
| 27 | `0x8010ed1b` | `_IOR(0xed, 27, struct edgetpu_fw_version)` | 16 | Copy 16B `fw_version` from `etdev+0x358` | Match (`-ENODEV` if unpopulated `-1`) | `libtachyon_core` |
| 28 | `0x8008ed1c` | `_IOR(0xed, 28, __u64)` | 8 | `edgetpu_ioctl_tpu_timestamp` | Match (reads `bar + 0x1a01c0`) | — |
| 29 | `—` | `(EDGETPU_GET_DRAM_USAGE)` | — | `-ENOTTY` | **Removed** vs AOSP | — |
| 30 | `0x4018ed1e` | `_IOW(0xed, 30, struct edgetpu_ext_mailbox_ioctl)` | 24 | `edgetpu_ioctl_acquire_ext_mailbox` | Accepts ONLY `type == 1 (TZ)` and `count == 1` | — |
| 31 | `0x4018ed1f` | `_IOW(0xed, 31, struct edgetpu_ext_mailbox_ioctl)` | 24 | `edgetpu_ioctl_release_ext_mailbox` | Match | — |
| 32 | `0x8004ed20` | `_IOR(0xed, 32, __u32)` | 4 | `edgetpu_ioctl_get_fatal_errors` | Match | — |
| **33** | `0x8004ed21` | `_IOR(0xed, 33, __u32)` | 4 | `edgetpu_ioctl_acquire_wakelock(client, (u32)arg)` | **New**: value-passed flags (`bit 0` = suspendable); `_IOR` encoding despite value input | — |
| 34 | `—` | `(EDGETPU_SET_DEVICE_PROPERTIES)` | — | `-ENOTTY` | **Removed** vs AOSP | — |
| 35 | `0xc050ed23` | `_IOWR(0xed, 35, struct edgetpu_vii_command_ioctl)` | 80 | Returns `-EOPNOTSUPP` (`-95`) | **Disabled** in `buenos.ko` | Probe/error path only |
| 36 | `0xc018ed24` | `_IOWR(0xed, 36, struct edgetpu_vii_response_ioctl)` | 24 | Returns `-EOPNOTSUPP` (`-95`) | **Disabled** in `buenos.ko` | Probe/error path only |
| 37 | `0xc038ed25` | `_IOWR(0xed, 37, struct edgetpu_vii_litebuf_command_ioctl)` | 56 | `edgetpu_ioctl_vii_litebuf_command` | Rejects `seq >= 2^32`; waits on merged IKF (`0x9c4` = 2500ms); `size==0` -> `type=3`; IKV root `0x20`B | `libtachyon_core` |
| 38 | `0xc018ed26` | `_IOWR(0xed, 38, struct edgetpu_vii_litebuf_response_ioctl)` | 24 | `edgetpu_ioctl_vii_litebuf_response` | Allows `litebuf_address == 0` (returns `seq` + `code`); copies `0x30`B payload | `libtachyon_core` |
| **39** | `0x8008ed27` | `_IOR(0xed, 39, __u64)` | 8 | Copies constant `__u64 16` (`0x10`) | **New**: VII credit size probe (`16`, userspace falls back to `8` on `ENOTTY`) | `libtachyon_core`, `libdarwinn_hal`, `libedgetpu_util` |
| **40** | `0xc004ed28` | `_IOWR(0xed, 40, __u32)` | 4 | `edgetpu_ioctl_add_limited_interface` | **New**: value-passed `int limited_fd` (`_IOWR` encoding, not pointer) | `libtachyon_core` (`0xed1b0`) |
| **41** | `0x0000ed29` | `_IO(0xed, 41)` | 0 | `edgetpu_group_remap_buffers` | **New**: remaps trimmed group buffers (`gcip_mapping_buffer_remap`) | (Note: `libedgetpu_util` probes `0xc008ed29` elsewhere) |
| **42** | `0x8044ed2a` | `_IOR(0xed, 42, struct edgetpu_interface_version_ioctl)` | 68 | `edgetpu_ioctl_get_interface_version` | **New**: returns `major=1, minor=4, version_build[64]` (`"gcc524543248c"`) | `libtachyon_core`, `libedgetpu_litert` |
| **43** | `0x4004ed2b` | `_IOW(0xed, 43, __u32)` | 4 | `edgetpu_client_trim_enable(client, (u32)arg)` | **New**: value-passed bool (`pixel_trim`) | — |
| **44** | `0x4004ed2c` | `_IOW(0xed, 44, __u32)` | 4 | `edgetpu_client_set_tgid(client, (u32)arg)` | **New**: value-passed `vpid`/`tgid` | `libtachyon_core` (`0xed4f0`) |

`/dev/edgetpu-limited` (`edgetpu_limited_ioctl`):
- Strictly allows ONLY `0xc020ed00`, `0x4020ed04`, `0xc028ed11`, `0x4028ed12` (each called with `limited=1`); all other ioctls return `-ENOTTY`.

`edgetpu_mmap` (`buenos.ko`):
- Accepts ONLY offsets `0x1b00000 + i * 0x100000` for `i = 0..8`:
  - `i = 0,2,4,6` (`0x1b00000, 0x1d00000, 0x1f00000, 0x2100000`): `telemetry_log[0..3]`
  - `i = 1,3,5,7` (`0x1c00000, 0x1e00000, 0x2000000, 0x2200000`): `telemetry_trace[0..3]`
  - `i = 8` (`0x2300000`): **HWTRACE** buffer (`etdev + 0x1c8`, initialized by `edgetpu_telemetry_hwtrace_init` at IOVA `0x18000000`, KCI `0x1d`); returns `-ENOENT` if `[etdev+0x1f0] == NULL`.
- Legacy AOSP CSR/queue mmap offsets (`0x1500000..0x1a00000`) are absent and rejected with `-EINVAL`.

---

## 4. GXP (`/dev/gxp`) Dispatch Matrix (`gxp.ko`)

Base magic: `GXP_IOCTL_BASE = 0xEE`. `gxp_ioctl` delegates first to `gxp->handle_ioctl` (`gxp_mcu_ioctl` in MCU mode) and falls back on `-EOPNOTSUPP` (`-95`). Note jump table `0x4004ee13..0x4004ee2b` validates unhandled entries -> `-ENOTTY`.

| nr | Observed Hex | Route | `_IOC_SIZE` | Handler / Behavior (`gxp.ko`) | AOSP `gxp/zuma` Status & Delta |
|---:|---|---|---:|---|---|
| 0, 1 | `0xc020ee00`, `0x4020ee01` | `gxp_ioctl` | 32 | `GXP_MAP_BUFFER`, `GXP_UNMAP_BUFFER` | Identical |
| 2 | `0x4018ee02` | `gxp_ioctl` | 24 | `GXP_SYNC_BUFFER` | Identical |
| 4 | `0xc018ee04` | `gxp_mcu_ioctl` | 24 | Returns `-EOPNOTSUPP` (`-95`) in MCU mode | Identical MCU rejection (`direct`-mode fallback if direct) |
| 5 | `0x8010ee05` | `gxp_ioctl` | 16 | `GXP_GET_SPECS` | **Delta**: `core_count=3` (Zuma: 4), `max_vd_allocation=15`, `secure_telemetry_buffer_size=1` |
| 6 | `0xc00cee06` | `gxp_ioctl` | 12 | `GXP_ALLOCATE_VIRTUAL_DEVICE` | **Delta**: `gxp_vd_allocate` rejects `core_count == 0 \|\| core_count >= 4` (only `1..3`) |
| 13, 14 | `0x4010ee0d`, `0x4010ee0e` | `gxp_ioctl` | 16 | `GXP_MAP_TPU_MBX_QUEUE`, `GXP_UNMAP_TPU_MBX_QUEUE` | Identical (`gxp_mcu_after_map_tpu_mbx_queue` queries `buenos.ko`) |
| 15, 16 | `0x4008ee0f`, `0x4008ee10` | `gxp_ioctl` | 8 | `GXP_{UN,}REGISTER_CORE_TELEMETRY_EVENTFD` | Identical |
| 17 | `0x8008ee11` | `gxp_ioctl` | 8 | `GXP_READ_GLOBAL_COUNTER` | Identical (used by `gxp_telemetry_reader.so`) |
| 19 | `0x4004ee13` | `gxp_ioctl` | 4 | `GXP_RELEASE_WAKE_LOCK` | Identical (value-passed `u32` flags) |
| 20, 21 | `0xc018ee14`, `0x4018ee15` | `gxp_ioctl` | 24 | `GXP_MAP_DMABUF`, `GXP_UNMAP_DMABUF` | Identical |
| 22, 24 | `0x400cee16`, `0x400cee18` | `gxp_ioctl` | 12 | `GXP_{UN,}REGISTER_MAILBOX_EVENTFD` | Identical |
| 23 | `0xc030ee17` | `gxp_mcu_ioctl` | 48 | Returns `-EOPNOTSUPP` (`-95`) in MCU mode | Identical MCU rejection |
| 25 | `0x4014ee19` | `gxp_ioctl` | 20 | `GXP_ACQUIRE_WAKE_LOCK` | Identical (`aur_memory_state_array` `{0..6}`) |
| 26 | `0x8044ee1a` | `gxp_ioctl` | 68 | `GXP_GET_INTERFACE_VERSION` | **Delta**: returns **`major=1, minor=33` (`0x21`)** (Zuma: `1.31`), `build="gcc97177d49ea"` |
| 27 | `0x4004ee1b` | `gxp_ioctl` | 4 | `GXP_TRIGGER_DEBUG_DUMP` | Identical |
| 28, 29 | `0x4008ee1c`, `0x4008ee1d` | `gxp_mcu_ioctl` | 8 | `GXP_{UN,}REGISTER_MCU_TELEMETRY_EVENTFD` | Identical (`gcip_telemetry_{set,unset}_event`) |
| 30 | `0xc040ee1e` | `gxp_mcu_ioctl` | 64 | `GXP_MAILBOX_UCI_COMMAND_COMPAT` | Identical |
| 31 | `0x8020ee1f` | `gxp_mcu_ioctl` | 32 | `GXP_MAILBOX_UCI_RESPONSE` | Identical |
| 32–34 | `0xc088ee20`, `0x4008ee21`, `0xc008ee22` | `gxp_ioctl` | 136, 8, 8 | `GXP_{CREATE,SIGNAL}_SYNC_FENCE`, `STATUS` | Identical |
| 35, 36 | `0x4004ee23`, `0x4004ee24` | `gxp_ioctl` | 4 | `GXP_{UN,}REGISTER_INVALIDATED_EVENTFD` | Identical |
| 37 | `0x4100ee25` | `gxp_mcu_ioctl` | 256 | `GXP_SET_DEVICE_PROPERTIES` | Identical (`256B` opaque, KCI `0x16`) |
| 38 | `0x8004ee26` | `gxp_ioctl` | 4 | `GXP_GET_INVALIDATED_REASON` | Identical |
| **39** | `0xc080ee27` | `gxp_mcu_ioctl` | 128 | `GXP_MAILBOX_UCI_COMMAND` | **Delta**: tail `+0x60` (`mid_in_fences`) & `+0x68` (`mid_out_fences`) parsed -> 64B `gxp_uci_additional_info_root` |
| 40 | `0xc008ee28` | `gxp_mcu_ioctl` | 8 | `GXP_CREATE_IIF_FENCE` | Identical |
| 41 | `0xc024ee29` | `gxp_mcu_ioctl` | 36 | `GXP_FENCE_REMAINING_SIGNALERS` | Identical |
| 42, 43 | `0xc010ee2a`, `0x4010ee2b` | `gxp_ioctl` | 16 | `GXP_{RESERVE,RETIRE}_IOVA_REGION` | Identical (unused in `libgxp.so`) |

Note on `GXP_MAILBOX_UCI_COMMAND` (`0xc080ee27`, 128B) in `gxp_mcu_ioctl` (`0x24fa0..0x25268`):
- `gxp_mcu_ioctl` passes `&cmd[+0x60]` and `&cmd[+0x68]` (8-byte spacing) to `get_num_fences()` and `gcip_fence_array_create()`, then passes both arrays AND the raw FD pointers (`&cmd[+0x60]`, `&cmd[+0x68]`) into `gxp_mcu_fs_send_uci_cmd` -> `gxp_uci_send_cmd`.
- In `gxp_uci_send_cmd` / `gxp_uci_fill_additional_info`, `gxp_uci_additional_info_root.object_size` increased from `0x20` (32B in Zuma) to `0x40` (64B) to hold `(mid_in_fence_fds, mid_in_fence_iif_ids, mid_out_fence_fds, mid_out_fence_iif_ids)` offset/count pairs.
- Subtle implementation note: `get_num_fences()` scans up to 4 `__s32` elements stopping at `0` or `-1`. When `mid_out_fences` starts at `0` or `-1` (e.g., `libgxp.so` zeros `+0x60..+0x7f`), `mid_in_fences` cleanly accommodates up to 2 FDs; if 2+ valid FDs were placed in `mid_in_fences` AND a non-zero/non-`-1` FD was placed in `mid_out_fences[0]`, `get_num_fences(&mid_in_fences)` would read across into `mid_out_fences[0]`.

---

## 5. Evidence Classification Ledger

### A. Observed from `.ko` Disassembly & Userland Binaries
- Exact ioctl dispatch tables, command numbers, directions, and `_IOC_SIZE` for `buenos.ko` (`edgetpu_ioctl`, `edgetpu_limited_ioctl`) and `gxp.ko` (`gxp_ioctl`, `gxp_mcu_ioctl`), cross-matched to `vendor/lib64/{libtachyon_core,libedgetpu_litert,libgxp,libtachyon_dsp_core,gxp_telemetry_reader}.so`.
- `buenos.ko` new/changed ioctls (`33, 39..44`), disabled VII legacy (`35, 36 -> -EOPNOTSUPP`), `FINALIZE_GROUP` no-op (`8`), `edgetpu-limited` 4-ioctl allowlist (`0, 4, 17, 18`), `edgetpu_mmap` indices `0..8` (`0x2300000` HWTRACE), Litebuf `seq < 2^32` constraint, `0x20B` `edgetpu_ikv_additional_info_root`, and KCI codes `6, 28, 29, 30` (`bcl_mitigation`, `fw_log_state`, `map_hwtrace_buffer`, `coresight_remote_cmd`).
- `gxp.ko` `core_count = 3`, `max_vd_allocation = 15`, `secure_telemetry_buffer_size = 1`, `GXP_GET_INTERFACE_VERSION = 1.33`, `GXP_MAILBOX_UCI_COMMAND` `+0x60`/`+0x68` mid-fence extraction, `0x40B` `gxp_uci_additional_info_root`, and `gxp.ko -> buenos.ko` `edgetpu_ext_driver_cmd` (`cmds 0..3`).

### B. From AOSP Source (`edgetpu-rio-android16`, `gxp-zuma-android16`)
- Exact C struct member names for unchanged ioctl payloads (`edgetpu_map_ioctl`, `edgetpu_event_register`, `edgetpu_sync_ioctl`, `edgetpu_map_dmabuf_ioctl`, `edgetpu_create_sync_fence_data`, `edgetpu_fw_version`, `edgetpu_ext_mailbox_ioctl`, `edgetpu_vii_litebuf_*`, `gxp_map_ioctl`, `gxp_specs_ioctl`, `gxp_acquire_wakelock_ioctl`, `gxp_tpu_mbx_queue_ioctl`, `gxp_*_fence*`, `iif_*`, `gcip_kci_*`, `gcip_fw_info`, `gcip_telemetry_header`), verified to match `.ko` sizes and field offsets accessed in disassembly.

### C. Unknowns / Unverified
- **Canonical preprocessor macro names** for `buenos.ko` additions `nr = 33, 39, 40, 41, 43, 44`, `EDGETPU_MMAP_HWTRACE_BUFFER_OFFSET`, `EDGETPU_VII_LITEBUF_NULL_COMMAND`, KCI codes `6, 28, 29, 30`, and the `mid_in_fences`/`mid_out_fences` field names in `gxp_mailbox_uci_command_ioctl` (module ELF preserves C function names like `edgetpu_client_trim_enable`, `edgetpu_client_set_tgid`, `edgetpu_group_remap_buffers`, `edgetpu_ioctl_add_limited_interface`, `edgetpu_kci_bcl_mitigation`, not `#define` identifiers; `EDGETPU_GET_CREDIT_SIZE` is inferred from `"Failed to probe for credit size: %s"`).
- **Exact payload layouts of opaque kernel-firmware buffers** when not parsed by host `.ko` code: `wait_data` passed to `edgetpu_ikv_additional_info_fill` (always `NULL/0` from `edgetpu_ioctl_vii_litebuf_command`), `bcl_mitigation` 32B KCI payload (`cmd 6`), `coresight_remote_cmd` payload (`cmd 30`), `edgetpu_ext_driver_cmd` `cmd 0` internal `in`/`out` structs beyond the observed `tpu_fd` (`+0`), `attr_ptr` (`+8`), and `flags` (`+0x18`), and the full `dt` property nodes from `boot`/`vendor_boot` DTB (`dtbo.img` only holds overlay fragments such as `qos_box_tpu_*` / `thermal_zones_laguna`).
- **Live runtime/device execution**: not tested (Phase 1 offline-only).

---

## 6. Verification Results

Compiled with NDK r26 `aarch64-linux-android34-clang` (`-x c -fsyntax-only -Wall -Wextra -Werror`):
- `docs/kernel-abi/gcip_uapi.h`: **PASS** (exit `0`, 0 warnings)
- `docs/kernel-abi/edgetpu_buenos_uapi.h`: **PASS** (exit `0`, 0 warnings)
- `docs/kernel-abi/gxp_laguna_uapi.h`: **PASS** (exit `0`, 0 warnings)
- `work/kernel-driver/verify_uapi.c`: **PASS** (exit `0`, 0 warnings — `_Static_assert` checks across all `/dev/iif`, `/dev/edgetpu-soc`, and `/dev/gxp` ioctl hex constants and wire/UAPI struct `sizeof()` values).
