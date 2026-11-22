# Tensor G5 (`buenos`) DVFS, Power States, Memory QoS & WakeLock Wire Specification

Clean-room technical specification of TPU core frequency scaling, memory/fabric power states, firmware priority, QoS class, WakeLocks, and cluster addressing on Pixel 10 (`rango`, Tensor G5 `buenos`), verified via clean-room wire protocol inspection and live bare-metal execution on silicon.

Companion header: [`litebuf_vii_wire.h`](litebuf_vii_wire.h). Kernel ioctl ABI: [`docs/kernel-abi/README.md`](../kernel-abi/README.md).

---

## 1. Critical Finding: 1-Byte Field Shift in Legacy `litebuf_vii_wire.h`

Wire-level analysis and live captures across `perf_mode 0..5` prove that bytes `+0x1c..+0x1f` of `struct litebuf_vii_execute` were shifted by **1 byte** in the earlier Phase 3/4 header [`litebuf_vii_wire.h`](litebuf_vii_wire.h):

| Outer Offset (80B pkt) | `litebuf_vii_execute` Offset | Old `litebuf_vii_wire.h` Name (WRONG) | **Actual Wire Field** | Valid Values / Default |
|---:|---:|---|---|---|
| `+0x16` | root `+0x06` | `priority` | `root_flags` / `is_preemptible` | `0x00` |
| `+0x20` | `+0x08` | `dive_cmd_address` (`u64`) | `dive_cmd_address` (`u64`) | 64-bit IOVA |
| `+0x28` | `+0x10` | `dive_cmd_size_bytes` (`u32`) | `dive_cmd_size_bytes` (`u32`) | inner DIVE byte size |
| `+0x30` | `+0x18` | `model_id` (`u32`) | `model_handle` (`u32`) | from `RegisterModelResponse` |
| **`+0x34`** | **`+0x1c`** | ~~`tpu_core_power_state`~~ | **`fw_priority`** (`uint8_t`) | **`0x0b`** (`11`, default; lower = higher prio; `0x0c`/`0x10` slower) |
| **`+0x35`** | **`+0x1d`** | ~~`memory_power_state`~~ | **`tpu_core_power_state`** (`uint8_t`) | `0x01`, **`0x03..0x19`** (`0x03`=38.4 MHz, `0x16`=1113.6 MHz, **`0x19`=1286.4 MHz MAX**) |
| **`+0x36`** | **`+0x1e`** | ~~`qos_class`~~ | **`memory_power_state`** (`uint8_t`) | `0x01`, **`0x02..0x15`** (`0x01`=min, `0x09`/`0x0d`/`0x11`/**`0x15`=MAX**) |
| **`+0x37`** | **`+0x1f`** | ~~`perf_metrics_flag`~~ | **`qos_class`** (`uint8_t`) | **`{0, 1, 2, 4, 5}`** (`3` invalid; default **`0x01`**) |
| **`+0x38`** | **`+0x20`** | `cluster_id` (`uint8_t`) | **`cluster_id`** (`uint8_t`) | **`0x02`** (`ctz(0x04)` on `rango`) |
| `+0x40` | `+0x28` | `core_session_handle` (`u32`) | `core_session_handle` (`u32`) | `0xffffffff` (`NONE`) |

**Root cause of Phase 5B's `38.4 MHz` bottleneck**: Bare-metal sent `(0x0b, 0x03, 0x01, 0x01, 0x02)` at `+0x1c..+0x20`, which accidentally set `tpu_core_power_state = 0x03` (`38.4 MHz` sleep OPP) and `memory_power_state = 0x01` (OFF/MIN DRAM).

---

## 2. Exact `edgetpu_performance_mode` (`0..5`) Wire Mapping

Coarse `(dev_ps, mem_ps)` performance presets map onto the wire fields `(+0x1c, +0x1d, +0x1e, +0x1f)` as follows:

| `edgetpu_performance_mode` | Coarse `(dev_ps, mem_ps)` | Wire `+0x1c` (`fw_prio`) | Wire `+0x1d` (`core_ps`) | Wire `+0x1e` (`mem_ps`) | Wire `+0x1f` (`qos`) | Observed `rango` Peak TPU Clock | Live `1x2048x2048` HW (`rango`) |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0` (MIN / LOWEST) | `(3, 1)` | `0x0b` | **`0x03`** | **`0x01`** | `0x01` | `38.4 MHz` | `5,869 us` (`0.71 GB/s`) |
| `1` (LOW / POWER_SAVE) | `(4, 3)` | `0x0b` | **`0x0c`** | **`0x09`** | `0x01` | mid OPP | `611 us` (`6.86 GB/s`) |
| `2` (MEDIUM / BALANCED) | `(6, 4)` | `0x0b` | **`0x16`** | **`0x0d`** | `0x01` | `1,113.6 MHz` | `177–490 us` (`8.6–23.7 GB/s`) |
| `3` (HIGH) | `(6, 6)` | `0x0b` | **`0x16`** | **`0x15`** | `0x01` | `1,113.6–1,286.4 MHz` | `360 us` (`11.65 GB/s`) |
| `4` (MAX / SUSTAINED) | `(6, 4)` | `0x0b` | **`0x16`** | **`0x0d`** | `0x01` | `1,113.6 MHz` | `177–490 us` |
| `5` (OVERDRIVE preset) | `(6, 6)` | `0x0b` | **`0x16`** | **`0x15`** | `0x01` | `1,113.6–1,286.4 MHz` | `360 us` |
| **Bare-Metal Peak (`core=0x19`)** | *(above `0x16` preset)* | `0x0b` | **`0x19`** | **`0x0d`** / **`0x15`** | `0x01` | **`1,286.4 MHz`** | **`158–160 us` (`26.55 GB/s`, `37.1x`)** |

---

## 3. Complete Litebuf VII Power-State & QoS Wire Enum Tables

### 3.1 `TpuCorePowerState` (`+0x1d` in `Execute`, `+0x02` in `AcquireWakeLock`)

Valid wire range is **`0x01` and `0x03..0x19`** (`0x00` = `UNDEFINED` rejected; `0x02` unused; **`>= 0x1a` rejected by firmware** on `rango`):

| Wire Byte | Fine Value(s) | Coarse Alias | Notes / `rango` OPP Behavior |
|---:|---|---|---|
| `0x00` | `0` | `kUndefined` | Rejected |
| `0x01` | `1` | `kOff` / min | Power-off / idle floor |
| **`0x03`** | `2, 3, 0x18` | `kVeryLow` (`perf_mode=0`) | **`38.4 MHz`** (OPP 0) |
| `0x04..0x08` | `4..8` | fine OPPs | intermediate frequencies |
| `0x09` | `9, 0x1c` | coarse alias | intermediate |
| `0x0a..0x0b` | `10..11` | fine OPPs | intermediate |
| **`0x0c`** | `12, 0x19` | `kLow` (`perf_mode=1`) | mid OPP (`611 us` on `1x2048x2048`) |
| `0x0d..0x0e` | `13..14` | fine OPPs | intermediate |
| `0x0f` | `15, 0x1d` | coarse alias | intermediate |
| `0x10` | `16` | fine OPP | intermediate |
| **`0x11`** | `17, 0x1a` | `kMedium` | mid-high OPP |
| `0x12` | `18` | fine OPP | intermediate |
| `0x13` | `19, 0x1e` | coarse alias | high OPP |
| `0x14..0x15` | `20..21` | fine OPPs | high OPPs |
| **`0x16`** | `22, 0x1b` | `kHigh` / `kMax` (`perf_mode=2..5`) | **`1,113.6 MHz`** (OPP 13; `862.8 INT8 GOPS` on `64x1536x1536` at `164.8 mW`) |
| **`0x17`** | `23, 0x1f` | `kOverdrive` | **`1,209.6–1,286.4 MHz`** (OPP 14–15) |
| **`0x18`** | `0x20` | fine OPP | **`1,286.4 MHz`** (OPP 15) |
| **`0x19`** | `0x21` | **FW MAX** | **`1,286.4 MHz`** (OPP 15; **`158 us` / `26.55 GB/s`** on `1x2048x2048` at `223.2 mW`) |
| `>= 0x1a` | *(invalid)* | — | **Rejected by FW (`err = -1`)** |

*Reference: `rango` `36500000.buenos` exposes 16 hardware OPPs: `[38.4, 134.4, 192, 288, 403.2, 480, 595.2, 691.2, 768, 825.6, 883.2, 960, 1036.8, 1113.6, 1209.6, 1286.4] MHz`.*

### 3.2 `MemoryPowerState` (`+0x1e` in `Execute`, `+0x03` in `AcquireWakeLock`)

Valid wire range is **`0x01..0x15`** (`0x00` = `UNDEFINED` rejected; **`>= 0x16` rejected by firmware**):

| Wire Byte | Fine Value | Coarse Preset (`perf_mode`) | Role |
|---:|---|---|---|
| `0x00` | `0` | `kUndefined` | Rejected |
| **`0x01`** | `1` | `mem_ps=1` (`perf_mode=0`) | Idle/minimum memory & fabric floor (`409 MHz` GMC) |
| `0x02..0x04` | `7..9` | — | Fine low memory levels |
| **`0x05`** | `10` (`0x0a`) | `mem_ps=2` | Coarse level 2 |
| `0x06..0x08` | `11..13` | — | Fine levels |
| **`0x09`** | `14` (`0x0e`) | `mem_ps=3` (`perf_mode=1`) | Coarse level 3 |
| `0x0a..0x0c` | `15..17` | — | Fine levels |
| **`0x0d`** (`13`) | `18` (`0x12`) | `mem_ps=4` (`perf_mode=2, 4`) | **Best sweet spot on `rango`** (`158 us` @ `core=0x19`, `350 us` @ `core=0x16`) |
| `0x0e..0x10` | `19..21` | — | Fine levels |
| **`0x11`** (`17`) | `22` (`0x16`) | `mem_ps=5` | Coarse level 5 |
| `0x12..0x14` | `23..25` | — | Fine high memory levels |
| **`0x15`** (`21`) | `26` (`0x1a`) | `mem_ps=6` (`perf_mode=3, 5`) | **FW MAX** memory/fabric vote (`160 us` @ `core=0x19`) |
| `>= 0x16` | *(invalid)* | — | **Rejected by FW (`err = -1`)** |

### 3.3 `QosClass` (`+0x1f` in `Execute`)

- Valid bitmask: `(v < 6) && ((0x37 >> v) & 1)` -> valid values are **`{0, 1, 2, 4, 5}`** (**`3` is INVALID**).
- Wire byte is identity on `{0, 1, 2, 4, 5}`. Default is **`0x01`**; `0x02` is used in Peak mode (`PM=5`).

### 3.4 `fw_priority` (`+0x1c` in `Execute`)

Clamped to `[0, 255]`, default **`0x0b` (`11`)**.
Lower values represent higher firmware scheduling priority (live `rango` test: `0x0b` -> `360 us`, `0x0c` -> `396 us`, `0x10` -> `513 us` at `core=0x16, mem=0x15`; `0x00`/`0x01` yield lowest jitter in `PM=4`/`PM=5`).

---

## 4. Firmware WakeLock (`Litebuf type=4 / type=5`) vs Kernel WakeLock & DVFS

### 4.1 Two Distinct WakeLock Layers

1. **Kernel ioctl `EDGETPU_ACQUIRE_WAKE_LOCK`** (`nr 26 = 0x0000ed1a` or `nr 33 = 0x8004ed21` with flags):
   - Powers on the TPU block (`edgetpu_pm_get`), checks thermal suspend, and attaches the client's VII mailbox.
   - Does **not** set TPU OPP or memory/fabric OPP — without a Litebuf power state or Litebuf WakeLock, the TPU idles/gates at `38.4 MHz` between executes.
2. **Firmware Litebuf `AcquireWakeLock` (`union_type = 4`, `32 B`) / `ReleaseWakeLock` (`union_type = 5`, `32 B`)**:
   - Payload `struct litebuf_vii_wakelock_cmd` (defined in [`litebuf_vii_wire.h`](litebuf_vii_wire.h)):
     - `+0x02`: `tpu_core_power_state` (`0x03..0x19`, e.g. **`0x19`**)
     - `+0x03`: `memory_power_state` (`0x01..0x15`, e.g. **`0x0d`** or **`0x15`**)
     - `+0x04`: `cluster_ids_bitset` (`uint8_t`, **bitmask `1 << cluster_id` = `0x04`** on `rango` — NOT `cluster_id = 0x02`)
   - Pins both TPU core clock and memory/fabric OPPs warm across back-to-back commands, avoiding per-command power-gate churn and DVFS ramp-up jitter.
   - **Live `rango` Silicon Proof (300 iters, `max_abs_err = 0`, zero Google `.so`s)**:

| Workload (`rango` bare-metal) | FW WL (`type=4`) | `(fw_prio, core_ps, mem_ps, qos)` | `hw_us` | **`wall_ms`** | **`WtBW`** | **`GOPS`** | Peak TPU MHz | ODPM TPU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `1x2048x2048` (Phase 5B old) | `0` | `(0x0b, 0x03, 0x01, 0x01)` | `5,869` | `6.594` | `0.71 GB/s` | `1.4` | `38.4` | `26.1 mW` |
| `1x2048x2048` (no FW WL) | `0` | `(0x0b, 0x19, 0x0d, 0x01)` | `341` | `1.184` | `12.30 GB/s` | `24.6` | `1,286.4` | `203.1 mW` |
| **`1x2048x2048` (+FW WL)** | **`1`** | **`(0x01, 0x19, 0x15, 0x02)`** | **`152`** | **`0.482`** (**`13.7x`**) | **`27.59 GB/s`** | **`55.2`** | **`1,286.4`** | **`114.2 mW`** |
| `64x1536x1536` (Phase 5B old) | `0` | `(0x0b, 0x03, 0x01, 0x01)` | `10,101` | `12.190` | `0.23 GB/s` | `29.8` | `38.4` | `41.4 mW` |
| **`64x1536x1536` (+FW WL)** | **`1`** | **`(0x00, 0x19, 0x0d, 0x01)`** | **`307`** | **`0.674`** (**`18.1x`**) | **`7.69 GB/s`** | **`983.6`** | **`1,286.4`** | **`177.2 mW`** |
| **`64x1536x1536` (+FW WL)** | **`1`** | **`(0x00, 0x19, 0x15, 0x01)`** | **`311`** | **`0.642`** (**`19.0x`**) | **`7.59 GB/s`** | **`971.0`** | **`1,286.4`** | **`220.3 mW`** |

   - Sending **1 Litebuf `AcquireWakeLock` (`union_type = 4`)** on context init/mode change (and `ReleaseWakeLock` `union_type = 5` on destroy) **cuts sync `wall_ms` `2.46x` (`1.184 -> 0.482 ms`)**, **doubles sustained BW (`12.3 -> 27.59 GB/s`)**, **nearly halves TPU power (`203 -> 114 mW`)** by eliminating per-call power-gate churn, and achieves **`983.6 INT8 GOPS` (`~1.0 TOPS`, `5.55 TOPS/W`)**.

### 4.2 Kernel Driver (`buenos.ko`) DVFS, MIF/GMC & Thermal Architecture

- **No `EDGETPU_SET_POWER_STATE` ioctl exists** in `buenos.ko` ([`docs/kernel-abi/README.md`](../kernel-abi/README.md) §3, `nr 0..44`).
- **No kernel RKCI MIF/INT vote on `buenos`**: Unlike older Exynos `mobile-soc-gsx01.c`, `buenos.ko`'s `edgetpu_soc_handle_reverse_kci` unconditionally returns `-95` (`-EOPNOTSUPP`). On Tensor G5, TPU firmware drives DVFS and memory/fabric (`GMC`/`FABHBW`/`MEMSS`) votes directly via SoC firmware channels using the Litebuf `tpu_core_power_state` and `memory_power_state` fields.
- **Current-frequency readout (`edgetpu_soc_pm_get_rate`)**: Reads BAR0 MMIO `+0x1e0030` (`readl_relaxed`): bit `0` = block active; bits `17:13` index `etdev->active_states[]` (the 16 OPPs).
- **Frequency floor/ceiling (`devfreq` & thermal)**: Root-writable cooling/devfreq caps exist via KCI opcode `0x18` (`{min_khz, max_khz}`) and `gcip-thermal.c` sysfs, but are not required for normal unprivileged bare-metal full-clock operation (Litebuf `core=0x19` reaches peak `1,286.4 MHz`).
- **Mailbox group creation (`EDGETPU_CREATE_GROUP`, `struct edgetpu_mailbox_attr`)**: `priority: 4`, `cmdq_tail_doorbell: 1`, `partition_type: 1`, `client_priv: 1`, `partition_type_high: 1`.

---

## 5. Cluster Addressing (`cluster_id` / `cluster_ids_bitset`) & Parallelism

- `cluster_id` is `ctz(cluster_ids_bitset)` (trailing zero count, valid range `0..6`).
- Tensor G5 (`buenos`) executables set `hardware_cluster_id = 2`, corresponding to `cluster_ids_bitset = (1u << 2) = 0x04` and Litebuf `cluster_id = 0x02`.
- **`0x02` is Tensor G5 `buenos`'s cluster index, not a 2-core mask.**
- **Concurrency on `rango`**:
  1. **16-credit async VII ring pipelining** (`Q/K/V`, `up/gate` queued back-to-back without waiting between submissions).
  2. **Single-VII multi-op chaining** (`0xa40` chained graphs like `fuse_swiglu_64` and `fuse_qkv_64` — 3 GEMMs in `65–73 us` total in 1 VII call).

---

## 6. Clean-Room C++17 Wire Preset Table for `libopenbuenos` (`ob_device.cpp`)

Drop-in mapping for `OPENBUENOS_PERF_MODE` (`0..5`, **default `5`**) using the corrected `+0x1c..+0x20` layout (implemented in [`libopenbuenos/src/ob_device.cpp`](../../libopenbuenos/src/ob_device.cpp)):

```cpp
struct ObPowerProfile {
  uint8_t fw_priority;           // Execute +0x1c (default 0x0b; 0x00..0x01 for PM=4..5)
  uint8_t tpu_core_power_state;  // Execute +0x1d (0x03..0x19; >=0x1a rejected)
  uint8_t memory_power_state;    // Execute +0x1e (0x01..0x15; >=0x16 rejected)
  uint8_t qos_class;             // Execute +0x1f ({0,1,2,4,5}; 3 invalid)
};

// Matches perf_mode 0..3 + unlocks full 1286.4 MHz OPP 15 + FW WakeLock on modes 4..5
static constexpr ObPowerProfile kObPowerProfiles[6] = {
  /* 0: MIN (38.4 MHz)             */ { 0x0b, 0x03, 0x01, 0x01 },
  /* 1: LOW / POWER_SAVE           */ { 0x0b, 0x0c, 0x09, 0x01 },
  /* 2: BALANCED (1113.6 MHz)      */ { 0x0b, 0x16, 0x0d, 0x01 },
  /* 3: HIGH (1113-1286 MHz)       */ { 0x0b, 0x16, 0x15, 0x01 },
  /* 4: MAX (1286.4 MHz, mem=0x0d) */ { 0x00, 0x19, 0x0d, 0x01 },
  /* 5: PEAK (1286.4 MHz, 27.6GB/s)*/ { 0x01, 0x19, 0x15, 0x02 },
};

// Issue 1 Litebuf AcquireWakeLock (union_type=4) on context init / perf_mode change,
// and ReleaseWakeLock (union_type=5) on teardown or PM=0:
//   tpu_core_power_state = prof.tpu_core_power_state
//   memory_power_state   = prof.memory_power_state
//   cluster_ids_bitset   = static_cast<uint8_t>(1u << cluster_id); // 0x04 on rango
```

Note: `litebuf_build_vii_execute` in [`litebuf_vii_wire.h`](litebuf_vii_wire.h) retains its original four `uint8_t` parameter names `(tpu_core_power_state, memory_power_state, qos_class, perf_metrics_flag)` for source compatibility at wire offsets `(+0x1c, +0x1d, +0x1e, +0x1f)`. Callers pass `(fw_priority, tpu_core_power_state, memory_power_state, qos_class)` in those four positional slots.
