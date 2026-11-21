/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Clean-room UAPI Header for Tensor G5 EdgeTPU Driver (`buenos.ko`)
 * Device nodes: /dev/edgetpu-soc (/dev/edgetpu symlink) and /dev/edgetpu-limited
 * Build provenance: rango CP3A.261005.005 (Linux 6.6.127-android15, DT alias google,edgetpu-laguna)
 *
 * Strictly distinguishes observed .ko ABI vs AOSP edgetpu/rio definitions:
 *   - Interface version returned by EDGETPU_GET_INTERFACE_VERSION (0x8044ed2a): 1.4
 *   - /dev/edgetpu-limited ONLY permits nrs {0, 4, 17, 18} once linked via nr 40.
 *   - Legacy VII ioctls 35 & 36 return -EOPNOTSUPP (-95); use Litebuf VII (37 & 38).
 *   - Removed AOSP rio ioctls (return -ENOTTY): 7, 19, 22, 23, 29, 34.
 *   - Added laguna ioctls: 33 (wakelock flags), 39 (credit size = 16), 40 (limited iface),
 *     41 (remap trimmed buffers), 42 (interface version 1.4), 43 (trim enable), 44 (set tgid).
 *   - Added mmap offset 0x2300000 (index 8) for hardware trace telemetry (hwtrace).
 */

#ifndef __EDGETPU_BUENOS_UAPI_H__
#define __EDGETPU_BUENOS_UAPI_H__

#include <linux/ioctl.h>
#include <linux/types.h>

#ifndef __edgetpu_packed
#define __edgetpu_packed __attribute__((__packed__))
#endif

#define EDGETPU_INTERFACE_VERSION_MAJOR 1
#define EDGETPU_INTERFACE_VERSION_MINOR 4

#define EDGETPU_IOCTL_BASE 0xED

/* =========================================================================
 * mmap() offsets (verified in buenos.ko `edgetpu_mmap`)
 * Note: 0x1500000..0x1a00000 CSR/queue offsets from AOSP rio are NOT mapped
 * in buenos.ko `edgetpu_mmap`, which strictly validates offset in
 * [0x1b00000 .. 0x2300000] aligned to 1 MiB (index 0..8).
 * ========================================================================= */
#define EDGETPU_MMAP_LOG_BUFFER_OFFSET    0x1B00000 /* index 0: telemetry log[0] */
#define EDGETPU_MMAP_TRACE_BUFFER_OFFSET  0x1C00000 /* index 1: telemetry trace[0] */
#define EDGETPU_MMAP_LOG1_BUFFER_OFFSET   0x1D00000 /* index 2: telemetry log[1] */
#define EDGETPU_MMAP_TRACE1_BUFFER_OFFSET 0x1E00000 /* index 3: telemetry trace[1] */
#define EDGETPU_MMAP_LOG2_BUFFER_OFFSET   0x1F00000 /* index 4: telemetry log[2] */
#define EDGETPU_MMAP_TRACE2_BUFFER_OFFSET 0x2000000 /* index 5: telemetry trace[2] */
#define EDGETPU_MMAP_LOG3_BUFFER_OFFSET   0x2100000 /* index 6: telemetry log[3] */
#define EDGETPU_MMAP_TRACE3_BUFFER_OFFSET 0x2200000 /* index 7: telemetry trace[3] */
/* Observed in buenos.ko (edgetpu_mmap case 8 -> etdev+0x1c8, edgetpu_telemetry_hwtrace_init): */
#define EDGETPU_MMAP_HWTRACE_BUFFER_OFFSET 0x2300000 /* index 8: telemetry hwtrace */

/* =========================================================================
 * Memory Mapping Flags (verified in edgetpu_mappings_encode_gcip_map_flags)
 * ========================================================================= */
typedef __u32 edgetpu_map_flag_t;

#define EDGETPU_MAP_DIR_MASK           3u
#define EDGETPU_MAP_DMA_BIDIRECTIONAL  0u
#define EDGETPU_MAP_DMA_TO_DEVICE      1u
#define EDGETPU_MAP_DMA_FROM_DEVICE    2u
#define EDGETPU_MAP_DMA_NONE           3u

/*
 * Bit 2 (0x04): legacy AOSP mirrored/nonmirrored (unobserved in buenos.ko encode)
 * Bit 3 (0x08): AOSP CPU_NONACCESSIBLE; in buenos.ko when bit 3 == 1, IOVA is
 *               NOT restricted to 32-bit (logs "36-bit" vs "32-bit").
 * Bit 4 (0x10): SKIP_CPU_SYNC / DMA_ATTR bit (passed shifted into dma_attr).
 * Bit 9 (0x200): COHERENT; buenos.ko forces (flags |= 0x10) when bit 9 is set.
 */
#define EDGETPU_MAP_MIRRORED           (0u << 2)
#define EDGETPU_MAP_NONMIRRORED        (1u << 2)
#define EDGETPU_MAP_CPU_ACCESSIBLE     (0u << 3)
#define EDGETPU_MAP_CPU_NONACCESSIBLE  (1u << 3)
#define EDGETPU_MAP_36BIT_IOVA         (1u << 3) /* Observed in buenos.ko error format */
#define EDGETPU_MAP_SKIP_CPU_SYNC      (1u << 4)
#define EDGETPU_MAP_ATTR_PBHA_SHIFT    5
#define EDGETPU_MAP_ATTR_PBHA_MASK     0xfu
#define EDGETPU_MAP_COHERENT           (1u << 9)

/* =========================================================================
 * Core Buffer Mapping / Group / Sync / Eventfd / Power Structs & Ioctls
 * ========================================================================= */

struct edgetpu_map_ioctl {
	__u64 host_address;
	__u64 size;
	__u64 device_address;
	edgetpu_map_flag_t flags;
	__u32 die_index;
}; /* sizeof == 32 (0x20) */

#define EDGETPU_MAP_BUFFER   _IOWR(EDGETPU_IOCTL_BASE, 0, struct edgetpu_map_ioctl) /* 0xc020ed00 */
#define EDGETPU_UNMAP_BUFFER _IOW(EDGETPU_IOCTL_BASE, 4, struct edgetpu_map_ioctl)  /* 0x4020ed04 */

#define EDGETPU_EVENT_RESPDATA    0
#define EDGETPU_EVENT_FATAL_ERROR 1

struct edgetpu_event_register {
	__u32 event_id;
	__u32 eventfd;
}; /* sizeof == 8 */

#define EDGETPU_SET_EVENTFD _IOW(EDGETPU_IOCTL_BASE, 5, struct edgetpu_event_register) /* 0x4008ed05 */

#define EDGETPU_PRIORITY_DETACHABLE (1u << 3)
#define EDGETPU_PARTITION_NORMAL    0
#define EDGETPU_PARTITION_EXTRA     1

/*
 * Mailbox/group attributes (20B). Bitfields verified in buenos.ko
 * `edgetpu_device_group_create` & `edgetpu_group_activate`:
 *   [3:0] priority (default 8 in libtachyon_core.so)
 *   [4]   cmdq_tail_doorbell / detachable flag (bit 4 set by libtachyon_core.so: 0x18)
 *   [5]   partition_type (1 = extra low partition VCID 0)
 *   [6]   client_priv / first_party (passed to edgetpu_ikv_activate_client)
 *   [7]   partition_type_high (1 = extra high partition VCID 1)
 */
struct edgetpu_mailbox_attr {
	__u32 cmd_queue_size;
	__u32 resp_queue_size;
	__u32 sizeof_cmd;
	__u32 sizeof_resp;
	__u32 priority            : 4;
	__u32 cmdq_tail_doorbell  : 1;
	__u32 partition_type      : 1;
	__u32 client_priv         : 1;
	__u32 partition_type_high : 1;
	__u32 reserved            : 24;
}; /* sizeof == 20 (0x14) */

#define EDGETPU_CREATE_GROUP \
	_IOW(EDGETPU_IOCTL_BASE, 6, struct edgetpu_mailbox_attr) /* 0x4014ed06 */

/* nr=7 (EDGETPU_JOIN_GROUP) removed in buenos.ko (-ENOTTY) */
/* nr=8 finalized automatically inside CREATE_GROUP; returns 0 as a no-op */
#define EDGETPU_FINALIZE_GROUP _IO(EDGETPU_IOCTL_BASE, 8) /* 0x0000ed08 */

#define EDGETPU_PERDIE_EVENT_LOGS_AVAILABLE   0x1000
#define EDGETPU_PERDIE_EVENT_TRACES_AVAILABLE 0x1001

#define EDGETPU_SET_PERDIE_EVENTFD \
	_IOW(EDGETPU_IOCTL_BASE, 9, struct edgetpu_event_register) /* 0x4008ed09 */

/* Note: nr=14 & nr=15 pass __u32 event_id directly by value as ioctl arg */
#define EDGETPU_UNSET_EVENT        _IOW(EDGETPU_IOCTL_BASE, 14, __u32) /* 0x4004ed0e */
#define EDGETPU_UNSET_PERDIE_EVENT _IOW(EDGETPU_IOCTL_BASE, 15, __u32) /* 0x4004ed0f */

#define EDGETPU_SYNC_FOR_DEVICE (0 << 2)
#define EDGETPU_SYNC_FOR_CPU    (1 << 2)

struct edgetpu_sync_ioctl {
	__u64 device_address;
	__u64 size;
	__u64 offset;
	__u32 die_index;
	__u32 flags;
}; /* sizeof == 32 (0x20) */

#define EDGETPU_SYNC_BUFFER _IOW(EDGETPU_IOCTL_BASE, 16, struct edgetpu_sync_ioctl) /* 0x4020ed10 */

struct edgetpu_map_dmabuf_ioctl {
	__u64 offset;
	__u64 size;
	__u64 device_address;
	__s32 dmabuf_fd;
	edgetpu_map_flag_t flags;
	__u32 die_index;
}; /* sizeof == 40 (0x28) */

#define EDGETPU_MAP_DMABUF \
	_IOWR(EDGETPU_IOCTL_BASE, 17, struct edgetpu_map_dmabuf_ioctl) /* 0xc028ed11 */
#define EDGETPU_UNMAP_DMABUF \
	_IOW(EDGETPU_IOCTL_BASE, 18, struct edgetpu_map_dmabuf_ioctl)  /* 0x4028ed12 */

/* nr=19 (ALLOCATE_DEVICE_BUFFER) removed in buenos.ko (-ENOTTY) */

#define EDGETPU_SYNC_TIMELINE_NAME_LEN 128

struct edgetpu_create_sync_fence_data {
	__u32 seqno;
	char timeline_name[EDGETPU_SYNC_TIMELINE_NAME_LEN];
	__s32 fence;
}; /* sizeof == 136 (0x88) */

#define EDGETPU_CREATE_SYNC_FENCE \
	_IOWR(EDGETPU_IOCTL_BASE, 20, struct edgetpu_create_sync_fence_data) /* 0xc088ed14 */

struct edgetpu_signal_sync_fence_data {
	__s32 fence;
	__s32 error;
}; /* sizeof == 8 */

#define EDGETPU_SIGNAL_SYNC_FENCE \
	_IOW(EDGETPU_IOCTL_BASE, 21, struct edgetpu_signal_sync_fence_data) /* 0x4008ed15 */

/* nr=22, 23 ({UN,}MAP_BULK_DMABUF) removed in buenos.ko (-ENOTTY) */

struct edgetpu_sync_fence_status {
	__s32 fence;
	__s32 status;
}; /* sizeof == 8 */

#define EDGETPU_SYNC_FENCE_STATUS \
	_IOWR(EDGETPU_IOCTL_BASE, 24, struct edgetpu_sync_fence_status) /* 0xc008ed18 */

#define EDGETPU_RELEASE_WAKE_LOCK _IO(EDGETPU_IOCTL_BASE, 25) /* 0x0000ed19 */
#define EDGETPU_ACQUIRE_WAKE_LOCK _IO(EDGETPU_IOCTL_BASE, 26) /* 0x0000ed1a */

struct edgetpu_fw_version {
	__u32 major_version;
	__u32 minor_version;
	__u32 vii_version;
	__u32 kci_version;
}; /* sizeof == 16 (0x10) */

#define EDGETPU_FIRMWARE_VERSION \
	_IOR(EDGETPU_IOCTL_BASE, 27, struct edgetpu_fw_version) /* 0x8010ed1b */

#define EDGETPU_GET_TPU_TIMESTAMP _IOR(EDGETPU_IOCTL_BASE, 28, __u64) /* 0x8008ed1c */

/* nr=29 (GET_DRAM_USAGE) removed in buenos.ko (-ENOTTY) */

#define EDGETPU_EXT_MAILBOX_TYPE_TZ  1
#define EDGETPU_EXT_MAILBOX_TYPE_GSA 2

struct edgetpu_ext_mailbox_ioctl {
	__u64 client_id;
	__u64 attrs;
	__u32 type;  /* buenos.ko only accepts EDGETPU_EXT_MAILBOX_TYPE_TZ (1) */
	__u32 count; /* buenos.ko only accepts count == 1 */
}; /* sizeof == 24 (0x18) */

#define EDGETPU_ACQUIRE_EXT_MAILBOX \
	_IOW(EDGETPU_IOCTL_BASE, 30, struct edgetpu_ext_mailbox_ioctl) /* 0x4018ed1e */
#define EDGETPU_RELEASE_EXT_MAILBOX \
	_IOW(EDGETPU_IOCTL_BASE, 31, struct edgetpu_ext_mailbox_ioctl) /* 0x4018ed1f */

#define EDGETPU_ERROR_FW_CRASH             0x1
#define EDGETPU_ERROR_WATCHDOG_TIMEOUT     0x2
#define EDGETPU_ERROR_THERMAL_STOP         0x4
#define EDGETPU_ERROR_HW_NO_ACCESS         0x8
#define EDGETPU_ERROR_HW_FAIL              0x10
#define EDGETPU_ERROR_RUNTIME_TIMEOUT      0x20
#define EDGETPU_ERROR_CLIENT_CONTEXT_CRASH 0x40

#define EDGETPU_GET_FATAL_ERRORS _IOR(EDGETPU_IOCTL_BASE, 32, __u32) /* 0x8004ed20 */

/* =========================================================================
 * Observed Laguna Additions & VII / IKV Structs (buenos.ko)
 * ========================================================================= */

/*
 * nr=33: EDGETPU_ACQUIRE_WAKE_LOCK_FLAGS (observed 0x8004ed21 in buenos.ko)
 * Note: Encoded with _IOR direction despite passing __u32 flags directly by
 * value in ioctl arg (`edgetpu_ioctl_acquire_wakelock(client, (__u32)arg)`).
 */
#define EDGETPU_WAKELOCK_FLAG_SUSPENDABLE (1u << 0)
#define EDGETPU_ACQUIRE_WAKE_LOCK_FLAGS _IOR(EDGETPU_IOCTL_BASE, 33, __u32) /* 0x8004ed21 */

/* nr=34 (SET_DEVICE_PROPERTIES) removed in buenos.ko (-ENOTTY) */

/*
 * Legacy VII structs (kept for ABI completeness; nr=35 & nr=36 return -EOPNOTSUPP
 * in buenos.ko).
 */
#define EDGETPU_NUM_VII_CREDITS_AOSP   8
#define EDGETPU_NUM_VII_CREDITS_LAGUNA 16

struct edgetpu_vii_dma_descriptor {
	__u64 address;
	__u32 size;
	__u32 flags;
};

struct edgetpu_vii_command {
	__u64 seq;
	__u16 code;
	__u8 priority;
	__u8 reserved_0[5];
	struct edgetpu_vii_dma_descriptor dma_descriptor;
	__u8 reserved_1[8];
	__u32 client_id;
	__u8 qos_class;
	__u8 cluster_ids_bitset;
	__u8 atomic;
	__u8 reserved_2[1];
} __edgetpu_packed;

struct edgetpu_vii_command_ioctl {
	struct edgetpu_vii_command command;
	__u64 in_fence_array;
	__u32 in_fence_count;
	__u64 out_fence_array;
	__u32 out_fence_count;
}; /* sizeof == 80 (0x50) */

#define EDGETPU_VII_COMMAND \
	_IOWR(EDGETPU_IOCTL_BASE, 35, struct edgetpu_vii_command_ioctl) /* 0xc050ed23 (-EOPNOTSUPP) */

#define VII_RESPONSE_CODE_KERNEL_BASE          (1u << 15)
#define VII_RESPONSE_CODE_KERNEL_CMD_TIMEOUT   (VII_RESPONSE_CODE_KERNEL_BASE + 0)
#define VII_RESPONSE_CODE_KERNEL_ENQUEUE_FAILED (VII_RESPONSE_CODE_KERNEL_BASE + 1)
#define VII_RESPONSE_CODE_KERNEL_FENCE_ERROR   (VII_RESPONSE_CODE_KERNEL_BASE + 2)
#define VII_RESPONSE_CODE_KERNEL_FENCE_TIMEOUT (VII_RESPONSE_CODE_KERNEL_BASE + 3)
#define VII_RESPONSE_CODE_KERNEL_CANCELED      (VII_RESPONSE_CODE_KERNEL_BASE + 4)

struct edgetpu_vii_response {
	__u64 seq;
	__u16 code;
	__s8 cluster_index;
	__u8 reserved;
	__u32 client_id;
	__u64 retval;
} __edgetpu_packed;

struct edgetpu_vii_response_ioctl {
	struct edgetpu_vii_response response;
}; /* sizeof == 24 (0x18) */

#define EDGETPU_VII_RESPONSE \
	_IOWR(EDGETPU_IOCTL_BASE, 36, struct edgetpu_vii_response_ioctl) /* 0xc018ed24 (-EOPNOTSUPP) */

/*
 * Litebuf VII ioctls (nr=37 & nr=38): primary inference submission path on buenos.ko.
 * Constraint observed in buenos.ko: `seq` MUST fit in 32 bits (`seq >> 32 == 0`);
 * IKF in-fences wait up to 0x9c4 (2500) ms via `iif_dma_fence_wait_timeout`.
 */
#define EDGETPU_VII_COMMAND_MAX_NUM_FENCES 64

struct edgetpu_vii_litebuf_command_ioctl {
	__u64 litebuf_address;
	__u32 litebuf_size;
	__u64 seq;
	__u64 in_fence_array;  /* __s32 * */
	__u32 in_fence_count;
	__u64 out_fence_array; /* __s32 * */
	__u32 out_fence_count;
	__u8 reserved[4];
}; /* sizeof == 56 (0x38) */

#define EDGETPU_VII_LITEBUF_COMMAND \
	_IOWR(EDGETPU_IOCTL_BASE, 37, struct edgetpu_vii_litebuf_command_ioctl) /* 0xc038ed25 */

struct edgetpu_vii_litebuf_response_ioctl {
	__u64 litebuf_address; /* Optional (>=48B buffer, or 0 to fetch seq/code only) */
	__u64 seq;
	__u16 code;
	__u8 reserved[6];
}; /* sizeof == 24 (0x18) */

#define EDGETPU_VII_LITEBUF_RESPONSE \
	_IOWR(EDGETPU_IOCTL_BASE, 38, struct edgetpu_vii_litebuf_response_ioctl) /* 0xc018ed26 */

/*
 * nr=39: EDGETPU_GET_CREDIT_SIZE (name inferred from libtachyon_core.so error:
 * "Failed to probe for credit size: %s"; exact macro symbol unverified).
 * buenos.ko copies u64 0x10 (16) to userspace (userspace falls back to 8 on ENOTTY).
 */
#define EDGETPU_GET_CREDIT_SIZE _IOR(EDGETPU_IOCTL_BASE, 39, __u64) /* 0x8008ed27 */

/*
 * nr=40: EDGETPU_ADD_LIMITED_INTERFACE (observed 0xc004ed28 in buenos.ko & libtachyon_core.so)
 * Note: Encoded with _IOWR(0xED, 40, __u32), but passes limited_device_fd
 * directly by integer value as ioctl arg. Links /dev/edgetpu-limited client
 * to the caller's device group on /dev/edgetpu-soc.
 */
#define EDGETPU_ADD_LIMITED_INTERFACE _IOWR(EDGETPU_IOCTL_BASE, 40, __u32) /* 0xc004ed28 */

/*
 * nr=41: EDGETPU_REMAP_BUFFERS (observed 0x0000ed29 in buenos.ko)
 * Calls `edgetpu_group_remap_buffers` to re-map trimmed group buffers.
 * (Note: libedgetpu_util.so/libdarwinn_hal.so probe 0xc008ed29 BFA fence elsewhere.)
 */
#define EDGETPU_REMAP_BUFFERS _IO(EDGETPU_IOCTL_BASE, 41) /* 0x0000ed29 */

/*
 * nr=42: EDGETPU_GET_INTERFACE_VERSION (observed 0x8044ed2a in buenos.ko,
 * libtachyon_core.so, libedgetpu_litert.so).
 * Returns version_major=1, version_minor=4, version_build=scmversion ("gcc524543248c").
 */
#define EDGETPU_INTERFACE_VERSION_BUILD_BUFFER_SIZE 64

struct edgetpu_interface_version_ioctl {
	__u16 version_major;
	__u16 version_minor;
	char version_build[EDGETPU_INTERFACE_VERSION_BUILD_BUFFER_SIZE];
}; /* sizeof == 68 (0x44) */

#define EDGETPU_GET_INTERFACE_VERSION \
	_IOR(EDGETPU_IOCTL_BASE, 42, struct edgetpu_interface_version_ioctl) /* 0x8044ed2a */

/*
 * nr=43: EDGETPU_CLIENT_TRIM_ENABLE (observed 0x4004ed2b in buenos.ko)
 * Value-passed __u32 boolean (arg != 0 enables pixel_trim callback tracking on client).
 */
#define EDGETPU_CLIENT_TRIM_ENABLE _IOW(EDGETPU_IOCTL_BASE, 43, __u32) /* 0x4004ed2b */

/*
 * nr=44: EDGETPU_CLIENT_SET_TGID (observed 0x4004ed2c in buenos.ko & libtachyon_core.so)
 * Value-passed __u32 vpid/tgid; binds client identity/name for proxy calls.
 */
#define EDGETPU_CLIENT_SET_TGID _IOW(EDGETPU_IOCTL_BASE, 44, __u32) /* 0x4004ed2c */

/* =========================================================================
 * Wire Formats: Litebuf VII Mailbox Packets & IKV Additional Info (DRAM/iremap)
 * (128B cmd / 64B resp per AOSP edgetpu-vii-litebuf.h; verified 0x30B payload
 *  copy, +0x34 seq, +0x3c code in buenos.ko `edgetpu_ioctl_vii_litebuf_*`).
 * ========================================================================= */

#define EDGETPU_VII_CMD_SIZE_BYTES           128
#define EDGETPU_VII_CMD_KERNEL_USABLE_BYTES  32
#define EDGETPU_VII_CMD_PAYLOAD_SIZE_BYTES   (EDGETPU_VII_CMD_SIZE_BYTES - EDGETPU_VII_CMD_KERNEL_USABLE_BYTES)

enum edgetpu_vii_litebuf_command_type {
	EDGETPU_VII_LITEBUF_RUNTIME_COMMAND = 0,
	EDGETPU_VII_LITEBUF_LARGE_RUNTIME_COMMAND = 1,
	/* Observed in buenos.ko when litebuf_size == 0 (null/fence-only command): */
	EDGETPU_VII_LITEBUF_NULL_COMMAND = 3,
};

struct edgetpu_vii_litebuf_large_runtime_command {
	__u32 address;
	__u32 size_bytes;
};

struct edgetpu_vii_litebuf_command {
	union {
		__u8 runtime_command[EDGETPU_VII_CMD_PAYLOAD_SIZE_BYTES];
		struct edgetpu_vii_litebuf_large_runtime_command large_runtime_command;
		__u8 max_payload_size[EDGETPU_VII_CMD_PAYLOAD_SIZE_BYTES];
	};
	__u8 reserved_0[16];
	__u32 seq;
	__u32 client_id;
	__u32 additional_info_address;
	__u16 additional_info_size;
	__u8 type;
	__u8 reserved_1[1];
}; /* sizeof == 128 (0x80) */

#define EDGETPU_VII_RESP_SIZE_BYTES          64
#define EDGETPU_VII_RESP_KERNEL_USABLE_BYTES 16
#define EDGETPU_VII_RESP_PAYLOAD_SIZE_BYTES  (EDGETPU_VII_RESP_SIZE_BYTES - EDGETPU_VII_RESP_KERNEL_USABLE_BYTES)

enum edgetpu_vii_litebuf_response_type {
	EDGETPU_VII_LITEBUF_RUNTIME_RESPONSE = 0,
};

struct edgetpu_vii_litebuf_response {
	union {
		__u8 runtime_response[EDGETPU_VII_RESP_PAYLOAD_SIZE_BYTES];
		__u8 max_payload_size[EDGETPU_VII_RESP_PAYLOAD_SIZE_BYTES];
	};
	__u8 reserved_0[4];
	__u32 seq;
	__u32 client_id;
	__u16 code;
	__u8 type;
	__u8 reserved_1[1];
}; /* sizeof == 64 (0x40) */

/*
 * IKV Additional Info wire layout (observed in buenos.ko `edgetpu_ikv_additional_info_fill`):
 * Root expanded from AOSP rio 16B -> 32B (`root_size = 0x20`) to add
 * out_fence_signal_data and wait_data offset/size fields.
 */
struct edgetpu_ikv_additional_info_header {
	__u16 identifier;
	__u8 reserved[6];
	__u64 root_offset; /* 0x10 */
} __edgetpu_packed;

struct edgetpu_ikv_additional_info_root {
	__u32 object_size;            /* 0x20 */
	__s32 in_fences_offset;
	__u32 in_fences_size;
	__s32 out_fences_offset;
	__u32 out_fences_size;
	__u32 out_fence_signal_data;  /* passed via param_6 in buenos.ko */
	__s32 wait_data_offset;
	__u32 wait_data_size;
} __edgetpu_packed; /* sizeof == 32 (0x20) */

#endif /* __EDGETPU_BUENOS_UAPI_H__ */
