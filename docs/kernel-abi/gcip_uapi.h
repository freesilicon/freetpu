/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Clean-room GCIP / IIF (Inter-IP Fence) Kernel-User & Shared Firmware ABI
 * Target: Tensor G5 (laguna / rango CP3A.261005.005: buenos.ko + gxp.ko + IIF)
 *
 * Provenance:
 *   - Shared IIF, KCI, firmware info, and telemetry wire/UAPI structs:
 *     From AOSP gcip-kernel-driver (edgetpu-rio-android16 / gxp-zuma-android16)
 *     cross-verified against disassembly/relocations in buenos.ko & gxp.ko.
 *   - Laguna extensions (KCI codes 6, 0x1c, 0x1d, 0x1e; GCIP IOMMU 36-bit/coherent
 *     flag encodings; /dev/iif 0x80126980 usage in vendor userspace):
 *     Observed from buenos.ko, gxp.ko, and vendor/lib64/{libedgetpu_litert,libtachyon_core}.so.
 */

#ifndef __GCIP_UAPI_H__
#define __GCIP_UAPI_H__

#include <linux/ioctl.h>
#include <linux/types.h>

#ifndef __gcip_packed
#define __gcip_packed __attribute__((__packed__))
#endif

/* =========================================================================
 * Inter-IP Fence (IIF) UAPI (/dev/iif & shared fence tables)
 * ========================================================================= */

#define IIF_INTERFACE_VERSION_MAJOR 1
#define IIF_INTERFACE_VERSION_MINOR 0

#define IIF_IOCTL_BASE 'i'
#define IIF_FENCE_IOCTL_NUM_BASE 0x80

/* Maximum fences per IIF ioctl batch; max per IP in shared fence table. */
#define IIF_MAX_NUM_FENCES 64
#define IIF_NUM_FENCES_PER_IP 1024

/* IP identifiers (matches AOSP iif-shared.h & observed buenos.ko/gxp.ko usage). */
enum iif_ip_type {
	IIF_IP_DSP = 0,
	IIF_IP_TPU = 1,
	IIF_IP_GPU = 2,
	IIF_IP_AP = 3,
	IIF_IP_NUM = 4,
	IIF_IP_RESERVED = 16,
};

enum iif_flag_bits {
	IIF_FLAG_ERROR_BIT = 0,
};

struct iif_wait_table_entry {
	__u8 waiting_ips;
	__u8 reserved[7];
} __gcip_packed;

struct iif_signal_table_entry {
	__u16 remaining_signals;
	__u8 flag;
	__u8 reserved[5];
} __gcip_packed;

/* /dev/iif ioctl payloads (AOSP iif.h; /dev/iif fd ioctl 0x80126980 observed in libedgetpu_litert.so) */
struct iif_create_fence_ioctl {
	__u8 signaler_ip;
	__u16 total_signalers;
	__s32 fence;
};

#define IIF_FENCE_REMAINING_SIGNALERS_NO_REGISTER_EVENTFD (~0u)

struct iif_fence_remaining_signalers_ioctl {
	__u64 fences; /* __s32 * */
	__u32 fences_count;
	__u32 eventfd;
	__u64 remaining_signalers; /* __u32 * */
};

struct iif_fence_get_information_ioctl {
	__u8 signaler_ip;
	__u16 total_signalers;
	__u16 submitted_signalers;
	__u16 signaled_signalers;
	__u16 outstanding_waiters;
	__s16 signal_status;
	__u8 reserved[5];
}; /* sizeof == 18 (0x12), matches _IOR('i', 0x80, ...) == 0x80126980 */

struct iif_fence_signal_ioctl {
	__s32 error;
	__u16 remaining_signals;
};

#define IIF_CREATE_FENCE \
	_IOWR(IIF_IOCTL_BASE, 0, struct iif_create_fence_ioctl)
#define IIF_FENCE_REMAINING_SIGNALERS \
	_IOWR(IIF_IOCTL_BASE, 1, struct iif_fence_remaining_signalers_ioctl)
#define IIF_FENCE_GET_INFORMATION \
	_IOR(IIF_IOCTL_BASE, IIF_FENCE_IOCTL_NUM_BASE, struct iif_fence_get_information_ioctl)
#define IIF_FENCE_SUBMIT_SIGNALER \
	_IO(IIF_IOCTL_BASE, IIF_FENCE_IOCTL_NUM_BASE + 1)
#define IIF_FENCE_SIGNAL \
	_IOWR(IIF_IOCTL_BASE, IIF_FENCE_IOCTL_NUM_BASE + 2, struct iif_fence_signal_ioctl)

/* =========================================================================
 * GCIP KCI (Kernel Control Interface) Wire Elements & Command Codes
 * (AP kernel <-> TPU/GXP MCU firmware mailbox protocol)
 * ========================================================================= */

#define GCIP_KCI_REVERSE_FLAG 0x8000000000000000ULL

struct gcip_kci_dma_descriptor {
	__u64 address;
	__u32 size;
	__u32 flags;
};

struct gcip_kci_command_element {
	__u64 seq;
	__u16 code;
	__u16 reserved[3];
	struct gcip_kci_dma_descriptor dma;
} __gcip_packed; /* sizeof == 24 */

struct gcip_kci_response_element {
	__u64 seq;
	__u16 code;
	__u16 status;
	__u32 retval;
} __gcip_packed; /* sizeof == 16 */

enum gcip_kci_code {
	GCIP_KCI_CODE_ACK = 0,
	GCIP_KCI_CODE_UNMAP_BUFFER = 1,
	GCIP_KCI_CODE_MAP_LOG_BUFFER = 2,
	GCIP_KCI_CODE_JOIN_GROUP = 3,
	GCIP_KCI_CODE_LEAVE_GROUP = 4,
	GCIP_KCI_CODE_MAP_TRACE_BUFFER = 5,
	/* Observed in buenos.ko (edgetpu_kci_bcl_mitigation, 0x20B DMA payload): */
	GCIP_KCI_CODE_BCL_MITIGATION = 6,
	GCIP_KCI_CODE_SHUTDOWN = 7,
	GCIP_KCI_CODE_GET_DEBUG_DUMP = 8,
	GCIP_KCI_CODE_OPEN_DEVICE = 9,
	GCIP_KCI_CODE_CLOSE_DEVICE = 10,
	GCIP_KCI_CODE_EXCHANGE_INFO = 11,
	GCIP_KCI_CODE_GET_USAGE_V1 = 12,
	GCIP_KCI_CODE_GET_USAGE = 12,
	GCIP_KCI_CODE_NOTIFY_THROTTLING = 13,
	GCIP_KCI_CODE_BLOCK_BUS_SPEED_CONTROL = 14,
	GCIP_KCI_CODE_ALLOCATE_VMBOX = 15,
	GCIP_KCI_CODE_RELEASE_VMBOX = 16,
	GCIP_KCI_CODE_LINK_OFFLOAD_VMBOX = 17,
	GCIP_KCI_CODE_UNLINK_OFFLOAD_VMBOX = 18,
	GCIP_KCI_CODE_FIRMWARE_TRACING_LEVEL = 19,
	GCIP_KCI_CODE_THERMAL_CONTROL = 20,
	GCIP_KCI_CODE_GET_USAGE_V2 = 21,
	GCIP_KCI_CODE_SET_DEVICE_PROPERTIES = 22,
	GCIP_KCI_CODE_FAULT_INJECTION = 23,
	GCIP_KCI_CODE_SET_FREQ_LIMITS = 24,
	GCIP_KCI_CODE_DEBUG_CMD = 25,
	GCIP_KCI_CODE_DEBUG_RESET = 26,
	GCIP_KCI_CODE_DEBUG_INIT = 27,
	/* Observed in buenos.ko (edgetpu_kci_fw_log_state / hwtrace / coresight): */
	GCIP_KCI_CODE_FW_LOG_STATE = 28,
	GCIP_KCI_CODE_MAP_HWTRACE_BUFFER = 29,
	GCIP_KCI_CODE_CORESIGHT_REMOTE_CMD = 30, /* Also observed in gxp.ko (0x1e) */

	GCIP_KCI_CODE_RKCI_ACK = 256,
};

enum gcip_reverse_kci_code {
	GCIP_RKCI_CHIP_CODE_FIRST = 0,
	GCIP_RKCI_PM_QOS_REQUEST = 1,
	GCIP_RKCI_CHANGE_BTS_SCENARIO = 2,
	GCIP_RKCI_PM_QOS_BTS_REQUEST = 3,
	GCIP_RKCI_DSP_CORE_TELEMETRY_TRY_READ = 4,
	GCIP_RKCI_CLIENT_FATAL_ERROR_NOTIFY = 5,
	GCIP_RKCI_COHERENT_FABRIC_QOS_REQUEST = 6,
	GCIP_RKCI_CHIP_CODE_LAST = 0x7FFF,
	GCIP_RKCI_GENERIC_CODE_FIRST = 0x8000,
	GCIP_RKCI_FIRMWARE_CRASH = 0x8000,
	GCIP_RKCI_JOB_LOCKUP = 0x8001,
	GCIP_RKCI_DEBUG_ASYNC_RESP = 0x8002,
	GCIP_RKCI_DEBUG_INIT_REQ = 0x8003,
	GCIP_RKCI_GENERIC_CODE_LAST = 0xFFFF,
};

enum gcip_kci_error {
	GCIP_KCI_ERROR_OK = 0,
	GCIP_KCI_ERROR_CANCELLED = 1,
	GCIP_KCI_ERROR_UNKNOWN = 2,
	GCIP_KCI_ERROR_INVALID_ARGUMENT = 3,
	GCIP_KCI_ERROR_DEADLINE_EXCEEDED = 4,
	GCIP_KCI_ERROR_NOT_FOUND = 5,
	GCIP_KCI_ERROR_ALREADY_EXISTS = 6,
	GCIP_KCI_ERROR_PERMISSION_DENIED = 7,
	GCIP_KCI_ERROR_RESOURCE_EXHAUSTED = 8,
	GCIP_KCI_ERROR_FAILED_PRECONDITION = 9,
	GCIP_KCI_ERROR_ABORTED = 10,
	GCIP_KCI_ERROR_OUT_OF_RANGE = 11,
	GCIP_KCI_ERROR_UNIMPLEMENTED = 12,
	GCIP_KCI_ERROR_INTERNAL = 13,
	GCIP_KCI_ERROR_UNAVAILABLE = 14,
	GCIP_KCI_ERROR_DATA_LOSS = 15,
	GCIP_KCI_ERROR_UNAUTHENTICATED = 16,
};

enum gcip_kci_offload_chip_type {
	GCIP_KCI_OFFLOAD_CHIP_TYPE_TPU = 0,
};

/* =========================================================================
 * GCIP Firmware Info & Telemetry Shared Ring Headers
 * (Verified identical 0x38B fw_info allocation in buenos.ko & gxp.ko)
 * ========================================================================= */

enum gcip_fw_flavor {
	GCIP_FW_FLAVOR_ERROR = -1,
	GCIP_FW_FLAVOR_UNKNOWN = 0,
	GCIP_FW_FLAVOR_BL1 = 1,
	GCIP_FW_FLAVOR_SYSTEST = 2,
	GCIP_FW_FLAVOR_PROD_DEFAULT = 3,
	GCIP_FW_FLAVOR_CUSTOM = 4,
};

struct gcip_fw_info {
	__u64 fw_build_time;
	__u32 fw_flavor; /* enum gcip_fw_flavor */
	__u32 fw_changelist;
	__u32 spare[10];
}; /* sizeof == 56 (0x38) */

/* Firmware telemetry ring buffer header (mmapped from EdgeTPU/GXP log/trace/hwtrace). */
struct gcip_telemetry_header {
	__u32 head;
	__u32 size;
	__u32 reserved0[14];
	__u32 tail;
	__u32 entries_dropped;
	__u32 reserved1[14];
}; /* sizeof == 128 (0x80) */

struct gcip_log_entry_header {
	__s16 code;
	__u16 length;
	__u64 timestamp;
	__u16 crc16;
} __gcip_packed;

/* =========================================================================
 * GCIP IOMMU Internal Flag Bitfields (verified via gcip_iommu_encode_gcip_map_flags)
 * ========================================================================= */

#define GCIP_MAP_FLAGS_DMA_DIRECTION_SHIFT 0
#define GCIP_MAP_FLAGS_DMA_DIRECTION_MASK  (0x3ULL << 0)
#define GCIP_MAP_FLAGS_DMA_COHERENT        (1ULL << 2)
#define GCIP_MAP_FLAGS_DMA_ATTR_SHIFT      3
#define GCIP_MAP_FLAGS_RESTRICT_IOVA       (1ULL << 13)
#define GCIP_MAP_FLAGS_MMIO                (1ULL << 14)

/*
 * Cross-module kernel interface exported by buenos.ko (EXPORT_SYMBOL) and
 * imported by gxp.ko (`edgetpu_ext_driver_cmd`):
 *   cmd 0: Acquire external mailbox (type 0x27 or 0x28), outputs bar/reg info
 *   cmd 1: Disable external mailbox (`edgetpu_mailbox_disable_ext`)
 *   cmd 2: Query TPU VMbox client_id / PASID (sets bit 30 if high partition)
 *   cmd 3: Acquire `struct iif_manager *` (`iif_manager_get`)
 */
#define EDGETPU_EXT_CMD_ACQUIRE_MAILBOX 0
#define EDGETPU_EXT_CMD_RELEASE_MAILBOX 1
#define EDGETPU_EXT_CMD_GET_CLIENT_ID   2
#define EDGETPU_EXT_CMD_GET_IIF_MANAGER 3

#endif /* __GCIP_UAPI_H__ */
