/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Clean-room UAPI Header for Tensor G5 GXP DSP Driver (`gxp.ko`)
 * Device node: /dev/gxp
 * Build provenance: rango CP3A.261005.005 (Linux 6.6.127-android15, DT alias google,gxp-lga)
 *
 * Strictly distinguishes observed .ko ABI vs AOSP gxp/zuma definitions:
 *   - Interface version returned by GXP_GET_INTERFACE_VERSION (0x8044ee1a): 1.33 (AOSP: 1.31)
 *   - GXP_GET_SPECS (0x8010ee05): core_count = 3 (AOSP: 4), max_vd_allocation = 15,
 *     secure_telemetry_buffer_size = 1. `GXP_ALLOCATE_VIRTUAL_DEVICE` rejects count 0 or >= 4.
 *   - Direct mode ioctls 4 (MAILBOX_RESPONSE) and 23 (MAILBOX_COMMAND) return -EOPNOTSUPP (-95)
 *     in MCU mode (`gxp_mcu_ioctl`).
 *   - GXP_MAILBOX_UCI_COMMAND (0xc080ee27, 128B): bytes +0x60..+0x6f in the AOSP `reserved[32]`
 *     tail now carry `mid_in_fences` and `mid_out_fences` (`gxp_mcu_ioctl` ->
 *     `gxp_uci_send_cmd`), expanding wire `gxp_uci_additional_info_root` from 32B to 64B.
 */

#ifndef __GXP_LAGUNA_UAPI_H__
#define __GXP_LAGUNA_UAPI_H__

#include <linux/ioctl.h>
#include <linux/types.h>

#ifndef __gxp_packed
#define __gxp_packed __attribute__((__packed__))
#endif

#define GXP_INTERFACE_VERSION_MAJOR 1
#define GXP_INTERFACE_VERSION_MINOR 33
#define GXP_INTERFACE_VERSION_BUILD 0

#define GXP_IOCTL_BASE 0xEE

/* =========================================================================
 * mmap() offsets (verified in gxp.ko `gxp_mmap` & `gxp_mcu_mmap`)
 * ========================================================================= */
#define GXP_MMAP_MCU_LOG_BUFFER_OFFSET          0x30000
#define GXP_MMAP_MCU_TRACE_BUFFER_OFFSET        0x40000
#define GXP_MMAP_CORE_LOG_BUFFER_OFFSET         0x50000
#define GXP_MMAP_SECURE_CORE_LOG_BUFFER_OFFSET  0x70000

/* =========================================================================
 * Buffer Mapping & Sync Flags (verified in gxp.ko `gxp_dma_encode_gcip_map_flags`)
 * ========================================================================= */
#define GXP_MAP_DIR_MASK          3u
#define GXP_MAP_DMA_BIDIRECTIONAL 0u
#define GXP_MAP_DMA_TO_DEVICE     1u
#define GXP_MAP_DMA_FROM_DEVICE   2u
#define GXP_MAP_COHERENT          (1u << 2)

#define GXP_SPEC_FEATURE_MODE_MCU (1u << 0)
#define GXP_ALLOCATE_VD_SECURE    (1u << 0)

#define GXP_CORE_TELEMETRY_BUFFER_UNIT_SIZE         0x10000u
#define GXP_TELEMETRY_BUFFER_VALID_MAGIC_CODE       0xC0DEC0DEu
#define GXP_TELEMETRY_SECURE_BUFFER_VALID_MAGIC_CODE 0xA0B0C0D0u

struct gxp_map_ioctl {
	__u16 virtual_core_list;
	__u64 host_address;
	__u32 size;
	__u32 flags;
	__u64 device_address;
}; /* sizeof == 32 (0x20) */

#define GXP_MAP_BUFFER   _IOWR(GXP_IOCTL_BASE, 0, struct gxp_map_ioctl) /* 0xc020ee00 */
#define GXP_UNMAP_BUFFER _IOW(GXP_IOCTL_BASE, 1, struct gxp_map_ioctl)  /* 0x4020ee01 */

#define GXP_SYNC_FOR_DEVICE 0
#define GXP_SYNC_FOR_CPU    1

struct gxp_sync_ioctl {
	__u64 device_address;
	__u32 size;
	__u32 offset;
	__u32 flags;
}; /* sizeof == 24 (0x18) */

#define GXP_SYNC_BUFFER _IOW(GXP_IOCTL_BASE, 2, struct gxp_sync_ioctl) /* 0x4018ee02 */

/* Response error codes shared with UCI & legacy mailbox response */
#define GXP_RESPONSE_ERROR_NONE     0
#define GXP_RESPONSE_ERROR_INTERNAL 1
#define GXP_RESPONSE_ERROR_TIMEOUT  2
#define GXP_RESPONSE_ERROR_NOENT    3
#define GXP_RESPONSE_ERROR_AGAIN    4
#define GXP_RESPONSE_ERROR_CANCELED 5

struct gxp_mailbox_response_ioctl {
	__u16 virtual_core_id;
	__u64 sequence_number;
	__u16 error_code;
	__u32 cmd_retval;
}; /* sizeof == 24 (0x18) */

/* Note: returns -EOPNOTSUPP (-95) in MCU mode on laguna gxp.ko */
#define GXP_MAILBOX_RESPONSE \
	_IOWR(GXP_IOCTL_BASE, 4, struct gxp_mailbox_response_ioctl) /* 0xc018ee04 */

struct gxp_specs_ioctl {
	__u8 core_count;                  /* 3 on laguna gxp.ko */
	__u8 features;
	__u8 telemetry_buffer_size;
	__u8 secure_telemetry_buffer_size; /* 1 on laguna gxp.ko */
	__u8 max_vd_allocation;           /* 15 (0xf) on laguna gxp.ko */
	__u8 max_vd_activation;
	__u16 total_iova_size;
	__u8 reserved[8];
}; /* sizeof == 16 (0x10) */

#define GXP_GET_SPECS _IOR(GXP_IOCTL_BASE, 5, struct gxp_specs_ioctl) /* 0x8010ee05 */

struct gxp_virtual_device_ioctl {
	__u8 core_count; /* gxp.ko validates 1 <= core_count <= 3 */
	__u8 flags;
	__u8 reserved[6];
	__u32 vdid;
}; /* sizeof == 12 (0x0c) */

#define GXP_ALLOCATE_VIRTUAL_DEVICE \
	_IOWR(GXP_IOCTL_BASE, 6, struct gxp_virtual_device_ioctl) /* 0xc00cee06 */

#define GXP_TELEMETRY_TYPE_LOGGING 0
#define GXP_TELEMETRY_TYPE_TRACING 1

struct gxp_tpu_mbx_queue_ioctl {
	__u32 tpu_fd;
	__u32 virtual_core_list;
	__u64 attr_ptr;
}; /* sizeof == 16 (0x10) */

#define GXP_MAP_TPU_MBX_QUEUE \
	_IOW(GXP_IOCTL_BASE, 13, struct gxp_tpu_mbx_queue_ioctl) /* 0x4010ee0d */
#define GXP_UNMAP_TPU_MBX_QUEUE \
	_IOW(GXP_IOCTL_BASE, 14, struct gxp_tpu_mbx_queue_ioctl) /* 0x4010ee0e */

struct gxp_register_telemetry_eventfd_ioctl {
	__u32 eventfd;
	__u8 type;
}; /* sizeof == 8 */

#define GXP_REGISTER_CORE_TELEMETRY_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 15, struct gxp_register_telemetry_eventfd_ioctl) /* 0x4008ee0f */
#define GXP_UNREGISTER_CORE_TELEMETRY_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 16, struct gxp_register_telemetry_eventfd_ioctl) /* 0x4008ee10 */

#define GXP_READ_GLOBAL_COUNTER _IOR(GXP_IOCTL_BASE, 17, __u64) /* 0x8008ee11 */
#define GXP_RELEASE_WAKE_LOCK   _IOW(GXP_IOCTL_BASE, 19, __u32) /* 0x4004ee13 */

struct gxp_map_dmabuf_ioctl {
	__u16 virtual_core_list;
	__s32 dmabuf_fd;
	__u32 flags;
	__u64 device_address;
}; /* sizeof == 24 (0x18) */

#define GXP_MAP_DMABUF   _IOWR(GXP_IOCTL_BASE, 20, struct gxp_map_dmabuf_ioctl) /* 0xc018ee14 */
#define GXP_UNMAP_DMABUF _IOW(GXP_IOCTL_BASE, 21, struct gxp_map_dmabuf_ioctl)  /* 0x4018ee15 */

struct gxp_register_mailbox_eventfd_ioctl {
	__u32 eventfd;
	__u32 flags;
	__u16 virtual_core_id;
}; /* sizeof == 12 (0x0c) */

#define GXP_REGISTER_MAILBOX_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 22, struct gxp_register_mailbox_eventfd_ioctl) /* 0x400cee16 */

struct gxp_mailbox_command_ioctl {
	__u16 virtual_core_id;
	__u16 num_cores;
	__u64 sequence_number;
	__u64 device_address;
	__u32 size;
	__u32 gxp_power_state;
	__u32 memory_power_state;
	__u32 flags;
	__u32 power_flags;
}; /* sizeof == 48 (0x30) */

/* Note: returns -EOPNOTSUPP (-95) in MCU mode on laguna gxp.ko */
#define GXP_MAILBOX_COMMAND \
	_IOWR(GXP_IOCTL_BASE, 23, struct gxp_mailbox_command_ioctl) /* 0xc030ee17 */

#define GXP_UNREGISTER_MAILBOX_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 24, struct gxp_register_mailbox_eventfd_ioctl) /* 0x400cee18 */

#define WAKELOCK_BLOCK          (1u << 0)
#define WAKELOCK_VIRTUAL_DEVICE (1u << 1)

#define GXP_POWER_STATE_OFF                 0
#define GXP_POWER_STATE_UUD                 1
#define GXP_POWER_STATE_SUD                 2
#define GXP_POWER_STATE_UD                  3
#define GXP_POWER_STATE_NOM                 4
#define GXP_POWER_STATE_READY               5
#define GXP_POWER_STATE_UUD_PLUS            6
#define GXP_POWER_STATE_SUD_PLUS            7
#define GXP_POWER_STATE_UD_PLUS             8
#define GXP_POWER_STATE_PERCENT_FREQUENCY_5 GXP_POWER_STATE_UUD
#define GXP_POWER_STATE_PERCENT_FREQUENCY_10 9
#define GXP_POWER_STATE_PERCENT_FREQUENCY_15 10
#define GXP_POWER_STATE_PERCENT_FREQUENCY_20 11
#define GXP_POWER_STATE_PERCENT_FREQUENCY_25 12
#define GXP_POWER_STATE_PERCENT_FREQUENCY_30 13
#define GXP_POWER_STATE_PERCENT_FREQUENCY_35 GXP_POWER_STATE_UUD_PLUS
#define GXP_POWER_STATE_PERCENT_FREQUENCY_40 14
#define GXP_POWER_STATE_PERCENT_FREQUENCY_45 15
#define GXP_POWER_STATE_PERCENT_FREQUENCY_50 GXP_POWER_STATE_SUD
#define GXP_POWER_STATE_PERCENT_FREQUENCY_55 16
#define GXP_POWER_STATE_PERCENT_FREQUENCY_60 17
#define GXP_POWER_STATE_PERCENT_FREQUENCY_65 GXP_POWER_STATE_SUD_PLUS
#define GXP_POWER_STATE_PERCENT_FREQUENCY_70 18
#define GXP_POWER_STATE_PERCENT_FREQUENCY_75 GXP_POWER_STATE_UD
#define GXP_POWER_STATE_PERCENT_FREQUENCY_80 19
#define GXP_POWER_STATE_PERCENT_FREQUENCY_85 GXP_POWER_STATE_UD_PLUS
#define GXP_POWER_STATE_PERCENT_FREQUENCY_90 20
#define GXP_POWER_STATE_PERCENT_FREQUENCY_95 21
#define GXP_POWER_STATE_MAX_FREQUENCY       GXP_POWER_STATE_NOM
#define GXP_POWER_STATE_OVERDRIVE           22
#define GXP_NUM_POWER_STATES                (GXP_POWER_STATE_OVERDRIVE + 1)

/* aur_memory_state_array in gxp.ko .rodata verified as {0, 1, 2, 3, 4, 5, 6} */
#define MEMORY_POWER_STATE_UNDEFINED 0
#define MEMORY_POWER_STATE_MIN       1
#define MEMORY_POWER_STATE_VERY_LOW  2
#define MEMORY_POWER_STATE_LOW       3
#define MEMORY_POWER_STATE_HIGH      4
#define MEMORY_POWER_STATE_VERY_HIGH 5
#define MEMORY_POWER_STATE_MAX       6

#define GXP_POWER_NON_AGGRESSOR   (1u << 0)
#define GXP_POWER_LOW_FREQ_CLKMUX (1u << 1)

struct gxp_acquire_wakelock_ioctl {
	__u32 components_to_wake;
	__u32 gxp_power_state;
	__u32 memory_power_state;
	__u32 reserved;
	__u32 flags;
}; /* sizeof == 20 (0x14) */

#define GXP_ACQUIRE_WAKE_LOCK \
	_IOW(GXP_IOCTL_BASE, 25, struct gxp_acquire_wakelock_ioctl) /* 0x4014ee19 */

#define GXP_INTERFACE_VERSION_BUILD_BUFFER_SIZE 64

struct gxp_interface_version_ioctl {
	__u16 version_major;
	__u16 version_minor;
	char version_build[GXP_INTERFACE_VERSION_BUILD_BUFFER_SIZE];
}; /* sizeof == 68 (0x44) */

#define GXP_GET_INTERFACE_VERSION \
	_IOR(GXP_IOCTL_BASE, 26, struct gxp_interface_version_ioctl) /* 0x8044ee1a */

#define GXP_TRIGGER_DEBUG_DUMP _IOW(GXP_IOCTL_BASE, 27, __u32) /* 0x4004ee1b */

/* =========================================================================
 * MCU Mode Ioctls (gxp_mcu_ioctl, nr 28..41)
 * ========================================================================= */

#define GXP_REGISTER_MCU_TELEMETRY_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 28, struct gxp_register_telemetry_eventfd_ioctl) /* 0x4008ee1c */
#define GXP_UNREGISTER_MCU_TELEMETRY_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 29, struct gxp_register_telemetry_eventfd_ioctl) /* 0x4008ee1d */

#define GXP_UCI_CMD_OPAQUE_SIZE 48

struct gxp_mailbox_uci_command_compat_ioctl {
	__u64 sequence_number;
	__u8 reserved[8];
	__u8 opaque[GXP_UCI_CMD_OPAQUE_SIZE];
}; /* sizeof == 64 (0x40) */

#define GXP_MAILBOX_UCI_COMMAND_COMPAT \
	_IOWR(GXP_IOCTL_BASE, 30, struct gxp_mailbox_uci_command_compat_ioctl) /* 0xc040ee1e */

struct gxp_mailbox_uci_response_ioctl {
	__u64 sequence_number;
	__u16 error_code;
	__u8 reserved[6];
	__u8 opaque[16];
}; /* sizeof == 32 (0x20) */

#define GXP_MAILBOX_UCI_RESPONSE \
	_IOR(GXP_IOCTL_BASE, 31, struct gxp_mailbox_uci_response_ioctl) /* 0x8020ee1f */

#define GXP_SYNC_TIMELINE_NAME_LEN 128

struct gxp_create_sync_fence_data {
	__u32 seqno;
	char timeline_name[GXP_SYNC_TIMELINE_NAME_LEN];
	__s32 fence;
}; /* sizeof == 136 (0x88) */

#define GXP_CREATE_SYNC_FENCE \
	_IOWR(GXP_IOCTL_BASE, 32, struct gxp_create_sync_fence_data) /* 0xc088ee20 */

struct gxp_signal_sync_fence_data {
	__s32 fence;
	__s32 error;
}; /* sizeof == 8 */

#define GXP_SIGNAL_SYNC_FENCE \
	_IOW(GXP_IOCTL_BASE, 33, struct gxp_signal_sync_fence_data) /* 0x4008ee21 */

struct gxp_sync_fence_status {
	__s32 fence;
	__s32 status;
}; /* sizeof == 8 */

#define GXP_SYNC_FENCE_STATUS \
	_IOWR(GXP_IOCTL_BASE, 34, struct gxp_sync_fence_status) /* 0xc008ee22 */

struct gxp_register_invalidated_eventfd_ioctl {
	__u32 eventfd;
}; /* sizeof == 4 */

#define GXP_REGISTER_INVALIDATED_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 35, struct gxp_register_invalidated_eventfd_ioctl) /* 0x4004ee23 */
#define GXP_UNREGISTER_INVALIDATED_EVENTFD \
	_IOW(GXP_IOCTL_BASE, 36, struct gxp_register_invalidated_eventfd_ioctl) /* 0x4004ee24 */

#define GXP_DEV_PROP_SIZE 256

struct gxp_set_device_properties_ioctl {
	__u8 opaque[GXP_DEV_PROP_SIZE];
}; /* sizeof == 256 (0x100) */

#define GXP_SET_DEVICE_PROPERTIES \
	_IOW(GXP_IOCTL_BASE, 37, struct gxp_set_device_properties_ioctl) /* 0x4100ee25 */

#define GXP_INVALIDATED_NONE                 0
#define GXP_INVALIDATED_MCU_CRASH            1
#define GXP_INVALIDATED_CLIENT_CRASH         2
#define GXP_INVALIDATED_VMBOX_RELEASE_FAILED 3

#define GXP_GET_INVALIDATED_REASON _IOR(GXP_IOCTL_BASE, 38, __u32) /* 0x8004ee26 */

#define GXP_MAX_FENCES_PER_UCI_COMMAND     4
#define GXP_MAX_MID_FENCES_PER_UCI_COMMAND 2
#define GXP_FENCE_ARRAY_TERMINATION        (~0u)

/*
 * GXP_MAILBOX_UCI_COMMAND (128B, 0xc080ee27)
 * Observed delta vs AOSP zuma in gxp.ko `gxp_mcu_ioctl` (0x24fa0..0x25248):
 *   Offset +0x60..+0x6f inside the old 32B `reserved` tail is now read as
 *   `mid_in_fences` (+0x60) and `mid_out_fences` (+0x68) and passed into
 *   `gxp_mcu_fs_send_uci_cmd` -> `gxp_uci_send_cmd`.
 *   Note: `get_num_fences()` in gxp.ko scans up to 4 s32 elements stopping at
 *   0 or -1 (~0u); libgxp.so zero-fills +0x60..+0x7f when unused.
 */
struct gxp_mailbox_uci_command_ioctl {
	__u64 sequence_number;                                  /* +0x00 */
	__u32 in_fences[GXP_MAX_FENCES_PER_UCI_COMMAND];        /* +0x08 */
	__u32 out_fences[GXP_MAX_FENCES_PER_UCI_COMMAND];       /* +0x18 */
	__u32 timeout_ms;                                       /* +0x28 */
	__u32 flags;                                            /* +0x2c */
	__u8 opaque[GXP_UCI_CMD_OPAQUE_SIZE];                   /* +0x30 */
	__u32 mid_in_fences[GXP_MAX_MID_FENCES_PER_UCI_COMMAND];  /* +0x60 (observed gxp.ko) */
	__u32 mid_out_fences[GXP_MAX_MID_FENCES_PER_UCI_COMMAND]; /* +0x68 (observed gxp.ko) */
	__u8 reserved[16];                                      /* +0x70 */
}; /* sizeof == 128 (0x80) */

#define GXP_MAILBOX_UCI_COMMAND \
	_IOWR(GXP_IOCTL_BASE, 39, struct gxp_mailbox_uci_command_ioctl) /* 0xc080ee27 */

enum gxp_iif_ip_type {
	GXP_IIF_IP_DSP = 0,
	GXP_IIF_IP_TPU = 1,
	GXP_IIF_IP_GPU = 2,
};

struct gxp_create_iif_fence_ioctl {
	__u8 signaler_ip;
	__u16 total_signalers;
	__s32 fence;
}; /* sizeof == 8 */

#define GXP_CREATE_IIF_FENCE \
	_IOWR(GXP_IOCTL_BASE, 40, struct gxp_create_iif_fence_ioctl) /* 0xc008ee28 */

#define GXP_FENCE_REMAINING_SIGNALERS_NO_REGISTER_EVENTFD (~0u)

struct gxp_fence_remaining_signalers_ioctl {
	__u32 fences[GXP_MAX_FENCES_PER_UCI_COMMAND];
	__u32 eventfd;
	__u32 remaining_signalers[GXP_MAX_FENCES_PER_UCI_COMMAND];
}; /* sizeof == 36 (0x24) */

#define GXP_FENCE_REMAINING_SIGNALERS \
	_IOWR(GXP_IOCTL_BASE, 41, struct gxp_fence_remaining_signalers_ioctl) /* 0xc024ee29 */

struct gxp_reserve_iova_region_ioctl {
	__u64 size;
	__u64 device_address;
}; /* sizeof == 16 (0x10) */

#define GXP_RESERVE_IOVA_REGION \
	_IOWR(GXP_IOCTL_BASE, 42, struct gxp_reserve_iova_region_ioctl) /* 0xc010ee2a */
#define GXP_RETIRE_IOVA_REGION \
	_IOW(GXP_IOCTL_BASE, 43, struct gxp_reserve_iova_region_ioctl)  /* 0x4010ee2b */

/* =========================================================================
 * Wire Formats: UCI Queue Elements & Laguna Additional Info Root
 * ========================================================================= */

enum gxp_uci_type {
	GXP_UCI_TYPE_CORE_COMMAND = 0,
	GXP_UCI_TYPE_NULL_COMMAND = 1,
	GXP_UCI_TYPE_IIF_UNBLOCK_COMMAND = 2,
};

struct gxp_uci_core_command_params {
	__u64 address;
	__u32 size;
	__u8 num_cores;
	__u8 dsp_operating_point;
	__u8 memory_operating_point;
};

struct gxp_uci_wakelock_command_params {
	__u8 dsp_operating_point;
	__u8 memory_operating_point;
};

struct gxp_uci_command {
	__u32 seq;
	__u32 additional_info_address;
	__u32 client_id;
	__u8 type; /* enum gxp_uci_type */
	__u8 reserved;
	__u16 additional_info_size;
	union {
		struct gxp_uci_core_command_params core_command_params;
		struct gxp_uci_wakelock_command_params wakelock_command_params;
		__u16 iif_id;
		__u8 opaque[GXP_UCI_CMD_OPAQUE_SIZE];
	};
}; /* sizeof == 64 (0x40), matches local_b0..uStack_78 in gxp_uci_send_cmd */

struct gxp_uci_response {
	__u64 seq;
	__u32 client_id;
	__u16 code;
	__u8 reserved[2];
	__u8 opaque[16];
}; /* sizeof == 32 (0x20) */

struct gxp_uci_additional_info_header {
	__u8 identifier;
	__u8 version;
	__u8 reserved[6];
	__u64 root_offset; /* 0x10 */
} __gxp_packed;

/*
 * Observed in gxp.ko `gxp_uci_fill_additional_info` & `gxp_uci_send_cmd`:
 * `object_size` expanded from 0x20 (AOSP zuma) to 0x40 (laguna) to carry
 * mid-in and mid-out fence FD arrays + resolved IIF ID arrays.
 */
struct gxp_uci_additional_info_root {
	__u32 object_size;                      /* 0x40 */
	__s32 in_fences_offset;                 /* +0x04 */
	__u32 in_fences_size;                   /* +0x08 */
	__s32 out_fences_offset;                /* +0x0c */
	__u32 out_fences_size;                  /* +0x10 */
	__u32 timeout_ms;                       /* +0x14 */
	__s32 runtime_additional_info_offset;   /* +0x18 */
	__u32 runtime_additional_info_size;     /* +0x1c (0 in gxp_uci_send_cmd) */
	__s32 mid_in_fence_fds_offset;          /* +0x20 */
	__u32 mid_in_fence_fds_size;            /* +0x24 */
	__s32 mid_in_fence_iif_ids_offset;      /* +0x28 */
	__u32 mid_in_fence_iif_ids_size;        /* +0x2c */
	__s32 mid_out_fence_fds_offset;         /* +0x30 */
	__u32 mid_out_fence_fds_size;           /* +0x34 */
	__s32 mid_out_fence_iif_ids_offset;     /* +0x38 */
	__u32 mid_out_fence_iif_ids_size;       /* +0x3c */
} __gxp_packed; /* sizeof == 64 (0x40) */

#endif /* __GXP_LAGUNA_UAPI_H__ */
