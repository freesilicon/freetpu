/*
 * FreeTPU Shared-Memory (MMF) Telemetry ABI for bare-metal libopentpu.so
 *
 * Copyright 2026 FreeTPU Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Writers (e.g. libopentpu.so) publish per-chip counters with mode 0600/0644
 * (never group/world-writable) owned by a trusted publisher UID to:
 *   /dev/shm/freetpu_metrics_<pci_bdf>   (e.g. /dev/shm/freetpu_metrics_0000:00:06.0)
 *
 * Concurrency & freshness protocol (lockless SeqLock, single writer / multi-reader):
 *   1. Before mutating payload fields, increment `seq` to an odd value with
 *      acquire-release ordering (e.g. `__atomic_fetch_add(&s->seq, 1u, __ATOMIC_ACQ_REL)`
 *      on GCC/Clang, or call `freetpu_shm_metrics_v1_publish()` below).
 *   2. Write `hbm_allocated_bytes`, `hbm_used_bytes`, `mxu_active_cycles`,
 *      `dma_bytes_transferred`, and `timestamp_ns` (CLOCK_REALTIME Unix epoch
 *      nanoseconds heartbeat; writers MUST refresh `timestamp_ns` periodically
 *      while alive and unlink or stop refreshing on exit).
 *   3. Increment `seq` to the next even value with release ordering
 *      (`__atomic_fetch_add(&s->seq, 1u, __ATOMIC_RELEASE)`).
 *
 * Readers (`ops/tpu-exporter/tpu_exporter.py`) open with `O_NOFOLLOW`, verify
 * regular-file type, trusted owner UID, and non-group/world-writable mode via
 * `fstat`, then read via `pread` (avoiding `mmap` SIGBUS on truncation),
 * checking `magic`, `version`, even/unchanged `seq`, and `timestamp_ns`
 * freshness (`now_ns - timestamp_ns <= max_age_ns`, default 30s). Expired
 * samples from stopped/crashed publishers are omitted.
 */

#ifndef FREETPU_METRICS_SHM_H
#define FREETPU_METRICS_SHM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FREETPU_METRICS_SHM_MAGIC                   0x46545055u /* "FTPU" little-endian */
#define FREETPU_METRICS_SHM_VERSION                 1u
#define FREETPU_METRICS_SHM_SIZE                    56u
#define FREETPU_METRICS_SHM_DEFAULT_MAX_AGE_SECONDS 30u

struct freetpu_shm_metrics_v1 {
    uint32_t magic;                 /* offset 0:  FREETPU_METRICS_SHM_MAGIC */
    uint32_t version;               /* offset 4:  FREETPU_METRICS_SHM_VERSION */
    uint64_t seq;                   /* offset 8:  SeqLock counter (odd = dirty) */
    uint64_t hbm_allocated_bytes;   /* offset 16: HBM pool/capacity bytes */
    uint64_t hbm_used_bytes;        /* offset 24: HBM live bytes in use */
    uint64_t mxu_active_cycles;     /* offset 32: cumulative MXU active cycles */
    uint64_t dma_bytes_transferred; /* offset 40: cumulative DMA bytes */
    uint64_t timestamp_ns;          /* offset 48: publisher heartbeat CLOCK_REALTIME ns */
};

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(struct freetpu_shm_metrics_v1) == FREETPU_METRICS_SHM_SIZE,
               "freetpu_shm_metrics_v1 must be 56 bytes");
#elif defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(struct freetpu_shm_metrics_v1) == FREETPU_METRICS_SHM_SIZE,
              "freetpu_shm_metrics_v1 must be 56 bytes");
#endif

#if defined(__GNUC__) || defined(__clang__)
static inline __attribute__((unused)) void freetpu_shm_metrics_v1_publish(
    struct freetpu_shm_metrics_v1 *s,
    uint64_t hbm_allocated_bytes,
    uint64_t hbm_used_bytes,
    uint64_t mxu_active_cycles,
    uint64_t dma_bytes_transferred,
    uint64_t timestamp_ns) {
    uint64_t cur = __atomic_load_n(&s->seq, __ATOMIC_RELAXED);
    uint64_t dirty = (cur + 1u) | 1u;
    __atomic_store_n(&s->seq, dirty, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_ACQ_REL);
    __atomic_store_n(&s->magic, FREETPU_METRICS_SHM_MAGIC, __ATOMIC_RELAXED);
    __atomic_store_n(&s->version, FREETPU_METRICS_SHM_VERSION, __ATOMIC_RELAXED);
    __atomic_store_n(&s->hbm_allocated_bytes, hbm_allocated_bytes, __ATOMIC_RELAXED);
    __atomic_store_n(&s->hbm_used_bytes, hbm_used_bytes, __ATOMIC_RELAXED);
    __atomic_store_n(&s->mxu_active_cycles, mxu_active_cycles, __ATOMIC_RELAXED);
    __atomic_store_n(&s->dma_bytes_transferred, dma_bytes_transferred, __ATOMIC_RELAXED);
    __atomic_store_n(&s->timestamp_ns, timestamp_ns, __ATOMIC_RELAXED);
    __atomic_store_n(&s->seq, dirty + 1u, __ATOMIC_RELEASE);
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* FREETPU_METRICS_SHM_H */
