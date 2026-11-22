#include "ob_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace ob {

void pin_thread_to_perf_cores() {
#if defined(__linux__)
  // Dynamically identify non-LITTLE cores via sysfs cpufreq max frequency.
  // On Tensor G5 (rango): cpu0..1=2.246GHz (LITTLE), cpu2..6=3.052GHz (MID), cpu7=3.782GHz (BIG).
  // Excluding min_freq cores keeps helper threads strictly on MID/BIG cores (cpu2..7),
  // while preserving any caller-imposed CPU affinity mask (e.g., taskset).
  cpu_set_t inherited;
  CPU_ZERO(&inherited);
  if (sched_getaffinity(0, sizeof(inherited), &inherited) != 0) {
    return;
  }

  const uint32_t max_cpus = std::min<uint32_t>(32u, static_cast<uint32_t>(CPU_SETSIZE));
  uint64_t freqs[32]{};
  bool valid[32]{};
  uint64_t min_freq = UINT64_MAX;
  uint64_t max_freq = 0ULL;
  uint32_t valid_cnt = 0u;
  for (uint32_t cpu = 0u; cpu < max_cpus; ++cpu) {
    char path[96]{};
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", cpu);
    FILE* f = std::fopen(path, "r");
    if (!f) continue;
    unsigned long khz = 0ul;
    if (std::fscanf(f, "%lu", &khz) == 1 && khz > 0ul) {
      freqs[cpu] = static_cast<uint64_t>(khz);
      valid[cpu] = true;
      ++valid_cnt;
      if (freqs[cpu] < min_freq) min_freq = freqs[cpu];
      if (freqs[cpu] > max_freq) max_freq = freqs[cpu];
    }
    std::fclose(f);
  }
  if (valid_cnt < 2u || max_freq <= min_freq) return;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  uint32_t pinned_cnt = 0u;
  for (uint32_t cpu = 0u; cpu < max_cpus; ++cpu) {
    if (valid[cpu] && freqs[cpu] > min_freq &&
        CPU_ISSET(static_cast<int>(cpu), &inherited)) {
      CPU_SET(static_cast<int>(cpu), &cpuset);
      ++pinned_cnt;
    }
  }
  if (pinned_cnt > 0u) {
    (void)sched_setaffinity(0, sizeof(cpuset), &cpuset);
  }
#endif
}

namespace {

inline void cpu_yield_hint() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#endif
}

struct WorkerLaunchArg {
  WorkerPool* pool = nullptr;
  uint32_t worker_idx = 0u;
};

void* worker_thread_main(void* raw) {
  WorkerLaunchArg* larg = static_cast<WorkerLaunchArg*>(raw);
  WorkerPool* p = larg->pool;
  const uint32_t tid = larg->worker_idx + 1u;
  delete larg;

  pin_thread_to_perf_cores();

  uint64_t seen_gen = 0u;
  for (;;) {
    WorkerPool::TaskFn fn = nullptr;
    void* arg = nullptr;
    uint32_t total = 0u;
    uint32_t step = 1u;
    uint32_t active_workers = 0u;

    pthread_mutex_lock(&p->mu);
    while (!p->stopping && p->job_gen == seen_gen) {
      pthread_cond_wait(&p->work_cv, &p->mu);
    }
    if (p->stopping) {
      pthread_mutex_unlock(&p->mu);
      return nullptr;
    }
    seen_gen = p->job_gen;
    active_workers = p->cur_active_workers;
    if (tid <= active_workers) {
      fn = p->cur_fn;
      arg = p->cur_arg;
      total = p->cur_total;
      step = p->cur_step;
    }
    pthread_mutex_unlock(&p->mu);

    if (fn != nullptr) {
      for (;;) {
        const uint32_t s = p->next_item.fetch_add(step, std::memory_order_relaxed);
        if (s >= total) break;
        const uint32_t e = (s + step < total) ? (s + step) : total;
        fn(arg, s, e, tid);
      }
      if (p->workers_remaining.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_signal(&p->done_cv);
        pthread_mutex_unlock(&p->mu);
      }
    }
  }
}

}  // namespace

WorkerPool::~WorkerPool() {
  shutdown();
  pthread_cond_destroy(&done_cv);
  pthread_cond_destroy(&work_cv);
  pthread_mutex_destroy(&mu);
}

void WorkerPool::init(uint32_t requested_threads) {
  std::lock_guard<std::mutex> dlk(dispatch_mu);
  if (requested_threads < 1u) requested_threads = 1u;
  if (requested_threads > 16u) requested_threads = 16u;
  num_threads = requested_threads;
}

void WorkerPool::shutdown() {
  std::lock_guard<std::mutex> dlk(dispatch_mu);
  if (!started) return;
  pthread_mutex_lock(&mu);
  stopping = true;
  pthread_cond_broadcast(&work_cv);
  pthread_mutex_unlock(&mu);
  for (pthread_t t : threads) {
    pthread_join(t, nullptr);
  }
  threads.clear();
  started = false;
  stopping = false;
}

void WorkerPool::parallel_for(uint32_t total, uint32_t min_chunk, TaskFn fn, void* arg) {
  if (total == 0u || !fn) return;
  if (min_chunk == 0u) min_chunk = 1u;
  if (num_threads <= 1u || total <= min_chunk) {
    fn(arg, 0u, total, 0u);
    return;
  }

  std::lock_guard<std::mutex> dlk(dispatch_mu);
  if (num_threads <= 1u || total <= min_chunk) {
    fn(arg, 0u, total, 0u);
    return;
  }

  if (!started) {
    const uint32_t want_workers = num_threads - 1u;
    threads.reserve(want_workers);
    stopping = false;
    for (uint32_t w = 0u; w < want_workers; ++w) {
      WorkerLaunchArg* larg = new (std::nothrow) WorkerLaunchArg{this, w};
      if (!larg) break;
      pthread_t tid_h{};
      if (pthread_create(&tid_h, nullptr, &worker_thread_main, larg) != 0) {
        delete larg;
        break;
      }
      threads.push_back(tid_h);
    }
    num_threads = static_cast<uint32_t>(threads.size()) + 1u;
    started = !threads.empty();
    if (!started) {
      fn(arg, 0u, total, 0u);
      return;
    }
  }

  const uint32_t max_useful = (total + min_chunk - 1u) / min_chunk;
  const uint32_t active_threads = std::min<uint32_t>(num_threads, max_useful);
  if (active_threads <= 1u) {
    fn(arg, 0u, total, 0u);
    return;
  }

  const uint32_t active_workers = active_threads - 1u;
  // Work-stealing granularity (~4 chunks/thread) balances heterogeneous ARM64 big+mid cores.
  const uint32_t target_chunks = active_threads * 4u;
  const uint32_t step = std::max<uint32_t>(min_chunk, (total + target_chunks - 1u) / target_chunks);

  pthread_mutex_lock(&mu);
  cur_fn = fn;
  cur_arg = arg;
  cur_total = total;
  cur_step = step;
  cur_active_workers = active_workers;
  next_item.store(0u, std::memory_order_relaxed);
  workers_remaining.store(active_workers, std::memory_order_release);
  job_gen += 1u;
  pthread_cond_broadcast(&work_cv);
  pthread_mutex_unlock(&mu);

  // Main thread participates as tid=0
  for (;;) {
    const uint32_t s = next_item.fetch_add(step, std::memory_order_relaxed);
    if (s >= total) break;
    const uint32_t e = (s + step < total) ? (s + step) : total;
    fn(arg, s, e, 0u);
  }

  // Brief bounded yield spin (~25 us) before sleeping on done_cv to avoid futex wakeup jitter
  if (workers_remaining.load(std::memory_order_acquire) > 0u) {
    const uint64_t spin_deadline = monotonic_now_ns() + 25000ULL;
    for (uint32_t it = 0u; workers_remaining.load(std::memory_order_acquire) > 0u; ++it) {
      if ((it & 15u) == 15u && monotonic_now_ns() >= spin_deadline) {
        break;
      }
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
      asm volatile("pause" ::: "memory");
#endif
    }
  }

  if (workers_remaining.load(std::memory_order_acquire) > 0u) {
    pthread_mutex_lock(&mu);
    while (workers_remaining.load(std::memory_order_acquire) > 0u) {
      pthread_cond_wait(&done_cv, &mu);
    }
    pthread_mutex_unlock(&mu);
  }
}

namespace {

void* async_dequant_thread_main(void* raw) {
  AsyncDequantWorker* w = static_cast<AsyncDequantWorker*>(raw);
  pin_thread_to_perf_cores();

  // ~1.5 ms active spin window keeps the helper awake across an entire pp512 forward pass
  // (~1 condvar wakeup total!) while dropping to 0% CPU during tg64/idle.
  constexpr uint64_t kIdleSpinNs = 1500000ULL;
  uint64_t c = w->cons_idx.load(std::memory_order_relaxed);

  for (;;) {
    uint64_t p = w->prod_idx.load(std::memory_order_acquire);
    if (c < p) {
      const AsyncDequantWorker::Slot s = w->ring[c & AsyncDequantWorker::kRingMask];
      if (s.fn) {
        s.fn(s.arg);
      }
      ++c;
      w->cons_idx.store(c, std::memory_order_seq_cst);
      if (w->main_waiting.load(std::memory_order_seq_cst)) {
        pthread_mutex_lock(&w->done_mu);
        pthread_cond_signal(&w->done_cv);
        pthread_mutex_unlock(&w->done_mu);
      }
      continue;
    }

    if (w->stopping.load(std::memory_order_acquire)) {
      return nullptr;
    }

    // Bounded lock-free spin before sleeping
    const uint64_t spin_deadline = monotonic_now_ns() + kIdleSpinNs;
    bool got_work = false;
    for (uint32_t it = 0u;; ++it) {
      if (w->prod_idx.load(std::memory_order_acquire) > c) {
        got_work = true;
        break;
      }
      if (w->stopping.load(std::memory_order_acquire)) {
        return nullptr;
      }
      if ((it & 63u) == 63u && monotonic_now_ns() >= spin_deadline) {
        break;
      }
      cpu_yield_hint();
    }
    if (got_work) continue;

    pthread_mutex_lock(&w->wake_mu);
    w->helper_sleeping.store(true, std::memory_order_seq_cst);
    while (!w->stopping.load(std::memory_order_acquire) &&
           w->prod_idx.load(std::memory_order_seq_cst) == c) {
      pthread_cond_wait(&w->wake_cv, &w->wake_mu);
    }
    w->helper_sleeping.store(false, std::memory_order_relaxed);
    pthread_mutex_unlock(&w->wake_mu);
  }
}

}  // namespace

AsyncDequantWorker::~AsyncDequantWorker() {
  shutdown();
  pthread_cond_destroy(&done_cv);
  pthread_mutex_destroy(&done_mu);
  pthread_cond_destroy(&wake_cv);
  pthread_mutex_destroy(&wake_mu);
}

bool AsyncDequantWorker::ensure_started() {
  if (started) return true;
  stopping.store(false, std::memory_order_relaxed);
  helper_sleeping.store(false, std::memory_order_relaxed);
  main_waiting.store(false, std::memory_order_relaxed);
  if (pthread_create(&thread, nullptr, &async_dequant_thread_main, this) != 0) {
    return false;
  }
  started = true;
  return true;
}

uint64_t AsyncDequantWorker::push(JobFn fn, void* arg) {
  if (!ensure_started()) {
    if (fn) fn(arg);
    const uint64_t next = prod_idx.load(std::memory_order_relaxed) + 1u;
    prod_idx.store(next, std::memory_order_relaxed);
    cons_idx.store(next, std::memory_order_release);
    return next;
  }
  const uint64_t p = prod_idx.load(std::memory_order_relaxed);
  if (p - cons_idx.load(std::memory_order_acquire) >= kRingCap) {
    wait_until(p - kRingCap + 1u);
  }
  ring[p & kRingMask] = Slot{fn, arg};
  const uint64_t next = p + 1u;
  prod_idx.store(next, std::memory_order_seq_cst);
  if (helper_sleeping.load(std::memory_order_seq_cst)) {
    pthread_mutex_lock(&wake_mu);
    pthread_cond_signal(&wake_cv);
    pthread_mutex_unlock(&wake_mu);
  }
  return next;
}

void AsyncDequantWorker::wait_until(uint64_t target_seq) {
  if (cons_idx.load(std::memory_order_acquire) >= target_seq) return;
  // Short ~60 us spin avoids futex latency when final tile finishes quickly
  const uint64_t spin_deadline = monotonic_now_ns() + 60000ULL;
  for (uint32_t it = 0u;; ++it) {
    if (cons_idx.load(std::memory_order_acquire) >= target_seq) return;
    if ((it & 31u) == 31u && monotonic_now_ns() >= spin_deadline) break;
    cpu_yield_hint();
  }
  pthread_mutex_lock(&done_mu);
  main_waiting.store(true, std::memory_order_seq_cst);
  while (cons_idx.load(std::memory_order_seq_cst) < target_seq) {
    pthread_cond_wait(&done_cv, &done_mu);
  }
  main_waiting.store(false, std::memory_order_relaxed);
  pthread_mutex_unlock(&done_mu);
}

void AsyncDequantWorker::shutdown() {
  if (!started) return;
  wait_until(prod_idx.load(std::memory_order_relaxed));
  pthread_mutex_lock(&wake_mu);
  stopping.store(true, std::memory_order_seq_cst);
  pthread_cond_broadcast(&wake_cv);
  pthread_mutex_unlock(&wake_mu);
  pthread_join(thread, nullptr);
  started = false;
  stopping.store(false, std::memory_order_relaxed);
}

}  // namespace ob
