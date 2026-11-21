#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <linux/ioctl.h>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace tpu_runner {

inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20 || c == 0x7f) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  return out;
}

inline std::string json_str(const std::string& s) {
  return "\"" + json_escape(s) + "\"";
}

inline std::string read_trimmed_file(const std::string& path, int* out_errno = nullptr) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (out_errno) *out_errno = errno;
    return "";
  }
  std::string out;
  char buf[512];
  ssize_t n = 0;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) {
    out.append(buf, static_cast<size_t>(n));
  }
  int saved_err = (n < 0) ? errno : 0;
  ::close(fd);
  if (out_errno) *out_errno = saved_err;
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == '\0' || out.back() == ' ')) {
    out.pop_back();
  }
  return out;
}

inline bool read_binary_file(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    err = std::string("open(") + path + "): " + std::strerror(errno);
    return false;
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    err = std::string("fstat(") + path + "): " + std::strerror(errno);
    ::close(fd);
    return false;
  }
  if (st.st_size < 0) {
    err = "negative file size";
    ::close(fd);
    return false;
  }
  out.resize(static_cast<size_t>(st.st_size));
  size_t total = 0;
  while (total < out.size()) {
    ssize_t n = ::read(fd, out.data() + total, out.size() - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      err = std::string("read(") + path + "): " + std::strerror(errno);
      ::close(fd);
      return false;
    }
    if (n == 0) break;
    total += static_cast<size_t>(n);
  }
  ::close(fd);
  out.resize(total);
  return true;
}

inline bool write_binary_file(const std::string& path, const void* data, size_t size, std::string& err) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    err = std::string("open(") + path + "): " + std::strerror(errno);
    return false;
  }
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t written = 0;
  while (written < size) {
    ssize_t n = ::write(fd, ptr + written, size - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      err = std::string("write(") + path + "): " + std::strerror(errno);
      ::close(fd);
      return false;
    }
    written += static_cast<size_t>(n);
  }
  ::close(fd);
  return true;
}

struct ThermalZoneReading {
  int zone_id = -1;
  std::string name;
  std::string type;
  bool temp_ok = false;
  int64_t temp_millic = 0;
  int err = 0;
};

inline std::vector<ThermalZoneReading> read_thermal_zones() {
  std::vector<ThermalZoneReading> zones;
  DIR* dir = ::opendir("/sys/class/thermal");
  if (!dir) return zones;
  struct dirent* ent = nullptr;
  while ((ent = ::readdir(dir)) != nullptr) {
    if (std::strncmp(ent->d_name, "thermal_zone", 12) != 0) continue;
    ThermalZoneReading z;
    z.name = ent->d_name;
    z.zone_id = std::atoi(ent->d_name + 12);
    std::string base = std::string("/sys/class/thermal/") + ent->d_name;
    z.type = read_trimmed_file(base + "/type");
    int t_err = 0;
    std::string temp_s = read_trimmed_file(base + "/temp", &t_err);
    if (!temp_s.empty() && t_err == 0) {
      char* endp = nullptr;
      long long val = std::strtoll(temp_s.c_str(), &endp, 10);
      if (endp != temp_s.c_str()) {
        z.temp_ok = true;
        z.temp_millic = static_cast<int64_t>(val);
      }
    } else {
      z.err = t_err;
    }
    zones.push_back(std::move(z));
  }
  ::closedir(dir);
  for (size_t i = 0; i < zones.size(); ++i) {
    for (size_t j = i + 1; j < zones.size(); ++j) {
      if (zones[j].zone_id < zones[i].zone_id) {
        std::swap(zones[i], zones[j]);
      }
    }
  }
  return zones;
}

inline double monotonic_ms_since(const struct timespec& start, const struct timespec& end) {
  int64_t ds = static_cast<int64_t>(end.tv_sec) - static_cast<int64_t>(start.tv_sec);
  int64_t dns = static_cast<int64_t>(end.tv_nsec) - static_cast<int64_t>(start.tv_nsec);
  return static_cast<double>(ds) * 1000.0 + static_cast<double>(dns) / 1e6;
}

// Kernel DMA-BUF heap & sync ioctl ABI (stable Linux UAPI)
struct LocalDmaHeapAllocationData {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};

struct LocalDmaBufSync {
  uint64_t flags;
};

#define LOCAL_DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, LocalDmaHeapAllocationData)
#define LOCAL_DMA_BUF_IOCTL_SYNC   _IOW('b', 0x0, LocalDmaBufSync)
#define LOCAL_DMA_BUF_SYNC_READ    (1ULL << 0)
#define LOCAL_DMA_BUF_SYNC_WRITE   (2ULL << 0)
#define LOCAL_DMA_BUF_SYNC_RW      (LOCAL_DMA_BUF_SYNC_READ | LOCAL_DMA_BUF_SYNC_WRITE)
#define LOCAL_DMA_BUF_SYNC_START   (0ULL << 2)
#define LOCAL_DMA_BUF_SYNC_END     (1ULL << 2)

inline int alloc_dma_heap_buffer(size_t bytes, int* out_dmabuf_fd, std::string& err,
                                 const char* heap_path = "/dev/dma_heap/system") {
  int heap_fd = ::open(heap_path, O_RDONLY | O_CLOEXEC);
  if (heap_fd < 0) {
    int e = errno;
    err = std::string("open(") + heap_path + "): " + std::strerror(e);
    return -e;
  }
  size_t alloc_len = bytes < 4096 ? 4096 : ((bytes + 4095) & ~static_cast<size_t>(4095));
  LocalDmaHeapAllocationData req{};
  req.len = alloc_len;
  req.fd = 0;
  req.fd_flags = O_RDWR | O_CLOEXEC;
  req.heap_flags = 0;
  if (::ioctl(heap_fd, LOCAL_DMA_HEAP_IOCTL_ALLOC, &req) != 0) {
    int e = errno;
    err = std::string("ioctl(DMA_HEAP_IOCTL_ALLOC): ") + std::strerror(e);
    ::close(heap_fd);
    return -e;
  }
  ::close(heap_fd);
  *out_dmabuf_fd = static_cast<int>(req.fd);
  return 0;
}

inline int sync_dmabuf(int dmabuf_fd, uint64_t flags) {
  LocalDmaBufSync sync_req{flags};
  if (::ioctl(dmabuf_fd, LOCAL_DMA_BUF_IOCTL_SYNC, &sync_req) != 0) {
    return -errno;
  }
  return 0;
}

}  // namespace tpu_runner
