import os, glob, json, subprocess, inspect, ctypes
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("=== #4 READ-ONLY VFIO / PCI HARDWARE SURFACE SURVEY ===")
for path in ["/dev/vfio/", "/dev/accel*", "/dev/tpu*"]:
    r_ls = subprocess.run(f"ls -la {path} 2>&1 || true", shell=True, text=True, capture_output=True)
    print(f"$ ls -la {path}\n{r_ls.stdout.strip()}")

r_ti = subprocess.run("tpu-info 2>&1 || python3 -m tpu_info 2>&1 || true", shell=True, text=True, capture_output=True)
print("$ tpu-info\n" + r_ti.stdout.strip())

print("--- JAX / TPU Topology & HBM ---")
print("Default backend:", jax.default_backend())
for d in jax.devices():
    print("Device:", d, "kind:", d.device_kind, "coords:", getattr(d, "coords", None))
    try:
        ms = {k: int(v) for k, v in d.memory_stats().items()}
        print("memory_stats:", json.dumps(ms))
    except Exception as e:
        print("memory_stats err:", e)
try:
    print("pltpu.get_tpu_info():", pltpu.get_tpu_info())
except Exception as e:
    print("get_tpu_info err:", e)

print("--- All PCI Devices (Google / VFIO / Non-Bridge) ---")
for pci_dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
    bdf = os.path.basename(pci_dev)
    vendor = open(os.path.join(pci_dev, "vendor")).read().strip() if os.path.exists(os.path.join(pci_dev, "vendor")) else ""
    device = open(os.path.join(pci_dev, "device")).read().strip() if os.path.exists(os.path.join(pci_dev, "device")) else ""
    cls = open(os.path.join(pci_dev, "class")).read().strip() if os.path.exists(os.path.join(pci_dev, "class")) else ""
    drv = os.path.basename(os.readlink(os.path.join(pci_dev, "driver"))) if os.path.exists(os.path.join(pci_dev, "driver")) else "none"
    grp = os.path.basename(os.readlink(os.path.join(pci_dev, "iommu_group"))) if os.path.exists(os.path.join(pci_dev, "iommu_group")) else "none"
    print(f"  {bdf}: vendor={vendor} device={device} class={cls} driver={drv} iommu_group={grp}")
    if drv == "vfio-pci" or (vendor == "0x1ae0" and cls == "0xff0000"):
        for field in ["subsystem_vendor", "subsystem_device", "revision",
                      "current_link_speed", "current_link_width", "max_link_speed", "max_link_width",
                      "numa_node", "irq", "dma_mask_bits", "consistent_dma_mask_bits", "enable"]:
            p = os.path.join(pci_dev, field)
            if os.path.exists(p):
                print(f"    pci.{field}: {open(p).read().strip()}")
        res_path = os.path.join(pci_dev, "resource")
        if os.path.exists(res_path):
            for idx, line in enumerate(open(res_path)):
                parts = line.strip().split()
                if len(parts) == 3:
                    s, e, fl = int(parts[0], 16), int(parts[1], 16), int(parts[2], 16)
                    if s or e:
                        sz = (e - s) + 1
                        print(f"    BAR{idx}: start=0x{s:016x} end=0x{e:016x} size={sz} ({sz/(1024*1024):.2f} MiB) flags=0x{fl:x}")
        msix = glob.glob(os.path.join(pci_dev, "msi_irqs", "*"))
        print("    msi_irqs count:", len(msix))

print("\n=== LIBTPU / PJRT C API SURFACE ===")
override_libtpu = os.environ.get("TPU_LIBRARY_PATH")
if override_libtpu and os.path.isfile(override_libtpu):
    so_path = override_libtpu
else:
    import libtpu
    so_path = libtpu.get_library_path()
print("libtpu.so:", so_path)
r2 = subprocess.run(f"nm -D '{so_path}' 2>&1 | grep -iE 'pjrt|tpu' | head -n 40", shell=True, text=True, capture_output=True)
print("--- exported symbols (pjrt/tpu) ---")
print(r2.stdout.strip())

lib = ctypes.CDLL(so_path)
lib.GetPjrtApi.restype = ctypes.c_void_p
api_ptr = lib.GetPjrtApi()
print(f"GetPjrtApi() ptr: 0x{api_ptr:x}")

class PJRT_Api_Header(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("ver_struct_size", ctypes.c_size_t),
        ("ver_ext", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]
hdr = PJRT_Api_Header.from_address(api_ptr)
print(f"PJRT_Api struct_size={hdr.struct_size} version={hdr.major_version}.{hdr.minor_version}")

r3 = subprocess.run("which g++ gcc clang++ cmake ninja curl 2>&1; find /usr /opt -name 'pjrt_c_api.h' 2>/dev/null | head -n 10", shell=True, text=True, capture_output=True)
print("--- build tools & pjrt headers ---")
print(r3.stdout.strip())

print("\n=== PALLAS TPU API SURFACE (JAX 0.11.2) ===")
print("pltpu attrs:", sorted([a for a in dir(pltpu) if not a.startswith("_")]))
print("pl.pallas_call sig:", inspect.signature(pl.pallas_call))
if hasattr(pltpu, "PrefetchScalarGridSpec"):
    print("PrefetchScalarGridSpec sig:", inspect.signature(pltpu.PrefetchScalarGridSpec))
if hasattr(pltpu, "CompilerParams"):
    print("CompilerParams sig:", inspect.signature(pltpu.CompilerParams))
if hasattr(pltpu, "async_copy"):
    print("async_copy sig:", inspect.signature(pltpu.async_copy))
try:
    from jax.experimental.pallas.ops.tpu import flash_attention
    print("flash_attention attrs:", [a for a in dir(flash_attention) if not a.startswith("_")][:30])
    if hasattr(flash_attention, "flash_attention"):
        print("flash_attention sig:", inspect.signature(flash_attention.flash_attention))
except Exception as e:
    print("flash_attention import:", repr(e))

try:
    from jax.experimental.pallas.ops.tpu import splash_attention
    print("splash_attention attrs:", [a for a in dir(splash_attention) if not a.startswith("_")][:30])
except Exception as e:
    print("splash_attention import:", repr(e))

print("=== SURVEY_COMPLETE ===")
