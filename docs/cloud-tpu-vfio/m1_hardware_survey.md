# Cloud TPU `v5e-1` M1 Bare-Metal `/dev/vfio/0` Hardware & VFIO Survey

Verified directly on ephemeral Google Colab `v5e-1` (`TPU v5 lite`) silicon using standalone C++ (`vfio_m1_probe.cpp`, `vfio_m1_scan2.cpp`) with zero `libtpu.so` and zero `libpython` (`MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`).

## 1. PCIe & VFIO Container / Group Surface

- **BDF / IDs**: `0000:00:05.0`, `vendor=0x1ae0`, `device=0x0063`, `subsystem=0x1ae0:0x00af`, `rev=0x00`, `class=0xff0000`
- **Kernel Binding**: `driver=vfio-pci`, `iommu_group=0` (`/dev/vfio/vfio`, `/dev/vfio/0` mode `0666`)
- **VFIO API & IOMMU**:
  - `VFIO_GET_API_VERSION = 0`
  - `VFIO_TYPE1_IOMMU = 1`, `VFIO_TYPE1v2_IOMMU = 1`, `VFIO_DMA_CC_IOMMU = 0`, `VFIO_NOIOMMU_IOMMU = 0`
  - `VFIO_SET_IOMMU(VFIO_TYPE1v2_IOMMU)` succeeds
  - `VFIO_IOMMU_GET_INFO`: `iova_pgsizes=0xffffff7ffffff000` (4 KiB+ pages), valid IOVA ranges:
    - `[0x0 .. 0xfedfffff]`
    - `[0xfef00000 .. 0xfcffffffff]`
    - `[0x10000000000 .. 0xffffffffffffffff]`
  - `VFIO_IOMMU_MAP_DMA` / `VFIO_IOMMU_UNMAP_DMA`: verified `rc=0` across 32-bit and >32-bit IOVAs (`0x100000`, `0x80000000`, `0x100000000`, `0x1000000000`).
- **PCI Config Space (`256 B` at `region[7]`, `offset=0x70000000000`)**:
  - At rest before driver attach: `cmd=0x0003` (`IO=1 MEM=1 MASTER=0 INTxDis=0`). Userspace driver must set `PCI_COMMAND_MASTER` (`0x0004`) via `pwrite` on config space for DMA bus mastering.
  - `cap@0x80` (`0x11`, MSI-X): `tbl_size=128`, `tbl_bir=0`, `tbl_off=0x1000`, `pba_bir=0`, `pba_off=0x0`.
  - `cap@0x8c` (`0x10`, PCIe): `devcap=0x10008fc2`.
- **VFIO Regions (`num_regions=9`)**:
  - `region[0]` (`BAR0`): `size=0x100000000` (`4096 MiB`), `offset=0x0`, `flags=0xf` (`READ | WRITE | MMAP | CAPS`, cap id `3` MSI-X mappable).
  - `region[4]` (`BAR4`): `size=0x1000000` (`16 MiB`), `offset=0x40000000000`, `flags=0x7` (`READ | WRITE | MMAP`).
  - `region[7]` (`PCI Config`): `size=0x100` (`256 B`), `offset=0x70000000000`, `flags=0x3` (`READ | WRITE`).
- **VFIO IRQs (`num_irqs=5`)**:
  - `irq[0]` (`INTx`): `count=1`, `flags=0x7` (`EVENTFD | MASKABLE | AUTOMASKED`)
  - `irq[2]` (`MSI-X`): `count=128`, `flags=0x1` (`EVENTFD`)

## 2. Pre-Booted Hardware State (Zero `libtpu.so` Prior Touch)

BAR0 and BAR4 are already populated immediately on VM boot before any userspace TPU driver runs, establishing that Cloud TPU `v5e-1` silicon is **pre-booted by the hypervisor/host**:

- **BAR4 (`16 MiB`)**:
  - `+0x00008`: `u32` byte length of serialized device descriptor protobuf (`0x76`–`0x77` bytes observed).
  - `+0x0000c`: `u32` header word (`0x00140001`).
  - `+0x00010..`: serialized protobuf containing repeated per-unit submessages (`field 2`) and an ASCII chip/instance identifier (`field 3`).
  - `+0x00814`: `0x00000001`.
  - `+0x1000..0xffffff`: all `0x00000000` at rest.
- **BAR0 (`4096 MiB`)**:
  - `+0x00000..0x00fff`: MSI-X PBA (`0` at rest).
  - `+0x01000..0x017ff`: 128 x 16-byte MSI-X vector table entries (masked `0x00000001` at `+0xc`).
  - `+0x10000..0x12fff`: active control/capability register block (`[0]=0x00000001`, `[4]=0x00000010`, `+0x10f08=0x0000ffff`).
  - `+0x0a0510000..0x0a9020000` (~2.5 GiB aperture): populated pre-booted CSRs including four symmetric 1 MiB / 4 MiB replicated blocks (`0x0a2c00000/0x0a2d00000/0x0a2e00000/0x0a2f00000` and `0x0a3400000..0x0a4200000`), matching the 4-MXU architecture.

## 3. Black-Box VFIO IOMMU DMA & IRQ Lifecycle Observations

Observed at the `/dev/vfio/*` kernel boundary during a full `PJRT_Client_Create` -> 4 KiB H2D -> 4 KiB D2H -> `128x128` BF16 matmul -> `PJRT_Client_Destroy` cycle:

- `VFIO_DEVICE_SET_IRQS(index=2, start=0, count=128, flags=EVENTFD|TRIGGER)` resets all 128 MSI-X vectors (`eventfd=-1`), indicating polling-capable completion in single-chip flow.
- `VFIO_IOMMU_MAP_DMA` establishes a persistent IOVA map:
  - `0x00000000` (`0x8000` = 32 KiB)
  - `0x00008000` (`0x2000` = 8 KiB)
  - `0x0000a000` (`0x1000` = 4 KiB)
  - `0x0000b000` (`0x100000` = 1 MiB)
  - `0x00fef00000` (`0x80000` = 512 KiB)
  - `0x00fef80000` (`0x1000` = 4 KiB, unmap/remap scratch/handshake)
  - `0x00ff000000` (`0x100000000` = 4 GiB host staging arena)
  - `0x01ff000000` (`0x10000000` = 256 MiB)

## 4. `BAR0 + 0x10000` SQ/CQ Ring Controller & `op=0x04` PCIe <-> HBM DMA

- **Controller Init & `seq=0` Reset**:
  - Issue `VFIO_DEVICE_RESET` (`ioctl`), enable `PCI_COMMAND_MASTER`, map SQ (`0x8000` bytes at IOVA `0x0`, 256 x 128 B descriptors), CQ (`0x2000` bytes at IOVA `0x8000`, 256 x 32 B completions), and status page (`0x1000` bytes at IOVA `0xa000`).
  - `VFIO_DEVICE_RESET` leaves `BAR0 + 0x1001c` (`sq_head`) and `+0x10028` (`cq_tail`) non-zero across runs; zeroing `BAR0 + 0x10018 = 0`, `+0x1002c = 0`, and `+0x10044 = 0` after setting `+0x10000 = 1` clears stale ring pointers so `seq=0` finishes in **`~10 us`** with zero `BAR0 + 0x0a...` CSR writes required.
- **`op=0x04` PCIe <-> HBM DMA Descriptor (128 B SQ Entry)**:
  - `qwords = len / 8`: `w[8] = ((qwords & 0xffff) << 16) | 0x1001`, `w[9] = 0x00400000 | (qwords >> 16)`.
  - Host IOVA bits `[31:0]` / `[63:32]` in `w[10..]` (supporting both `<4 GiB` and `>=4 GiB` host IOVAs).
  - HBM `[35:32]` high bits encoded in `w[25]` (`D2H`: `0x00104000 | hi`) or `w[27]` (`H2D`: `0x40000000 | hi`).
  - Verified usable HBM aperture from `0x0_00100000` through `0x3_ef000000`, with **`match=1`** byte-exact round-trips across `4 KiB`, `64 KiB`, and `1 MiB` (`opentpu_vfio_smoke`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`) and over `GetPjrtApi()` (`PJRT_Client_BufferFromHostBuffer` / `PJRT_Buffer_ToHostBuffer`, `64 KiB` `match=1`).

## 5. `v5e-1` `512 B` Launch Descriptor, `{1,0:T(2,128)}` Tiling & 512-bit VLIW Matmul

- **`512 B` Launch Descriptor (`OtV5eLaunchDesc` at `0x3_fff00000`)**:
  - Uploaded via `op=0x04` (`w28=0x0001fb10`, `w29=0x01000000`):
  - `+0x170 = rhs_hbm >> 9`, `+0x174 = lhs_hbm >> 9`, `+0x17c = 0xffffffff`, `+0x190 = out_hbm >> 9`, `+0x1a0 = 0x20`, `+0x1a4 = prog_hbm >> 9`, `+0x1a8 = 1`, `+0x1ac = 1`, `+0x1e4 = prog_hbm >> 9`, `+0x1e8 = exec_seq`.
  - Runs cleanly with zeroed timestamp/fingerprint slots (`+0x180`, `+0x1f4..+0x1fc`).
- **`{1,0:T(2,128)}` BF16 HBM Layout**:
  - Pair-major `128x128` BF16 packing: `tiled_u16_idx(r, c) = (r >> 1) * 256 + c * 2 + (r & 1)` (`ot::tile_pack_128x128_bf16` / `tile_unpack_128x128_bf16` in `libopentpu/src/ot_vliw.h`).
- **58-Bundle `128x128` BF16 MXU Kernel (`ot::emit_v5e_matmul_128x128_body()`)**:
  - Clean-room 512-bit (`16 x uint32_t`) VLIW bundle sequence performing RHS/LHS HBM->VMEM DMA, `16 x 8`-row MXU weight pushes, `4 x 4 KiB` activation loads, 8 MXU matmul-accumulate steps, 8 VSTORE accumulator flushes, VMEM->HBM output DMA, sync restore, and scalar return (`0/16384` errors, `max_diff=0.000000` against random `128x128` BF16 `A @ B` on live `v5e-1` silicon with `MAPS_LIBTPU=0`).
