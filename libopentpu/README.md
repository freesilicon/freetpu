# `libopentpu` — Clean-Room Bare-Metal Cloud TPU Userspace Driver

Open-source (Apache-2.0) bare-metal Cloud TPU userspace driver over Linux `/dev/vfio/*`, exporting `GetPjrtApi()` (`PJRT_Api` v0.116) with zero runtime `libtpu.so` or `libpython` dependencies (`MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`).

## Build & Offline Tests

```bash
make -C libopentpu verify-header-host host test
```

Outputs in `libopentpu/build/host/`:
- `libopentpu.so` (or `.dylib` on macOS) — shared library exporting `ot_*` C API and `GetPjrtApi()`
- `libopentpu.a` — static library
- `opentpu_host_test` — offline unit tests (IOVA allocator, BAR4 protobuf descriptor parser, mock H2D/D2H + `128x128` BF16 matmul, `{1,0:T(2,128)(2,1)}` tiling/aliasing/chaining parity, IMEM stub structure, and `GetPjrtApi()` mock execution/validation)
- `opentpu_vfio_smoke` — standalone VFIO / `MAPS_LIBTPU=0` DMA smoke binary (`--mock` for offline host verification)
- `opentpu_matmul_proof` — standalone `dlopen` live proof binary exercising both C API `ot_matmul_bf16` and `GetPjrtApi()` across consecutive random `128x128` BF16 runs (`--mock` for offline host verification)

## Milestones (Track 2 / Issue #16)

- **M1 (`/dev/vfio/0` Bare-Metal Access & Protocol Survey)**: Verified on `v5e-1` (`0x1ae0:0x0063`, `BAR0=4 GiB`, `BAR4=16 MiB`, `128` MSI-X vectors, `VFIO_TYPE1v2_IOMMU` + `MAP_DMA`, pre-booted BAR0 CSRs & BAR4 device descriptor proto, `VFIO_DEVICE_RESET` + `BAR0 + 0x10000` 256-slot SQ/CQ controller pointer reset). See [m1_hardware_survey.md](../docs/cloud-tpu-vfio/m1_hardware_survey.md).
- **M2 (Host <-> HBM DMA Round-Trip over `/dev/vfio/0`)**: Verified byte-exact (`match=1`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`) across `4 KiB`, `64 KiB`, and `1 MiB` transfers via `op=0x04` SQ/CQ DMA descriptors in `ot_vfio.cpp` + `ot_device.cpp`.
- **M3 (Self-Contained Bare-Metal `128x128` BF16 MXU Matmul over `/dev/vfio/0`)**: Complete clean-room 120-bundle (`7,680 B`) direct-in-IMEM resident boot/poller + 58-bundle `512`-bit VLIW MXU matmul loop (`ot::emit_v5e_resident_matmul_imem()` in `src/ot_vliw.h`), direct Host -> IMEM DMA (`op=0x04`), core start (`op=0x03 d[8]=0x121`), slot-0 `OtV5eLaunchDesc` doorbell (`0x3fff00000`), and transparent `{1,0:T(2,128)(2,1)}` HBM layout handling in `ot_device.cpp` + `ot_pjrt.cpp`. See [imem_boot_stub_spec.md](../docs/cloud-tpu-vfio/imem_boot_stub_spec.md).
- **M4 (`GetPjrtApi()` `0.116`)**: Verified via `dlopen("./build/host/libopentpu.so")` -> `GetPjrtApi()` over live `/dev/vfio/0` (`platform=tpu`, `devices=1`, `PJRT_Client_BufferFromHostBuffer` -> `PJRT_Client_Compile` -> `PJRT_LoadedExecutable_Execute` -> `PJRT_Buffer_ToHostBuffer`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`). Because `libopentpu` is intentionally zero-dependency (no LLVM/MLIR/StableHLO runtime linked), `PJRT_Client_Compile` lowers single-op **textual** StableHLO `bf16` `dot_general` (`format="mlir"`) for both live VFIO (`128x128x128`) and mock clients, and returns `PJRT_Error_Code_UNIMPLEMENTED` for binary MLIR bytecode (`ML\xefR`) pending a zero-dependency bytecode decoder.
- **Multi-Gen HAL & Ironwood (`tpu7x`) Prep**: `ot_chip_geometry_for_device_id()` / `ot_device_get_geometry()` parameterize MXU tiles (`128x128` for `0x0063` v5e, `256x256` for `0x006f` v6e and provisional `0x0075` Ironwood `tpu7x` with `(4,1)` FP8 inner tile and dual-chiplet `96 GiB/chiplet`), `clamp_bar_mmap_len()` / `parse_pcie_rebar_cap()` guard 64-bit Resizable BAR mappings, and `GetPjrtApi()` enumerates dual-chiplet `coords` + `core_on_chip` attributes (`OPENTPU_MOCK_TOPOLOGY=tpu7x-1t` / `tpu7x-4t`). See [`docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md`](../docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md).
