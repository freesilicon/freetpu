# `libopentpu` — Clean-Room Bare-Metal Cloud TPU Userspace Driver

Open-source (Apache-2.0) bare-metal Cloud TPU userspace driver over Linux `/dev/vfio/*`, exporting `GetPjrtApi()` (`PJRT_Api` v0.116) with zero runtime `libtpu.so` or `libpython` dependencies (`MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`).

## Build & Offline Tests

```bash
make -C libopentpu verify-header-host host test
```

Outputs in `libopentpu/build/host/`:
- `libopentpu.so` (or `.dylib` on macOS) — shared library exporting `ot_*` C API and `GetPjrtApi()`
- `libopentpu.a` — static library
- `opentpu_host_test` — offline unit tests (IOVA allocator, BAR4 protobuf descriptor parser, mock H2D/D2H + `128x128` BF16 matmul, VLIW/tiling emitter, and `GetPjrtApi()` mock execution)
- `opentpu_vfio_smoke` — standalone VFIO / `MAPS_LIBTPU=0` smoke binary (`--mock` for offline host verification)

## Milestones (Track 2 / Issue #16)

- **M1 (`/dev/vfio/0` Bare-Metal Access & Protocol Survey)**: Verified on `v5e-1` (`0x1ae0:0x0063`, `BAR0=4 GiB`, `BAR4=16 MiB`, `128` MSI-X vectors, `VFIO_TYPE1v2_IOMMU` + `MAP_DMA`, pre-booted BAR0 CSRs & BAR4 device descriptor proto, `VFIO_DEVICE_RESET` + `BAR0 + 0x10000` 256-slot SQ/CQ controller pointer reset). See [m1_hardware_survey.md](../docs/cloud-tpu-vfio/m1_hardware_survey.md).
- **M2 (Host <-> HBM DMA Round-Trip over `/dev/vfio/0`)**: Verified byte-exact (`match=1`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`) across `4 KiB`, `64 KiB`, and `1 MiB` transfers via `op=0x04` SQ/CQ DMA descriptors in `ot_vfio.cpp` + `ot_device.cpp`.
- **M3 (Bare-Metal `128x128` BF16 MXU Matmul over `/dev/vfio/0`)**: Cracked the `512 B` HBM launch descriptor (`OtV5eLaunchDesc`), `{1,0:T(2,128)}` BF16 HBM layout (`tile_pack_128x128_bf16` / `tile_unpack_128x128_bf16`), and 58-bundle `512`-bit VLIW MXU kernel (`ot::emit_v5e_matmul_128x128_body()` in `src/ot_vliw.h`). Clean-room emitter verified `0/16384` mismatches (`max_diff=0.000000`, `MAPS_LIBTPU=0`) on live `v5e-1` silicon; `ot_matmul_bf16` returns `OT_ERR_UNSUPPORTED` on live hardware until a standalone clean-room IMEM boot stub replaces resident command-listener init (`s8`) without vendor blobs.
- **M4 (`GetPjrtApi()` `0.116`)**: Verified via `dlopen("./build/host/libopentpu.so")` -> `GetPjrtApi()` over live `/dev/vfio/0` (`platform=tpu`, `devices=1`, `PJRT_Client_BufferFromHostBuffer` -> `PJRT_Buffer_ToHostBuffer` 64 KiB `match=1`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`). Because `libopentpu` is intentionally zero-dependency (no LLVM/MLIR/StableHLO runtime linked), `PJRT_Client_Compile` in M4 lowers single-op **textual** StableHLO `bf16` `dot_general` (`format="mlir"`) in `OPENTPU_MOCK=1` mode only for offline/direct C++ callers, returns `PJRT_Error_Code_UNIMPLEMENTED` on live VFIO clients until standalone clean-room IMEM boot lands, and returns `PJRT_Error_Code_UNIMPLEMENTED` for binary MLIR bytecode (`ML\xefR` portable StableHLO artifacts serialized by OpenXLA's `PjRtCApiClient`) pending a zero-dependency bytecode decoder.
