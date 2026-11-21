# Pixel Tensor G5 (`buenos_a0` / `laguna`) Hardware & Runtime Specifications

Clean-room hardware, kernel ABI, and execution parameters for Google Pixel 10 series Tensor G5 TPU (`buenos_a0` / `laguna`), verified against `/dev/edgetpu-soc` (`buenos.ko` ABI `1.4`) on Pixel 10 Pro Fold (`rango`).

---

## 1. Compute & Memory Architecture

| Parameter | Verified Value | Source / Verification |
| --- | --- | --- |
| **TPU Target / Family** | `buenos_a0` (`laguna`, ChipFamily `0xb`) | [kernel-abi](kernel-abi/README.md) |
| **Core Clock** | `1.12 GHz` (`2` DIVE control cores) | `ArchitectureConfig` / live silicon |
| **PE Array Geometry** | `4x4` (`16` physical tiles) in serpentine traversal | [multitile_gemm_isa.md](isa-gemm/multitile_gemm_isa.md) |
| **Serpentine Tile Order** | `SNAKE_16 = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]` | Closed-form 81/81 bit-exact |
| **Core Narrow SRAM (NM)** | `768 KiB` per core | On-chip SRAM tiling ceiling |
| **Unit Memory** | `1 MiB` | Tile/DMA scratchpad budget |
| **IOMMU Contexts** | `58` host IOMMU contexts (`36-bit` DMA-BUF IOVAs) | [edgetpu_buenos_uapi.h](kernel-abi/edgetpu_buenos_uapi.h) |
| **Control Core ISA** | `RV64GCV` (`ext[1]` DIVE VM interpreter, `638,977 B`) | Extracted locally on-device |

---

## 2. Kernel Driver Interface (`/dev/edgetpu-soc`)

- **Driver**: `buenos.ko` (`EDGETPU_GET_INTERFACE_VERSION` = `1.4`).
- **Command Transport**: Two-layer **Litebuf VII** (`schema 0x0008` -> `0x00fe` `silicon.dive.api.runtime_commands`) via ioctl `nr=37` (`0xc038ed25`) and `nr=38` (`0xc018ed26`).
  - Legacy `EDGETPU_VII_COMMAND` (`nr=35`) / `EDGETPU_VII_RESPONSE` (`nr=36`) return `-EOPNOTSUPP`.
  - Inline commands (`<= 96 B`, e.g. `Execute` `80 B`, `UnregisterModel` `32 B`) use `EDGETPU_VII_LITEBUF_RUNTIME_COMMAND`.
  - Large commands (`> 96 B`, e.g. `RegisterModel` `144 B`) use `EDGETPU_VII_LITEBUF_LARGE_RUNTIME_COMMAND`.
- **Mailbox Concurrency**: `EDGETPU_GET_CREDIT_SIZE` (`nr=39`) = **`16` concurrent credits**, signaled out-of-order via `eventfd` (`RESPDATA` + `FATAL_ERROR`).
- **Constraint**: `buenos.ko` enforces `seq >> 32 == 0` (32-bit sequence numbers).
- **Memory Allocation**: `/dev/dma_heap/system` (`DMA_HEAP_IOCTL_ALLOC`) mapped into TPU address space via `EDGETPU_MAP_DMABUF` (`0xc028ed11`).

See [litebuf_vii_protocol.md](baremetal-vii/litebuf_vii_protocol.md) and [litebuf_vii_wire.h](baremetal-vii/litebuf_vii_wire.h) for exact wire struct layouts.

---

## 3. Arithmetic & VLIW ISA Summary

- **VLIW Packet Encoding**: `16 B` flit-aligned (`64-bit` little-endian LSB-first sub-words).
- **Shadow-CSR Bank**: `98 x 25-bit` registers (`0x00..0x61`) packed into `HDR_DATA` (`0x200`) starting at bit `77`, updated incrementally by `VICA_CSR` and `CSR_040` (`0x040`).
- **Rounding Mode**: **IEEE-754 `float32` Round-Half-to-Even (RNE)** (`std::nearbyint` / `np.rint`) with post-round `zp_out` addition (`stream125` registers `0x5b..0x5f`), **not** fixed-point `(m, shift)` RHU.
- **Activation & Splines (`0x640`)**: `208 B` (`1664-bit`) `LoadCoefficientTables` encoding 8 piecewise degree-4 Horner polynomials + 7 breakpoints, supporting zero-overhead fused `GEMM + SiLU/GELU` in `chunk[5]`.
- **INT4 (`W4A8`)**: Verified on-device with `w_i4 + 8` (`CSR 0x56 = 0x0010000`) and `w_i4 + 128` (`0x0100000`), plus native `UINT4 -> BF16` nibble-pack pipeline (`CSR 0x56 = 0x0000026`, `50%` weight bandwidth savings).

See:
- [isa-gemm/multitile_gemm_isa.md](isa-gemm/multitile_gemm_isa.md)
- [isa-gemm/int4_and_fusion_spec.md](isa-gemm/int4_and_fusion_spec.md)
- [isa-dma/dma_ttu_bitfields.md](isa-dma/dma_ttu_bitfields.md)
- [isa-vec/vector_alu_isa.md](isa-vec/vector_alu_isa.md)
- [int4-splines/splines_silu_gelu_exp.md](int4-splines/splines_silu_gelu_exp.md)

---

## 4. Extracting `dive_interp_rv64.bin` Locally On-Device

`freetpu` is strictly clean-room and **does not ship** the `638,977 B` RV64 DIVE VM interpreter binary (`ext[1]`) from Pixel firmware/compiled containers.

Users with an unlocked Pixel 10 device can extract their own `ext[1]` locally using `libopenbuenos`:

```c
size_t sz = 0;
ob_gemm_extract_dive_interp_from_dgc0("/path/to/local_model.dgc0", NULL, 0, &sz);
uint8_t *buf = malloc(sz);
ob_gemm_extract_dive_interp_from_dgc0("/path/to/local_model.dgc0", buf, sz, &sz);
```

Or pass any locally compiled `.dgc0` directly to `ob_gemm_register_builtin_emitter(ctx, "/path/to/local_model.dgc0")` or via `OPENBUENOS_DIVE_INTERP_PATH=/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin` — `libopenbuenos` automatically extracts `ext[1]` in memory without embedding proprietary firmware in the library or repository.
