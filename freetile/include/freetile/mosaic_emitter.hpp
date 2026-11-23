#pragma once

#include "freetile/tile_ir.hpp"

#include <cstdint>
#include <string>

namespace freetile {

struct MosaicEmitResult {
    std::string mosaic_mlir;
    std::string mosaic_mlir_b64;
    std::string backend_config_json;
    std::string kernel_name;
    std::string cache_key;
};

// Encodes raw bytes to standard RFC 4648 base64 (zero external dependencies).
std::string base64_encode(const std::string& input);

// Escapes a UTF-8 string for embedding inside an MLIR quoted string attribute.
std::string escape_mlir_string_literal(const std::string& raw);

// Lowers a validated FreeTile TileKernel IR into Mosaic MLIR (`stable_mosaic.*`)
// and builds the `tpu_custom_call` backend_config JSON payload (`serialization_format = 1`).
// Throws std::runtime_error if the kernel is invalid or target does not support Mosaic.
MosaicEmitResult emit_mosaic_kernel(const TileKernel& kernel, int64_t mosaic_version_override = -1);

}  // namespace freetile
