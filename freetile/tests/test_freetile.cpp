#include "freetile/mosaic_emitter.hpp"
#include "freetile/tile_ir.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#define CHECK_TRUE(cond, msg)                                                                  \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));             \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

int main() {
    // 1. Target profiles & chip detection
    {
        const auto v5e = freetile::target_profile(freetile::ChipTarget::V5E);
        const auto v6e = freetile::target_profile(freetile::ChipTarget::V6E);
        const auto t7x = freetile::target_profile(freetile::ChipTarget::TPU7X);
        const auto b0 = freetile::target_profile(freetile::ChipTarget::BUENOS_A0);
        CHECK_TRUE(v5e.mxu_m == 128 && v5e.vmem_bytes == 16ULL * 1024ULL * 1024ULL && v5e.supports_mosaic, "V5E profile");
        CHECK_TRUE(v6e.mxu_m == 256 && v6e.vmem_bytes == 32ULL * 1024ULL * 1024ULL && v6e.supports_mosaic, "V6E profile");
        CHECK_TRUE(t7x.mxu_m == 256 && t7x.vmem_bytes == 64ULL * 1024ULL * 1024ULL && t7x.supports_mosaic, "TPU7X profile");
        CHECK_TRUE(!b0.supports_mosaic && b0.lane_align == 64, "BUENOS_A0 hook profile");
        CHECK_TRUE(freetile::detect_chip_target("TPU v5 lite") == freetile::ChipTarget::V5E, "detect v5e");
        CHECK_TRUE(freetile::detect_chip_target("TPU v6e Trillium") == freetile::ChipTarget::V6E, "detect v6e");
        CHECK_TRUE(freetile::detect_chip_target("tpu7x-4t") == freetile::ChipTarget::TPU7X, "detect tpu7x");
        CHECK_TRUE(freetile::detect_chip_target("Pixel Tensor G5 buenos") == freetile::ChipTarget::BUENOS_A0, "detect buenos");
    }

    // 2. Auto-tiling planner + Mosaic emission for Llama-3.1-8B Q4_K projections
    {
        freetile::TileKernel k_ffn;
        std::string reason;
        const bool ok = freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 14336, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_ffn, &reason);
        CHECK_TRUE(ok, ("Q4_K 4096x14336 plan failed: " + reason).c_str());
        CHECK_TRUE(k_ffn.matmul.m_padded == 8, "M=1 pads to 8");
        CHECK_TRUE(k_ffn.grid.bn == 2048 && k_ffn.grid.bk == 2048, "Picks 2048x2048 tile for 4096x14336");
        CHECK_TRUE(k_ffn.grid.dims.size() == 2 && k_ffn.grid.dims[0] == 7 && k_ffn.grid.dims[1] == 2, "Grid 7x2");
        CHECK_TRUE(k_ffn.scratch.size() == 1, "Multi-K allocates scratch");

        const auto emitted = freetile::emit_mosaic_kernel(k_ffn);
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.version = 17") != std::string::npos, "Mosaic v17 attr");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.arith.sitofp") != std::string::npos, "S4 sitofp");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.tpu.matmul") != std::string::npos, "tpu.matmul");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.vector.multi_reduction") != std::string::npos, "Q4_K factored x_sum");
        CHECK_TRUE(emitted.mosaic_mlir.find("@transform_4") != std::string::npos, "5 window transforms");
        CHECK_TRUE(emitted.mosaic_mlir.find("memref<8x2048xf32, #tpu.memory_space<vmem>>") != std::string::npos, "F32 output");
        CHECK_TRUE(emitted.backend_config_json.find("\"serialization_format\": 1") != std::string::npos, "serialization_format 1");
    }

    // 3. Single-K Q6_K and Q4_0 and Q8_0
    {
        freetile::TileKernel k_q6;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 4, 2048, 2048, freetile::QuantFormat::Q6_K,
            freetile::DType::BF16, freetile::DType::BF16, false, &k_q6), "Q6_K 2048x2048 single-K");
        CHECK_TRUE(k_q6.grid.dims.size() == 1 && k_q6.scratch.empty(), "Single-K 1D grid, zero scratch");
        const auto em6 = freetile::emit_mosaic_kernel(k_q6);
        CHECK_TRUE(em6.mosaic_mlir.find("stable_mosaic.arith.truncf") != std::string::npos, "BF16 output truncf");
        CHECK_TRUE(em6.mosaic_mlir.find("stable_mosaic.vector.multi_reduction") == std::string::npos, "Q6_K has no min term");

        freetile::TileKernel k_q40;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 1024, 1024, freetile::QuantFormat::Q4_0,
            freetile::DType::BF16, freetile::DType::F32, true, &k_q40), "Q4_0 U4 1024x1024");
        const auto em40 = freetile::emit_mosaic_kernel(k_q40);
        CHECK_TRUE(em40.mosaic_mlir.find("stable_mosaic.arith.uitofp") != std::string::npos, "Q4_0 U4 uitofp");
        CHECK_TRUE(em40.mosaic_mlir.find("stable_mosaic.arith.subf") != std::string::npos, "Q4_0 U4 -8");

        freetile::TileKernel k_q80;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 8, 4096, 1024, freetile::QuantFormat::Q8_0,
            freetile::DType::BF16, freetile::DType::F32, false, &k_q80), "Q8_0 4096x1024");
        const auto em80 = freetile::emit_mosaic_kernel(k_q80);
        CHECK_TRUE(em80.mosaic_mlir.find("stable_mosaic.scf.if") != std::string::npos, "Multi-K scf.if");
    }

    // 4. Ineligible shapes & BUENOS_A0 hook behavior
    {
        freetile::TileKernel bad;
        std::string reason;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 128256, freetile::QuantFormat::Q6_K,
            freetile::DType::BF16, freetile::DType::F32, false, &bad, &reason),
            "128256 is divisible by 128 so Q6_K succeeds");

        CHECK_TRUE(!freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 32064, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &bad, &reason),
            "N=32064 (not div by 128) cleanly rejected");
        CHECK_TRUE(!reason.empty(), "Rejection provides reason");

        freetile::TileKernel b0_k;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::BUENOS_A0, 1, 1024, 1024, freetile::QuantFormat::Q8_0,
            freetile::DType::BF16, freetile::DType::BF16, false, &b0_k), "BUENOS_A0 plans TileKernel IR");
        CHECK_TRUE(b0_k.preferred_lowering == freetile::LoweringTarget::OPENBUENOS_DGC0_HOOK,
                   "BUENOS_A0 selects OPENBUENOS_DGC0_HOOK");
        bool threw = false;
        try {
            (void)freetile::emit_mosaic_kernel(b0_k);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_TRUE(threw, "BUENOS_A0 cleanly rejects Mosaic emission");
    }

    // 5. Base64 sanity check ("Man" -> "TWFu", "Ma" -> "TWE=", "M" -> "TQ==")
    {
        CHECK_TRUE(freetile::base64_encode("Man") == "TWFu", "b64 Man");
        CHECK_TRUE(freetile::base64_encode("Ma") == "TWE=", "b64 Ma");
        CHECK_TRUE(freetile::base64_encode("M") == "TQ==", "b64 M");
    }

    // 6. TilePlannerConfig extension hook seam (allows external tuning overrides with zero fork)
    {
        auto custom_hook = [](freetile::ChipTarget, int64_t, int64_t, int64_t, freetile::QuantFormat,
                              freetile::DType, freetile::DType, bool, void* user_data) -> freetile::TileShapeOverride {
            int* called = static_cast<int*>(user_data);
            *called += 1;
            return {true, 16, 512, 512};
        };
        int hook_calls = 0;
        freetile::TilePlannerConfig cfg;
        cfg.schedule_hook = custom_hook;
        cfg.hook_user_data = &hook_calls;
        freetile::TileKernel k_ov;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 14336, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_ov, nullptr, &cfg),
            "Hook override plans Q4_K kernel");
        CHECK_TRUE(hook_calls == 1, "Schedule hook invoked once");
        CHECK_TRUE(k_ov.grid.bm == 16 && k_ov.grid.bn == 512 && k_ov.grid.bk == 512,
                   "Schedule hook overrides (bm, bn, bk)");
    }

    std::printf("PASS: all freetile unit tests succeeded!\n");
    return 0;
}
