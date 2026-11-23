#include "freetile/mosaic_emitter.hpp"
#include "freetile/tile_ir.hpp"

#include <cstdio>
#include <locale>
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
        CHECK_TRUE(emitted.mosaic_mlir.find("value = 8.000000e+00 : bf16") != std::string::npos, "Q4_K S4 +8 unshift const");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.arith.addf") != std::string::npos, "Q4_K S4 +8 addf");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.tpu.matmul") != std::string::npos, "tpu.matmul");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.vector.multi_reduction") != std::string::npos, "Q4_K F32 x_sum");
        CHECK_TRUE(emitted.mosaic_mlir.find("stable_mosaic.arith.truncf") == std::string::npos,
                   "F32-output Q4_K never truncates F32 x_sum to BF16");
        CHECK_TRUE(emitted.mosaic_mlir.find("@transform_4") != std::string::npos, "5 window transforms");
        CHECK_TRUE(emitted.mosaic_mlir.find("memref<8x2048xf32, #tpu.memory_space<vmem>>") != std::string::npos, "F32 output");
        CHECK_TRUE(emitted.backend_config_json.find("\"serialization_format\": 1") != std::string::npos, "serialization_format 1");

        freetile::TileKernel k_q4k_u4;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 1024, 1024, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, true, &k_q4k_u4), "Q4_K U4 1024x1024");
        const auto em_q4k_u4 = freetile::emit_mosaic_kernel(k_q4k_u4);
        CHECK_TRUE(em_q4k_u4.mosaic_mlir.find("stable_mosaic.arith.uitofp") != std::string::npos, "Q4_K U4 uitofp");
        CHECK_TRUE(em_q4k_u4.mosaic_mlir.find("vector<1024x1024xi4>") != std::string::npos, "Q4_K U4 signless vector i4");
        CHECK_TRUE(em_q4k_u4.mosaic_mlir.find("memref<1024x1024xi4, #tpu.memory_space<vmem>>") != std::string::npos,
                   "Q4_K U4 signless memref i4");
        CHECK_TRUE(em_q4k_u4.mosaic_mlir.find("xui4") == std::string::npos, "Q4_K U4 emits zero ui4 inside Mosaic");
        CHECK_TRUE(em_q4k_u4.mosaic_mlir.find("value = 8.000000e+00 : bf16") == std::string::npos, "Q4_K U4 has no +8 shift");
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
        CHECK_TRUE(em40.mosaic_mlir.find("vector<1024x1024xi4>") != std::string::npos, "Q4_0 U4 signless vector i4");
        CHECK_TRUE(em40.mosaic_mlir.find("xui4") == std::string::npos, "Q4_0 U4 emits zero ui4 inside Mosaic");
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

    // 7. Validation rejects mismatched operand & scratch windows against grid tile dims
    {
        freetile::TileKernel k_val;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 512, 256, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_val), "Plan Q4_K 512x256");
        CHECK_TRUE(k_val.grid.bn == 256, "bn == 256");

        // Mutate output window from 256 -> 128 on bn=256 grid: divides global N=256, must still fail
        freetile::TileKernel bad_out = k_val;
        bad_out.operands.back().window_bounds[1] = 128;
        std::string verr;
        CHECK_TRUE(!bad_out.validate(&verr), "Output window 128 != bn 256 rejected");
        bool threw = false;
        try {
            (void)freetile::emit_mosaic_kernel(bad_out);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_TRUE(threw, "emit_mosaic_kernel throws on mismatched output window");

        // Mutate scale window rows
        freetile::TileKernel bad_d = k_val;
        bad_d.operands[2].window_bounds[0] = 8;
        CHECK_TRUE(!bad_d.validate(&verr), "Scale window mismatch rejected");

        // Mutate scratch accumulator window on multi-K kernel
        freetile::TileKernel k_mk;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 1024, freetile::QuantFormat::Q8_0,
            freetile::DType::BF16, freetile::DType::F32, false, &k_mk), "Plan multi-K Q8_0");
        CHECK_TRUE(k_mk.scratch.size() == 1, "Multi-K has scratch");
        k_mk.scratch[0].window_bounds[1] = 128;
        CHECK_TRUE(!k_mk.validate(&verr), "Scratch window mismatch rejected");
    }

    // 8. Default dense planner handles aligned K < 128 (e.g. M=8, K=64, N=128 and K=96) without schedule hook
    {
        freetile::TileKernel k_d64;
        std::string reason;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 8, 64, 128, freetile::QuantFormat::NONE,
            freetile::DType::BF16, freetile::DType::F32, false, &k_d64, &reason),
            ("Dense M=8 K=64 N=128 plan failed: " + reason).c_str());
        CHECK_TRUE(k_d64.grid.bm == 8 && k_d64.grid.bk == 64 && k_d64.grid.bn == 128,
                   "Dense K=64 selects 8x64x128 tile");
        CHECK_TRUE(k_d64.grid.dims.size() == 1 && k_d64.grid.dims[0] == 1, "Dense K=64 single-K 1D grid");
        CHECK_TRUE(k_d64.validate(&reason), "Dense K=64 validates cleanly");
        const auto em_d64 = freetile::emit_mosaic_kernel(k_d64);
        CHECK_TRUE(em_d64.mosaic_mlir.find("vector<8x64xbf16>") != std::string::npos, "Dense K=64 vector<8x64xbf16>");

        freetile::TileKernel k_d96;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 8, 96, 128, freetile::QuantFormat::NONE,
            freetile::DType::BF16, freetile::DType::BF16, false, &k_d96, &reason),
            ("Dense M=8 K=96 N=128 plan failed: " + reason).c_str());
        CHECK_TRUE(k_d96.grid.bk == 96 && k_d96.validate(&reason), "Dense K=96 selects bk=96 and validates");
    }

    // 9. Positional activation window transforms (renaming operand 0 to "act" preserves transforms)
    {
        // Single-K multi-N kernel (N=4096, bn=2048 -> 2 N tiles)
        freetile::TileKernel k_sk;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 512, 4096, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_sk), "Plan single-K multi-N Q4_K");
        CHECK_TRUE(k_sk.grid.dims.size() == 1 && k_sk.grid.dims[0] == 2, "Single-K 2 N tiles");
        const auto em_sk_orig = freetile::emit_mosaic_kernel(k_sk);
        k_sk.operands[0].name = "act";
        const auto em_sk_renamed = freetile::emit_mosaic_kernel(k_sk);
        CHECK_TRUE(em_sk_renamed.mosaic_mlir == em_sk_orig.mosaic_mlir,
                   "Single-K multi-N window transforms ignore activation name");

        // Multi-K kernel (K=4096, N=4096 -> 2D grid)
        freetile::TileKernel k_mk;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 4096, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_mk), "Plan multi-K Q4_K");
        CHECK_TRUE(k_mk.grid.dims.size() == 2, "Multi-K 2D grid");
        const auto em_mk_orig = freetile::emit_mosaic_kernel(k_mk);
        k_mk.operands[0].name = "act";
        const auto em_mk_renamed = freetile::emit_mosaic_kernel(k_mk);
        CHECK_TRUE(em_mk_renamed.mosaic_mlir == em_mk_orig.mosaic_mlir,
                   "Multi-K window transforms ignore activation name");

        // Multi-K with PARALLEL K axis must fail validation & throw on emit
        freetile::TileKernel bad_sem = k_mk;
        bad_sem.grid.semantics[1] = freetile::DimensionSemantics::PARALLEL;
        std::string verr;
        CHECK_TRUE(!bad_sem.validate(&verr), "Multi-K PARALLEL K axis rejected by validate()");
        bool threw = false;
        try {
            (void)freetile::emit_mosaic_kernel(bad_sem);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_TRUE(threw, "Multi-K PARALLEL K axis throws in emit_mosaic_kernel()");
    }

    // 10. Unknown QuantFormat rejected cleanly before modulo
    {
        freetile::TileKernel bad_fmt;
        std::string reason;
        const bool ok = freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 512, 256, static_cast<freetile::QuantFormat>(99),
            freetile::DType::BF16, freetile::DType::F32, false, &bad_fmt, &reason);
        CHECK_TRUE(!ok && !reason.empty(), "Unknown QuantFormat cleanly rejected with diagnostic");
    }

    // 11. F32 scales/mins preserved without BF16 narrowing in TileKernel & Mosaic MLIR
    {
        freetile::TileKernel k_f32s;
        std::string reason;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 256, 128, freetile::QuantFormat::Q4_K,
            freetile::DType::F32, freetile::DType::F32, false, &k_f32s, &reason),
            ("Q4_K F32 scales plan failed: " + reason).c_str());
        CHECK_TRUE(k_f32s.matmul.scale_dtype == freetile::DType::F32 &&
                   k_f32s.operands[2].dtype == freetile::DType::F32 &&
                   k_f32s.operands[3].dtype == freetile::DType::F32,
                   "Q4_K preserves F32 d/m operand dtypes");
        const auto em_f32s = freetile::emit_mosaic_kernel(k_f32s);
        CHECK_TRUE(em_f32s.mosaic_mlir.find("memref<8x128xf32, #tpu.memory_space<vmem>>") != std::string::npos,
                   "F32 scale/min memrefs emitted as f32");
        CHECK_TRUE(em_f32s.mosaic_mlir.find("vector<1x128xbf16>) -> vector<1x128xf32>") == std::string::npos,
                   "F32 scale/min skips bf16->f32 extf");
    }

    // 12. Unsupported memory spaces (e.g. 4096x4096 BF16 HBM weight window on V5E) rejected
    {
        freetile::TileKernel bad_hbm;
        bad_hbm.name = "ft_dense_hbm_bypass";
        bad_hbm.target = freetile::ChipTarget::V5E;
        bad_hbm.grid.dims = {1};
        bad_hbm.grid.semantics = {freetile::DimensionSemantics::PARALLEL};
        bad_hbm.grid.bm = 8;
        bad_hbm.grid.bn = 4096;
        bad_hbm.grid.bk = 4096;
        bad_hbm.matmul.m = 8;
        bad_hbm.matmul.m_padded = 8;
        bad_hbm.matmul.n = 4096;
        bad_hbm.matmul.k = 4096;
        bad_hbm.matmul.quant = freetile::QuantFormat::NONE;
        bad_hbm.matmul.quant_block_size = 1;
        bad_hbm.matmul.has_min = false;
        bad_hbm.matmul.act_dtype = freetile::DType::BF16;
        bad_hbm.matmul.weight_dtype = freetile::DType::BF16;
        bad_hbm.matmul.out_dtype = freetile::DType::F32;
        bad_hbm.operands.push_back({"x", freetile::MemorySpace::VMEM, freetile::DType::BF16, {8, 4096}, {8, 4096}, false, false});
        bad_hbm.operands.push_back({"w", freetile::MemorySpace::HBM, freetile::DType::BF16, {4096, 4096}, {4096, 4096}, false, false});
        bad_hbm.operands.push_back({"out", freetile::MemorySpace::VMEM, freetile::DType::F32, {8, 4096}, {8, 4096}, true, false});

        std::string verr;
        CHECK_TRUE(!bad_hbm.validate(&verr), "4096x4096 HBM-tagged weight window rejected by validate()");
        bool threw = false;
        try {
            (void)freetile::emit_mosaic_kernel(bad_hbm);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_TRUE(threw, "4096x4096 HBM-tagged weight window throws in emit_mosaic_kernel()");

        // Even a small SMEM-tagged operand must be rejected on Mosaic targets
        freetile::TileKernel bad_smem;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 8, 64, 128, freetile::QuantFormat::NONE,
            freetile::DType::BF16, freetile::DType::F32, false, &bad_smem), "Plan small dense kernel");
        bad_smem.operands[1].space = freetile::MemorySpace::SMEM;
        CHECK_TRUE(!bad_smem.validate(&verr), "SMEM-tagged weight window rejected on Mosaic target");
    }

    // 13. Classic locale invariant (digit-grouping global locale must never emit "2,048")
    {
        struct CommaNumpunct : std::numpunct<char> {
            char do_thousands_sep() const override { return ','; }
            std::string do_grouping() const override { return "\3"; }
        };
        const std::locale prev_loc = std::locale::global(std::locale(std::locale::classic(), new CommaNumpunct()));
        struct LocaleGuard {
            std::locale prev;
            ~LocaleGuard() { std::locale::global(prev); }
        } guard{prev_loc};

        freetile::TileKernel k_loc;
        std::string reason;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 1, 4096, 14336, freetile::QuantFormat::Q4_K,
            freetile::DType::BF16, freetile::DType::F32, false, &k_loc, &reason), "Plan under grouping locale");
        CHECK_TRUE(k_loc.name.find(',') == std::string::npos, "Kernel name contains no locale commas");
        const auto em_loc = freetile::emit_mosaic_kernel(k_loc);
        CHECK_TRUE(em_loc.mosaic_mlir.find("2,048") == std::string::npos, "Mosaic MLIR contains no grouped 2,048");
        CHECK_TRUE(em_loc.mosaic_mlir.find("memref<8x2048xf32, #tpu.memory_space<vmem>>") != std::string::npos,
                   "Mosaic MLIR preserves clean 2048 dimension under grouping locale");
        CHECK_TRUE(em_loc.cache_key.find(',') == std::string::npos, "Mosaic cache_key contains no locale commas");
    }

    // 14. Unsafe kernel names (quotes, backslashes, whitespace, leading digits) rejected
    {
        freetile::TileKernel k_name;
        CHECK_TRUE(freetile::plan_quant_matmul_kernel(
            freetile::ChipTarget::V5E, 8, 64, 128, freetile::QuantFormat::NONE,
            freetile::DType::BF16, freetile::DType::F32, false, &k_name), "Plan dense kernel for name test");
        std::string verr;
        for (const char* bad : {"bad\"name\\x", "bad name", "123leading_digit", "bad\nname"}) {
            k_name.name = bad;
            CHECK_TRUE(!k_name.validate(&verr), "Unsafe kernel.name rejected by validate()");
            bool threw = false;
            try {
                (void)freetile::emit_mosaic_kernel(k_name);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            CHECK_TRUE(threw, "Unsafe kernel.name throws in emit_mosaic_kernel()");
        }
    }

    std::printf("PASS: all freetile unit tests succeeded!\n");
    return 0;
}
