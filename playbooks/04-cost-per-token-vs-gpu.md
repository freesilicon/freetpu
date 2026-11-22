# Playbook 04: `$/1M Tokens` & `TFLOP/s-per-$` Scorecard vs H100 & RTX 5090

Generate a tweetable, apples-to-apples JSON + Markdown economics scorecard comparing your measured `freetpu-bench` run against Pixel Tensor G5 (`buenos_a0`), Cloud TPU `v5e-1`, Cloud TPU `v6e-1`, NVIDIA H100 SXM, and NVIDIA RTX 5090.

---

## 60-Second Copy-Paste Commands

### 1. Interactive Single-User Decode (`B=1`, `8 GB` Llama-3-8B W8A16 Footprint)

```bash
python3 bench/freetpu_bench.py \
  --suite all \
  --backend buenos_mock \
  --model-gb 8.0 \
  --bw-efficiency 0.70 \
  --batch-size 1 \
  --json runs/econ-b1.json \
  --markdown runs/econ-b1.md

cat runs/econ-b1.md
```

### 2. Batched Serving Projection (`B=32`, `8 GB` Weights)

```bash
python3 bench/freetpu_bench.py \
  --suite compute,bandwidth \
  --backend colab_tpu \
  --model-gb 8.0 \
  --bw-efficiency 0.70 \
  --batch-size 32 \
  --markdown runs/econ-b32.md
```

### 3. Compact 4-Bit Edge Footprint (`4.0 GB` Q4 Model)

```bash
python3 bench/freetpu_bench.py \
  --suite all \
  --backend buenos_mock \
  --model-gb 4.0 \
  --bw-efficiency 0.70 \
  --batch-size 1 \
  --markdown runs/econ-q4.md
```

---

## Economics Formulas (100% Transparent)

`freetpu-bench` computes normalized economics using three explicit equations printed in every Markdown/JSON output:

1. **Compute Efficiency (`TFLOP/s per $/hr`)**:
   $$\text{TFLOP/s per \$/hr} = \frac{\text{BF16 TFLOP/s}}{\text{Hourly Cost (\$/hr)}}$$
2. **Memory-Bandwidth-Bound Decode Throughput (`tok/s`)**:
   $$\text{est\_tok/s} = \frac{\text{BW}_{\text{GB/s}} \times \eta_{\text{BW}} \times B}{M_{\text{weights\_GB}}}$$
   Default parameters: $M_{\text{weights\_GB}} = 8.0\text{ GB}$ (Llama-3-8B W8A16), $\eta_{\text{BW}} = 0.70$, $B = 1$ (pass `--batch-size 32` for batched serving).
3. **Normalized Token Cost (`$/1M decode tokens`)**:
   $$\text{\$/1M tok} = \frac{\text{Hourly Cost (\$/hr)}}{\text{est\_tok/s} \times 3600} \times 10^6$$

---

## Published Reference Baselines & Pricing Citations

Every scorecard explicitly tags live runs as `measured` and reference rows as `published-reference`:

| Hardware | Peak / Verified Compute | Memory BW | Hourly Cost Basis | Public Pricing & Source Citation |
| --- | --- | --- | --- | --- |
| **Pixel Tensor G5 (`buenos_a0`)** | `0.993 TOPS` INT8 (live silicon) | `27.96 GB/s` weight DMA (live) | **`$0.05/hr`** amortized (`$0` cloud) | `$1,299` Pixel 10 Pro amortized 3yr 24/7 (`$0.049/hr`) + `~$0.001/hr` power (`~5W`); FreeTPU Phase 6 Report |
| **Cloud TPU `v5e-1`** | `179.36 TF/s` BF16 / `350.69 TOPS` INT8 (live, `197/394` peak) | `655.55 GB/s` triad (live, `819` peak) | **`$1.20/hr`** on-demand | Google Cloud TPU `v5e` public on-demand (`us-central1`/`us-east5`/`us-west1`: `$1.20/chip-hr`) |
| **Cloud TPU `v6e-1` (Trillium)** | `918 TF/s` BF16 / `1,836 TOPS` INT8 (spec) | `1,638 GB/s` HBM (spec) | **`$2.70/hr`** on-demand | Google Cloud TPU `v6e` public on-demand (`us-east1`/`us-east5`: `$2.70/chip-hr`) |
| **NVIDIA H100 SXM** | `989 TF/s` BF16 dense (spec) | `3,350 GB/s` HBM3 (spec) | **`$3.49/hr`** spec. cloud (`~$7–11/hr` hyperscaler) | NVIDIA H100 SXM datasheet (non-sparse Tensor Core) & public specialized GPU cloud rates |
| **NVIDIA RTX 5090** | `419 TF/s` BF16 dense (spec) | `1,792 GB/s` GDDR7 (spec) | **`$0.50/hr`** cloud (`~$0.20/hr` self-hosted) | NVIDIA Blackwell spec (non-sparse), community cloud midpoint (`$0.32–$0.80/hr`), `$1,999` FE MSRP |

---

## Reporting Rule

When posting a scorecard to GitHub, X, or a benchmark thread:
- Keep **`measured`** rows strictly separate from **`published-reference`** rows.
- Include the exact `--model-gb`, `--bw-efficiency`, and `--batch-size` assumptions whenever quoting `$/1M tok`.
- Cite live `B=1` vs `B=32` separately (for example, `v5e-1` measured `97.53 tok/s` at `B=1` = `$3.42 / 1M tok` vs `1,571.15 tok/s` at `B=32` = `$0.21 / 1M tok` on Llama-3-8B W8A16).
