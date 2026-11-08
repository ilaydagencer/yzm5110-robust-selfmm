# YZM5110 — Sentiment analysis with Self-MM under missing and noisy modalities

İlayda Gençer · YZM5110 research project · fork of [THUIAR/MMSA](https://github.com/thuiar/MMSA)

This file is a short summary. Setup, data access, commands and all tables are in
[research/README.md](research/README.md).

## Research question

Can the agreement between Self-MM's self-supervised unimodal predictions (text, audio,
visual) be used to set the fusion contributions when audio or visual input is missing or
corrupted? Does this rule reduce the error on corrupted input while preserving performance on clean input?
Does audio/visual dropout during training change this behavior?

## Main paper

Yu, W., Xu, H., Yuan, Z. and Wu, J. (2021). *Learning Modality-Specific
Representations with Self-Supervised Multi-Task Learning for Multimodal
Sentiment Analysis.* AAAI. [arXiv:2102.04830](https://arxiv.org/abs/2102.04830)

## Methods

| Method | Training | Inference |
|---|---|---|
| A | Standard Self-MM | Original fusion |
| B | Per-example 20% audio / 20% visual input dropout, during training only | Original fusion |
| C | Same checkpoint as A | Fusion weighted by the agreement of the unimodal predictions (τ=1), two forward passes |
| D | Same checkpoint as B | Same weighting as C |

Weight rule: d_m = \|y_m − mean(y_other)\|, w = 3 × softmax(−d/τ).
The weights are applied to the modality representations before the first fusion layer; no new
learned parameters are added.

## Data and conditions

- CMU-MOSI, MMSA's precomputed features: `MOSI/Processed/unaligned_50.pkl`
  (SHA-256 `78e0f8b5…2e02524`). Train/valid/test: 1284/229/686 examples.
  The data are not distributed in this repository; access is through the [MMSA datasets section](https://github.com/thuiar/MMSA#2-datasets).
- Conditions: K0 clean; K1 audio, K2 visual, K3 both set to zero; K4/K5/K6
  30%/50%/70% masking of valid time steps; K7 Gaussian noise at 0.5 times
  the train std. Text is kept intact in K0–K7; protocol v2 adds text corruption
  with `[UNK]` tokens (K8 30%, K9 60%, K10 all tokens).
- Training seeds 1111–1115; corruption seed 1111.

## Measured results (five seeds, test, mean ± sample SD)

| K0 method | MAE ↓ |
|---|---:|
| A | 0.7240 ± 0.0146 |
| B | 0.7574 ± 0.0244 |
| C | 0.8424 ± 0.0412 |
| D | 0.9163 ± 0.0747 |

- τ=1 weighting increased MAE in every seed and in every condition of K0–K7.
- The small validation MAE gain from dropout (B−A = −0.0119 ± 0.0281) did not hold on test.
- A's test MAE stayed almost the same even when audio and visual were both set to zero
  (K0 0.7240, K3 0.7235).
- Because of the two forward passes, the per-example inference time ratio was 2.05 for C/A and
  2.04 for D/B; the difference in peak allocated memory was about 6.5 KiB (seed 1111, K0, batch 4).

Source files: [test summary](research/results_package_v1/evaluation_test/summary_metrics.csv),
[validation summary](research/results_package_v1/validation/summary_metrics.csv),
[inference cost](research/results_package_v1/inference/paired_summary.csv).
The scores published in the paper and these measurements are given in separate tables in the report
([research/report/Gencer_YZM5110_Research_Report.docx](research/report/Gencer_YZM5110_Research_Report.docx)).

## Environment

Windows 11 · NVIDIA RTX 3050 Ti Laptop GPU (4 GB) · Python 3.10.5 ·
PyTorch 2.3.1 + CUDA 12.1 · MMSA 2.2.1. Package versions:
[requirements-research.txt](research/requirements-research.txt),
[environment-observed.txt](research/environment-observed.txt).

## Status (8 November 2026)

- Done: A/B training for five seeds; A/B/C/D K0–K7 validation and test
  evaluation; inference cost measurement; a real GPU rerun of the five seeds from their
  checkpoints; a rerun of seed 1111 in a new venv; pre-registered diagnostic experiments
  that separate why the weighting failed (protocol v2, validation and a single test run);
  bootstrap intervals; scripts that produce the report tables and figures
  ([research/report_tables/tables_en.md](research/report_tables/tables_en.md),
  [research/figures_en](research/figures_en)); a one-command CPU check
  (`research/run_cpu_checks.py`); the English report and slides in [research/report](research/report).
- The submitted version is fixed by the tag `v1.0-submission`.

## Result of the diagnostic experiments (protocol v2)

All six decisions written in advance came out as expected on validation and
replicated on test. The harm comes from rescaling the dominant text representation;
the agreement signal is no better than randomly shuffled weights. The predictions of the audio and
visual heads vary very little across examples (K0 test SD: audio
0.001–0.17, visual 0.004–0.44; text 0.64–1.00), so the rule compares the text with weak
predictions.
Under audio/visual corruption even the oracle gives no improvement; when text is corrupted, the rule
gives more weight to text. Details: [DIAGNOSTICS_V2.md](research/DIAGNOSTICS_V2.md).

AI assistance was used in preparing the code and text; the tools and
their contributions are described in an appendix of the report.
