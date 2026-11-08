> English translation made on 8 November 2026, after all runs, for readers of the
> submission. The pre-registered document is the Turkish original,
> `research/selfmm_diagnostics_protocol_v2.md`, committed before the first run
> (commit e3e73b9, SHA-256 f45d39f6…e0eb1); its hash is recorded in
> `research/diagnostics_v2/*/manifest.json`. If the two differ, the Turkish
> original is authoritative. Nothing was added to or removed from the content.

# Self-MM weighting diagnostic protocol, version 2

This protocol was written on 25 October 2026, before any result of the diagnostic
experiments had been seen. The document and [`diagnose_selfmm_weighting.py`](diagnose_selfmm_weighting.py)
are committed before the GPU run; the Git history is the evidence of the order. This
document is not changed after the results have been seen. If needed, a new version
is opened and the difference is written in the report as a deviation.

## 1. Why these experiments?

Fixed results of version 1 (five seeds, validation and test):

- τ=1 agreement weighting (C/D) increased MAE in every condition of K0–K7 and in
  every seed. On clean validation, C−A = +0.1497 ± 0.0414.
- A's test MAE remained almost unchanged even when audio and visual were zeroed
  together (K0 0.7240; K3 0.7235).

These two observations do not explain the cause of the negative result. To separate
the possible causes from one another, this protocol defines controlled experiments
that, without new training, perform only inference on the existing 10 checkpoints.

**Seen beforehand:** the v1 validation and test results have been seen. The H4 and
H5a expectations are partly based on these observations; they are not presented as
an independent discovery. The K8–K10 text conditions and all new weight variants
have not been run on any split before.

## 2. What does not change

- Checkpoints: training seeds 1111–1115, baseline and dropout (10 in total).
  The identifiers are read from the existing records:
  `configs/selfmm_portable_k0_dependencies.json` (1111) and
  `configs/selfmm_portable_seed_dependencies.json` (1112–1115).
- Batch 4, `shuffle=False`, FP32, `eval()` and `inference_mode()`.
- The K0–K7 audio/visual corruptions are generated with the same module as v1 and
  corruption seed 1111. The noise scale is recomputed from the train data only.
- As in v1, the weights are applied to the text (768), audio (16) and visual (32)
  representation blocks at the input of the first fusion layer. The unimodal heads
  do not change.

## 3. New conditions: text corruption

| Condition | Text | Audio and visual |
|---|---|---|
| K8 | Each of the valid tokens independently becomes `[UNK]` with 30% probability | K0 (clean) |
| K9 | The same operation at 60% | K0 |
| K10 | All valid tokens become `[UNK]` | K0 |

- The `[CLS]` (first) and `[SEP]` (last valid) tokens are preserved; the attention
  mask, segment id and padding do not change. This is the same as the missing-text
  rule that MMSA uses for TFR-Net (`[UNK]`=100, CLS/SEP preserved).
- The randomness is derived from the key `selfmm-diagnostics-v2|1111|bölme|örnek|text|unk` [translator: bölme = split, örnek = sample];
  the model, batch and run order play no part. The K8 mask is a subset of the K9
  mask, and the K9 mask is a subset of the K10 mask.
- The realized masking rates are reported separately.

## 4. Weight variants

For every checkpoint and every condition (K0–K10), with the same inputs:

| Variant | Weight (text, audio, visual) | Question it isolates |
|---|---|---|
| `plain` | No weight (A or B) | Reference |
| `uniform` | (1, 1, 1) | Identity check; should come out identical to `plain` |
| `agree_tau{τ}` | 3·softmax(−d/τ), τ ∈ {0.25, 0.5, 1, 2, 4, 8, 16} | Does the harm grow with the amount of scaling? τ=1 is C/D |
| `agree_av_tau1` | Text 1; audio and visual 2·softmax(−[d_A, d_V]/1) | Does the harm come from the scaling of the text block? |
| `shuffled_tau1` | The τ=1 weight vectors swap places across samples according to a fixed permutation | Does the agreement signal carry sample-specific information? |
| `oracle` | Known missing modality 0, the others 1. Only K1 (audio), K2 (visual), K3 (both), K10 (text) | If the signal were perfect, would there have been room for improvement? |

d_m = |ŷ_m − mean(ŷ_other)|, where ŷ_m is the first-pass prediction of the unimodal
heads. The permutation is generated from the key `selfmm-diagnostics-v2|1111|bölme|koşul|shuffle` [translator: bölme = split, koşul = condition];
the two checkpoints and the five seeds use the same permutation. The metrics of the
unimodal heads (`head_T`, `head_A`, `head_V`) against the labels are also reported.

## 5. Hypotheses and decision criteria

The primary metric is MAE. The criteria are applied on validation (229 samples), with
the **baseline checkpoint** and across the five seeds. "Mean" is the mean of the five
seeds. The paired difference is first taken within the same seed. The dropout
checkpoint (B/D) is reported as secondary with the same tables; it has no separate
decision criterion.

| Code | Hypothesis | Support criterion | Prior expectation |
|---|---|---|---|
| H1 | The harm increases with the amount of scaling | K0–K7 mean MAE decreases at every step from τ 0.25 to 16 (equality up to 1e-4 accepted) **and** in K0 \|agree_tau16 − plain\| ≤ 0.01 | Supported |
| H2 | The main source of the harm is the scaling of the text block | In K0, mean (agree_av_tau1 − plain) < 0.25 × mean (agree_tau1 − plain) **and** agree_av_tau1 < agree_tau1, 5/5 seeds | Supported |
| H3 | The agreement signal carries useful sample-specific information | In K0, agree_tau1 < shuffled_tau1 in at least 4/5 seeds **and** mean difference ≤ −0.01 | Not supported |
| H4 | Under audio/visual corruption, the room for improvement is small | In each of K1, K2 and K3, \|mean (oracle − plain)\| ≤ 0.01 | Supported |
| H5a | The model relies predominantly on text | For plain, mean MAE(K10) − MAE(K0) ≥ 0.10 | Supported |
| H5b | The rule is useful when the dominant modality is corrupted | In at least one of K9 or K10, agree_tau1 < plain in at least 4/5 seeds **and** mean difference ≤ −0.01 | Not supported |

An expectation that does not hold is not an error but a finding; it is written in the
report as "not verified". The criteria are descriptive: no p-value is computed, no
multiple-comparison correction is made, and the five seeds are not interpreted as an
independent sample size. Sample-level bootstrap intervals are added in a separate
analysis.

## 6. τ selection and test rule

- **τ\*** = the τ that, on the baseline checkpoint, taking the equally weighted mean
  over the K0–K10 conditions, gives the lowest five-seed mean validation MAE.
  In a tie within 1e-6, the larger τ is chosen. The selection is made from validation
  only and is committed together with the validation summary.
- The test runs **once** for each seed, with this file given via `--selection`.
  The test run computes all variants; in the report, τ\* is given as the headline
  result and the other τ values as the full curve. A variant or τ that looks good on
  test is not selected.
- If an error is found after the test, the correction is written as a deviation; the
  initial and corrected results are kept together.

## 7. Run checks

A run counts as completed only if all of these pass:

- **Reference replication:** in K0–K7, `plain` and `agree_tau1` match the A/B/C/D
  metrics in the result package (validation or test, the selected seed) with a
  tolerance of 1e-5: 32 rows × 4 metrics = 128 checks. The new numbers are trusted
  only if this passes.
- On the test split, the K0–K7 corrupted-input digests are identical to the v1
  references.
- The train statistics match the v1 references.
- `uniform` is identical to `plain` in every batch (1e-6); temporary hooks are
  removed; the unimodal heads are not affected by the weighting; the model inputs do
  not change; the two checkpoints receive the same corrupted inputs.
- The SHA256 digests of the data, checkpoint, config, source code and result package
  files do not change during the run.

## 8. Outputs

For each seed and split, into a new, empty folder:

- `manifest.json`: versions, input digests, protocol and script digest, status.
- `metrics.csv`: checkpoint × condition × variant; MAE, Non0 accuracy, Non0 F1, Corr.
- `reference_checks.csv`: 128 reference checks.
- `predictions.csv`: per sample, the label, all variant predictions, the unimodal
  predictions, d and the τ=1 weights.
- `weight_summary.csv`: weight mean/SD/min/max per condition and variant.
- `input_summary.csv`: input summary per condition, masking rates.
- `controls.csv`, `train_feature_stats.json`, `train_stats_checks.csv`.

The summary of the five seeds, the hypothesis decisions and τ\* are produced by a
separate summary script.

## 9. Limitations

There is a single dataset, a single corruption seed, synthetic feature-level
corruption and a small validation set (229 samples). All variants are applied after
training (post-hoc); retraining with the weighting is outside the scope of this
protocol. `oracle` uses information that is unknown in real use and is only an
upper-bound diagnostic.
