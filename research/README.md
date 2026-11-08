# YZM5110 Self-MM research project

İlayda Gençer — an adapted reproduction of Self-MM on CMU-MOSI, and a study of
whether the agreement between its unimodal predictions can serve as a fusion
weight when modalities are missing or noisy. The code base is
[MMSA](https://github.com/thuiar/MMSA); the base paper is
[Self-MM](https://arxiv.org/abs/2102.04830).

## Short path for the reader

**What was done?** Self-MM was reproduced on CMU-MOSI with settings adapted to a
4 GB GPU, using five training seeds. A parameter-free fusion weight based on the
agreement of the unimodal predictions (C/D) and audio/video dropout during
training (B) were tested in eleven conditions (K0–K10). The cause of the negative
result was separated with a diagnostic protocol committed before the results
([protocol v2](selfmm_diagnostics_protocol_v2.md), pre-registered Turkish original;
[English translation](selfmm_diagnostics_protocol_v2_en.md)).

**What was found?**

- Clean test MAE 0.7240 ± 0.0146 (0.713 in the paper); Non0 accuracy 2.08 percentage points below the paper.
- τ=1 weighting increased MAE in every K0–K7 condition and every seed
  (+0.118 at K0, 95% bootstrap interval [+0.091; +0.148]) and roughly doubled
  inference time. The small validation gain of dropout did not carry over to test.
- All six decisions of protocol v2 matched the expectation written beforehand and
  replicated on test: the harm comes from rescaling the dominant text block; the
  agreement signal is no better than randomly shuffled weights; even oracle weights
  do not help under audio/video corruption; when text is corrupted, the rule gives
  text more weight.

**Report and presentation.** The submitted report is
[report/Gencer_YZM5110_Research_Report.docx](report) (the PDF export sits next to it);
the slides are in the same folder.

**One-command check.** No GPU, dataset, checkpoint or internet connection is needed.
Run it from the repository root (the folder above this one, `MMSA`):

```powershell
py -3.10 -B .\research\run_cpu_checks.py
```

The last line should be `ALL_CHECKS_PASSED`. The script runs seven checks in
separate processes and reports each as `[PASS]`, `[FAIL]` or `[SKIP]`. A missing
optional package does not fail a check; the `[SKIP]` line says what to install.
After each `[PASS]` the script repeats the status line of the checked script; some
of these scripts print Turkish ASCII tokens (see [Language note](#language-note)).

| Check | Needs |
|---|---|
| v1 result package, the two protocol v2 summaries, report tables (Turkish and English) | Python 3.10 standard library only |
| Bootstrap intervals (byte for byte) | numpy 1.26.4 |
| Figures redrawn into a temporary folder, Turkish and English labels | [requirements-figures.txt](requirements-figures.txt) |
| Diagnostic script self-test (CPU pipeline with a fake model) | The research environment in the GPU environment section below |

A separate venv is enough to run the first six checks in one go:

```powershell
py -3.10 -m venv ..\.venv-check
& '..\.venv-check\Scripts\python.exe' -m pip install -r '.\research\requirements-figures.txt'
& '..\.venv-check\Scripts\python.exe' -B '.\research\run_cpu_checks.py'
```

**Where is what?**

| Topic | Location |
|---|---|
| Report (English) | [report/](report) |
| All report tables | [report_tables/tables_en.md](report_tables/tables_en.md) (English); [tables.md](report_tables/tables.md) has the same numbers with Turkish labels |
| Report figures | [figures_en/](figures_en) (English, `make_selfmm_figures.py --lang en`); [figures/](figures) has Turkish labels |
| Pre-registration: hypotheses and decision criteria | [selfmm_diagnostics_protocol_v2.md](selfmm_diagnostics_protocol_v2.md) (Turkish original), [English translation](selfmm_diagnostics_protocol_v2_en.md) |
| Results and commands of the diagnostic experiments | [DIAGNOSTICS_V2.md](DIAGNOSTICS_V2.md) |
| v1 result package (A/B/C/D, K0–K7, cost) | [results_package_v1/](results_package_v1) |
| v2 packages, compressed test predictions, bootstrap | [diagnostics_v2/](diagnostics_v2) |
| Checkpoint re-runs that need a GPU | [PORTABLE_MODEL.md](PORTABLE_MODEL.md), [PORTABLE_K0_K7.md](PORTABLE_K0_K7.md), [PORTABLE_SEED_TEST.md](PORTABLE_SEED_TEST.md), [CLEAN_ENVIRONMENT.md](CLEAN_ENVIRONMENT.md) (Turkish records, summarized below) |
| Setup, versions, data access | Further down in this file |

**Results that need extra resources.** Training and model evaluation need a CUDA
environment, the MOSI feature file downloaded separately under its licence terms,
and ten checkpoints (five seeds × baseline/dropout). The checkpoints are not in the
repository because of their size; these results are verified against the records in
the repository, and reproducing them from scratch requires repeating the training.
A full re-run from training on another computer has not been verified.

## Language note

The course language is English, and all submission documents are in English. Some
files were written in Turkish during the work and are kept unchanged, either because
their SHA-256 is recorded elsewhere (changing them would break the evidence chain) or
because they are dated historical records:

| File | Content in one line |
|---|---|
| [selfmm_diagnostics_protocol_v2.md](selfmm_diagnostics_protocol_v2.md) | Pre-registered diagnostic protocol (hash recorded in the run manifests); see the [English translation](selfmm_diagnostics_protocol_v2_en.md) |
| [selfmm_corruption_protocol_v1.md](selfmm_corruption_protocol_v1.md) | The fixed corruption protocol of v1 (conditions K0–K7, seeds, masks, noise scale), summarized in the table below |
| [results_package_v1/DATA_DICTIONARY.md](results_package_v1/DATA_DICTIONARY.md) | Column definitions of the v1 result package (hash-checked part of the package) |
| [PORTABLE_MODEL.md](PORTABLE_MODEL.md) | How to re-run seed 1111 K0 A/B/C/D inference from explicit file paths; 16/16 metrics matched |
| [PORTABLE_K0_K7.md](PORTABLE_K0_K7.md) | The same for seed 1111 on all K0–K7 conditions; 128/128 metrics, 16/16 inputs matched |
| [PORTABLE_SEED_TEST.md](PORTABLE_SEED_TEST.md) | The same for seeds 1112–1115 with one shared entry point; all passed |
| [CLEAN_ENVIRONMENT.md](CLEAN_ENVIRONMENT.md) | Seed 1111 K0–K7 GPU re-run in a fresh venv from a separate source copy; byte-identical outputs |
| [report_tables/tables.md](report_tables/tables.md), [figures/](figures) | Turkish-labelled tables and figures; the English versions have identical numbers |

The older scripts print Turkish ASCII status tokens. The ones you will see:
`PAKET_KONTROLU_BASARILI` (package check passed), `OZET_DOGRULANDI` (summary verified),
`TABLOLAR_DOGRULANDI` (tables verified), `BOOTSTRAP_DOGRULANDI` (bootstrap verified),
`FIGURLER_YAZILDI` (figures written), `TANI_OZTESTLERI_BASARILI` (diagnostic self-tests
passed), `OZETLER_YENIDEN_HESAPLANDI` (summaries recomputed). Error messages start with
`HATA` (error).

## Verification history

8 November 2026, fresh GitHub clone: the repository was cloned from the
`yzm5110-selfmm` branch (commit `81c543d`) into an empty folder separate from the
development folders; Git converted line endings as on Windows (`core.autocrlf=true`).
Three environments were tried:

| Environment | Setup | `run_cpu_checks.py` |
|---|---|---|
| Python 3.10 venv without packages | None | 4 passed, 3 skipped (no numpy, matplotlib or research environment) |
| Small check environment | The `.venv-check` commands above | 6 passed, 1 skipped (diagnostic self-test) |
| Full research environment | The “GPU environment” commands below, as written (torch 2.3.1+cu121, `pip check` clean) | 6 passed, 1 skipped (figures); with `--python ..\.venv-check` the figures passed too |

No check failed; each of the seven checks passed in at least one environment. In the
same clone, with local data and checkpoints, `diagnose_selfmm_weighting.py --check`
(seed 1113) and `evaluate_selfmm_portable_seed_test.py --check` (seed 1114) also
passed: the source code in the clone matches the frozen SHA256 records despite the
CRLF conversion. No tracked file changed in the clone’s working tree. This test does
not cover running models on a GPU or repeating training. At that commit the runner
printed Turkish labels and ended with `KONTROLLER_TAMAM`; it now prints English labels
and ends with `ALL_CHECKS_PASSED`, with the same seven checks.

19 October 2026: on the same computer, with a new isolated venv and a separate source
copy, the **seed 1111 K0–K7 real GPU re-run was completed**. 128/128 metric, 16/16
input and 2/2 train-statistic checks passed. A byte-level match with the eight output
files of the earlier portable seed 1111 run was verified; the prediction/label/weight
difference over 10976 rows is zero.
[Setup, result and scope](CLEAN_ENVIRONMENT.md),
[short verification record](verification/clean_environment_seed1111_k0_k7_v1.json).
Re-running the other four seeds in a new venv, another computer, a public Git checkout
and a re-run from training are outside the scope of this verification.

## Quick check

Run the commands in the repository root folder, i.e. one level above the `research`
folder. On a computer with Python 3.10 installed:

```powershell
py -3.10 -I -S -B '.\research\verify_results_package.py' --check
```

The same with the existing project venv:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\verify_results_package.py' --check
```

`-I -S` runs without loading user/site packages; `-B` prevents writing bytecode
files. The check uses only the Python standard library. `PAKET_KONTROLU_BASARILI`
should appear at the end. `--check` writes no files.

The check verifies the SHA-256/size/row count of the 19 package files; the 160
seed/method/condition rows of each of the two splits; the means, sample SDs and paired
differences; and the 4,128 batch and 24 round records of the resource measurement. It
does not compute MAE/F1 from sample predictions and does not re-verify old checkpoint
or data hashes locally. The recorded correctness flags are historical evidence.

## Recomputing the summaries

After the check passes, the command below writes the same computations to a new folder
as 11 CSV files. The original package does not change. An existing target is not
overwritten.

```powershell
if (-not (Test-Path -LiteralPath '..\runs')) {
    New-Item -ItemType Directory -Path '..\runs' | Out-Null
}

& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\verify_results_package.py' `
    --output-dir '..\runs\selfmm_public_summary_recheck_v1'
```

A reader without the venv can use `py -3.10` as the launcher in the same command.
Wait for the `OZETLER_YENIDEN_HESAPLANDI` message. If the target exists, choose a new
folder name instead of deleting the earlier result. The recomputed numbers match
within the comparison tolerance; because of CSV row order or number formatting, the
file bytes need not be identical to the original CSV.

## Package contents and reading the results

```text
research/
  .gitattributes
  README.md
  verify_results_package.py
  requirements-research.txt
  environment-observed.txt
  configs/selfmm_mosi_batch4.example.json
  results_package_v1/
    manifest.json
    protocol.json
    provenance.json
    DATA_DICTIONARY.md
    validation/            # Five-seed validation, 5 CSV
    evaluation_test/       # Five-seed test, 5 CSV
    inference/             # Seed 1111 K0 resource measurement, 6 CSV
```

The name `evaluation_test` is used so that the `test*` ignore rule in the repository
root does not hide the results. Detailed column descriptions are in the
[data dictionary](results_package_v1/DATA_DICTIONARY.md) (Turkish).
`.gitattributes` turns off Git’s automatic LF/CRLF conversion for the hash-checked
package and check script, so that a checkout on another computer does not change the
file digests merely by converting line endings; this file is committed as well.

Example view:

```powershell
Import-Csv '.\research\results_package_v1\evaluation_test\summary_metrics.csv' |
    Where-Object { $_.condition -eq 'K0' } |
    Format-Table method, MAE_mean, MAE_sd, Non0_acc_2_mean -AutoSize
```

| K0 method | Validation MAE ± SD | Test MAE ± SD |
|---|---:|---:|
| A | 0.7168 ± 0.0202 | 0.7240 ± 0.0146 |
| B | 0.7049 ± 0.0117 | 0.7574 ± 0.0244 |
| C | 0.8665 ± 0.0513 | 0.8424 ± 0.0412 |
| D | 0.9293 ± 0.0667 | 0.9163 ± 0.0747 |

Source: the [validation](results_package_v1/validation/summary_metrics.csv) and
[test](results_package_v1/evaluation_test/summary_metrics.csv) summaries.
Lower MAE is better. Accuracy/F1 are on a 0–1 scale in the CSV.
`Non0_acc_pp_delta` is already in percentage points. The SD is computed over the five
training seeds with ddof=1. A paired difference is first taken within the same seed.
The SD is not a confidence interval; eight related conditions do not count as eight
independent repetitions.

In the resource measurement, the A/B/C/D times are 4.9833 / 5.0731 / 10.1926 /
10.3496 ms per sample. These are costs amortised over batches of 4, not batch-1
latency. C/A=2.0454 and D/B=2.0402 are means of the ratios within six rounds. Loading,
transfer, warm-up and check passes are not included in the time. Memory is the PyTorch
allocated/reserved measure, not total system GPU memory. The resource SD belongs to
the timing repetitions.
[Resource summary](results_package_v1/inference/summary.csv),
[paired cost](results_package_v1/inference/paired_summary.csv).

## Methods and conditions

| Method | Training | Inference |
|---|---|---|
| A | Standard Self-MM | Original fusion, one forward pass |
| B | Independent 20% audio / 20% video input dropout per sample, in training only | Original fusion, one forward pass |
| C | The same checkpoint as A | Two forward passes with the agreement weight |
| D | The same checkpoint as B | Two forward passes with the agreement weight |

Text and lengths are kept. Dropout is off during evaluation. For the three auxiliary
predictions, d_m = abs(y_m − mean(y_other)); w = 3 × softmax(−d/τ), τ=1. The weights
are applied to the representation blocks before the first fusion layer. They sum to 3;
they are not probabilities or calibrated confidence values. A missing input does not
automatically get zero weight. No newly learned parameters are added.

| Condition | Audio/video operation |
|---|---|
| K0 | Clean inputs |
| K1 | Whole audio input zero |
| K2 | Whole video input zero |
| K3 | Both zero |
| K4/K5/K6 | Independent 30%/50%/70% masking of valid time steps; masks nested across rates |
| K7 | Gaussian noise scaled to 0.5 × the feature std found on valid train steps |

Padding is kept in K4–K7. The audio and video RNGs are independent; the inputs of the
same split/condition are shared by all models. The training seeds are 1111–1115 and
the corruption seed is 1111. The validation and test RNG spaces are separate.
Train/valid/test are 1284/229/686 samples; Non0 valid/test are 216/656 samples.
Protocol v2 adds text corruption with `[UNK]`: K8 (30%), K9 (60%), K10 (all tokens).

## Diagnostic experiments: why did the weighting not work? (protocol v2)

The results above show that the τ=1 rule is harmful, not why.
[Protocol v2](selfmm_diagnostics_protocol_v2.md) ([English translation](selfmm_diagnostics_protocol_v2_en.md))
separates the following questions on the same 10 checkpoints, with inference only:
a τ sweep (amount of rescaling), a variant with the text weight fixed (the dominant
text block), weights shuffled across samples (signal content), an oracle that gives
zero weight to the known missing modality (room to improve), and text corruption
K8–K10 with `[UNK]` (when the dominant modality is corrupted). The hypotheses and
decision criteria were written and committed before the results; τ is selected on
validation only, and the test runs once. [How to run it](DIAGNOSTICS_V2.md).

Validation result (five seeds): all six decisions came out as written beforehand. The
harm comes from rescaling the dominant text block (it disappears when the text weight
is held fixed); the agreement signal is no better than randomly shuffled weights; even
the oracle gives no improvement under audio/video corruption; and the rule does not
help when text is corrupted either. The selected τ\* = 16 (closest to unweighted). The
test ran once, after the validation package had been committed; all six decisions
replicated on test.
[Validation package](diagnostics_v2/validation), [test package](diagnostics_v2/evaluation_test),
[detailed tables](DIAGNOSTICS_V2.md#validation-result-25-october-2026-five-seeds-baseline-checkpoint).

## Uncertainty, report tables and figures

All tables and figures in the report are produced by scripts from files in the
repository; no GPU, dataset or checkpoint is needed. The commands run in the repository
root.

| Output | Script | Input |
|---|---|---|
| [Compressed test predictions](diagnostics_v2/predictions) (2.97 MB, lossless float32) | `export_selfmm_predictions.py` | The five test runs in `runs/` (run once) |
| [Bootstrap intervals](diagnostics_v2/bootstrap/bootstrap_test.csv) | `bootstrap_selfmm_diagnostics.py` | Compressed test predictions |
| [Report tables](report_tables/tables_en.md) | `make_selfmm_report_tables.py --lang en` (Turkish: without `--lang`) | Result packages, bootstrap, [training records](report_tables/training_resources.csv), [published scores](configs/published_reference.json) |
| [Figures](figures_en) | `make_selfmm_figures.py --lang en` (Turkish: without `--lang`) | Result packages, bootstrap, compressed predictions |

The bootstrap resamples the 686 test samples 2000 times; the same sample set is
applied to all seeds, methods and conditions (paired). The statistic is the five-seed
mean difference; the interval is the 95% percentile interval. The seed SD shows
training variability and the bootstrap interval test sampling; both are reported.

Re-verification (writes no files):

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\bootstrap_selfmm_diagnostics.py' --verify '.\research\diagnostics_v2\bootstrap'
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\make_selfmm_report_tables.py' --verify
```

Expect `BOOTSTRAP_DOGRULANDI` and `TABLOLAR_DOGRULANDI`. The bootstrap verification is
byte-level and depends on numpy 1.26.4. The table verification checks both
`tables.md` and `tables_en.md`.

The figures are drawn in a separate venv so as not to change the verified GPU
environment:

```powershell
py -3.10 -m venv ..\.venv-figures
& '..\.venv-figures\Scripts\python.exe' -m pip install -r '.\research\requirements-figures.txt'
& '..\.venv-figures\Scripts\python.exe' -B '.\research\make_selfmm_figures.py' --lang en --replace
& '..\.venv-figures\Scripts\python.exe' -B '.\research\make_selfmm_figures.py' --replace
```

PNG bytes can vary with the font build; the input digests are in
`figures_en/figures_manifest.json` and `figures/figures_manifest.json`.
`--transparent` draws the figures without a background (used for the slides).

## Preparing the GPU environment on a new computer

This section is for readers who plan to run real training/evaluation. The package
check above does not need this setup. Apply it in a separate, empty project folder
without changing a working project venv.

```powershell
# In an empty project folder; skip the clone step if you use an existing MMSA folder.
# The submitted version is the v1.0-submission tag; use --branch yzm5110-selfmm for the working branch.
git clone --branch v1.0-submission https://github.com/ilaydagencer/yzm5110-robust-selfmm.git MMSA
py -3.10 -m venv .venv
Set-Location '.\MMSA'

& '..\.venv\Scripts\python.exe' -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
& '..\.venv\Scripts\python.exe' -m pip install -r '.\research\requirements-research.txt'
& '..\.venv\Scripts\python.exe' -m pip install --no-deps -e .
& '..\.venv\Scripts\python.exe' -m pip check
& '..\.venv\Scripts\python.exe' -B -c "import sentencepiece; import torch; import MMSA; print(torch.__version__); print(torch.cuda.is_available()); print(MMSA.__file__)"
```

The PyTorch 2.3.1 CUDA 12.1 wheel command follows the
[official previous-versions guide](https://pytorch.org/get-started/previous-versions/#v231).
CUDA visibility depends on a suitable NVIDIA driver. torchvision and torchaudio are not
used in this pre-extracted-feature experiment and are not installed. `MMSA.__file__`
should point to the `src/MMSA` folder of this clone.
For Windows compatibility, **sentencepiece is imported before torch/MMSA**.

`requirements-research.txt` pins the package versions observed in the working Windows
Python 3.10.5 environment; torch and the local MMSA are installed in the separate steps
above. Full environment information is in
[environment-observed.txt](environment-observed.txt). This list is not a hash-locked
lockfile. Installation into a new venv on the same computer, package isolation and a
seed 1111 GPU re-run were verified. The installation commands, versions and limits
used are in the [clean environment record](CLEAN_ENVIRONMENT.md). Installation on
another computer has not been verified yet.

## Data and configuration

Folder layout:

```text
project_root/
  .venv/
  MMSA/                     # This repository
  data/MOSI/Processed/unaligned_50.pkl
  local_configs/            # Machine-specific; not added to Git
  runs/                     # Experiments; not added to Git as a whole
```

The pre-extracted MOSI features are accessed through the official links in the
[MMSA data access section](https://github.com/thuiar/MMSA#2-datasets). The data
provider’s access/usage terms apply. This repository does not distribute the raw
videos or the `.pkl` file. This work uses `unaligned_50.pkl` with BERT input; scores
are not expected to match with another feature file of a similar name. After
downloading:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '..\data\MOSI\Processed\unaligned_50.pkl'
```

Expected SHA-256:
`78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`.
The first model setup may download the `bert-base-uncased` files. The resource
benchmark runs offline and needs a prepared BERT cache.

The [config example](configs/selfmm_mosi_batch4.example.json) is a copy of the config
used, with only the data root set to `../data`; running from the repository root is
assumed. Batch=4, BERT fine-tuning=True, early_stop=8 and update_epochs=4 are kept.
This example is not copied over old experiment records. The existing training scripts
do not take a `--config` option; they generate the local config from the source config
and compare it with the previous config. The example JSON is for inspecting the
experiment settings without personal paths; it is not claimed to be a fully portable
GPU launcher.

## The actual experiment chain and the existing scripts

These scripts build an experiment archive that depends on the outputs of earlier steps.
Checkpoints, `local_configs`, raw prediction CSVs and the original manifests are not in
the small package. The manifests contain machine-specific absolute paths and SHA-256
guards. `--repo-dir` alone does not make moved old records compatible; switching off
these guards is not a reproduction method.

| Step | Script | Prerequisite / evidence produced |
|---|---|---|
| 1 | run_selfmm_smoke.py | Data and source config → seed 1111 baseline/config |
| 2 | run_selfmm_dropout.py | Baseline config/checkpoint → seed 1111 dropout |
| 3 | evaluate_selfmm_checkpoints.py | Two checkpoints → old A/B K0–K3 test reference |
| 4 | evaluate_selfmm_abcd_valid.py | Two checkpoints → A/B/C/D K0–K3 validation |
| 5 | check_selfmm_corruptions.py | Data → train std and validation corruption protocol |
| 6 | evaluate_selfmm_abcd_corrupt_valid.py | 4 and 5 → seed 1111 K0–K7 validation |
| 7 | run_selfmm_seed.py and evaluate_selfmm_abcd_seed_valid.py | Seed 1111 references → 1112–1115 training/validation |
| 8 | analyze_selfmm_five_seeds.py | Five complete validation records → validation summaries |
| 9 | evaluate_selfmm_abcd_test.py | Seed 1111 validation/protocol/old test records → K0–K7 test |
| 10 | evaluate_selfmm_abcd_seed_test.py | Seed 1111 test, selected seed training/validation records → 1112–1115 test |
| 11 | analyze_selfmm_test_five_seeds.py | Five complete test records → test summaries |
| 12 | benchmark_selfmm_inference.py | Completed seed 1111 test records → K0 resource measurement |

**Special behaviour of the old baseline script:** `run_selfmm_smoke.py --check` writes
the local config file; it is not the same as the other, newer `--check` flows. This old
script also lacks the guard in the newer scripts that prevents overwriting a
checkpoint/output. It is not run again in the existing seed 1111 archive. A re-run from
scratch needs a separate, empty project location and the absence of these two targets:
`../local_configs/selfmm_mosi_batch4.json`, `../runs/selfmm_seed1111`.

The seed 1112–1115 training script stops on an existing target, including with
`--check`. This is the expected behaviour that prevents damaging completed work. This
README does not give a single command that restarts all experiments.

If the full local archive exists, the following checks can be run with new, empty
output names:

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\analyze_selfmm_five_seeds.py' `
    --output-dir '..\runs\selfmm_valid_recheck_v1' --check

& '..\.venv\Scripts\python.exe' -B '.\research\analyze_selfmm_test_five_seeds.py' `
    --output-dir '..\runs\selfmm_test_recheck_v1' --check

& '..\.venv\Scripts\python.exe' -B '.\research\benchmark_selfmm_inference.py' `
    --seed 1111 --output-dir '..\runs\selfmm_inference_repeat_v1' --check
```

Without `--check`, the first two commands write new summary files; no model is run.
Without `--check`, the last command starts a real GPU inference measurement. The small
package does not replace this local archive. A reader without the full archive can use
the quick package check. Verifying that the first training runs and the archive chain
can be rebuilt from scratch on a new machine remains open.

`eval_robust.py`, `evaluate_consistency.py`, `evaluate_mask_only.py`,
`probe_consistency_fusion.py`, `inspect_*` and `run_lmf_smoke.py` are early exploration
files. They are not entry points of the final A/B/C/D result chain in this package;
LMF/MulT are not presented here as retrained.

## Limits of interpretation and submission status

MMSA ran an automatic test at the end of training, and there are earlier test
inspections too. The test set cannot be presented as a never-seen independent holdout.
No new τ, dropout rate or good-looking seed was chosen on the basis of the final
fixed-protocol comparisons. Text is kept in K0–K7; there is one dataset and one
corruption draw. Low sensitivity to corruption is not, on its own, strong evidence of
multimodal robustness.

In the source code the last incomplete gradient-accumulation group is not processed,
and the validation selection Loss may differ from the equally weighted MAE. These are
inherited limitations explained in the report; the old sources were not changed after
the results.

The submitted version is pinned by the `v1.0-submission` tag. `provenance.json`
records the HEAD before the v1 packaging; the commit order of protocol v2 is in
[DIAGNOSTICS_V2.md](DIAGNOSTICS_V2.md) and Section 6 of the report. All checks that do
not need a GPU run with `run_cpu_checks.py`. A re-run from training on another computer
has not been verified.

Some of the old exploration and training scripts (e.g. `evaluate_selfmm_abcd_valid.py`,
`run_selfmm_dropout.py`) keep paths of the development computer as default values;
they are overridden with `--repo-dir` / `--project-dir`. Because the SHA256 digests of
these files are kept in frozen records and training records, they were not changed, so
that the evidence chain stays intact.

AI tools were used for code drafts, checks, table generation and documentation:
ChatGPT/Codex in the v1 stage, and from 25 October 2026 Claude (Claude Code) for the
protocol v2, diagnostic, summary, bootstrap, table and figure scripts, the
documentation, the report draft and the English translation of the report, slides and
documentation. The tools and their contributions are listed in Appendix B of the report.
The student ran all GPU commands and carried out all Git operations; explaining the
methods and sources is the student’s responsibility.

## Real model re-run with explicit file paths: K0

A new entry point was added for clean K0 A/B/C/D inference for seed 1111:
[`evaluate_selfmm_portable_k0.py`](evaluate_selfmm_portable_k0.py).
The [step-by-step guide](PORTABLE_MODEL.md) (Turkish) explains the required config and
dependency manifest, the data/checkpoint/local BERT paths, the offline `--check` and
the real run. This entry point does not need the old `runs` manifests; it compares the
metrics computed from new predictions with the 16 K0 metrics in the public result
package. The dependencies of the old archive scripts above are unchanged.

A non-writing `--check` was verified in the existing environment and in a separate
source copy; 10 CPU contract tests passed. On 19 October 2026 the user completed the
seed 1111 K0 re-run in the existing environment with an RTX 3050 Ti GPU. All 16
reference metric checks of the four methods passed; the largest absolute difference
was 5.55e-16. The metrics were also recomputed from the 1372 saved prediction rows; the
SHA256 digests of 4 output and 103 input files were verified.
[Verification record](verification/portable_k0_seed1111_v1.json).
This entry point is limited to K0; the seed 1111 K1–K7 re-run was verified with the
separate entry point below. The portable re-runs of seeds 1112–1115 in the existing
environment were also completed with the shared entry point below. Later, a seed 1111
K0–K7 re-run with a new venv and a separate source copy was verified; this does not
widen the scope of the K0 helper. A re-run from training has not been verified yet.

## K0–K7 re-run with explicit file paths

[`evaluate_selfmm_portable_k0_k7.py`](evaluate_selfmm_portable_k0_k7.py) was prepared to
run the seed 1111 checkpoints on all K0–K7 test conditions. The
[step-by-step guide](PORTABLE_K0_K7.md) (Turkish) explains the recomputation of the
train-only noise statistics and the checks of 128 metrics and 16 input digests. The K0
helper and the old experiments are not changed; old run manifests are not read.

During preparation 10 CPU check tests passed. In a separate real-data CPU audit, the
two train statistics and the input digests of the eight conditions matched the
reference. A non-writing --check was verified in a separate source copy in which model
and data loading were blocked.

On 19 October 2026 the user completed the real K0–K7 re-run in the existing RTX 3050 Ti
environment: 32 metric rows, 128/128 reference metric checks, 16/16 corrupted-input
digests and 2/2 train-statistic checks were verified. The largest reference metric
difference was 5.55e-16 and the train-statistic difference zero. The metrics were also
recomputed from 10976 prediction rows; the SHA256 digests of 8 output and 106 recorded
input files matched at audit time.
[Verification record](verification/portable_k0_k7_seed1111_v1.json).

This re-run shows that the earlier results can be produced in the existing environment
without depending on the old run manifests. It is not new training or an independent
experiment. The K0–K7 re-runs of seeds 1112–1115 in the existing environment were
completed with the entry point below. The later separate source copy and new venv
audit was completed for seed 1111 with a real GPU result;
[evidence and scope](CLEAN_ENVIRONMENT.md).


## Portable test re-run by training seed: 1112–1115

[`evaluate_selfmm_portable_seed_test.py`](evaluate_selfmm_portable_seed_test.py)
evaluates the recorded baseline/dropout checkpoints on the K0–K7 test conditions with
`--seed 1112`, `--seed 1113`, `--seed 1114` or `--seed 1115`.
[Step-by-step guide](PORTABLE_SEED_TEST.md) (Turkish). Seed 1111 uses its own entry
points; the other four seeds use this shared script.

`configs/selfmm_portable_seed_dependencies.json` stores the checkpoint identities
cross-checked against the training/test records and the shared dependencies. The real
run does not read old run manifests or personal config files. The corruption seed 1111,
batch 4 and τ=1 are fixed. The train-only std is recomputed. 128 metrics are compared
with the public package rows of the selected training seed, and 16 corrupted-input
digests with the shared seed 1111 reference.

On 19 October 2026 the user completed the seed 1112 K0–K7 re-run in the existing
RTX 3050 Ti environment: 128/128 metric, 16/16 input and 2/2 train-statistic checks
passed. The largest reference metric difference was 8.88e-16 and the train-statistic
difference zero. The metrics were also recomputed from 10976 prediction rows; the
SHA256 digests of 8 outputs and 106 recorded inputs matched in the audit before the
update.
[Seed 1112 verification record](verification/portable_k0_k7_seed1112_v1.json).

On 19 October 2026 the seed 1113 K0–K7 re-run was also completed in the existing
environment. 128/128 metric, 16/16 input and 2/2 train-statistic checks passed. The
largest reference metric difference is below 1.23e-15; the MAE and accuracy differences
are zero. The metrics were also recomputed from 10976 prediction rows. The digests of
8 outputs and 106 recorded inputs matched in the audit before this update.
[Seed 1113 verification record](verification/portable_k0_k7_seed1113_v1.json).

On 19 October 2026 the seed 1114 K0–K7 re-run was also completed in the existing
environment. 128/128 metric, 16/16 input and 2/2 train-statistic checks passed. The
largest reference metric difference is below 1e-15; the MAE and accuracy differences
are zero. The metrics were also recomputed from 10976 prediction rows. The digests of
8 outputs and 106 recorded inputs matched in the audit before this update.
[Seed 1114 verification record](verification/portable_k0_k7_seed1114_v1.json).

## Seed 1115 and the completed five-seed portable re-runs — 19 October 2026

For seed 1115, 128/128 metric, 16/16 corrupted-input and 2/2 train-statistic checks
passed. The largest reference difference is 1.11023e-15 (Corr); the MAE and accuracy
differences are zero. The difference is zero in the train statistics and in the
uniform-weight identity check. An independent metric computation from 10976 prediction
rows was verified; the largest difference from the float32 MAE is below 1.05e-7
(tolerance 1e-6). The file digests of 8 outputs and 106 inputs matched at audit time.
[Seed 1115 record](verification/portable_k0_k7_seed1115_v1.json).

| Seed 1115, K0 | Non0 accuracy (%) | MAE | Corr |
|---|---:|---:|---:|
| A | 84.45 | 0.7149 | 0.7843 |
| B | 83.08 | 0.7463 | 0.7692 |
| C | 84.45 | 0.7965 | 0.7681 |
| D | 82.93 | 0.8539 | 0.7512 |

The K0 MAE differences are about B−A +0.0314, C−A +0.0816 and D−B +0.1076. In all of
K0–K7, the MAE of B is higher than A, C higher than A and D higher than B. C keeps the
K0 accuracy while increasing the sentiment intensity error. This is a reproduction of
the earlier test finding; no new observation is added to the five-seed statistics.
200.42 seconds is the total re-run time; it is not an inference benchmark.

All of seeds 1111–1115 were re-run in the existing environment. In total, 640 metric,
80 input and 10 train-statistic checks succeeded. The manifest identities of the five
runs and 40 output files in total were re-audited. The largest reference metric
difference is below 1.23e-15. These totals are not a number of independent experiments.
[Five-seed verification summary](verification/portable_five_seeds_v1.json).

This step does not change the evaluation scripts, the recorded checkpoint identities or
the fixed protocol. The next clean environment stage was completed for seed 1111:
[results and setup guide](CLEAN_ENVIRONMENT.md) (Turkish).
The old verification JSONs are kept with their historical scope; the new evidence is in
a separate record.


Two standard-library tools were added to prepare the clean environment:
`prepare_selfmm_clean_copy.py` copies selected small sources into a separate folder;
`check_selfmm_clean_environment.py` audits the new venv’s isolation, versions,
package/source paths, the `pip check` result and CUDA visibility. The user completed
the package installation of the new environment and the seed 1111 real GPU re-run. The
in-environment paths of 51 packages and the file digests of 125 copied sources and four
installation logs were audited. The --check result of these tools and the real GPU
result are separate pieces of evidence; both can be traced together in the new
verification record.
