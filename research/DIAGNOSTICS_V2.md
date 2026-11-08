# Weighting diagnostics (protocol v2): run instructions

What and why: [protocol v2](selfmm_diagnostics_protocol_v2.md) (English translation: [research/selfmm_diagnostics_protocol_v2_en.md](selfmm_diagnostics_protocol_v2_en.md)).
Script: [`diagnose_selfmm_weighting.py`](diagnose_selfmm_weighting.py).

This step does no new training. It runs the existing 10 checkpoints (5 seeds × baseline/dropout)
first on **validation**, under conditions K0–K10 and with the weight variants in the
protocol. Test is run only once, after the validation summary has been committed.

## Order

1. The protocol and the script are committed (**before** the GPU run).
2. Validation runs (this document, sections 2–3).
3. The five-seed validation summary, the hypothesis decisions and τ\* are committed (separate step).
4. The test runs are done once with `--selection` (separate step).

## 1. Self-test without a GPU

In the repository root (the `MMSA` folder):

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\diagnose_selfmm_weighting.py' --self-test
```

The expected last lines are `TANI_OZTESTLERI_BASARILI` and "Ozdenetim dosya yazmadi" (the self-test wrote no files).
This test runs the whole pipeline on the CPU with a dummy model; it does not open real
data, checkpoints or BERT.

## 2. --check for the five seeds

`--check` reads the data and checkpoints as bytes and verifies their SHA256 hashes, the source
code, the config and the result package. No model or dataset is built, and no
file is written.

Open PowerShell in the repository root (the `MMSA` folder); the `..\` paths in the commands
are relative to this folder.

```powershell
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
function Invoke-Diagnostics([int]$Seed, [string]$Split, [string[]]$Extra) {
    $baseline = if ($Seed -eq 1111) { '..\runs\selfmm_seed1111' } else { "..\runs\selfmm_baseline_seed$Seed" }
    & '..\.venv\Scripts\python.exe' -B '.\research\diagnose_selfmm_weighting.py' `
        --seed $Seed --split $Split `
        --data-file '..\data\MOSI\Processed\unaligned_50.pkl' `
        --baseline-checkpoint "$baseline\saved_models\self_mm-mosi.pth" `
        --dropout-checkpoint "..\runs\selfmm_dropout_a20_v20_seed$Seed\saved_models\self_mm-mosi.pth" `
        --bert-dir $bertDir `
        --output-dir "..\runs\selfmm_diagnostics_v2_${Split}_seed$Seed" @Extra
}
foreach ($seed in 1111, 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'valid' @('--check')
    if ($LASTEXITCODE -ne 0) { break }
}
```

`KONTROL_BASARILI: seed…/valid` is expected for each seed.

## 3. Validation runs (GPU)

First run only seed 1111 and look at the run time and the reference check:

```powershell
Invoke-Diagnostics 1111 'valid' @()
```

For each condition a line `baseline/K… : plain MAE=…; tau1 MAE=…; … sn` is printed (sn: seconds).
At the end, `TANI_V2_TAMAMLANDI: seed1111/valid; 128/128 referans kontrolu gecti`
is expected. This line shows that the new script reproduces the v1 validation A/B/C/D results
on K0–K7 within a tolerance of 1e-5; only after that can the new numbers
be trusted.

Then the remaining four seeds:

```powershell
foreach ($seed in 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'valid' @()
    if ($LASTEXITCODE -ne 0) { break }
}
```

If a run fails, look at the `status` and `error` fields in `manifest.json` in the
output folder. Do not change the tolerance or the protocol; share the error.
An existing output folder is not overwritten; to retry, delete the folder
or give it a new name, and note this.

## Outputs (`..\runs\selfmm_diagnostics_v2_valid_seed<seed>\`)

| File | Contents |
|---|---|
| `metrics.csv` | 294 rows: 2 checkpoints × 11 conditions × variants (+ unimodal heads) |
| `reference_checks.csv` | 128 rows: K0–K7 plain/τ=1 compared with package A/B/C/D |
| `predictions.csv` | 5038 rows: per-example predictions of all variants, unimodal predictions, d and weights |
| `weight_summary.csv` | Weight mean/SD/min/max per variant |
| `input_summary.csv` | Masking rates and input hash per condition |
| `controls.csv`, `manifest.json`, `train_*` | Checks, versions, input hashes |

The `runs` folder is not added to Git. A small summary of the five seeds enters the repository in a separate step.

## 4. Five-seed validation summary and choice of τ (no GPU)

[`summarize_selfmm_diagnostics.py`](summarize_selfmm_diagnostics.py) uses only the Python
standard library. It checks that the five runs finished, ran with the same protocol/script
version, received the same corrupted inputs and that their output hashes have not
changed; it then applies the rules in sections 5 and 6 of the protocol.

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --runs-dir '..\runs' --split valid --output-dir '.\research\diagnostics_v2\validation'
```

Package ([diagnostics_v2/validation](diagnostics_v2/validation)): per-seed
metrics, five-seed mean/SD, paired variant − plain differences, hypothesis
decisions (`hypotheses.csv`), summaries of the weights and of the spread of the unimodal heads, and
`selection.json` for the test. Per-example predictions stay in `runs`.

To re-verify the package without a GPU, data or checkpoints:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --verify '.\research\diagnostics_v2\validation'
```

`OZET_DOGRULANDI` is expected. This command recomputes all summary tables, the hypothesis decisions
and τ\* from the per-seed metrics and checks the file hashes.

### Validation result (25 October 2026, five seeds, baseline checkpoint)

| Code | Decision | Key number |
|---|---|---|
| H1 | Supported | K0–K7 mean MAE 1.230 at τ=0.25 → 0.722 at τ=16; at K0, τ16 − plain = +0.003 |
| H2 | Supported | At K0, text-fixed variant − plain = +0.0005; τ=1 − plain = +0.150; 5/5 seeds |
| H3 | Not supported | At K0, τ=1 − shuffled = +0.040; τ=1 is not better than shuffled in any seed |
| H4 | Supported | Oracle − plain: K1 +0.0001, K2 −0.0003, K3 +0.0002 |
| H5a | Supported | plain MAE K0 0.717 → K10 1.464 (+0.748); Non0 accuracy 84.9% → 45.6% |
| H5b | Not supported | τ=1 − plain: K9 +0.036 (0/5 seeds), K10 +0.001 (3/5 seeds) |

All six decisions match the expectations written in advance. The selected τ\* = 16, the
largest value in the grid: on validation, the best weighting is the one closest to
unweighted fusion. The decisions are descriptive; there are no p-values.

## 5. Test runs (once, GPU)

Only after the validation package and `selection.json` have been committed and pushed.
With the function from section 2 defined:

```powershell
foreach ($seed in 1111, 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'test' @('--selection', '.\research\diagnostics_v2\validation\selection.json')
    if ($LASTEXITCODE -ne 0) { break }
}
```

Because the test split has 686 examples, a run takes about 15–20 minutes per seed. On
K0–K7, each run makes 128 metric checks and eight input-hash checks against the v1 test A/B/C/D results.
Then the summary:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --runs-dir '..\runs' --split test `
    --selection '.\research\diagnostics_v2\validation\selection.json' `
    --output-dir '.\research\diagnostics_v2\evaluation_test'
```

The hypothesis rows on test are not new decisions; they describe whether the
validation decisions replicate on test.

### Test result (25 October 2026, five seeds, baseline checkpoint, τ\* = 16)

Each of the five test runs ran once, after the validation package had been committed, with the
committed `selection.json`. Every run passed 128/128 metric checks and 16/16
input-hash checks. [Test package](diagnostics_v2/evaluation_test).

| Code | Validation | Test | Key test number |
|---|---|---|---|
| H1 | Supported | Replicated | K0–K7 mean MAE 1.205 at τ=0.25 → 0.724 at τ=16 |
| H2 | Supported | Replicated | At K0, text fixed − plain = −0.0004; τ=1 − plain = +0.118 |
| H3 | Not supported | Replicated | At K0, τ=1 − shuffled = +0.040; 0/5 seeds |
| H4 | Supported | Replicated | Oracle − plain: K1 −0.0007, K2 −0.0002, K3 −0.0009 |
| H5a | Supported | Replicated | plain MAE K0 0.724 → K10 1.399; Non0 accuracy 83.9% → 55.1% |
| H5b | Not supported | Replicated | τ=1 − plain: K9 +0.029 (0/5), K10 −0.0004 (2/5) |

- On test, τ\* = 16 is the same as plain fusion (K0 difference +0.0002); the selected "best"
  weighting in practice means no weighting.
- Inconsistent detail: at K10, the oracle that gives text zero weight is better than plain on
  validation (1.430 vs. 1.464) and worse on test (1.422 vs. 1.399). There was no decision
  criterion for this comparison in advance; the direction is not consistent across splits.
- At τ=1, the mean text weight increases from K0 to K10 in all five seeds
  (on test 0.71–0.81 → 0.91–1.18). The more the text is corrupted, the more weight the rule gives to text.
