"""Summarize frozen Self-MM validation exports for training seeds 1111-1115.

Usage from MMSA: python -B research/analyze_selfmm_five_seeds.py --check
Then run without --check to save a new summary directory under ../runs.
Only Python's standard library is used. No MMSA/torch/sentencepiece import,
checkpoint loading, dataset loading, training, inference or test evaluation.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1
SEEDS = (1111, 1112, 1113, 1114, 1115)
CONDITIONS = tuple(f'K{i}' for i in range(8))
METHODS = ('A', 'B', 'C', 'D')
METHOD_PAIRS = {'baseline': ('A', 'C'), 'dropout': ('B', 'D')}
COMPARISONS = (('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'A'))
METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')
N, NON0_N = 229, 216
MAE_TOLERANCE = 1e-6
TIE_TOLERANCE = 1e-12
DEFAULT_OUTPUT = 'selfmm_abcd_valid_five_seeds_tau1_summary_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), f'Sonlu olmayan sayi: {key}')
    return value


class Sources:
    """Snapshot exactly the bytes parsed, then detect changes before saving."""
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def read(self, path):
        raw = path.read_bytes()
        self.hashes[path.relative_to(self.root).as_posix()] = hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8-sig')

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path, columns, count):
        reader = csv.DictReader(io.StringIO(self.read(path), newline=''))
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)) and set(columns) <= set(fields),
                f'CSV sutunlari eksik/tekrarli: {path.name}')
        rows = list(reader)
        require(len(rows) == count, f'Satir sayisi {len(rows)}, beklenen {count}: {path}')
        require(all(None not in r and all(v is not None for v in r.values()) for r in rows),
                f'Bozuk CSV satiri: {path}')
        return rows

    def verify(self):
        for relative, expected in self.hashes.items():
            require(sha256(self.root / relative) == expected, f'Kaynak analiz sirasinda degisti: {relative}')


def seed_fields(row, seed):
    # Seed 1111's older CSV schema has no seed columns; its manifest is checked.
    for key, expected in (('training_seed', seed), ('corruption_seed', 1111)):
        if seed != 1111 or key in row:
            require(int(row[key]) == expected, f'CSV seed uyusmazligi: {key}, seed {seed}')


def keyed(rows, fields, expected):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        require(key not in result, f'Tekrarlanan kayit: {key}')
        result[key] = row
    require(set(result) == expected, f'Eksik/fazla anahtarlar: {fields}')
    return result


def prediction_metrics(labels, predictions):
    pairs = [(y, p) for y, p in zip(labels, predictions) if y != 0]
    require(len(labels) == N and len(pairs) == NON0_N, '229/216 ornek sayisi uyusmuyor')
    truth = [y > 0 for y, _ in pairs]
    predicted = [p > 0 for _, p in pairs]
    accuracy = sum(a == b for a, b in zip(truth, predicted)) / len(pairs)
    weighted_f1 = 0.0
    for label in (False, True):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        denominator = 2 * tp + fp + fn
        weighted_f1 += (2 * tp / denominator if denominator else 0) * truth.count(label) / len(pairs)
    ym, pm = statistics.mean(labels), statistics.mean(predictions)
    yc, pc = [y - ym for y in labels], [p - pm for p in predictions]
    denominator = math.sqrt(math.fsum(y*y for y in yc) * math.fsum(p*p for p in pc))
    require(denominator > 0, 'Korelasyon tanimsiz: sabit etiket/tahmin')
    return dict(MAE=math.fsum(abs(p-y) for y, p in zip(labels, predictions)) / N,
                Non0_acc_2=accuracy, Non0_F1_score=weighted_f1,
                Corr=math.fsum(y*p for y, p in zip(yc, pc)) / denominator)


def load_runs(runs):
    sources = Sources(runs)
    records, reference_labels, reference_manifest = {}, None, None
    control_keys = {(cp, c) for cp in METHOD_PAIRS for c in CONDITIONS}
    metric_keys = {(m, c) for m in METHODS for c in CONDITIONS}
    for seed in SEEDS:
        folder = runs / f'selfmm_abcd_corrupt_valid_seed{seed}_tau1_v1'
        manifest = sources.json(folder / 'manifest.json')
        for key, expected in dict(status='completed', split='valid', n=N, batch_size=4,
                                  training_seed=seed, corruption_seed=1111, tau=1.0,
                                  protocol_version=1, conditions=list(CONDITIONS)).items():
            require(manifest[key] == expected, f'Manifest uyusmazligi: seed {seed}, {key}')
        for key in ('shared_corrupted_inputs', 'protected_files_unchanged', 'data_file_unchanged', 'inference_only'):
            require(manifest[key] is True, f'Manifest kontrolu basarisiz: seed {seed}, {key}')
        if seed == 1111:
            reference_manifest = manifest
            require(set(manifest['corrupted_input_sha256']) == set(CONDITIONS), 'Girdi ozetleri eksik')
        else:
            for key in ('corrupted_input_sha256', 'data_sha256', 'train_stats_sha256', 'protocol_manifest_sha256'):
                require(manifest[key] == reference_manifest[key], f'Seed 1111 ile protokol/girdi farki: {seed}, {key}')
            ref_key = 'selfmm_abcd_corrupt_valid_seed1111_tau1_v1/manifest.json'
            require(manifest['reference_seed1111_manifest_sha256'] == sources.hashes[ref_key],
                    f'Seed 1111 referans manifest ozeti farkli: {seed}')
            for key in ('same_corruptions_as_seed1111', 'reference_labels_matched'):
                require(manifest[key] is True, f'Referans kontrolu basarisiz: {seed}, {key}')
            require(manifest['test_evaluation'] is False and manifest['hyperparameter_tuning'] is False,
                    f'Validation/tuning protokolu farkli: {seed}')
            for key, count in (('metric_rows', 32), ('prediction_rows', 3664),
                               ('control_rows', 16), ('reference_input_checks', 16)):
                require(manifest[key] == count, f'Manifest satir sayisi farkli: {seed}, {key}')

        controls = sources.csv(folder / 'controls.csv',
            ('checkpoint', 'condition', 'n', 'max_uniform_error', 'hooks_restored', 'corrupted_input_sha256'), 16)
        keyed(controls, ('checkpoint', 'condition'), control_keys)
        for row in controls:
            seed_fields(row, seed)
            require(int(row['n']) == N and number(row, 'max_uniform_error') == 0
                    and row['hooks_restored'] == 'True', f'Identity/hook kontrolu basarisiz: {seed}')
            require(row['corrupted_input_sha256'] == manifest['corrupted_input_sha256'][row['condition']],
                    f'Kontrol girdi ozeti farkli: {seed}')
        if seed != 1111:
            checks = sources.csv(folder / 'reference_input_checks.csv',
                ('checkpoint', 'condition', 'actual_sha256', 'reference_seed1111_sha256', 'passed'), 16)
            keyed(checks, ('checkpoint', 'condition'), control_keys)
            for row in checks:
                seed_fields(row, seed)
                expected = reference_manifest['corrupted_input_sha256'][row['condition']]
                require(row['passed'] == 'True' and row['actual_sha256'] == expected
                        and row['reference_seed1111_sha256'] == expected, f'Referans girdisi farkli: {seed}')

        metrics = sources.csv(folder / 'metrics.csv',
            ('method', 'checkpoint', 'split', 'condition', 'tau', 'n', 'MAE_exact') + METRICS, 32)
        lookup = keyed(metrics, ('method', 'condition'), metric_keys)
        for row in metrics:
            seed_fields(row, seed)
            require(row['checkpoint'] in METHOD_PAIRS
                    and row['method'] in METHOD_PAIRS[row['checkpoint']]
                    and row['split'] == 'valid' and int(row['n']) == N and number(row, 'tau') == 1,
                    f'Metrik satiri protokolle uyusmuyor: {seed}')
            for field in METRICS + ('MAE_exact',):
                number(row, field)

        rows = sources.csv(folder / 'predictions_and_weights.csv',
            ('checkpoint', 'condition', 'index', 'label', 'plain_method', 'weighted_method',
             'plain_prediction', 'weighted_prediction'), 3664)
        groups = {k: {} for k in control_keys}
        for row in rows:
            seed_fields(row, seed)
            key, index = (row['checkpoint'], row['condition']), int(row['index'])
            require(key in groups and index not in groups[key], f'Bilinmeyen/tekrarli tahmin: {seed}, {key}, {index}')
            require((row['plain_method'], row['weighted_method']) == METHOD_PAIRS[key[0]],
                    f'Tahmin-yontem eslesmesi farkli: {seed}')
            groups[key][index] = tuple(number(row, field) for field in
                                      ('label', 'plain_prediction', 'weighted_prediction'))
        for (checkpoint, condition), group in groups.items():
            require(set(group) == set(range(N)), f'Eksik/fazla ornek indeksleri: {seed}, {checkpoint}, {condition}')
            labels = [group[i][0] for i in range(N)]
            if reference_labels is None:
                reference_labels = labels
            require(labels == reference_labels, f'Seed/kosullar arasinda etiket farki: {seed}, {condition}')
            for column, method in enumerate(METHOD_PAIRS[checkpoint], start=1):
                calculated = prediction_metrics(labels, [group[i][column] for i in range(N)])
                saved = lookup[method, condition]
                mae_exact = number(saved, 'MAE_exact')
                require(abs(calculated['MAE'] - mae_exact) < MAE_TOLERANCE,
                        f'MAE tahminlerden dogrulanamadi: {seed}, {method}, {condition}')
                # Preserve the evaluator's float32 MAE; avoid loss from rounded MAE.
                calculated['MAE'] = mae_exact
                for metric in METRICS:
                    require(abs(round(calculated[metric], 4) - number(saved, metric)) < 1e-12,
                            f'Metrik tahminlerden dogrulanamadi: {seed}, {method}, {condition}, {metric}')
                records[seed, method, condition] = calculated
        print(f'SEED_DOGRULANDI: {seed}; 32 metrik, 3664 tahmin, 16 kontrol; valid=229, Non0=216')
    require(len(records) == 160, 'Bes seed tamamlanmadi')
    sources.verify()
    return records, sources, reference_manifest


def mean_sd(values):
    values = list(values)
    require(len(values) == 5 and all(math.isfinite(x) for x in values), 'Istatistik icin bes sonlu deger gerekli')
    return statistics.mean(values), statistics.stdev(values)


def self_check():
    mean, sd = mean_sd([1, 2, 3, 4, 5])
    require(mean == 3 and abs(sd-math.sqrt(2.5)) < 1e-12, 'Orneklem SS kontrolu basarisiz')
    first, second = [1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]
    require(mean_sd([b-a for a, b in zip(first, second)]) == (0.5, 0.0), 'Eslestirilmis SS kontrolu basarisiz')


def summarize(records):
    per_seed = [dict(training_seed=s, method=m, condition=c, n=N, non0_n=NON0_N, **records[s, m, c])
                for s in SEEDS for c in CONDITIONS for m in METHODS]
    summary, paired_by_seed, paired_summary, degradation = [], [], [], []
    for condition in CONDITIONS:
        for method in METHODS:
            row = dict(method=method, condition=condition, n_seeds=5, n=N, non0_n=NON0_N)
            change = dict(method=method, condition=condition, reference='K0', n_seeds=5)
            for metric in METRICS:
                row[metric+'_mean'], row[metric+'_sd'] = mean_sd(records[s, method, condition][metric] for s in SEEDS)
                change[metric+'_delta_mean'], change[metric+'_delta_sd'] = mean_sd(
                    records[s, method, condition][metric] - records[s, method, 'K0'][metric] for s in SEEDS)
            summary.append(row)
            degradation.append(change)
        for second, first in COMPARISONS:
            local = []
            for seed in SEEDS:
                row = dict(training_seed=seed, condition=condition, comparison=f'{second}-minus-{first}')
                for metric in METRICS:
                    column, factor = ('Non0_acc_pp', 100) if metric == 'Non0_acc_2' else (metric, 1)
                    row[column+'_delta'] = factor * (records[seed, second, condition][metric] - records[seed, first, condition][metric])
                local.append(row)
            paired_by_seed.extend(local)
            row = dict(condition=condition, comparison=f'{second}-minus-{first}', n_pairs=5)
            for metric in ('MAE', 'Non0_acc_pp', 'Non0_F1_score', 'Corr'):
                values = [r[metric+'_delta'] for r in local]
                row[metric+'_delta_mean'], row[metric+'_delta_sd'] = mean_sd(values)
                better = [(-v if metric == 'MAE' else v) for v in values]
                row[metric+'_improved_seeds'] = sum(v > TIE_TOLERANCE for v in better)
                row[metric+'_tied_seeds'] = sum(abs(v) <= TIE_TOLERANCE for v in better)
                row[metric+'_worsened_seeds'] = sum(v < -TIE_TOLERANCE for v in better)
            paired_summary.append(row)
    return {'per_seed_metrics.csv': per_seed, 'summary_metrics.csv': summary,
            'paired_differences_by_seed.csv': paired_by_seed, 'paired_summary.csv': paired_summary,
            'degradation_summary.csv': degradation}


NOTES = '''Self-MM / CMU-MOSI: five-seed validation analysis
Training seeds: 1111, 1112, 1113, 1114, 1115. Corruption seed: 1111. Tau: 1.
A=baseline; B=input dropout; C=weighted baseline; D=weighted dropout.
K0=clean; K1=no audio; K2=no vision; K3=no audio/vision;
K4/K5/K6=30/50/70% valid-frame masking; K7=Gaussian noise (factor 0.5).
Text is unchanged by the recorded corruption protocol.

Means weight all five training seeds equally; no seed or condition is excluded.
SD is sample standard deviation across five seeds (ddof=1), not SEM or a CI.
Paired deltas: subtract within the SAME training seed, then compute mean and SD.
Positive MAE deltas mean worse; positive accuracy/F1/Corr deltas mean better.
degradation_summary.csv uses condition minus K0 of the SAME method and seed;
its accuracy deltas are fractions, not percentage points.
Non0_acc_2 in per_seed_metrics/summary_metrics is a fraction (0..1).
Non0_acc_pp_delta in paired tables is a percentage-point difference.
Non0 accuracy and support-weighted binary F1 exclude label=0; prediction>0
is positive. They use 216 examples. MAE and Pearson Corr use all 229 examples.
MAE uses the recorded unrounded MAE_exact after recomputation within 1e-6.
Accuracy, F1 and Corr are recomputed from predictions before aggregation.
Small differences from averaging four-decimal displayed CSV metrics are expected.

Independent repetition unit: training seed (n=5), not 8 conditions or 229 rows.
The same validation examples and one corruption realization are shared.
No significance test, confidence interval, model selection or test evaluation.
This is validation evidence, not a final test-performance claim.

The script checks frozen CSV/JSON exports and their input-hash/label/control
consistency. Historical data/checkpoint-protection flags are read from manifests;
large datasets, checkpoints and current MMSA sources are NOT opened/re-audited.
Source SHA256 values fingerprint the exports consumed by this analysis. They
detect changes during analysis; they do not authenticate the original evaluation.
No source file is modified. Existing output directories are never overwritten.
'''


def save(output, tables, sources, reference):
    require(not output.exists(), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(output.parent.is_dir(), f'Cikti ust klasoru yok: {output.parent}')
    sources.verify()
    output.mkdir(exist_ok=False)
    # A failure leaves a partial directory without a completed manifest.
    for filename, rows in tables.items():
        with (output / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'analysis_notes.txt').open('x', encoding='utf-8') as stream:
        stream.write(NOTES)
    sources.verify()
    manifest = dict(status='completed', analysis_version=VERSION, created_at_utc=datetime.now(timezone.utc).isoformat(),
        training_seeds=list(SEEDS), split='valid', n=N, non0_n=NON0_N, n_seeds=5, sd_ddof=1,
        corruption_seed=1111, tau=1.0, conditions=list(CONDITIONS), comparisons=[f'{b}-minus-{a}' for b,a in COMPARISONS],
        source_files_unchanged=True, source_file_count=len(sources.hashes), source_hashes=sources.hashes,
        shared_inputs_verified=True, labels_matched=True, metrics_recomputed=True,
        data_and_checkpoints_loaded=False, test_evaluation=False, hyperparameter_tuning=False,
        source_integrity_scope='Frozen CSV/JSON exports only; historical protection flags checked, not large assets.',
        corrupted_input_sha256=reference['corrupted_input_sha256'],
        mae_recompute_tolerance=MAE_TOLERANCE, paired_tie_tolerance=TIE_TOLERANCE,
        row_counts={name:len(rows) for name,rows in tables.items()}, python_version=sys.version.split()[0],
        script_sha256=sha256(Path(__file__)),
        output_hashes={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runs-dir', type=Path, help='Optional location of the five frozen evaluation folders')
    parser.add_argument('--output-dir', type=Path, help='Optional NEW output directory; parent must exist')
    parser.add_argument('--check', action='store_true', help='Validate and calculate without writing files')
    cli = parser.parse_args(argv)
    try:
        runs = (cli.runs_dir or cli.repo_dir.resolve().parent / 'runs').resolve()
        output = (cli.output_dir or runs / DEFAULT_OUTPUT).resolve()
        require(not output.exists(), f'Cikti zaten var; uzerine yazilmaz: {output}')
        require(output.parent.is_dir(), f'Cikti ust klasoru yok: {output.parent}')
        self_check()
        records, sources, reference = load_runs(runs)
        tables = summarize(records)
        sources.verify()
        print('K0: ORTALAMA +/- ORNEKLEM SS; n=5; accuracy yuzde olarak gosterilir')
        for row in tables['summary_metrics.csv']:
            if row['condition'] == 'K0':
                print(f"{row['method']} MAE={row['MAE_mean']:.4f}+/-{row['MAE_sd']:.4f} "
                      f"ACC={100*row['Non0_acc_2_mean']:.2f}+/-{100*row['Non0_acc_2_sd']:.2f} "
                      f"F1={row['Non0_F1_score_mean']:.4f}+/-{row['Non0_F1_score_sd']:.4f} "
                      f"Corr={row['Corr_mean']:.4f}+/-{row['Corr_sd']:.4f}")
        for row in tables['paired_summary.csv']:
            if row['condition'] == 'K0':
                print(f"K0 {row['comparison']}: MAE_DELTA={row['MAE_delta_mean']:+.4f}"
                      f"+/-{row['MAE_delta_sd']:.4f}; IYILESEN_SEED={row['MAE_improved_seeds']}/5")
        print('CIKTI:', output)
        if cli.check:
            print('KONTROL_BASARILI: bes seed, protokol, etiketler, metrikler, eslestirme ve SS; dosya yazilmadi')
            return 0
        save(output, tables, sources, reference)
        print('ANALIZ_TAMAMLANDI:', output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f'ANALIZ_DURDURULDU: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
