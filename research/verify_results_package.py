"""Verify the small Self-MM evidence package using only Python's standard library.

--check reads and recalculates without writing. --output-dir writes NEW summary
CSVs only after validation. No MMSA, dataset, checkpoint, network or GPU access.
This validates aggregation of recorded metrics, not predictions or model quality.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics as stats
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
SEEDS = range(1111, 1116)
CONDITIONS = tuple(f'K{i}' for i in range(8))
METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')
PAIRS = {'B-minus-A': ('B', 'A'), 'C-minus-A': ('C', 'A'),
         'D-minus-B': ('D', 'B'), 'D-minus-A': ('D', 'A')}
QUALITY_FILES = ('per_seed_metrics.csv', 'summary_metrics.csv',
                 'paired_differences_by_seed.csv', 'paired_summary.csv', 'degradation_summary.csv')
INFERENCE_FILES = ('batch_timings.csv', 'rounds.csv', 'summary.csv',
                   'paired_rounds.csv', 'paired_summary.csv', 'correctness_checks.csv')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), f'CSV header: {path.name}')
        rows = list(reader)
        require(all(None not in r and all(v is not None for v in r.values()) for r in rows), f'CSV columns: {path.name}')
        return rows


def number(value):
    v = float(value)
    require(math.isfinite(v), 'Non-finite numeric value')
    return v


def close(a, b):
    require(math.isclose(number(a), number(b), rel_tol=1e-10, abs_tol=1e-10), f'Numeric mismatch: {a} != {b}')


def indexed(rows, fields):
    result = {tuple(str(r[k]) for k in fields): r for r in rows}
    require(len(result) == len(rows), f'Duplicate key: {fields}')
    return result


def compare(expected, recorded, keys):
    left, right = indexed(expected, keys), indexed(recorded, keys)
    require(left.keys() == right.keys(), f'Missing/extra keys: {keys}')
    for key, row in left.items():
        require(row.keys() == right[key].keys(), f'Columns differ: {key}')
        for name, value in row.items():
            if isinstance(value, (int, float)):
                close(value, right[key][name])
            else:
                require(str(value) == right[key][name], f'Field mismatch: {key}/{name}')


def mean_sd(row, key, values):
    row[key+'_mean'] = stats.mean(values)
    row[key+'_sd'] = stats.stdev(values)


def quality(folder, split):
    n, non0 = (229, 216) if split == 'validation' else (686, 656)
    raw = read_csv(folder/'per_seed_metrics.csv')
    rows = indexed(raw, ('training_seed', 'method', 'condition'))
    require(set(rows) == {(str(s), m, c) for s in SEEDS for m in 'ABCD' for c in CONDITIONS}, 'Expected 160 metrics per split')
    for r in raw:
        require(int(r['n']) == n and int(r['non0_n']) == non0, 'Wrong split sample count')
        for metric in METRICS:
            v = number(r[metric])
            require((v >= 0 if metric == 'MAE' else -1 <= v <= 1 if metric == 'Corr' else 0 <= v <= 1), 'Metric range')
    summary, paired, paired_summary, degradation = [], [], [], []
    for c in CONDITIONS:
        for method in 'ABCD':
            r = dict(method=method, condition=c, n_seeds=5, n=n, non0_n=non0)
            d = dict(method=method, condition=c, reference='K0', n_seeds=5)
            for metric in METRICS:
                values = [number(rows[str(s), method, c][metric]) for s in SEEDS]
                mean_sd(r, metric, values)
                mean_sd(d, metric+'_delta', [v-number(rows[str(s), method, 'K0'][metric]) for s, v in zip(SEEDS, values)])
            summary.append(r); degradation.append(d)
        for comparison, (new, old) in PAIRS.items():
            local = []
            for seed in SEEDS:
                p = dict(training_seed=seed, condition=c, comparison=comparison)
                for metric in METRICS:
                    name = 'Non0_acc_pp' if metric == 'Non0_acc_2' else metric
                    factor = 100 if metric == 'Non0_acc_2' else 1
                    p[name+'_delta'] = (number(rows[str(seed), new, c][metric])-number(rows[str(seed), old, c][metric]))*factor
                local.append(p)
            paired.extend(local)
            r = dict(condition=c, comparison=comparison, n_pairs=5)
            for metric in ('MAE', 'Non0_acc_pp', 'Non0_F1_score', 'Corr'):
                values = [p[metric+'_delta'] for p in local]
                mean_sd(r, metric+'_delta', values)
                benefit = [-v if metric == 'MAE' else v for v in values]
                r[metric+'_improved_seeds'] = sum(v > 1e-12 for v in benefit)
                r[metric+'_tied_seeds'] = sum(abs(v) <= 1e-12 for v in benefit)
                r[metric+'_worsened_seeds'] = sum(v < -1e-12 for v in benefit)
            paired_summary.append(r)
    outputs = {'summary_metrics.csv': (summary, ('method', 'condition')),
               'paired_differences_by_seed.csv': (paired, ('training_seed', 'condition', 'comparison')),
               'paired_summary.csv': (paired_summary, ('condition', 'comparison')),
               'degradation_summary.csv': (degradation, ('method', 'condition'))}
    for name, (values, keys) in outputs.items():
        compare(values, read_csv(folder/name), keys)
    print(f'{split.upper()}_DOGRULANDI: 160 seed/yontem/kosul; ortalama, orneklem SS, eslestirilmis farklar')
    return {split+'/'+name: values for name, (values, _) in outputs.items()}


def percentile(values, percent):
    values = sorted(values)
    pos = (len(values)-1)*percent/100
    lower = math.floor(pos); upper = math.ceil(pos)
    return values[lower]+(values[upper]-values[lower])*(pos-lower)


def inference(folder):
    batches = read_csv(folder/'batch_timings.csv')
    lookup = indexed(read_csv(folder/'rounds.csv'), ('method', 'round'))
    bi = indexed(batches, ('method', 'round', 'batch_index'))
    require(set(bi) == {(m, str(r), str(b)) for m in 'ABCD' for r in range(1, 7) for b in range(172)}, 'Missing/extra timing batches')
    require(set(lookup) == {(m, str(r)) for m in 'ABCD' for r in range(1, 7)}, 'Missing timing rounds')
    for (m, repeat), r in lookup.items():
        cp = 'baseline' if m in 'AC' else 'dropout'
        position = 1 if (m in 'AB') == (int(repeat) % 2 == 1) else 2
        current = [bi[m, repeat, str(i)] for i in range(172)]
        require([int(b['batch_n']) for b in current] == [4]*171+[2], 'Tail batch or sample count')
        require(all(b['checkpoint'] == cp and int(b['order_position']) == position for b in current), 'Timing input/order')
        require(r['checkpoint'] == cp and int(r['order_position']) == position and int(r['n']) == 686, 'Round context')
        require(int(r['batches']) == 172 and int(r['full_batches']) == 171 and int(r['tail_batch_n']) == 2, 'Round batch counts')
        require(int(r['forward_passes_per_batch']) == (1 if m in 'AB' else 2) and int(r['warmup_batches']) == 10, 'Timing passes/warmup')
        times = [number(b['latency_ms']) for b in current]
        require(min(times) > 0, 'Nonpositive latency')
        total = math.fsum(times)
        for k, v in dict(timed_total_ms=total, ms_per_sample=total/686, samples_per_second=686000/total,
                         full_batch_ms_mean=stats.mean(times[:-1]), tail_batch_ms=times[-1]).items(): close(r[k], v)
        close(r['peak_extra_allocated_mib'], number(r['peak_allocated_mib'])-number(r['resident_allocated_mib']))
        require(number(r['peak_reserved_mib']) >= number(r['peak_allocated_mib']) >= number(r['resident_allocated_mib']), 'Memory ordering')
        close(r['end_allocated_mib'], r['resident_allocated_mib'])
        require(r['hooks_restored'] == 'True' and number(r['max_prediction_error']) == 0, 'Recorded round controls')
    summaries = []
    for m in 'ABCD':
        r = dict(method=m, condition='K0', n=686, repeats=6, batch_size=4)
        rounds = [lookup[m, str(i)] for i in range(1, 7)]
        for key in ('ms_per_sample', 'samples_per_second', 'full_batch_ms_mean', 'peak_allocated_mib', 'peak_extra_allocated_mib', 'peak_reserved_mib'):
            mean_sd(r, key, [number(t[key]) for t in rounds])
        for key in ('peak_allocated_mib', 'peak_reserved_mib'): r[key+'_max'] = max(number(t[key]) for t in rounds)
        full = [number(b['latency_ms']) for b in batches if b['method'] == m and int(b['batch_n']) == 4]
        r['full_batch_ms_p50'] = percentile(full, 50); r['full_batch_ms_p95'] = percentile(full, 95)
        summaries.append(r)
    pairs, pair_summary = [], []
    for weighted, plain in (('C', 'A'), ('D', 'B')):
        local = []
        for i in range(1, 7):
            a, b = lookup[plain, str(i)], lookup[weighted, str(i)]
            local.append(dict(comparison=weighted+'-vs-'+plain, round=i,
                ms_per_sample_delta=number(b['ms_per_sample'])-number(a['ms_per_sample']),
                latency_ratio=number(b['ms_per_sample'])/number(a['ms_per_sample']),
                peak_allocated_mib_delta=number(b['peak_allocated_mib'])-number(a['peak_allocated_mib'])))
        pairs.extend(local)
        r = dict(comparison=weighted+'-vs-'+plain, repeats=6)
        for key in ('ms_per_sample_delta', 'latency_ratio', 'peak_allocated_mib_delta'): mean_sd(r, key, [p[key] for p in local])
        pair_summary.append(r)
    checks = indexed(read_csv(folder/'correctness_checks.csv'), ('method',))
    require(set(checks) == {(m,) for m in 'ABCD'}, 'Correctness methods')
    for r in checks.values():
        require(int(r['n']) == 686 and r['passed'] == r['hooks_restored'] == 'True', 'Recorded correctness')
        close(r['max_prediction_error'], 0); close(r['uniform_control_max_error'], 0)
    output = {'summary.csv': (summaries, ('method',)), 'paired_rounds.csv': (pairs, ('comparison', 'round')),
              'paired_summary.csv': (pair_summary, ('comparison',))}
    for name, (values, keys) in output.items(): compare(values, read_csv(folder/name), keys)
    print('INFERENCE_DOGRULANDI: 4128 batch, 24 tur; sure, bellek ozeti, eslestirilmis oranlar')
    return {'inference/'+name: values for name, (values, _) in output.items()}


def verify(package):
    manifest = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['package_version'] == 1 and manifest['sd_ddof'] == 1, 'Package version/statistics')
    expected = {f'{s}/{n}' for s in ('validation', 'evaluation_test') for n in QUALITY_FILES}
    expected.update('inference/'+n for n in INFERENCE_FILES)
    expected.update(('protocol.json', 'provenance.json', 'DATA_DICTIONARY.md'))
    require(set(manifest['files']) == expected, 'Manifest file list')
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    require(actual == expected | {'manifest.json'}, 'Unexpected or missing package files')
    require(manifest['verifier_sha256'] == sha(__file__), 'Verifier differs from package version')
    for name, entry in manifest['files'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name, 'Unsafe package path')
        path = package.joinpath(*relative.parts)
        require(path.resolve().is_relative_to(package.resolve()), 'Path leaves package')
        require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'File changed/missing: '+name)
        if name.endswith('.csv'): require(len(read_csv(path)) == entry['rows'], 'Row count: '+name)
    protocol = json.loads((package/'protocol.json').read_text(encoding='utf-8'))
    require(protocol['training_seeds'] == list(SEEDS) and protocol['conditions'] == list(CONDITIONS), 'Protocol seeds/conditions')
    require(protocol['tau'] == 1 and protocol['corruption_seed'] == 1111 and protocol['batch_size'] == 4, 'Protocol parameters')
    output = quality(package/'validation', 'validation')
    output.update(quality(package/'evaluation_test', 'evaluation_test')); output.update(inference(package/'inference'))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent/'results_package_v1')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Read-only verification and recalculation')
    group.add_argument('--output-dir', type=Path, help='Write 11 recalculated CSVs to a NEW directory')
    args = parser.parse_args(argv)
    try:
        package = args.package_dir.resolve()
        if args.output_dir:
            require(not os.path.lexists(args.output_dir), 'Output already exists; never overwritten')
            require(args.output_dir.parent.is_dir(), 'Output parent must already exist')
            require(not args.output_dir.resolve().is_relative_to(package), 'Output must be outside the source package')
        output = verify(package)
        if args.check:
            print('PAKET_KONTROLU_BASARILI: 19 dosya; ML/GPU/veri/checkpoint gerektirmedi; dosya yazilmadi')
        else:
            args.output_dir.mkdir(exist_ok=False)
            for name, rows in output.items():
                target = args.output_dir/name; target.parent.mkdir(exist_ok=True)
                with target.open('x', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            print('OZETLER_YENIDEN_HESAPLANDI: 11 CSV;', args.output_dir)
        print('KAPSAM: kayitli metriklerin ozetlenmesi; tahminler yeniden uretilmedi, model/veri dogrulanmadi')
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print('PAKET_KONTROLU_DURDU:', error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
