"""Five-seed summary and pre-registered decisions for Self-MM diagnostics v2.

Standard library only. Build mode reads the five completed run directories of one
split and writes a compact package: per-seed metrics, five-seed summaries,
variant-minus-plain paired differences, hypothesis decisions (protocol section 5)
and, for validation, the tau selection used by the single test run (section 6).
--verify recomputes every derived table from the package's per-seed CSV and the
stored hashes; it needs no GPU, PyTorch, dataset or checkpoint.
See research/selfmm_diagnostics_protocol_v2.md.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROTOCOL = 'selfmm-diagnostics-v2'
SEEDS = (1111, 1112, 1113, 1114, 1115)
AV_CONDITIONS = tuple(f'K{i}' for i in range(8))
CONDITIONS = AV_CONDITIONS + ('K8', 'K9', 'K10')
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ORACLE_CONDITIONS = ('K1', 'K2', 'K3', 'K10')
CHECKPOINTS = ('baseline', 'dropout')
METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')
SPLITS = {'valid': dict(n=229, folder='validation'), 'test': dict(n=686, folder='evaluation_test')}
RUN_TABLES = ('metrics.csv', 'controls.csv', 'weight_summary.csv', 'predictions.csv')
TOLERANCE = 1e-12


def tau_name(tau):
    return f'agree_tau{tau:g}'


VARIANTS = ('plain',) + tuple(tau_name(t) for t in TAUS) + ('agree_av_tau1', 'shuffled_tau1', 'oracle')
HEADS = ('head_T', 'head_A', 'head_V')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({c: r.get(c, '') for c in columns} for r in rows)


def number(value):
    return float(value) if value not in ('', None) else float('nan')


def mean_sd(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float('nan'), float('nan'), 0
    return statistics.fmean(finite), statistics.stdev(finite) if len(finite) > 1 else float('nan'), len(finite)


# ------------------------------------------------------------- derivations

def index_metrics(rows):
    table = {}
    for row in rows:
        key = row['checkpoint'], row['condition'], row['variant'], int(row['training_seed'])
        require(key not in table, f'Tekrarlanan metrik satiri: {key}')
        table[key] = {m: number(row[m]) for m in METRICS}
    expected = {(c, k, v, s) for c in CHECKPOINTS for k in CONDITIONS for s in SEEDS
                for v in VARIANTS + HEADS if v != 'oracle' or k in ORACLE_CONDITIONS}
    require(set(table) == expected, 'Bes seed metrik tablosu eksik veya fazla')
    return table


def mae(table, checkpoint, condition, variant):
    return [table[checkpoint, condition, variant, s]['MAE'] for s in SEEDS]


def summary_rows(table):
    rows = []
    for checkpoint in CHECKPOINTS:
        for condition in CONDITIONS:
            for variant in VARIANTS + HEADS:
                if (checkpoint, condition, variant, SEEDS[0]) not in table:
                    continue
                row = dict(checkpoint=checkpoint, condition=condition, variant=variant, n_seeds=len(SEEDS))
                for metric in METRICS:
                    m, sd, count = mean_sd([table[checkpoint, condition, variant, s][metric] for s in SEEDS])
                    row.update({f'{metric}_mean': m, f'{metric}_sd': sd, f'{metric}_finite_seeds': count})
                rows.append(row)
    return rows


def paired_rows(table):
    by_seed, summary = [], []
    for checkpoint in CHECKPOINTS:
        for condition in CONDITIONS:
            for variant in VARIANTS[1:]:
                if (checkpoint, condition, variant, SEEDS[0]) not in table:
                    continue
                deltas = {}
                for s in SEEDS:
                    a, b = table[checkpoint, condition, variant, s], table[checkpoint, condition, 'plain', s]
                    deltas[s] = {m: a[m] - b[m] for m in METRICS}
                    by_seed.append(dict(training_seed=s, checkpoint=checkpoint, condition=condition,
                                        comparison=f'{variant}-minus-plain',
                                        **{f'{m}_delta': deltas[s][m] for m in METRICS}))
                row = dict(checkpoint=checkpoint, condition=condition, comparison=f'{variant}-minus-plain',
                           n_seeds=len(SEEDS), MAE_lower_seeds=sum(deltas[s]['MAE'] < 0 for s in SEEDS))
                for metric in METRICS:
                    m, sd, _ = mean_sd([deltas[s][metric] for s in SEEDS])
                    row.update({f'{metric}_delta_mean': m, f'{metric}_delta_sd': sd})
                summary.append(row)
    return by_seed, summary


def hypotheses(table):
    """Protocol v2 section 5, applied to the baseline checkpoint."""
    ck = 'baseline'

    def diff(condition, first, second):
        return [a - b for a, b in zip(mae(table, ck, condition, first), mae(table, ck, condition, second))]

    results = []
    curve = {t: statistics.fmean(statistics.fmean(mae(table, ck, c, tau_name(t))[i] for c in AV_CONDITIONS)
                                 for i in range(len(SEEDS))) for t in TAUS}
    steps = [curve[b] - curve[a] for a, b in zip(TAUS, TAUS[1:])]
    k0_tau16 = statistics.fmean(diff('K0', 'agree_tau16', 'plain'))
    results.append(dict(code='H1', expected=True,
        supported=all(step <= 1e-4 for step in steps) and abs(k0_tau16) <= 0.01,
        evidence=dict(k0_k7_mean_mae_by_tau={f'{t:g}': curve[t] for t in TAUS}, largest_step=max(steps),
                      k0_tau16_minus_plain_mean=k0_tau16)))
    av, c = diff('K0', 'agree_av_tau1', 'plain'), diff('K0', 'agree_tau1', 'plain')
    av_below_c = sum(x < y for x, y in zip(mae(table, ck, 'K0', 'agree_av_tau1'), mae(table, ck, 'K0', 'agree_tau1')))
    results.append(dict(code='H2', expected=True,
        supported=statistics.fmean(av) < 0.25 * statistics.fmean(c) and av_below_c == len(SEEDS),
        evidence=dict(k0_av_minus_plain_mean=statistics.fmean(av), k0_tau1_minus_plain_mean=statistics.fmean(c),
                      seeds_av_below_tau1=av_below_c)))
    vs_shuffled = diff('K0', 'agree_tau1', 'shuffled_tau1')
    lower = sum(d < 0 for d in vs_shuffled)
    results.append(dict(code='H3', expected=False,
        supported=lower >= 4 and statistics.fmean(vs_shuffled) <= -0.01,
        evidence=dict(k0_tau1_minus_shuffled_mean=statistics.fmean(vs_shuffled), seeds_tau1_below_shuffled=lower)))
    oracle = {k: statistics.fmean(diff(k, 'oracle', 'plain')) for k in ('K1', 'K2', 'K3')}
    results.append(dict(code='H4', expected=True, supported=all(abs(v) <= 0.01 for v in oracle.values()),
                        evidence=dict(oracle_minus_plain_mean=oracle)))
    text_loss = statistics.fmean(a - b for a, b in zip(mae(table, ck, 'K10', 'plain'), mae(table, ck, 'K0', 'plain')))
    results.append(dict(code='H5a', expected=True, supported=text_loss >= 0.10,
                        evidence=dict(plain_k10_minus_k0_mean=text_loss)))
    h5b = {}
    for k in ('K9', 'K10'):
        d = diff(k, 'agree_tau1', 'plain')
        h5b[k] = dict(tau1_minus_plain_mean=statistics.fmean(d), seeds_tau1_below_plain=sum(x < 0 for x in d))
    results.append(dict(code='H5b', expected=False,
        supported=any(v['seeds_tau1_below_plain'] >= 4 and v['tau1_minus_plain_mean'] <= -0.01 for v in h5b.values()),
        evidence=h5b))
    for row in results:
        row['matches_expectation'] = row['supported'] == row['expected']
    return results


def select_tau(table):
    """Protocol v2 section 6: baseline, K0-K10 equally weighted, five-seed mean; ties -> larger tau."""
    score = {t: statistics.fmean(statistics.fmean(mae(table, 'baseline', c, tau_name(t))[i] for c in CONDITIONS)
                                 for i in range(len(SEEDS))) for t in TAUS}
    best = min(score.values())
    selected = max(t for t in TAUS if score[t] - best <= 1e-6)
    return selected, score


def derive(table, split):
    by_seed, paired = paired_rows(table)
    derived = dict(summary_metrics=summary_rows(table), paired_differences_by_seed=by_seed, paired_summary=paired,
                   hypotheses=hypotheses(table))
    if split == 'valid':
        derived['selected_tau'], derived['tau_scores'] = select_tau(table)
    return derived


def hypothesis_rows(results):
    return [dict(code=r['code'], expected=r['expected'], supported=r['supported'],
                 matches_expectation=r['matches_expectation'], evidence=json.dumps(r['evidence'], sort_keys=True))
            for r in results]


# ------------------------------------------------------------------- build

def load_runs(runs_dir, split, selection):
    runs, identities = {}, set()
    input_shas, permutation_shas = {}, {}
    for seed in SEEDS:
        directory = runs_dir / f'selfmm_diagnostics_v2_{split}_seed{seed}'
        manifest_path = directory / 'manifest.json'
        require(manifest_path.is_file(), f'Kosu bulunamadi: {directory.name}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        require(manifest.get('protocol') == PROTOCOL and manifest.get('split') == split
                and manifest.get('training_seed') == seed and manifest.get('status') == 'completed'
                and manifest.get('reference_metrics_matched') is True and manifest.get('reference_inputs_matched') is True
                and manifest.get('protected_files_unchanged') is True, f'Kosu tamamlanmamis/dogrulanmamis: {directory.name}')
        for name, expected in manifest['output_hashes'].items():
            require(digest(directory / name) == expected, f'Kosu ciktisi degismis: {directory.name}/{name}')
        identities.add((manifest['protocol_doc_sha256_lf'], manifest['script_sha256_lf']))
        if split == 'test':
            require(manifest.get('selection') == selection, f'Test kosusu farkli secimle calismis: {seed}')
        for row in read_csv(directory / 'controls.csv'):
            input_shas.setdefault(row['condition'], set()).add(row['input_sha256'])
            permutation_shas.setdefault(row['condition'], set()).add(row['shuffle_permutation_sha256'])
            require(row['shared_inputs'] == 'True' and float(row['max_uniform_error']) <= 1e-6, 'Kontrol satiri basarisiz')
        runs[seed] = dict(directory=directory, manifest=manifest, manifest_sha256=digest(manifest_path))
    require(len(identities) == 1, 'Kosular farkli protokol/betik surumleriyle calismis')
    require(all(len(v) == 1 for v in input_shas.values()) and len(input_shas) == len(CONDITIONS),
            'Seedler ayni bozulmus girdileri almamis')
    require(all(len(v) == 1 for v in permutation_shas.values()), 'Karistirma permutasyonu seedler arasinda farkli')
    return runs, identities.pop()


def head_spread_rows(runs, split):
    rows = []
    for seed, run in runs.items():
        groups = {}
        for row in read_csv(run['directory'] / 'predictions.csv'):
            groups.setdefault((row['checkpoint'], row['condition']), []).append(row)
        for (checkpoint, condition), records in sorted(groups.items()):
            require(len(records) == SPLITS[split]['n'], 'Tahmin satiri sayisi')
            out = dict(training_seed=seed, checkpoint=checkpoint, condition=condition, n=len(records))
            for key in ('T', 'A', 'V'):
                values = [float(r['y_' + key]) for r in records]
                out[f'y_{key}_sd'] = statistics.pstdev(values)
                out[f'd_{key}_mean'] = statistics.fmean(float(r['d_' + key]) for r in records)
            rows.append(out)
    return rows


def build(cli):
    split = cli.split
    output = cli.output_dir.absolute()
    require(not output.exists(), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(output.parent.is_dir(), f'Ust klasor yok: {output.parent}')
    selection = None
    if split == 'test':
        require(cli.selection is not None, 'Test ozeti icin --selection gerekli')
        selection = json.loads(cli.selection.read_text(encoding='utf-8'))
    else:
        require(cli.selection is None, 'Validation ozeti --selection almaz')
    runs, (protocol_sha, script_sha) = load_runs(cli.runs_dir.resolve(), split, selection)
    metric_rows = []
    for seed, run in runs.items():
        rows = read_csv(run['directory'] / 'metrics.csv')
        require(len(rows) == 294, f'Seed {seed} metrik satiri sayisi')
        metric_rows.extend(rows)
    table = index_metrics(metric_rows)
    derived = derive(table, split)
    weights = [row for run in runs.values() for row in read_csv(run['directory'] / 'weight_summary.csv')]
    output.mkdir()
    tables = {'per_seed_metrics.csv': metric_rows, 'summary_metrics.csv': derived['summary_metrics'],
              'paired_differences_by_seed.csv': derived['paired_differences_by_seed'],
              'paired_summary.csv': derived['paired_summary'], 'hypotheses.csv': hypothesis_rows(derived['hypotheses']),
              'weight_summary_by_seed.csv': weights, 'head_spread_by_seed.csv': head_spread_rows(runs, split)}
    for name, rows in tables.items():
        write_csv(output / name, rows)
    files = list(tables)
    if split == 'valid':
        chosen = dict(protocol=PROTOCOL, split='valid', training_seeds=list(SEEDS), selected_tau=derived['selected_tau'],
                      rule='baseline checkpoint; K0-K10 equally weighted; five-seed mean validation MAE; ties -> larger tau',
                      tau_scores={f'{t:g}': v for t, v in derived['tau_scores'].items()},
                      source_run_manifest_sha256={str(s): r['manifest_sha256'] for s, r in runs.items()})
        (output / 'selection.json').write_text(json.dumps(chosen, indent=2) + '\n', encoding='utf-8')
        files.append('selection.json')
    manifest = dict(protocol=PROTOCOL, split=split, training_seeds=list(SEEDS), conditions=list(CONDITIONS),
                    variants=list(VARIANTS), protocol_doc_sha256_lf=protocol_sha, run_script_sha256_lf=script_sha,
                    summary_script_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                    source_runs={str(s): dict(directory=r['directory'].name, manifest_sha256=r['manifest_sha256'],
                                              created_at_utc=r['manifest']['created_at_utc'])
                                 for s, r in runs.items()},
                    selection=selection, sd_ddof=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    scope='Compact copy of five run outputs plus derived tables; per-sample predictions stay in runs/.',
                    output_sha256={name: digest(output / name) for name in files})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    report(derived, split)
    print('OZET_YAZILDI:', output)


# ------------------------------------------------------------------ verify

def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        x, y = float(a), float(b)
        return (math.isnan(x) and math.isnan(y)) or abs(x - y) <= TOLERANCE * max(1.0, abs(x), abs(y))
    return str(a) == str(b)


def compare(stored, computed, name):
    require(len(stored) == len(computed), f'{name}: satir sayisi farkli')
    for s, c in zip(stored, computed):
        require(set(s) == {str(k) for k in c}, f'{name}: kolonlar farkli')
        for key, value in c.items():
            require(same(s[key], value), f'{name}: {key} farkli ({s[key]} != {value})')


def verify(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['output_sha256'].items():
        require(digest(directory / name) == expected, f'Paket dosyasi degismis: {name}')
    split = manifest['split']
    table = index_metrics(read_csv(directory / 'per_seed_metrics.csv'))
    derived = derive(table, split)
    compare(read_csv(directory / 'summary_metrics.csv'), derived['summary_metrics'], 'summary_metrics')
    compare(read_csv(directory / 'paired_differences_by_seed.csv'), derived['paired_differences_by_seed'], 'paired_by_seed')
    compare(read_csv(directory / 'paired_summary.csv'), derived['paired_summary'], 'paired_summary')
    stored = read_csv(directory / 'hypotheses.csv')
    computed = hypothesis_rows(derived['hypotheses'])
    require([r['code'] for r in stored] == [r['code'] for r in computed], 'Hipotez listesi farkli')
    for s, c in zip(stored, computed):
        require(s['supported'] == str(c['supported']) and s['expected'] == str(c['expected']), f'Hipotez karari farkli: {c["code"]}')
        evidence = json.loads(s['evidence'])
        require(json.dumps(evidence, sort_keys=True) == c['evidence']
                or all(same(a, b) for a, b in zip(_flatten(evidence), _flatten(json.loads(c['evidence'])))),
                f'Hipotez kaniti farkli: {c["code"]}')
    if split == 'valid':
        selection = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
        require(selection['selected_tau'] == derived['selected_tau'], 'Secilen tau yeniden uretilemedi')
    report(derived, split)
    print('OZET_DOGRULANDI:', directory.name, '; GPU/veri/checkpoint kullanilmadi, dosya yazilmadi')


def _flatten(value):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key])
    else:
        yield value


def report(derived, split):
    for row in derived['hypotheses']:
        verdict = 'DESTEKLENDI' if row['supported'] else 'DESTEKLENMEDI'
        note = 'beklendigi gibi' if row['matches_expectation'] else 'BEKLENTININ TERSI'
        print(f"{row['code']}: {verdict} ({note})")
    if split == 'valid':
        print('SECILEN_TAU:', f"{derived['selected_tau']:g}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verify', type=Path, help='Recompute and check an existing summary package')
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--split', choices=tuple(SPLITS), default='valid')
    parser.add_argument('--selection', type=Path, help='Committed validation selection.json; test split only')
    parser.add_argument('--output-dir', type=Path)
    cli = parser.parse_args(argv)
    if cli.verify is None and (cli.runs_dir is None or cli.output_dir is None):
        parser.error('--verify veya --runs-dir ile --output-dir gerekli')
    return cli


def main(argv=None):
    cli = parse_args(argv)
    if cli.verify is not None:
        verify(cli.verify)
    else:
        build(cli)


if __name__ == '__main__':
    main()
