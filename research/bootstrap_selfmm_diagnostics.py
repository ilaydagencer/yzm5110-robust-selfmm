"""Paired test-example bootstrap for the fixed Self-MM comparisons.

Seed SD in the summary packages describes training variation. This script adds
test-sampling uncertainty: the 686 test examples are resampled with replacement
(2000 times, numpy PCG64 seed 1111) and the SAME resampled index set is applied to
every training seed, method and condition, so all differences stay paired. Each
statistic is the five-seed mean difference: MAE (primary) and Non0 accuracy in
percentage points. Intervals are 95% percentile intervals.

The comparison list is fixed in COMPARISONS below and does not depend on results.
Input: research/diagnostics_v2/predictions/predictions_test.csv.gz (CPU only).
--verify recomputes into memory and checks the stored table byte for byte.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

SEEDS = (1111, 1112, 1113, 1114, 1115)
CONDITIONS = tuple(f'K{i}' for i in range(11))
N = 686
RESAMPLES = 2000
RNG_SEED = 1111


def build_comparisons():
    out = []

    def add(name, first, second):
        out.append(dict(comparison=name, first=first, second=second))
    for c in CONDITIONS:
        add('B-minus-A', ('dropout', c, 'plain'), ('baseline', c, 'plain'))
        add('C-minus-A', ('baseline', c, 'agree_tau1'), ('baseline', c, 'plain'))
        add('D-minus-B', ('dropout', c, 'agree_tau1'), ('dropout', c, 'plain'))
        add('tau16-minus-plain', ('baseline', c, 'agree_tau16'), ('baseline', c, 'plain'))
        add('textfixed-minus-plain', ('baseline', c, 'agree_av_tau1'), ('baseline', c, 'plain'))
        add('tau1-minus-shuffled', ('baseline', c, 'agree_tau1'), ('baseline', c, 'shuffled_tau1'))
    for c in ('K1', 'K2', 'K3', 'K10'):
        add('oracle-minus-plain', ('baseline', c, 'oracle'), ('baseline', c, 'plain'))
    for c in ('K3', 'K8', 'K9', 'K10'):
        add(f'plain-{c}-minus-K0', ('baseline', c, 'plain'), ('baseline', 'K0', 'plain'))
    return out


COMPARISONS = build_comparisons()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(path):
    columns = {}
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = int(row['training_seed']), row['checkpoint'], row['condition']
            columns.setdefault(key, []).append(row)
    arrays = {}
    for key, rows in columns.items():
        rows.sort(key=lambda r: int(r['index']))
        require([int(r['index']) for r in rows] == list(range(N)), f'Ornek sirasi eksik: {key}')
        label = np.array([float(r['label']) for r in rows])
        arrays[key] = dict(label=label)
        for name in ('plain', 'agree_tau1', 'agree_tau16', 'agree_av_tau1', 'shuffled_tau1', 'oracle'):
            values = [r['pred_' + name] for r in rows]
            if all(v != '' for v in values):
                arrays[key][name] = np.array([float(v) for v in values])
    require(len(arrays) == len(SEEDS) * 2 * len(CONDITIONS), 'Seed/checkpoint/kosul sayisi')
    labels = {tuple(a['label']) for a in arrays.values()}
    require(len(labels) == 1, 'Etiketler satirlar arasinda farkli')
    return arrays


def per_sample(arrays, spec):
    """Seed-mean per-example differences (first minus second) for abs error and Non0 correctness."""
    abs_diff, correct_diff = [], []
    for seed in SEEDS:
        a, b = (arrays[(seed,) + spec[k][:2]] for k in ('first', 'second'))
        y = a['label']
        pa, pb = a[spec['first'][2]], b[spec['second'][2]]
        abs_diff.append(np.abs(pa - y) - np.abs(pb - y))
        correct_diff.append(((pa > 0) == (y > 0)).astype(float) - ((pb > 0) == (y > 0)).astype(float))
    abs_diff, correct_diff = np.array(abs_diff), np.array(correct_diff)
    return abs_diff, correct_diff, y != 0


def compute(arrays):
    rng = np.random.default_rng(RNG_SEED)
    index = rng.integers(0, N, size=(RESAMPLES, N))
    rows = []
    for spec in COMPARISONS:
        abs_diff, correct_diff, mask = per_sample(arrays, spec)
        mae_i = abs_diff.mean(axis=0)
        acc_i = correct_diff.mean(axis=0) * 100
        per_seed_mae = abs_diff.mean(axis=1)
        point_acc = acc_i[mask].mean()
        boot_mae = mae_i[index].mean(axis=1)
        m = mask[index].astype(float)
        boot_acc = (acc_i[index] * m).sum(axis=1) / m.sum(axis=1)
        low, high = np.percentile(boot_mae, [2.5, 97.5])
        acc_low, acc_high = np.percentile(boot_acc, [2.5, 97.5])
        rows.append(dict(comparison=spec['comparison'], condition=spec['first'][1],
                         first='/'.join(spec['first']), second='/'.join(spec['second']), n=N, non0_n=int(mask.sum()),
                         resamples=RESAMPLES, MAE_delta=float(mae_i.mean()), MAE_ci_low=float(low), MAE_ci_high=float(high),
                         MAE_boot_share_above_zero=float((boot_mae > 0).mean()),
                         MAE_seed_sd=float(np.std(per_seed_mae, ddof=1)),
                         MAE_seeds_below_zero=int((per_seed_mae < 0).sum()),
                         Non0_acc_delta_pp=float(point_acc), Non0_acc_ci_low_pp=float(acc_low),
                         Non0_acc_ci_high_pp=float(acc_high)))
    return rows


def render(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def check_points(rows, summary_dir):
    """Point estimates must equal the committed five-seed paired summary (float32 vs float64 means)."""
    with (summary_dir / 'paired_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        paired = {(r['checkpoint'], r['condition'], r['comparison']): float(r['MAE_delta_mean'])
                  for r in csv.DictReader(stream)}
    mapping = {'C-minus-A': ('baseline', 'agree_tau1-minus-plain'), 'D-minus-B': ('dropout', 'agree_tau1-minus-plain'),
               'tau16-minus-plain': ('baseline', 'agree_tau16-minus-plain'),
               'textfixed-minus-plain': ('baseline', 'agree_av_tau1-minus-plain'),
               'oracle-minus-plain': ('baseline', 'oracle-minus-plain')}
    worst, checked = 0.0, 0
    for row in rows:
        if row['comparison'] in mapping:
            checkpoint, name = mapping[row['comparison']]
            worst = max(worst, abs(row['MAE_delta'] - paired[checkpoint, row['condition'], name]))
            checked += 1
    require(worst <= 1e-6, f'Bootstrap nokta tahmini ozet paketle eslesmiyor: {worst}')
    return checked, worst


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions-dir', type=Path, default=here / 'diagnostics_v2/predictions')
    parser.add_argument('--summary-dir', type=Path, default=here / 'diagnostics_v2/evaluation_test')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify', type=Path, help='Existing bootstrap folder to recompute and compare')
    cli = parser.parse_args(argv)
    require((cli.output_dir is None) != (cli.verify is None), '--output-dir veya --verify secin')
    source = cli.predictions_dir.resolve() / 'predictions_test.csv.gz'
    source_manifest = json.loads((cli.predictions_dir / 'manifest.json').read_text(encoding='utf-8'))
    require(digest(source) == source_manifest['output_sha256']['predictions_test.csv.gz'], 'Tahmin dosyasi degismis')
    rows = compute(load_predictions(source))
    checked, worst = check_points(rows, cli.summary_dir.resolve())
    data = render(rows)
    if cli.verify:
        stored = cli.verify.resolve() / 'bootstrap_test.csv'
        manifest = json.loads((cli.verify / 'manifest.json').read_text(encoding='utf-8'))
        require(manifest['input_sha256'] == digest(source), 'Bootstrap farkli tahmin dosyasiyla uretilmis')
        require(stored.read_bytes() == data, 'Bootstrap tablosu yeniden uretilemedi (numpy surumu farkli olabilir)')
        print(f'BOOTSTRAP_DOGRULANDI: {len(rows)} karsilastirma byte olarak ayni; dosya yazilmadi')
        return
    output = cli.output_dir.absolute()
    require(not output.exists() and output.parent.is_dir(), f'Cikti var veya ust klasor yok: {output}')
    output.mkdir()
    (output / 'bootstrap_test.csv').write_bytes(data)
    manifest = dict(split='test', training_seeds=list(SEEDS), n=N, resamples=RESAMPLES, rng='numpy PCG64',
                    rng_seed=RNG_SEED, numpy_version=np.__version__, interval='95% percentile, numpy linear',
                    statistic='five-seed mean of per-example differences; same resampled indices for every comparison',
                    comparisons=len(rows), point_estimates_checked=checked, point_max_abs_difference=worst,
                    input='diagnostics_v2/predictions/predictions_test.csv.gz', input_sha256=digest(source),
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    output_sha256={'bootstrap_test.csv': digest(output / 'bootstrap_test.csv')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'BOOTSTRAP_YAZILDI: {len(rows)} karsilastirma; {checked} nokta tahmini paketle eslesti '
          f'(en buyuk fark {worst:.1e}): {output}')


if __name__ == '__main__':
    main()
