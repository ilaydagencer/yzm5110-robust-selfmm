"""Export a compact, lossless copy of the five protocol-v2 test predictions.

Reads runs/selfmm_diagnostics_v2_test_seed{1111..1115}, checks each run manifest
and output hash, and writes one gzip CSV with the columns the bootstrap and
figures need. Values are float32 in the runs; each is written as the shortest
decimal that parses back to the same float32, so nothing is rounded away.
The export recomputes every MAE from the written file and compares it with the
committed test summary package. Existing outputs are never overwritten.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

SEEDS = (1111, 1112, 1113, 1114, 1115)
KEYS = ('training_seed', 'checkpoint', 'condition', 'index')
VALUES = ('label', 'pred_plain', 'pred_agree_tau1', 'pred_agree_tau16', 'pred_agree_av_tau1',
          'pred_shuffled_tau1', 'pred_oracle', 'y_T', 'w1_T')
ROWS_PER_SEED = 2 * 11 * 686


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shortest_float32(text):
    if text == '':
        return ''
    value = np.float32(float(text))
    require(float(value) == float(text), f'float32 olmayan deger: {text}')
    out = np.format_float_positional(value, unique=True, trim='-')
    require(float(np.float32(out)) == float(text), 'float32 gidis-donus hatasi')
    return out


def read_package_mae(package):
    with (package / 'per_seed_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        return {(int(r['training_seed']), r['checkpoint'], r['condition'], r['variant']): float(r['MAE'])
                for r in csv.DictReader(stream)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--package-dir', type=Path, required=True, help='research/diagnostics_v2/evaluation_test')
    parser.add_argument('--output-dir', type=Path, required=True)
    cli = parser.parse_args(argv)
    output = cli.output_dir.absolute()
    require(not output.exists() and output.parent.is_dir(), f'Cikti var veya ust klasor yok: {output}')
    package = cli.package_dir.resolve()
    package_manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    require(package_manifest['split'] == 'test', 'Test ozet paketi gerekli')
    reference = read_package_mae(package)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(KEYS + VALUES)
    sums, sources = {}, {}
    for seed in SEEDS:
        run = cli.runs_dir.resolve() / f'selfmm_diagnostics_v2_test_seed{seed}'
        manifest_path = run / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        require(manifest['status'] == 'completed' and manifest['split'] == 'test' and manifest['training_seed'] == seed,
                f'Kosu tamamlanmamis: {run.name}')
        require(digest(manifest_path) == package_manifest['source_runs'][str(seed)]['manifest_sha256'],
                f'Kosu, test ozet paketindeki kosu degil: {run.name}')
        require(digest(run / 'predictions.csv') == manifest['output_hashes']['predictions.csv'], 'Tahmin dosyasi degismis')
        sources[str(seed)] = dict(directory=run.name, manifest_sha256=digest(manifest_path),
                                  predictions_sha256=manifest['output_hashes']['predictions.csv'])
        count = 0
        with (run / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                writer.writerow([row[k] for k in KEYS] + [shortest_float32(row[k]) for k in VALUES])
                label = float(row['label'])
                for column in VALUES[1:7]:
                    if row[column] != '':
                        key = seed, row['checkpoint'], row['condition'], column[len('pred_'):]
                        total, n = sums.get(key, (0.0, 0))
                        sums[key] = total + abs(float(np.float32(shortest_float32(row[column]))) - label), n + 1
                count += 1
        require(count == ROWS_PER_SEED, f'Seed {seed} satir sayisi {count}')
    worst = 0.0
    for key, (total, n) in sums.items():
        require(n == 686 and key in reference, f'Eksik metrik anahtari: {key}')
        worst = max(worst, abs(total / n - reference[key]))
    # Package MAE is a float32 torch mean; this is a float64 mean of the same values.
    require(worst <= 1e-6, f'Disa aktarilan tahminler paket MAE degerlerini uretmiyor: {worst}')
    data = gzip.compress(buffer.getvalue().encode('utf-8'), compresslevel=9, mtime=0)
    output.mkdir()
    (output / 'predictions_test.csv.gz').write_bytes(data)
    manifest = dict(protocol='selfmm-diagnostics-v2', split='test', training_seeds=list(SEEDS), rows=len(SEEDS) * ROWS_PER_SEED,
                    columns=list(KEYS + VALUES), float_format='shortest decimal that round-trips to the run float32 value',
                    pred_oracle_note='empty unless condition is K1, K2, K3 or K10',
                    mae_recomputed=len(sums), mae_max_abs_difference_vs_package=worst,
                    summary_package_manifest_sha256=digest(package / 'manifest.json'), source_runs=sources,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    output_sha256={'predictions_test.csv.gz': digest(output / 'predictions_test.csv.gz')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'TAHMINLER_DISA_AKTARILDI: {manifest["rows"]} satir; {len(data) / 1e6:.2f} MB; '
          f'{len(sums)} MAE paketle eslesti (en buyuk fark {worst:.1e})')


if __name__ == '__main__':
    main()
