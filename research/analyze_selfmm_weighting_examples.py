"""Paired validation error analysis for C-A and D-B; no model/data loading.

error_change = abs(weighted_prediction-label) - abs(plain_prediction-label).
Positive means harm; negative means improvement. Rankings are made separately
for each checkpoint and K0-K3 condition, without mixing validation and test.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PAIRS = {'baseline': ('A', 'C'), 'dropout': ('B', 'D')}
CONDITIONS = ('K0', 'K1', 'K2', 'K3')
NUMERIC = ('label', 'plain_prediction', 'weighted_prediction', 'y_text', 'y_audio',
           'y_vision', 'w_text', 'w_audio', 'w_vision', 'deviation_text',
           'deviation_audio', 'deviation_vision')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def analyze(run, epsilon):
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('status') == 'completed' and manifest.get('split') == 'valid',
            'Tamamlanmis VALIDATION deneyi bekleniyor')
    require(manifest.get('n') == 229 and manifest.get('protected_files_unchanged') is True,
            'Beklenmeyen ornek sayisi veya koruma kontrolu')
    required = {'checkpoint', 'condition', 'index', 'plain_method', 'weighted_method', *NUMERIC}
    groups = {(name, c): {} for name in PAIRS for c in CONDITIONS}
    source = read_csv(run / 'predictions_and_weights.csv')
    require(bool(source) and required <= source[0].keys(), 'Tahmin CSV sutunlari eksik')
    for row in source:
        key = row['checkpoint'], row['condition']
        require(key in groups, f'Bilinmeyen grup: {key}')
        require((row['plain_method'], row['weighted_method']) == PAIRS[key[0]], 'Yontem cifti yanlis')
        index = int(row['index'])
        require(index not in groups[key], f'Tekrarlanan indeks: {key}/{index}')
        for field in NUMERIC:
            row[field] = float(row[field])
            require(math.isfinite(row[field]), f'Gecersiz sayi: {key}/{index}/{field}')
        require(all(row['w_'+m] >= 0 for m in ('text', 'audio', 'vision'))
                and abs(sum(row['w_'+m] for m in ('text', 'audio', 'vision'))-3) < 1e-5,
                'Agirlik toplami/pozitifligi gecersiz')
        row['index'] = index
        groups[key][index] = row
    reference = groups['baseline', 'K0']
    require(set(reference) == set(range(229)), 'Validation indeksleri 0..228 olmali')
    for key, group in groups.items():
        require(group.keys() == reference.keys(), f'Ornekler eslesmiyor: {key}')
        require(all(group[i]['label'] == reference[i]['label'] for i in reference), f'Etiket farki: {key}')
    metric_rows = read_csv(run / 'metrics.csv')
    require(len(metric_rows) == 16, '16 metrik satiri bekleniyor')
    metric_lookup = {(r['method'], r['condition']): r for r in metric_rows}
    require(len(metric_lookup) == 16, 'Tekrarlanan metrik satiri')
    summaries, details = [], []
    for (name, condition), group in groups.items():
        plain, weighted = PAIRS[name]
        local = []
        for index in sorted(group):
            row = group[index]
            label, p, q = row['label'], row['plain_prediction'], row['weighted_prediction']
            old_error, new_error = abs(p-label), abs(q-label)
            delta = new_error-old_error
            status = 'harmed' if delta > epsilon else 'helped' if delta < -epsilon else 'unchanged'
            non0 = label != 0
            old_correct = (p > 0) == (label > 0)
            new_correct = (q > 0) == (label > 0)
            if not non0:
                transition = 'neutral_label_excluded'
            elif old_correct and not new_correct:
                transition = 'correct_to_wrong'
            elif not old_correct and new_correct:
                transition = 'wrong_to_correct'
            elif old_correct:
                transition = 'correct_to_correct'
            else:
                transition = 'wrong_to_wrong'
            local.append(dict(comparison=f'{weighted}-minus-{plain}', condition=condition,
                              index=index, **{k: row[k] for k in NUMERIC},
                              plain_abs_error=old_error, weighted_abs_error=new_error,
                              error_change=delta, status=status,
                              abs_prediction_shift=abs(q-p),
                              has0_class_flip=int((p >= 0) != (q >= 0)),
                              non0_class_flip=int(non0 and ((p > 0) != (q > 0))),
                              non0_transition=transition))
        n = len(local)
        mean_old = math.fsum(r['plain_abs_error'] for r in local)/n
        mean_new = math.fsum(r['weighted_abs_error'] for r in local)/n
        for method, calculated in ((plain, mean_old), (weighted, mean_new)):
            require(abs(float(metric_lookup[method, condition]['MAE_exact'])-calculated) < 1e-6,
                    f'MAE kaynak metrikle eslesmiyor: {method}/{condition}')
        summary = dict(comparison=f'{weighted}-minus-{plain}', condition=condition, n=n,
                       non0_n=sum(r['label'] != 0 for r in local),
                       harmed_n=sum(r['status'] == 'harmed' for r in local),
                       helped_n=sum(r['status'] == 'helped' for r in local),
                       unchanged_n=sum(r['status'] == 'unchanged' for r in local),
                       plain_MAE=mean_old, weighted_MAE=mean_new, mean_error_change=mean_new-mean_old,
                       max_harm=max(r['error_change'] for r in local),
                       min_error_change=min(r['error_change'] for r in local),
                       has0_class_flips=sum(r['has0_class_flip'] for r in local),
                       non0_class_flips=sum(r['non0_class_flip'] for r in local),
                       non0_correct_to_wrong=sum(r['non0_transition']=='correct_to_wrong' for r in local),
                       non0_wrong_to_correct=sum(r['non0_transition']=='wrong_to_correct' for r in local),
                       mean_abs_prediction_shift=math.fsum(r['abs_prediction_shift'] for r in local)/n)
        require(summary['harmed_n']+summary['helped_n']+summary['unchanged_n'] == n, 'Sayim hatasi')
        summaries.append(summary)
        details.extend(local)
    return summaries, details


def write_csv(path, rows, fields=None):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=Path('C:/Users/ilayda/YZM5110/runs/selfmm_abcd_valid_seed1111_tau1'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--check', action='store_true')
    cli = parser.parse_args()
    require(cli.top > 0 and math.isfinite(cli.epsilon) and cli.epsilon >= 0, 'Top/epsilon gecersiz')
    output = cli.output_dir or cli.run_dir / 'weighting_error_analysis'
    require(not output.exists(), f'Cikti klasoru zaten var; uzerine yazilmaz: {output}')
    paths = [cli.run_dir / name for name in ('manifest.json', 'metrics.csv', 'predictions_and_weights.csv')]
    hashes = {str(p.resolve()): sha(p) for p in paths}
    summaries, details = analyze(cli.run_dir, cli.epsilon)
    print('COMPARISON       KOSUL  N    ZARAR  FAYDA  AYNI  MAE_DEGISIMI')
    for r in summaries:
        print(f"{r['comparison']:<16} {r['condition']:<5} {r['n']:<4} {r['harmed_n']:<6} "
              f"{r['helped_n']:<6} {r['unchanged_n']:<5} {r['mean_error_change']:+.6f}")
    require(all(sha(Path(p)) == value for p, value in hashes.items()), 'Kaynak dosya degisti')
    if cli.check:
        print('KONTROL_BASARILI: 8 grup, 229 ornek/grup, etiketler, agirliklar, MAE eslesmesi; dosya yazilmadi')
        return
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'all_examples.csv', details)
    for comparison in ('C-minus-A', 'D-minus-B'):
        for condition in CONDITIONS:
            group = [r for r in details if r['comparison']==comparison and r['condition']==condition]
            harmed = sorted((r for r in group if r['status']=='harmed'), key=lambda r:(-r['error_change'],r['index']))[:cli.top]
            helped = sorted((r for r in group if r['status']=='helped'), key=lambda r:(r['error_change'],r['index']))[:cli.top]
            fields = ['rank', *details[0].keys()]
            for label, selected in (('harmed', harmed), ('helped', helped)):
                write_csv(output/f'{comparison}_{condition}_{label}_top{cli.top}.csv',
                          [dict(rank=i, **r) for i,r in enumerate(selected,1)], fields=fields)
    (output / 'analysis_manifest.json').write_text(json.dumps(dict(
        status='completed', split='valid', n_per_group=229, groups=8, top=cli.top,
        epsilon=cli.epsilon, protected_hashes=hashes, input_files_unchanged=True,
        criterion='abs(weighted-label)-abs(plain-label); positive=harm, negative=help',
        notes=['Ranks are separate for each comparison and condition.',
               'No model, raw dataset or test split is loaded.',
               'Index is validation dataset index, not an original video ID.',
               'No raw text is included. Example rankings are descriptive, not causal.',
               'Non0 excludes label=0 and uses prediction>0; Has0 uses prediction>=0.']),
        indent=2), encoding='utf-8')
    print('ORNEK_ANALIZI_TAMAMLANDI:', output)


if __name__ == '__main__':
    main()
