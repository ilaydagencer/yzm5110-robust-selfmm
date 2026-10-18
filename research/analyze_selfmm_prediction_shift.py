"""Analyze frozen predictions; no torch, GPU, training or inference required."""
import argparse
import csv
import hashlib
import math
from pathlib import Path

MODELS = ('baseline', 'dropout')
CONDITIONS = ('K0', 'K1', 'K2', 'K3')
DEFAULT_INPUT = Path('C:/Users/ilayda/YZM5110/runs/selfmm_checkpoint_comparison_test_seed1111/predictions.csv')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_predictions(path):
    groups = {(m, c): {} for m in MODELS for c in CONDITIONS}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require({'model', 'condition', 'index', 'label', 'prediction'} <= set(reader.fieldnames or []),
                'CSV sutunlari eksik')
        for line, row in enumerate(reader, start=2):
            key = (row['model'], row['condition'])
            require(key in groups, f'Bilinmeyen model/kosul: satir {line}')
            index = int(row['index'])
            label, prediction = float(row['label']), float(row['prediction'])
            require(math.isfinite(label) and math.isfinite(prediction), f'Gecersiz sayi: satir {line}')
            require(index not in groups[key], f'Tekrarlanan indeks: {key}, {index}')
            groups[key][index] = (label, prediction)
    reference = groups[('baseline', 'K0')]
    require(bool(reference), 'CSV bos')
    require(set(reference) == set(range(len(reference))), 'Indeksler 0..n-1 olmali')
    for key, group in groups.items():
        require(group.keys() == reference.keys(), f'Eksik/fazla ornek: {key}')
        require(all(group[i][0] == reference[i][0] for i in reference), f'Etiket farki: {key}')
    return groups


def analyze(groups):
    summaries, details = [], []
    for model in MODELS:
        clean = groups[(model, 'K0')]
        for condition in CONDITIONS:
            local = []
            for index in sorted(clean):
                label, p0 = clean[index]
                _, p = groups[(model, condition)][index]
                # Match MMSA: Has0 uses >=0; Non0 excludes label=0 and uses >0.
                has0_flip = (p >= 0) != (p0 >= 0)
                non0_flip = label != 0 and ((p > 0) != (p0 > 0))
                old_correct = (p0 > 0) == (label > 0)
                new_correct = (p > 0) == (label > 0)
                local.append(dict(model=model, condition=condition, index=index, label=label,
                                  prediction_K0=p0, prediction=p, signed_change=p-p0,
                                  abs_change=abs(p-p0), has0_class_flip=int(has0_flip),
                                  non0_eligible=int(label != 0), non0_class_flip=int(non0_flip),
                                  non0_correct_to_wrong=int(label != 0 and old_correct and not new_correct),
                                  non0_wrong_to_correct=int(label != 0 and not old_correct and new_correct)))
            n = len(local)
            non0_n = sum(r['non0_eligible'] for r in local)
            has0_flips = sum(r['has0_class_flip'] for r in local)
            non0_flips = sum(r['non0_class_flip'] for r in local)
            maximum = max(r['abs_change'] for r in local)
            summaries.append(dict(model=model, condition=condition, n=n,
                                  mean_abs_change=math.fsum(r['abs_change'] for r in local)/n,
                                  max_abs_change=maximum,
                                  has0_class_flips=has0_flips, has0_flip_pct=100*has0_flips/n,
                                  non0_n=non0_n, non0_class_flips=non0_flips,
                                  non0_flip_pct=100*non0_flips/non0_n if non0_n else '',
                                  non0_correct_to_wrong=sum(r['non0_correct_to_wrong'] for r in local),
                                  non0_wrong_to_correct=sum(r['non0_wrong_to_correct'] for r in local)))
            details.extend(local)
    return summaries, details


def save_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    cli = parser.parse_args()
    output = cli.output_dir or cli.input.parent / 'prediction_shift_analysis'
    require(not output.exists(), f'Cikti klasoru zaten var; uzerine yazilmaz: {output}')
    raw_hash = hashlib.sha256(cli.input.read_bytes()).hexdigest()
    groups = load_predictions(cli.input)
    summary, details = analyze(groups)
    require(all(r['max_abs_change'] == 0 and r['has0_class_flips'] == 0
                for r in summary if r['condition'] == 'K0'), 'K0 kontrolu basarisiz')
    print('MODEL     KOSUL  N    ORT_MUTLAK       MAKS_MUTLAK      HAS0_FLIP  NON0_FLIP')
    for r in summary:
        print(f"{r['model']:<9} {r['condition']:<5} {r['n']:<4} "
              f"{r['mean_abs_change']:<16.9g} {r['max_abs_change']:<16.9g} "
              f"{r['has0_class_flips']:<10} {r['non0_class_flips']}")
    require(hashlib.sha256(cli.input.read_bytes()).hexdigest() == raw_hash, 'Girdi dosyasi degisti')
    if cli.check:
        print('KONTROL_BASARILI: eslesme, etiketler, sonlu sayilar ve K0; dosya yazilmadi')
        return
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / 'shift_summary.csv', summary)
    save_csv(output / 'shift_per_sample.csv', details)
    largest = []
    for model in MODELS:
        for condition in CONDITIONS[1:]:
            selected = [r for r in details if r['model'] == model and r['condition'] == condition]
            largest.extend(sorted(selected, key=lambda r: (-r['abs_change'], r['index']))[:10])
    save_csv(output / 'largest_changes_top10.csv', largest)
    (output / 'analysis_notes.txt').write_text(
        f'Input: {cli.input.resolve()}\nSHA256: {raw_hash}\n'
        'Change is relative to K0 of the SAME checkpoint.\n'
        'Mean/max use absolute prediction differences, not MAE differences.\n'
        'Has0: all labels, positive prediction >= 0.\n'
        'Non0: exclude label=0, positive prediction > 0.\n'
        'Class flips are binary; no 5/7-class rounding.\n'
        'Input preserved. No model inference or training.\n', encoding='utf-8')
    print('ANALIZ_TAMAMLANDI:', output)


if __name__ == '__main__':
    main()
