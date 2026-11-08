"""Build the report's tables from committed result files (standard library only).

Every number comes from a file in this repository; nothing is typed by hand
except the published Self-MM paper scores in configs/published_reference.json.
Output is one Markdown file: tables.md (Turkish, decimal commas) by default or
tables_en.md (English, decimal points) with --lang en. --collect-training copies the training-run resource records from runs/
into a small committed CSV once. --verify regenerates the tables in memory and
compares them with the stored files (both languages unless --lang is given).
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEEDS = (1111, 1112, 1113, 1114, 1115)
CONDITIONS = tuple(f'K{i}' for i in range(11))
METHODS = {'A': ('baseline', 'plain'), 'B': ('dropout', 'plain'), 'C': ('baseline', 'agree_tau1'),
           'D': ('dropout', 'agree_tau1')}
INPUTS = {
    'paper': 'research/configs/published_reference.json',
    'mmsa': 'results/result-stat.md',
    'diag_valid': 'research/diagnostics_v2/validation/summary_metrics.csv',
    'diag_test': 'research/diagnostics_v2/evaluation_test/summary_metrics.csv',
    'hyp_valid': 'research/diagnostics_v2/validation/hypotheses.csv',
    'hyp_test': 'research/diagnostics_v2/evaluation_test/hypotheses.csv',
    'weights_test': 'research/diagnostics_v2/evaluation_test/weight_summary_by_seed.csv',
    'heads_test': 'research/diagnostics_v2/evaluation_test/head_spread_by_seed.csv',
    'bootstrap': 'research/diagnostics_v2/bootstrap/bootstrap_test.csv',
    'selection': 'research/diagnostics_v2/validation/selection.json',
    'inference': 'research/results_package_v1/inference/summary.csv',
    'inference_paired': 'research/results_package_v1/inference/paired_summary.csv',
    'v1_test': 'research/results_package_v1/evaluation_test/summary_metrics.csv',
    'training': 'research/report_tables/training_resources.csv',
}
OUTPUTS = {'tr': ('tables.md', 'tables_manifest.json'), 'en': ('tables_en.md', 'tables_en_manifest.json')}
LANG = 'tr'
TRAINING_RUNS = {(1111, 'dropout'): 'selfmm_dropout_a20_v20_seed1111',
                 **{(s, 'baseline'): f'selfmm_baseline_seed{s}' for s in SEEDS[1:]},
                 **{(s, 'dropout'): f'selfmm_dropout_a20_v20_seed{s}' for s in SEEDS[1:]}}


def T(tr, en):
    return tr if LANG == 'tr' else en


def dec(text):
    return text.replace('.', ',') if LANG == 'tr' else text


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(path):
    # Git may check out text inputs (e.g. results/result-stat.md) with CRLF on Windows.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def read_csv(name):
    with (REPO / INPUTS[name]).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(x, digits=4, sign=False):
    text = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    return dec(text)


def pm(mean, sd, digits=4, scale=1.0):
    return f'{num(mean * scale, digits)} ± {num(sd * scale, digits)}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


# ----------------------------------------------------------------- sources

def mmsa_selfmm_mosi():
    text = (REPO / INPUTS['mmsa']).read_text(encoding='utf-8')
    section = text.split('## Regression', 1)[1].split('- MOSEI', 1)[0]
    header = [c.strip() for c in re.search(r'^\| Model .*$', section, re.M).group(0).strip('|').split('|')]
    row = [c.strip() for c in re.search(r'^\| self_mm .*$', section, re.M).group(0).strip('|').split('|')]
    values = dict(zip(header, row))
    return dict(MAE=float(values['MAE']) / 100, Corr=float(values['Corr']) / 100,
                Non0_acc_2_pct=float(values['Non0_acc_2']), Non0_F1_pct=float(values['Non0_F1_score']))


def summary(name):
    return {(r['checkpoint'], r['condition'], r['variant']): r for r in read_csv(name)}


def method_cell(rows, method, condition, metric='MAE', digits=4, scale=1.0):
    r = rows[METHODS[method][0], condition, METHODS[method][1]]
    return pm(float(r[f'{metric}_mean']), float(r[f'{metric}_sd']), digits, scale)


def bootstrap():
    return {(r['comparison'], r['condition']): r for r in read_csv('bootstrap')}


def ci(row, metric='MAE', digits=4):
    if metric == 'MAE':
        return f"{num(float(row['MAE_delta']), digits, True)} [{num(float(row['MAE_ci_low']), digits, True)}; {num(float(row['MAE_ci_high']), digits, True)}]"
    return (f"{num(float(row['Non0_acc_delta_pp']), 2, True)} [{num(float(row['Non0_acc_ci_low_pp']), 2, True)}; "
            f"{num(float(row['Non0_acc_ci_high_pp']), 2, True)}]")


# ------------------------------------------------------------------ tables

def build():
    paper = json.loads((REPO / INPUTS['paper']).read_text(encoding='utf-8'))['self_mm_paper']
    mmsa = mmsa_selfmm_mosi()
    test, valid = summary('diag_test'), summary('diag_valid')
    boot = bootstrap()
    out = [T('# Rapor tabloları', '# Report tables'), '',
           T('Bu dosya `research/make_selfmm_report_tables.py` ile üretilir; elle düzenlenmez.',
             'This file is generated by `research/make_selfmm_report_tables.py --lang en`; do not edit it by hand.'),
           T('Ondalık ayırıcı virgüldür. "±" beş eğitim seed’inin örneklem SS’sidir (ddof=1).',
             'The decimal separator is a point. "±" is the sample SD over the five training seeds (ddof=1).'),
           T('Köşeli parantez, 686 test örneği üzerinde eşleştirilmiş bootstrap %95 aralığıdır (2000 tekrar).',
             'Square brackets give the paired bootstrap 95% interval over the 686 test examples (2000 resamples).'), '']

    # T1 reproduction
    a = test['baseline', 'K0', 'plain']
    ours = dict(MAE=float(a['MAE_mean']), Corr=float(a['Corr_mean']), acc=float(a['Non0_acc_2_mean']) * 100,
                f1=float(a['Non0_F1_score_mean']) * 100)
    rows = [[T('Self-MM makalesi, Tablo 1 (yazarların bildirdiği)', 'Self-MM paper, Table 1 (reported by the authors)'), num(paper['MAE'], 3), num(paper['Non0_acc_2_pct'], 2),
             num(paper['Non0_F1_pct'], 2), num(paper['Corr'], 3)],
            [T('MMSA sonuç tablosu (result-stat.md)', 'MMSA results table (result-stat.md)'), num(mmsa['MAE'], 4), num(mmsa['Non0_acc_2_pct'], 2),
             num(mmsa['Non0_F1_pct'], 2), num(mmsa['Corr'], 4)],
            [T('Bu çalışma, A (ölçülen; 5 seed)', 'This work, A (measured; 5 seeds)'), pm(ours['MAE'], float(a['MAE_sd'])),
             pm(float(a['Non0_acc_2_mean']), float(a['Non0_acc_2_sd']), 2, 100),
             pm(float(a['Non0_F1_score_mean']), float(a['Non0_F1_score_sd']), 2, 100), pm(ours['Corr'], float(a['Corr_sd']))],
            [T('Fark: bu çalışma − makale', 'Difference: this work − paper'), num(ours['MAE'] - paper['MAE'], 4, True), num(ours['acc'] - paper['Non0_acc_2_pct'], 2, True),
             num(ours['f1'] - paper['Non0_F1_pct'], 2, True), num(ours['Corr'] - paper['Corr'], 4, True)]]
    out += [T('## Tablo 1. Yayımlanan ve ölçülen temiz test sonuçları (MOSI, K0)',
              '## Table 1. Published and measured clean test results (MOSI, K0)'), '',
            table([T('Kaynak', 'Source'), 'MAE ↓', 'Non0 Acc % ↑', 'Non0 F1 % ↑', 'Corr ↑'], rows), '',
            T('Makale satırı yazarların beş koşu ortalamasıdır; SS bildirilmediği için yazılmamıştır. '
              'MMSA tablosundaki MAE ve Corr 100 ölçeklidir ve 100’e bölünmüştür.',
              'The paper row is the authors’ mean over five runs; no SD was reported, so none is given. '
              'MAE and Corr in the MMSA table are on a 100 scale and were divided by 100.'), '']

    # T2/T3 A-D by condition
    for metric, title, digits, scale in (('MAE', 'MAE ↓', 4, 1.0), ('Non0_acc_2', 'Non0 accuracy % ↑', 2, 100.0)):
        rows = [[c] + [method_cell(test, m, c, metric, digits, scale) for m in 'ABCD'] for c in CONDITIONS]
        number = 2 if metric == 'MAE' else 3
        out += [T(f'## Tablo {number}. Koşullara göre A/B/C/D, test, {title}',
                  f'## Table {number}. A/B/C/D by condition, test, {title}'), '',
                table([T('Koşul', 'Condition'), 'A', 'B', 'C', 'D'], rows), '']
    v1 = {(r['method'], r['condition']): float(r['MAE_mean']) for r in read_csv('v1_test')}
    worst = max(abs(v1[m, c] - float(test[METHODS[m][0], c, METHODS[m][1]]['MAE_mean'])) for m in 'ABCD' for c in CONDITIONS[:8])
    require(worst <= 1e-6, f'v1 ve v2 test A/B/C/D farkli: {worst}')
    out += [T('K0–K7 değerleri v1 sonuç paketindekiyle aynıdır (en büyük MAE farkı ',
              'K0–K7 values are identical to the v1 result package (largest MAE difference ')
            + dec(f'{worst:.1e}')
            + T('); K8–K10 yalnız protokol v2’de ölçülmüştür. A: baseline, B: eğitimde ses/görüntü dropout’u, '
                'C/D: A/B checkpoint’ine τ=1 uyum ağırlığı.',
                '); K8–K10 were measured only in protocol v2. A: baseline, B: audio/visual dropout during training, '
                'C/D: τ=1 agreement weights on the A/B checkpoint.'), '']

    # T4 paired with bootstrap
    rows = []
    for name in ('B-minus-A', 'C-minus-A', 'D-minus-B'):
        for c in ('K0', 'K3', 'K7', 'K9', 'K10'):
            r = boot[name, c]
            rows.append([name.replace('-minus-', '−'), c, ci(r), num(float(r['MAE_seed_sd']), 4), f"{r['MAE_seeds_below_zero']}/5",
                         ci(r, 'acc')])
    out += [T('## Tablo 4. Eşleştirilmiş farklar ve belirsizlik, test', '## Table 4. Paired differences and uncertainty, test'), '',
            table(T(['Fark', 'Koşul', 'MAE farkı [%95 BA]', 'Seed SS', 'MAE’si düşen seed', 'Non0 Acc farkı yp [%95 BA]'],
                    ['Difference', 'Condition', 'MAE difference [95% CI]', 'Seed SD', 'Seeds with lower MAE',
                     'Non0 Acc difference pp [95% CI]']), rows), '',
            T('Negatif MAE farkı iyileşmedir. Seed SS eğitim değişkenliğini, bootstrap aralığı test örneklemesini yansıtır.',
              'A negative MAE difference is an improvement. The seed SD reflects training variability; the bootstrap '
              'interval reflects test sampling.'), '']

    # T5 hypotheses
    labels = T({'H1': 'Zarar ölçekleme miktarıyla artar (τ taraması)',
                'H2': 'Zararın ana kaynağı metin bloğunun ölçeklenmesidir',
                'H3': 'Uyum sinyali örneğe özgü yararlı bilgi taşır',
                'H4': 'Ses/görüntü bozulmasında iyileşme alanı küçüktür (oracle)',
                'H5a': 'Model metne baskın biçimde dayanır',
                'H5b': 'Kural, metin bozulduğunda yararlıdır'},
               {'H1': 'Harm grows with the amount of rescaling (τ sweep)',
                'H2': 'Rescaling the text block is the main source of harm',
                'H3': 'The agreement signal carries useful per-example information',
                'H4': 'Little room to improve under audio/visual corruption (oracle)',
                'H5a': 'The model relies mainly on text',
                'H5b': 'The rule helps when text is corrupted'})
    hyp_v = {r['code']: r for r in read_csv('hyp_valid')}
    hyp_t = {r['code']: r for r in read_csv('hyp_test')}

    def verdict(r):
        return T('Desteklendi', 'Supported') if r['supported'] == 'True' else T('Desteklenmedi', 'Not supported')
    rows = [[code, text, T('Desteklenir', 'Supported') if hyp_v[code]['expected'] == 'True' else T('Desteklenmez', 'Not supported'),
             verdict(hyp_v[code]),
             verdict(hyp_t[code])] for code, text in labels.items()]
    out += [T('## Tablo 5. Önceden kaydedilmiş hipotezler (protokol v2)', '## Table 5. Pre-registered hypotheses (protocol v2)'), '',
            table(T(['Kod', 'Hipotez', 'Önceden beklenti', 'Validation kararı', 'Test (tekrar)'],
                    ['Code', 'Hypothesis', 'Expected beforehand', 'Validation decision', 'Test (replication)']), rows), '',
            T('Kararlar validation’da verilmiştir; test satırı aynı kuralın testte tekrar edip etmediğini gösterir.',
              'Decisions were made on validation; the test column shows whether the same rule gives the same outcome '
              'on test.'), '']

    # T6 variants
    rows = []
    for name, label in (('C-minus-A', 'τ=1 (C) − plain'), ('tau16-minus-plain', 'τ=16 − plain'),
                        ('textfixed-minus-plain', T('Metin sabit, A/V ağırlıklı − plain', 'Text fixed, A/V weighted − plain')),
                        ('tau1-minus-shuffled', T('τ=1 − karıştırılmış τ=1', 'τ=1 − shuffled τ=1'))):
        for c in ('K0', 'K3', 'K10'):
            rows.append([label, c, ci(boot[name, c]), f"{boot[name, c]['MAE_seeds_below_zero']}/5"])
    for c in ('K1', 'K2', 'K3', 'K10'):
        rows.append(['Oracle − plain', c, ci(boot['oracle-minus-plain', c]),
                     f"{boot['oracle-minus-plain', c]['MAE_seeds_below_zero']}/5"])
    for c in ('K8', 'K9', 'K10'):
        rows.append([f'plain {c} − plain K0', c, ci(boot[f'plain-{c}-minus-K0', c]),
                     f"{boot[f'plain-{c}-minus-K0', c]['MAE_seeds_below_zero']}/5"])
    out += [T('## Tablo 6. Tanı varyantları, baseline checkpoint, test', '## Table 6. Diagnostic variants, baseline checkpoint, test'),
            '', table(T(['Karşılaştırma', 'Koşul', 'MAE farkı [%95 BA]', 'MAE’si düşen seed'],
                        ['Comparison', 'Condition', 'MAE difference [95% CI]', 'Seeds with lower MAE']), rows), '']

    # T7 tau curve
    taus =('0.25', '0.5', '1', '2', '4', '8', '16')
    rows = []
    for label, data in (('Validation', valid), ('Test', test)):
        def mean_over(variant):
            return statistics.fmean(float(data['baseline', c, variant]['MAE_mean']) for c in CONDITIONS[:8])
        rows.append([label, num(mean_over('plain'), 4)] + [num(mean_over(f'agree_tau{t}'), 4) for t in taus])
    selection = json.loads((REPO / INPUTS['selection']).read_text(encoding='utf-8'))
    out += [T('## Tablo 7. τ taraması, baseline, K0–K7 ortalama MAE', '## Table 7. τ sweep, baseline, mean MAE over K0–K7'), '',
            table([T('Bölme', 'Split'), 'plain'] + [f'τ={dec(t)}' for t in taus], rows), '',
            T(f"Validation’da K0–K10 ölçütüyle seçilen τ* = {num(selection['selected_tau'], 0)} (ızgaranın en büyük değeri).",
              f"τ* = {num(selection['selected_tau'], 0)} was selected on validation with the K0–K10 criterion "
              '(the largest value in the grid).'), '']

    # T8 resources
    inference = {r['method']: r for r in read_csv('inference')}
    paired = {r['comparison']: r for r in read_csv('inference_paired')}
    rows = [[m, pm(float(inference[m]['ms_per_sample_mean']), float(inference[m]['ms_per_sample_sd']), 3),
             num(float(inference[m]['samples_per_second_mean']), 1), num(float(inference[m]['peak_allocated_mib_mean']), 2),
             {'A': 1, 'B': 1, 'C': 2, 'D': 2}[m]] for m in 'ABCD']
    ratio = (f"C/A = {num(float(paired['C-vs-A']['latency_ratio_mean']), 3)} ± {num(float(paired['C-vs-A']['latency_ratio_sd']), 3)}; "
             f"D/B = {num(float(paired['D-vs-B']['latency_ratio_mean']), 3)} ± {num(float(paired['D-vs-B']['latency_ratio_sd']), 3)}; "
             + T('tepe allocated bellek farkı ', 'peak allocated memory difference ')
             + f"{num(float(paired['C-vs-A']['peak_allocated_mib_delta_mean']) * 1024, 1)} KiB.")
    out += [T('## Tablo 8. Çıkarım maliyeti (seed 1111, K0, test, batch 4, RTX 3050 Ti, 6 tur)',
              '## Table 8. Inference cost (seed 1111, K0, test, batch 4, RTX 3050 Ti, 6 rounds)'), '',
            table(T(['Yöntem', 'ms/örnek', 'örnek/sn', 'Tepe allocated MiB', 'Forward sayısı'],
                    ['Method', 'ms/example', 'examples/s', 'Peak allocated MiB', 'Forward passes']), rows), '', ratio,
            T('Batch 4 üzerinden amortize süredir; batch=1 gecikmesi değildir. Yükleme ve ısınma dahil değildir.',
              'Time is amortised over batches of 4; it is not batch-1 latency. Loading and warm-up are excluded.'), '']
    training = read_csv('training')
    rows = [[r['training_seed'], r['variant'], num(float(r['mmsa_total_seconds']), 1), num(float(r['peak_gpu_allocated_mib']), 1)]
            for r in training]
    by = {v: [float(r['mmsa_total_seconds']) for r in training if r['variant'] == v and r['training_seed'] != '1111']
          for v in ('baseline', 'dropout')}
    base_time = pm(statistics.fmean(by['baseline']), statistics.stdev(by['baseline']), 1)
    drop_time = pm(statistics.fmean(by['dropout']), statistics.stdev(by['dropout']), 1)
    out += [T('## Tablo 9. Eğitim süresi ve bellek (MMSA_run toplamı)', '## Table 9. Training time and memory (MMSA_run total)'), '',
            table(T(['Seed', 'Model', 'Toplam sn', 'Tepe allocated MiB'], ['Seed', 'Model', 'Total s', 'Peak allocated MiB']),
                  rows), '',
            T(f"Seed 1112–1115: baseline {base_time} sn, dropout {drop_time} sn (n=4). "
              'Seed 1111 baseline için süre kaydı yoktur. Süre yükleme, eğitim, validation, son test ve temizliği içerir; '
              'dropout modelleri farklı epoch’ta durduğu için fark yalnız dropout maliyeti değildir.',
              f"Seeds 1112–1115: baseline {base_time} s, dropout {drop_time} s (n=4). "
              'There is no time record for the seed 1111 baseline. Time covers loading, training, validation, the final '
              'test and cleanup; the dropout models stopped at different epochs, so the difference is not only the cost '
              'of dropout.'), '']

    # T10 mechanism
    heads = [r for r in read_csv('heads_test') if r['checkpoint'] == 'baseline' and r['condition'] == 'K0']
    weights = read_csv('weights_test')
    spread = {k: [float(r[f'y_{k}_sd']) for r in heads] for k in 'TAV'}
    names = T({'T': 'Metin', 'A': 'Ses', 'V': 'Görüntü'}, {'T': 'Text', 'A': 'Audio', 'V': 'Visual'})
    rows = [[names[k], f'{num(min(v), 3)} – {num(max(v), 3)}'] for k, v in spread.items()]
    out += [T('## Tablo 10. Mekanizma: tek modlu başlıkların örnekler arası yayılımı (baseline, K0, test)',
              '## Table 10. Mechanism: spread of unimodal head predictions across examples (baseline, K0, test)'), '',
            table(T(['Başlık', 'Tahmin SS aralığı (5 seed)'], ['Head', 'Prediction SD range (5 seeds)']), rows), '']
    rows = []
    for c in ('K0', 'K3', 'K8', 'K9', 'K10'):
        per_seed = [float(r['w_text_mean']) for r in weights
                    if r['checkpoint'] == 'baseline' and r['condition'] == c and r['variant'] == 'agree_tau1']
        rows.append([c, pm(statistics.fmean(per_seed), statistics.stdev(per_seed), 3),
                     f'{num(min(per_seed), 3)} – {num(max(per_seed), 3)}'])
    out += [T('## Tablo 11. τ=1 kuralında ortalama metin ağırlığı (baseline, test)',
              '## Table 11. Mean text weight under the τ=1 rule (baseline, test)'), '',
            table(T(['Koşul', 'Ortalama ± seed SS', 'Seed aralığı'], ['Condition', 'Mean ± seed SD', 'Seed range']), rows), '',
            T('Ağırlıkların toplamı 3’tür; eşit ağırlık 1’dir. Metin bozuldukça metin ağırlığı artar.',
              'The weights sum to 3; equal weight is 1. The text weight rises as the text is corrupted.'), '']
    return '\n'.join(out) + '\n'


# ------------------------------------------------------------------ actions

def collect_training(runs_dir):
    target = REPO / INPUTS['training']
    require(not target.exists(), f'Zaten var; uzerine yazilmaz: {target}')
    seed1111 = json.loads((HERE / 'configs/selfmm_portable_k0_dependencies.json').read_text(encoding='utf-8-sig'))
    rows = []
    for (seed, variant), name in sorted(TRAINING_RUNS.items()):
        path = runs_dir / name / 'experiment.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        require(record['status'] == 'completed' and record['seed'] == seed, f'Egitim kaydi uyumsuz: {name}')
        if 'variant' in record:
            require(record['variant'] == variant and record['batch_size'] == 4, f'Egitim kaydi uyumsuz: {name}')
            row = dict(seconds=record['mmsa_total_seconds'], allocated=record['peak_gpu_allocated_mib'],
                       reserved=record['peak_gpu_reserved_mib'], checkpoint=record['checkpoint_sha256'],
                       head=record['git_head'], schema='run_selfmm_seed.py')
        else:
            # The first dropout runner (seed 1111) used elapsed_seconds/peak_*_mb for the same MMSA_run scope.
            require((seed, variant) == (1111, 'dropout') and record['audio_probability'] == record['vision_probability'] == 0.2,
                    f'Beklenmeyen eski kayit: {name}')
            row = dict(seconds=record['elapsed_seconds'], allocated=record['peak_gpu_allocated_mb'],
                       reserved=record['peak_gpu_reserved_mb'], checkpoint=seed1111['checkpoint_sha256']['dropout'],
                       head='', schema='run_selfmm_dropout.py (checkpoint SHA256 from portable K0 registry)')
        rows.append(dict(training_seed=seed, variant=variant, run=name, mmsa_total_seconds=row['seconds'],
                         peak_gpu_allocated_mib=row['allocated'], peak_gpu_reserved_mib=row['reserved'], gpu=record['gpu'],
                         checkpoint_sha256=row['checkpoint'], git_head=row['head'], record_schema=row['schema'],
                         experiment_json_sha256=digest(path)))
    target.parent.mkdir(exist_ok=True)
    with target.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'EGITIM_KAYITLARI_YAZILDI: {len(rows)} kosu; seed 1111 baseline icin kayit yok: {target}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collect-training', type=Path, metavar='RUNS_DIR')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'report_tables')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--lang', choices=tuple(OUTPUTS), help='tr: tables.md (default when writing), en: tables_en.md; '
                        '--verify without --lang checks both')
    cli = parser.parse_args(argv)
    if cli.collect_training:
        collect_training(cli.collect_training.resolve())
        return
    global LANG
    if cli.verify:
        checked = []
        for LANG in ((cli.lang,) if cli.lang else tuple(OUTPUTS)):
            name, manifest_name = OUTPUTS[LANG]
            target = cli.output_dir / name
            require(target.read_bytes() == build().encode('utf-8'), f'{name} yeniden uretilemedi; kaynaklar degismis olabilir')
            stored = json.loads((cli.output_dir / manifest_name).read_text(encoding='utf-8'))
            require(all(text_digest(REPO / p) == stored['inputs'][p] for p in stored['inputs']),
                    f'Kaynak dosya ozeti farkli: {name}')
            checked.append(name)
        print(f"TABLOLAR_DOGRULANDI: {', '.join(checked)} byte olarak ayni; dosya yazilmadi")
        return
    LANG = cli.lang or 'tr'
    text = build().encode('utf-8')
    target = cli.output_dir / OUTPUTS[LANG][0]
    manifest_path = cli.output_dir / OUTPUTS[LANG][1]
    require(not target.exists(), f'Zaten var; uzerine yazilmaz: {target}')
    target.write_bytes(text)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    inputs={p: text_digest(REPO / p) for p in INPUTS.values()}, output_sha256=digest(target))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('TABLOLAR_YAZILDI:', target)


if __name__ == '__main__':
    main()
