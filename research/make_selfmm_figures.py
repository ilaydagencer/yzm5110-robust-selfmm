"""Draw the report figures from committed result files (separate figure venv).

Inputs are only files in this repository: the protocol-v2 summary packages, the
bootstrap table, the compact test predictions and the v1 inference summary.
Run with the figure environment (research/requirements-figures.txt):
    ..\\.venv-figures\\Scripts\\python.exe -B research\\make_selfmm_figures.py
Writes research/figures/*.png (200 dpi, light background, for print and Word);
--lang en writes English labels to research/figures_en/ (same data and numbers).
Existing figure files are never overwritten unless --replace is given.
"""
import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DIAG = HERE / 'diagnostics_v2'
CONDITIONS = tuple(f'K{i}' for i in range(11))
SEEDS = (1111, 1112, 1113, 1114, 1115)
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
# Reference palette slots 1-2 (documented to pass the all-pairs CVD gates in light mode);
# the weighting is also encoded by line style / hatch, so identity is never colour alone.
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, INK_3, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SURFACE = '#ffffff'
SHOW_TITLES = False
TRANSPARENT = False
LANG = 'tr'
TEXT = {
    'tr': dict(text_corr='metin bozulması', av_corr='ses/görüntü bozulması', c_label='C: A + τ=1 ağırlık',
               d_label='D: B + τ=1 ağırlık', mae_axis='Test MAE (düşük iyi)', agree='uyum ağırlığı',
               plain='ağırlıksız (plain)', tau_axis='Sıcaklık τ (log ölçek; büyük τ = eşit ağırlığa yakın)',
               mean_mae='K0–K7 ortalama MAE', selected='validation’da seçilen τ* = {tau}',
               v_c='τ=1 kuralı (C) − ağırlıksız', v_shuf='τ=1 kuralı − karıştırılmış ağırlık',
               v_text='Metin sabit, yalnız ses/görüntü ağırlıklı − ağırlıksız', v_t16='τ=16 − ağırlıksız',
               v_oracle='Oracle (ses+görüntü yok, K3) − ağırlıksız', v_k10='τ=1 kuralı − ağırlıksız, metin yok (K10)',
               diff_axis='Test MAE farkı (pozitif = daha kötü)' + chr(10) + 'nokta: beş seed ortalaması; çizgi: %95 bootstrap aralığı',
               equal='eşit ağırlık', strength='Metin başlığının tahmin şiddeti |ŷ_T| (n = {n})',
               text_weight='τ=1 metin ağırlığı', panel_a='(a) Güçlü metin tahmini bastırılır (K0)',
               panel_b='(b) Metin bozuldukça ağırlığı artar', cond_axis='Koşul (K8–K10: metin bozulur)',
               ms_axis='ms / örnek (batch 4)'),
    'en': dict(text_corr='text corruption', av_corr='audio/visual corruption', c_label='C: A + τ=1 weights',
               d_label='D: B + τ=1 weights', mae_axis='Test MAE (lower is better)', agree='agreement weights',
               plain='unweighted (plain)', tau_axis='Temperature τ (log scale; large τ ≈ equal weights)',
               mean_mae='Mean MAE over K0–K7', selected='τ* = {tau} selected on validation',
               v_c='τ=1 rule (C) − unweighted', v_shuf='τ=1 rule − shuffled weights',
               v_text='Text fixed, audio/visual weighted only − unweighted', v_t16='τ=16 − unweighted',
               v_oracle='Oracle (no audio+visual, K3) − unweighted', v_k10='τ=1 rule − unweighted, no text (K10)',
               diff_axis='Test MAE difference (positive = worse)' + chr(10) + 'dot: five-seed mean; line: 95% bootstrap interval',
               equal='equal weight', strength='Strength of the text-head prediction |ŷ_T| (n = {n})',
               text_weight='τ=1 text weight', panel_a='(a) Strong text predictions are suppressed (K0)',
               panel_b='(b) Corrupted text gets more weight', cond_axis='Condition (K8–K10: text corrupted)',
               ms_axis='ms / sample (batch 4)'),
}


def t(key, **values):
    return TEXT[LANG][key].format(**values)
INPUTS = ('diagnostics_v2/validation/summary_metrics.csv', 'diagnostics_v2/evaluation_test/summary_metrics.csv',
          'diagnostics_v2/evaluation_test/weight_summary_by_seed.csv', 'diagnostics_v2/bootstrap/bootstrap_test.csv',
          'diagnostics_v2/predictions/predictions_test.csv.gz', 'diagnostics_v2/validation/selection.json',
          'results_package_v1/inference/summary.csv')


def read_csv(relative):
    with (HERE / relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summary(split):
    return {(r['checkpoint'], r['condition'], r['variant']): r for r in read_csv(f'diagnostics_v2/{split}/summary_metrics.csv')}


def comma(x, digits=2):
    text = f'{x:.{digits}f}'
    return text.replace('.', ',') if LANG == 'tr' else text


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.edgecolor': INK_3, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': True,
        'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 8, 'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.formatter.use_locale': False,
        'axes.axisbelow': True})


def decimal_axis(ax, axis='y', digits=2):
    formatter = matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, digits))
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(formatter)


def title(ax, text):
    # The report adds numbered captions in Word; titles inside the PNG are optional.
    if SHOW_TITLES:
        ax.set_title(text)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches='tight', metadata={'Software': None}, transparent=TRANSPARENT)
    plt.close(fig)


# ----------------------------------------------------------------- figures

def fig_conditions(out):
    test = summary('evaluation_test')
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.axvspan(7.5, 10.5, color='#f0efec', zorder=0)
    ax.text(9, 1.56, t('text_corr'), ha='center', va='top', color=INK_2, fontsize=8)
    ax.text(3.5, 1.56, t('av_corr'), ha='center', va='top', color=INK_2, fontsize=8)
    series = (('A', 'A: baseline', 'baseline', 'plain', BLUE, '-', 'o'),
              ('B', 'B: dropout', 'dropout', 'plain', ORANGE, '-', 's'),
              ('C', t('c_label'), 'baseline', 'agree_tau1', BLUE, '--', 'o'),
              ('D', t('d_label'), 'dropout', 'agree_tau1', ORANGE, '--', 's'))
    for short, label, checkpoint, variant, color, line, marker in series:
        mean = np.array([float(test[checkpoint, c, variant]['MAE_mean']) for c in CONDITIONS])
        sd = np.array([float(test[checkpoint, c, variant]['MAE_sd']) for c in CONDITIONS])
        ax.errorbar(x, mean, yerr=sd, color=color, linestyle=line, marker=marker, markersize=4.5, linewidth=1.5,
                    elinewidth=0.8, capsize=2, markerfacecolor=SURFACE if line == '--' else color, label=label, zorder=3)
        ax.annotate(short, (x[7], mean[7]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8,
                    color=INK, fontweight='bold')
    ax.set_xticks(x, CONDITIONS)
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(0.6, 1.58)
    decimal_axis(ax)
    ax.set_ylabel(t('mae_axis'))
    title(ax, 'Şekil 1. Koşullara göre A/B/C/D, beş seed ortalaması ± SS')
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0.01, 0.9), handlelength=2.6)
    save(fig, out / 'fig1_conditions_abcd.png')


def fig_tau(out):
    selection = json.loads((DIAG / 'validation/selection.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for split, label, color, marker in (('validation', 'Validation', BLUE, 'o'), ('evaluation_test', 'Test', ORANGE, 's')):
        rows = summary(split)
        curve = [statistics.fmean(float(rows['baseline', c, f'agree_tau{t:g}']['MAE_mean']) for c in CONDITIONS[:8])
                 for t in TAUS]
        plain = statistics.fmean(float(rows['baseline', c, 'plain']['MAE_mean']) for c in CONDITIONS[:8])
        ax.plot(TAUS, curve, color=color, marker=marker, markersize=4.5, linewidth=1.5, label=f"{label}: {t('agree')}")
        ax.axhline(plain, color=color, linestyle=':', linewidth=1.2, label=f"{label}: {t('plain')}")
    ax.set_xscale('log', base=2)
    ax.set_xticks(TAUS, [f'{tau:g}'.replace('.', ',') if LANG == 'tr' else f'{tau:g}' for tau in TAUS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    decimal_axis(ax)
    ax.set_xlabel(t('tau_axis'))
    ax.set_ylabel(t('mean_mae'))
    ax.annotate(t('selected', tau=f"{selection['selected_tau']:g}"), (16, 0.73), xytext=(-8, 30),
                textcoords='offset points', ha='right', fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle='-', color=INK_3, linewidth=0.8))
    title(ax, 'Şekil 2. Ağırlıklandırmanın dozu: τ büyüdükçe zarar kaybolur (baseline)')
    ax.legend(loc='upper right', ncol=2)
    save(fig, out / 'fig2_tau_sweep.png')


def fig_variants(out):
    boot = {(r['comparison'], r['condition']): r for r in read_csv('diagnostics_v2/bootstrap/bootstrap_test.csv')}
    items = (('C-minus-A', 'K0', t('v_c')), ('tau1-minus-shuffled', 'K0', t('v_shuf')),
             ('textfixed-minus-plain', 'K0', t('v_text')), ('tau16-minus-plain', 'K0', t('v_t16')),
             ('oracle-minus-plain', 'K3', t('v_oracle')), ('C-minus-A', 'K10', t('v_k10')))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(items))[::-1]
    for yi, (name, cond, label) in zip(y, items):
        r = boot[name, cond]
        point, low, high = float(r['MAE_delta']), float(r['MAE_ci_low']), float(r['MAE_ci_high'])
        ax.plot([low, high], [yi, yi], color=BLUE, linewidth=1.8, solid_capstyle='round')
        ax.plot(point, yi, 'o', color=BLUE, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0)
        text = f'{point:+.3f}'.replace('-0.000', '0.000').replace('+0.000', '0.000')
        text = text.replace('.', ',') if LANG == 'tr' else text
        ax.annotate(text, (high, yi), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8, color=INK)
    ax.axvline(0, color=INK_3, linewidth=0.9)
    ax.set_yticks(y, [i[2] for i in items])
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(-0.02, 0.18)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    decimal_axis(ax, 'x', 2)
    ax.set_xlabel(t('diff_axis'))
    title(ax, 'Şekil 3. Zarar metin ölçeklemesinden gelir; uyum sinyali rastgeleden iyi değil')
    save(fig, out / 'fig3_variants_bootstrap.png')


def fig_mechanism(out):
    with gzip.open(DIAG / 'predictions/predictions_test.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['checkpoint'] == 'baseline' and r['condition'] == 'K0']
    # The top bin pools |y_T| >= 2 (above 2.5 there are only a handful of examples).
    edges = (0, 0.5, 1.0, 1.5, 2.0, 99)
    labels = tuple(b if LANG == 'tr' else b.replace(',', '.') for b in ('0–0,5', '0,5–1', '1–1,5', '1,5–2', '≥2'))
    per_seed = {s: [[] for _ in labels] for s in SEEDS}
    for r in rows:
        strength = abs(float(r['y_T']))
        k = next(i for i in range(len(labels)) if edges[i] <= strength < edges[i + 1])
        per_seed[int(r['training_seed'])][k].append(float(r['w1_T']))
    counts = [sum(len(per_seed[s][k]) for s in SEEDS) for k in range(len(labels))]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(labels))
    means = np.array([[statistics.fmean(per_seed[s][k]) if per_seed[s][k] else np.nan for k in range(len(labels))]
                      for s in SEEDS])
    for row in means:
        left.plot(x, row, color=BLUE, alpha=0.25, linewidth=1.0)
    left.plot(x, np.nanmean(means, axis=0), color=BLUE, marker='o', markersize=4.5, linewidth=1.8)
    left.axhline(1, color=INK_3, linestyle=':', linewidth=1)
    left.text(len(labels) - 1, 1.01, t('equal'), ha='right', va='bottom', fontsize=7.5, color=INK_2)
    left.set_xticks(x, labels)
    left.set_xlabel(t('strength', n=', '.join(str(c) for c in counts)), fontsize=7.5)
    left.set_ylabel(t('text_weight'))
    decimal_axis(left)
    left.set_title(t('panel_a'), fontsize=9)
    weights = read_csv('diagnostics_v2/evaluation_test/weight_summary_by_seed.csv')
    conds = ('K0', 'K3', 'K8', 'K9', 'K10')
    values = [[float(r['w_text_mean']) for r in weights if r['checkpoint'] == 'baseline' and r['condition'] == c
               and r['variant'] == 'agree_tau1'] for c in conds]
    xr = np.arange(len(conds))
    for i, v in enumerate(values):
        right.plot([xr[i]] * len(v), v, 'o', color=BLUE, alpha=0.35, markersize=3.5)
    right.plot(xr, [statistics.fmean(v) for v in values], color=BLUE, marker='o', markersize=5, linewidth=1.8)
    right.axhline(1, color=INK_3, linestyle=':', linewidth=1)
    right.set_xticks(xr, conds)
    right.set_xlabel(t('cond_axis'))
    decimal_axis(right)
    right.set_title(t('panel_b'), fontsize=9)
    if SHOW_TITLES:
        fig.suptitle('Şekil 4. Uyum kuralının mekanizması (baseline, test; ince çizgi/nokta = seed)', x=0.01, ha='left',
                     fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout()
    save(fig, out / 'fig4_mechanism.png')


def fig_cost(out):
    inference = {r['method']: r for r in read_csv('results_package_v1/inference/summary.csv')}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    methods = ('A', 'B', 'C', 'D')
    colors = (BLUE, ORANGE, BLUE, ORANGE)
    hatches = ('', '', '////', '////')
    for i, (m, color, hatch) in enumerate(zip(methods, colors, hatches)):
        mean, sd = float(inference[m]['ms_per_sample_mean']), float(inference[m]['ms_per_sample_sd'])
        ax.bar(i, mean, width=0.62, color=color if not hatch else SURFACE, edgecolor=color, hatch=hatch, linewidth=1.2)
        ax.errorbar(i, mean, yerr=sd, color=INK, capsize=2, linewidth=0.8)
        ax.text(i, mean + 0.35, comma(mean, 2), ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(range(4), ['A\n1 forward', 'B\n1 forward', 'C\n2 forward', 'D\n2 forward'])
    ax.set_ylabel(t('ms_axis'))
    ax.set_ylim(0, 12)
    decimal_axis(ax, 'y', 0)
    title(ax, 'Şekil 5. Çıkarım maliyeti (seed 1111, K0, 6 tur)')
    save(fig, out / 'fig5_inference_cost.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='Default: research/figures (tr) or research/figures_en (en)')
    parser.add_argument('--lang', choices=tuple(TEXT), default='tr', help='Label language; numbers are identical')
    parser.add_argument('--replace', action='store_true', help='Overwrite existing figure files')
    parser.add_argument('--transparent', action='store_true', help='Transparent background, e.g. for slides')
    parser.add_argument('--with-titles', action='store_true', help='Draw titles inside the PNGs (report captions omit them)')
    cli = parser.parse_args(argv)
    global LANG
    LANG = cli.lang
    out = cli.output_dir or HERE / ('figures' if LANG == 'tr' else 'figures_en')
    if out.exists() and any(out.iterdir()) and not cli.replace:
        raise SystemExit(f'Figur klasoru dolu; yeniden cizmek icin --replace verin: {out}')
    out.mkdir(exist_ok=True)
    global SHOW_TITLES, TRANSPARENT
    SHOW_TITLES, TRANSPARENT = cli.with_titles, cli.transparent
    style()
    for draw in (fig_conditions, fig_tau, fig_variants, fig_mechanism, fig_cost):
        draw(out)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), matplotlib=matplotlib.__version__,
                    numpy=np.__version__, language=LANG, inputs={p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in INPUTS},
                    figures=sorted(p.name for p in out.glob('*.png')),
                    note='PNG bytes can differ across font/freetype builds; the inputs are hash-pinned.')
    (out / 'figures_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('FIGURLER_YAZILDI:', ', '.join(manifest['figures']))


if __name__ == '__main__':
    main()
