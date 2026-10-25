"""Validation-first diagnostics for Self-MM agreement weighting (protocol v2).

Runs the frozen baseline/dropout checkpoints of one training seed on K0-K10:
the v1 audio/vision conditions plus [UNK] text corruption (K8-K10). Every
condition is scored with the plain fusion, the v1 tau=1 rule and the protocol-v2
controls: tau sweep, text-fixed, shuffled and oracle weights. Inference only.

--self-test runs CPU checks of the new functions and a dummy-model pipeline.
--check also verifies assets, registries and sources; it loads no model/dataset.
Test split requires --selection from the committed validation summary.
Existing output directories are never overwritten.
See research/selfmm_diagnostics_protocol_v2.md.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

PROTOCOL = 'selfmm-diagnostics-v2'
PROTOCOL_DOC = 'research/selfmm_diagnostics_protocol_v2.md'
SEEDS = (1111, 1112, 1113, 1114, 1115)
CORRUPTION_SEED = 1111
AV_CONDITIONS = tuple(f'K{i}' for i in range(8))
TEXT_RATES = {'K8': 0.3, 'K9': 0.6, 'K10': 1.0}
CONDITIONS = AV_CONDITIONS + tuple(TEXT_RATES)
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ORACLE_ZERO = {'K1': (1,), 'K2': (2,), 'K3': (1, 2), 'K10': (0,)}
UNK_ID, CLS_ID, SEP_ID = 100, 101, 102
SPLITS = {'valid': dict(n=229, non0_n=216, package='validation'),
          'test': dict(n=686, non0_n=656, package='evaluation_test')}
METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')
TOLERANCE = 1e-5
MODALITIES = ('text', 'audio', 'vision')


def tau_name(tau):
    return f'agree_tau{tau:g}'


VARIANTS = ('plain',) + tuple(tau_name(t) for t in TAUS) + ('agree_av_tau1', 'shuffled_tau1', 'oracle')
WEIGHTED_VARIANTS = VARIANTS[1:]
REFERENCE_METHODS = {('baseline', 'plain'): 'A', ('baseline', 'agree_tau1'): 'C',
                     ('dropout', 'plain'): 'B', ('dropout', 'agree_tau1'): 'D'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_stack(repo):
    """Import order is mandatory on this Windows environment: sentencepiece first."""
    global np, torch, core, helper, corruptions, seedmod
    import sentencepiece  # noqa: F401
    import numpy as np
    import torch
    import MMSA
    require(Path(MMSA.__file__).resolve() == (repo / 'src/MMSA/__init__.py').resolve(), 'Yanlis yerel MMSA kurulumu')
    research = repo / 'research'
    core = load_module(research / 'evaluate_selfmm_portable_k0.py', 'diagnostics_core')
    helper = load_module(research / 'evaluate_selfmm_abcd_valid.py', 'diagnostics_weighting')
    corruptions = load_module(research / 'selfmm_corruptions.py', 'diagnostics_corruptions')
    seedmod = load_module(research / 'evaluate_selfmm_portable_seed_test.py', 'diagnostics_seed_runner')
    core.np, core.torch = np, torch
    # The seed runner's train-statistics helpers read these module globals.
    seedmod.core, seedmod.corruptions = core, corruptions
    require(corruptions.CONDITIONS == AV_CONDITIONS, 'Bozulma modulu kosullari degismis')


# ---------------------------------------------------------------- new pieces

def rng_for(*parts):
    key = '|'.join([PROTOCOL, *map(str, parts)])
    value = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')
    return np.random.default_rng(value)


def corrupt_text(text, condition, index, split, seed=CORRUPTION_SEED):
    """text: [3, length] float array of BERT ids, attention mask and segment ids.

    Keeps [CLS], the last valid [SEP], the mask, segments and padding unchanged.
    One uniform draw per position makes K8 a subset of K9 and K9 of K10.
    """
    require(condition in CONDITIONS, f'Bilinmeyen kosul: {condition}')
    x = np.asarray(text)
    require(x.ndim == 2 and x.shape[0] == 3, 'Metin girdisi [3, uzunluk] olmali')
    ids, mask = x[0], x[1]
    length = int(mask.sum())
    require(2 <= length <= len(ids) and (mask[:length] == 1).all() and (mask[length:] == 0).all(),
            'Attention mask bitisik degil')
    require(ids[0] == CLS_ID and ids[length - 1] == SEP_ID, 'CLS/SEP yerinde degil')
    eligible = np.zeros(len(ids), dtype=bool)
    eligible[1:length - 1] = True
    chosen = np.zeros(len(ids), dtype=bool)
    if condition in TEXT_RATES:
        u = rng_for(seed, split, index, 'text', 'unk').random(len(ids))
        chosen = eligible & (u < TEXT_RATES[condition])
    out = x.copy()
    out[0, chosen] = UNK_ID
    return out, chosen, eligible


def shuffle_permutation(n, split, condition, seed=CORRUPTION_SEED):
    perm = rng_for(seed, split, condition, 'shuffle').permutation(n)
    require(not np.array_equal(perm, np.arange(n)), 'Permutasyon identity cikti')
    return perm


def text_fixed_weights(deviations, tau=1.0):
    """Text weight 1; audio/vision share 2 via the v1 rule restricted to their deviations."""
    weights = torch.ones_like(deviations)
    weights[:, 1:] = 2 * torch.softmax(-deviations[:, 1:] / tau, dim=1)
    return weights


def oracle_weights(n, condition, like):
    weights = torch.ones(n, 3, dtype=like.dtype, device=like.device)
    for column in ORACLE_ZERO[condition]:
        weights[:, column] = 0
    return weights


def safe_metrics(pred, truth):
    """Same formulas as the portable runner; Corr is NaN for a constant head output."""
    from sklearn.metrics import accuracy_score, f1_score
    p, y = pred.numpy(), truth.numpy()
    mask = y != 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        values = dict(MAE=float((pred - truth).abs().mean()),
                      Non0_acc_2=float(accuracy_score(y[mask] > 0, p[mask] > 0)),
                      Non0_F1_score=float(f1_score(y[mask] > 0, p[mask] > 0, average='weighted')),
                      Corr=float(np.corrcoef(p, y)[0, 1]) if float(np.std(p)) > 0 else float('nan'))
    return values


# ------------------------------------------------------------- per condition

def prepare_condition(loader, condition, split, stats, totals):
    """Corrupts every batch once. Fingerprint byte order matches the v1 K0-K7 runner."""
    fingerprint = hashlib.sha256()
    batches = []
    for batch in loader:
        prepared = dict(index=batch['index'].clone(), label=batch['labels']['M'].reshape(-1).clone())
        for modality in ('audio', 'vision'):
            original = batch[modality].numpy()
            saved = original.copy()
            av_condition = condition if condition in AV_CONDITIONS else 'K0'
            scale = np.asarray(stats['modalities'][modality]['std'], dtype=np.float64)
            summary = totals.setdefault(modality, dict(valid_frames=0, masked_frames=0, z_square=0., z_count=0))
            altered = []
            for j, index in enumerate(batch['index'].tolist()):
                length = int(batch[modality + '_lengths'][j])
                out, mask = corruptions.corrupt_modality(original[j], length, modality, av_condition, scale,
                                                         index, split=split, seed=CORRUPTION_SEED)
                require(np.array_equal(out[length:], original[j, length:], equal_nan=True), 'Padding degisti')
                summary['valid_frames'] += length
                summary['masked_frames'] += int(mask[:length].sum())
                if av_condition == 'K7' and (scale > 0).any():
                    positive = scale > 0
                    z = (out[:length, positive].astype(np.float64)
                         - original[j, :length][:, positive].astype(np.float64)) / scale[positive]
                    summary['z_square'] += float((z * z).sum())
                    summary['z_count'] += z.size
                altered.append(out)
            require(np.array_equal(original, saved, equal_nan=True), 'Kaynak batch yerinde degisti')
            values = np.stack(altered)
            fingerprint.update(values.tobytes())
            fingerprint.update(batch[modality + '_lengths'].numpy().tobytes())
            prepared[modality] = torch.from_numpy(values)
            prepared[modality + '_lengths'] = batch[modality + '_lengths'].clone()
        original = batch['text'].numpy()
        saved = original.copy()
        text_summary = totals.setdefault('text', dict(eligible_tokens=0, masked_tokens=0))
        altered, masked_counts = [], []
        for j, index in enumerate(batch['index'].tolist()):
            out, chosen, eligible = corrupt_text(original[j], condition, index, split)
            text_summary['eligible_tokens'] += int(eligible.sum())
            text_summary['masked_tokens'] += int(chosen.sum())
            masked_counts.append((int(eligible.sum()), int(chosen.sum())))
            altered.append(out)
        require(np.array_equal(original, saved), 'Kaynak metin yerinde degisti')
        values = np.stack(altered)
        fingerprint.update(values.tobytes())
        fingerprint.update(batch['index'].numpy().tobytes())
        prepared['text'] = torch.from_numpy(values)
        prepared['text_counts'] = masked_counts
        batches.append(prepared)
    return batches, fingerprint.hexdigest()


def evaluate_condition(model, layer, dims, batches, condition, split, n, device):
    """Two passes: (1) plain, identity control and per-sample variants; (2) shuffled weights."""
    hooks_before = len(layer._forward_pre_hooks)
    records = {}
    weights_by_variant = {v: {} for v in WEIGHTED_VARIANTS}
    max_identity = 0.0
    gpu_batches = []

    def run(prepared, weights=None):
        def forward():
            return model(prepared['text_dev'], (prepared['audio_dev'], prepared['audio_lengths']),
                         (prepared['vision_dev'], prepared['vision_lengths']))
        if weights is None:
            return forward()
        with helper.fusion_weights(layer, weights, dims):
            return forward()

    def check_heads(base, changed):
        for key in ('T', 'A', 'V'):
            require(torch.allclose(base[key], changed[key], atol=1e-6, rtol=1e-6), 'Tek modlu baslik degisti')

    with torch.inference_mode():
        for prepared in batches:
            require(all(not m.training for m in model.modules())
                    and all(not p.requires_grad for p in model.parameters()), 'Model eval/dondurulmus degil')
            prepared = dict(prepared, text_dev=prepared['text'].to(device), audio_dev=prepared['audio'].to(device),
                            vision_dev=prepared['vision'].to(device))
            saved = tuple(prepared[k].clone() for k in ('text_dev', 'audio_dev', 'vision_dev',
                                                       'audio_lengths', 'vision_lengths'))
            base = run(prepared)
            w1, deviations = helper.weights_from_outputs(base, 1.0)
            uniform = run(prepared, torch.ones_like(w1))
            for key in ('M', 'T', 'A', 'V'):
                max_identity = max(max_identity, float((base[key] - uniform[key]).abs().max()))
                require(torch.allclose(base[key], uniform[key], atol=1e-6, rtol=1e-6), 'Identity kontrolu basarisiz')
            outputs, batch_weights = {'plain': base['M']}, {}
            for tau in TAUS:
                batch_weights[tau_name(tau)], _ = helper.weights_from_outputs(base, tau)
            require(torch.equal(batch_weights['agree_tau1'], w1), 'tau=1 agirligi v1 kuraliyla ayni degil')
            batch_weights['agree_av_tau1'] = text_fixed_weights(deviations)
            if condition in ORACLE_ZERO:
                batch_weights['oracle'] = oracle_weights(len(w1), condition, w1)
            for variant, weights in batch_weights.items():
                changed = run(prepared, weights)
                check_heads(base, changed)
                outputs[variant] = changed['M']
            require(len(layer._forward_pre_hooks) == hooks_before, 'Hook kaldirilmadi')
            current = tuple(prepared[k] for k in ('text_dev', 'audio_dev', 'vision_dev',
                                                  'audio_lengths', 'vision_lengths'))
            require(all(torch.equal(a, b) for a, b in zip(current, saved)), 'Model girdiyi degistirdi')
            values = {k: v.reshape(-1).cpu() for k, v in outputs.items()}
            heads = {k: base[k].reshape(-1).cpu() for k in ('T', 'A', 'V')}
            d_cpu, w_cpu = deviations.cpu(), w1.cpu()
            require(all(bool(torch.isfinite(v).all()) for v in [*values.values(), *heads.values(), prepared['label']]),
                    'Gecersiz tahmin/etiket')
            for j, index in enumerate(prepared['index'].tolist()):
                require(index not in records, 'Tekrarlanan ornek')
                eligible, masked = prepared['text_counts'][j]
                record = dict(index=index, label=float(prepared['label'][j]), text_eligible=eligible,
                              text_masked=masked)
                for i, key in enumerate(('T', 'A', 'V')):
                    record[f'y_{key}'] = float(heads[key][j])
                    record[f'd_{key}'] = float(d_cpu[j, i])
                    record[f'w1_{key}'] = float(w_cpu[j, i])
                for variant, value in values.items():
                    record['pred_' + variant] = float(value[j])
                records[index] = record
                for variant, weights in batch_weights.items():
                    weights_by_variant[variant][index] = weights[j].cpu()
            prepared['heads_dev'] = {k: base[k] for k in ('T', 'A', 'V')}
            gpu_batches.append(prepared)
        require(sorted(records) == list(range(n)), 'Eksik/tekrar/sirasi farkli ornek')
        perm = shuffle_permutation(n, split, condition)
        all_w1 = torch.stack([weights_by_variant['agree_tau1'][i] for i in range(n)])
        shuffled = all_w1[torch.from_numpy(perm)]
        for prepared in gpu_batches:
            index = prepared['index']
            weights = shuffled[index].to(device)
            changed = run(prepared, weights)
            check_heads(prepared['heads_dev'], changed)
            for j, i in enumerate(index.tolist()):
                records[i]['pred_shuffled_tau1'] = float(changed['M'].reshape(-1)[j].cpu())
                weights_by_variant['shuffled_tau1'][i] = shuffled[i]
        require(len(layer._forward_pre_hooks) == hooks_before, 'Hook kaldirilmadi')
    ordered = [records[i] for i in range(n)]
    labels = torch.tensor([r['label'] for r in ordered], dtype=torch.float32)
    predictions = {}
    for variant in VARIANTS:
        if variant == 'oracle' and condition not in ORACLE_ZERO:
            continue
        predictions[variant] = torch.tensor([r['pred_' + variant] for r in ordered], dtype=torch.float32)
    heads = {f'head_{k}': torch.tensor([r[f'y_{k}'] for r in ordered], dtype=torch.float32) for k in ('T', 'A', 'V')}
    weights = {v: torch.stack([w[i] for i in range(n)]) for v, w in weights_by_variant.items() if w}
    return dict(records=ordered, labels=labels, predictions=predictions, heads=heads, weights=weights,
                max_identity=max_identity, permutation_sha256=hashlib.sha256(perm.astype('<i8').tobytes()).hexdigest())


def condition_rows(result, seed, split, checkpoint, condition, n, non0_n):
    rows = []
    for variant, pred in result['predictions'].items():
        measured = core.prediction_metrics(pred, result['labels'])
        rows.append(dict(training_seed=seed, split=split, checkpoint=checkpoint, condition=condition,
                         variant=variant, method=REFERENCE_METHODS.get((checkpoint, variant), ''),
                         n=n, non0_n=non0_n, **measured))
    for variant, pred in result['heads'].items():
        rows.append(dict(training_seed=seed, split=split, checkpoint=checkpoint, condition=condition,
                         variant=variant, method='', n=n, non0_n=non0_n, **safe_metrics(pred, result['labels'])))
    return rows


def weight_rows(result, seed, split, checkpoint, condition):
    rows = []
    for variant, w in result['weights'].items():
        row = dict(training_seed=seed, split=split, checkpoint=checkpoint, condition=condition, variant=variant)
        for i, modality in enumerate(MODALITIES):
            column = w[:, i].double()
            row.update({f'w_{modality}_mean': float(column.mean()), f'w_{modality}_sd': float(column.std(unbiased=False)),
                        f'w_{modality}_min': float(column.min()), f'w_{modality}_max': float(column.max())})
        rows.append(row)
    return rows


# --------------------------------------------------------------- self tests

def build_dummy_model():
    nn = torch.nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_proj = nn.Linear(1, 4)
            self.audio_proj = nn.Linear(5, 2)
            self.video_proj = nn.Linear(20, 2)
            self.post_fusion_layer_1 = nn.Linear(8, 4)
            self.fusion_out = nn.Linear(4, 1)
            self.heads = nn.ModuleList([nn.Linear(4, 1), nn.Linear(2, 1), nn.Linear(2, 1)])

        def forward(self, text, audio, video):
            (a, _), (v, _) = audio, video
            t = self.text_proj(text[:, 0, :].mean(1, keepdim=True) / 1000)
            a, v = self.audio_proj(a.mean(1)), self.video_proj(v.mean(1))
            fused = torch.relu(self.post_fusion_layer_1(torch.cat([t, a, v], dim=-1)))
            return dict(M=self.fusion_out(fused), T=self.heads[0](t), A=self.heads[1](a), V=self.heads[2](v))

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.Model = Inner()

        def forward(self, *args):
            return self.Model(*args)

    return Outer()


def dummy_loader(n, length=12, frames=6):
    batches = []
    generator = torch.Generator().manual_seed(1111)
    for start in range(0, n, 4):
        size = min(4, n - start)
        text = torch.zeros(size, 3, length)
        for j in range(size):
            valid = 4 + (start + j) % (length - 4)
            text[j, 0, 0], text[j, 0, valid - 1] = CLS_ID, SEP_ID
            text[j, 0, 1:valid - 1] = torch.randint(1000, 2000, (valid - 2,), generator=generator).float()
            text[j, 1, :valid] = 1
        audio_lengths = torch.tensor([2 + (start + j) % (frames - 1) for j in range(size)])
        vision_lengths = torch.tensor([1 + (start + j) % frames for j in range(size)])
        batches.append(dict(index=torch.arange(start, start + size), text=text,
                            audio=torch.randn(size, frames, 5, generator=generator),
                            vision=torch.randn(size, frames, 20, generator=generator),
                            audio_lengths=audio_lengths, vision_lengths=vision_lengths,
                            labels=dict(M=torch.randn(size, 1, generator=generator) * 2)))
    return batches


def self_test():
    with torch.random.fork_rng(devices=[]):
        _self_test()


def _self_test():
    helper.self_check()
    seedmod.synthetic_check()
    require(len(VARIANTS) == len(set(VARIANTS)) == 11 and 'agree_tau1' in VARIANTS
            and tau_name(0.25) == 'agree_tau0.25' and tau_name(16.0) == 'agree_tau16', 'Varyant adlari')
    # Text corruption.
    text = np.zeros((3, 40), dtype=np.float32)
    text[0, 0], text[0, 1:29], text[0, 29], text[1, :30] = CLS_ID, np.arange(1000, 1028), SEP_ID, 1
    saved = text.copy()
    masks = {}
    for condition in CONDITIONS:
        out, chosen, eligible = corrupt_text(text, condition, 7, 'valid')
        again, chosen2, _ = corrupt_text(text, condition, 7, 'valid')
        require(np.array_equal(out, again) and np.array_equal(chosen, chosen2), 'Metin seed tekrari')
        require(np.array_equal(out[1:], text[1:]) and out[0, 0] == CLS_ID and out[0, 29] == SEP_ID
                and (out[0, 30:] == 0).all(), 'Mask/segment/CLS/SEP/padding degisti')
        require(eligible.sum() == 28 and not (chosen & ~eligible).any(), 'Uygun token kumesi')
        require((out[0, chosen] == UNK_ID).all() and np.array_equal(out[0, ~chosen], text[0, ~chosen]), 'UNK uygulamasi')
        if condition in AV_CONDITIONS:
            require(not chosen.any() and np.array_equal(out, text), 'K0-K7 metni degistirdi')
        masks[condition] = chosen
    require(masks['K10'].sum() == 28 and (~masks['K8'] | masks['K9']).all()
            and (~masks['K9'] | masks['K10']).all(), 'Metin maskeleri ic ice degil')
    other, _, _ = corrupt_text(text, 'K9', 7, 'test')
    require(not np.array_equal(other, corrupt_text(text, 'K9', 7, 'valid')[0]), 'Valid/test metin RNG ayni')
    require(np.array_equal(text, saved), 'Kaynak metin degisti')
    rates = {c: [] for c in TEXT_RATES}
    for index in range(2000):
        for condition in TEXT_RATES:
            rates[condition].append(corrupt_text(text, condition, index, 'valid')[1].sum() / 28)
    for condition, rate in TEXT_RATES.items():
        require(abs(float(np.mean(rates[condition])) - rate) < 0.01, f'Metin oranı sapti: {condition}')
    # Permutation.
    perm = shuffle_permutation(229, 'valid', 'K0')
    require(sorted(perm.tolist()) == list(range(229)) and np.array_equal(perm, shuffle_permutation(229, 'valid', 'K0'))
            and not np.array_equal(perm, shuffle_permutation(229, 'valid', 'K1'))
            and not np.array_equal(perm, shuffle_permutation(229, 'test', 'K0')), 'Permutasyon')
    # Weights.
    d = torch.tensor([[0., 0., 0.], [3., 0., 1.]])
    w = text_fixed_weights(d)
    require(torch.allclose(w[0], torch.ones(3)) and torch.allclose(w.sum(1), torch.full((2,), 3.))
            and bool((w[:, 0] == 1).all()) and w[1, 1] > w[1, 2], 'Metin sabit agirlik')
    require(torch.equal(oracle_weights(2, 'K1', d), torch.tensor([[1., 0., 1.]] * 2))
            and torch.equal(oracle_weights(1, 'K3', d), torch.tensor([[1., 0., 0.]]))
            and torch.equal(oracle_weights(1, 'K10', d), torch.tensor([[0., 1., 1.]])), 'Oracle agirlik')
    outputs = dict(T=torch.tensor([[2.0], [0.]]), A=torch.tensor([[-0.3], [0.]]), V=torch.tensor([[-0.4], [0.]]))
    spread = [float((helper.weights_from_outputs(outputs, tau)[0] - 1).abs().max()) for tau in TAUS]
    require(all(a > b for a, b in zip(spread, spread[1:])) and spread[-1] < 0.2, 'Tau buyudukce agirlik 1e yaklasmali')
    # Pipeline with a dummy model on CPU.
    torch.manual_seed(1111)
    model = build_dummy_model().eval().requires_grad_(False)
    layer, dims = model.Model.post_fusion_layer_1, (4, 2, 2)
    stats = dict(modalities={'audio': dict(std=[1.] * 5), 'vision': dict(std=[0.] + [1.] * 19)})
    loader = dummy_loader(9)
    fingerprints, results = {}, {}
    for condition in ('K0', 'K3', 'K7', 'K9', 'K10'):
        totals = {}
        batches, fingerprints[condition] = prepare_condition(loader, condition, 'valid', stats, totals)
        again, repeat = prepare_condition(loader, condition, 'valid', stats, {})
        require(repeat == fingerprints[condition], 'Girdi parmak izi tekrarlanamadi')
        results[condition] = result = evaluate_condition(model, layer, dims, batches, condition, 'valid', 9, 'cpu')
        require(set(result['predictions']) == set(VARIANTS) - ({'oracle'} if condition not in ORACLE_ZERO else set()),
                'Varyant kumesi')
        require(result['max_identity'] == 0.0, 'Dummy identity')
        rows = condition_rows(result, 1111, 'valid', 'baseline', condition, 9, 9)
        require(len(rows) == len(result['predictions']) + 3, 'Metrik satiri sayisi')
        require(len(weight_rows(result, 1111, 'valid', 'baseline', condition)) == len(result['weights']), 'Agirlik satiri')
        w_shuffled, w1 = result['weights']['shuffled_tau1'], result['weights']['agree_tau1']
        perm = shuffle_permutation(9, 'valid', condition)
        require(torch.equal(w_shuffled, w1[torch.from_numpy(perm)]), 'Karistirilmis agirlik permutasyonu')
        if condition == 'K10':
            require(totals['text']['masked_tokens'] == totals['text']['eligible_tokens'] > 0, 'K10 tum metni maskelemeli')
        if condition == 'K3':
            require(totals['audio']['masked_frames'] == totals['audio']['valid_frames'], 'K3 sesi silmeli')
    require(len(set(fingerprints.values())) == len(fingerprints), 'Kosullar ayni girdiyi uretti')
    k0 = results['K0']['records']
    gap = {t: np.mean([abs(r['pred_plain'] - r['pred_' + tau_name(t)]) for r in k0]) for t in (0.25, 16.0)}
    require(gap[16.0] < gap[0.25], 'Dummy: buyuk tau plain tahmine daha yakin olmali')
    # Oracle on K3 equals manually zeroed audio/vision blocks.
    batch = prepare_condition(loader, 'K3', 'valid', stats, {})[0][0]
    with torch.inference_mode():
        t = model.Model.text_proj(batch['text'][:, 0, :].mean(1, keepdim=True) / 1000)
        manual = model.Model.fusion_out(torch.relu(layer(torch.cat([t, torch.zeros(len(t), 4)], dim=-1)))).reshape(-1)
    got = torch.tensor([results['K3']['records'][i]['pred_oracle'] for i in batch['index'].tolist()])
    require(torch.allclose(manual, got, atol=1e-6), 'Oracle K3 elle hesaplanan sonucla ayni degil')
    print('TANI_OZTESTLERI_BASARILI: metin [UNK], ic ice maske, permutasyon, agirlik varyantlari, dummy model hatti')


# ---------------------------------------------------------- real run set-up

def checkpoint_registry(repo):
    k0 = core.load_json(repo / 'research/configs/selfmm_portable_k0_dependencies.json')
    frozen = core.load_json(repo / 'research/configs/selfmm_portable_seed_dependencies.json')
    require(k0['training_seed'] == 1111 and k0['batch_size'] == 4 and k0['tau'] == 1, 'Seed 1111 kaydi uyumsuz')
    require(k0['effective_config_sha256'] == frozen['effective_config_sha256'] and k0['data_sha256'] == frozen['data_sha256']
            and k0['versions'] == frozen['versions'], 'Iki checkpoint kaydi ayni ortami tanimlamiyor')
    registry = {1111: k0['checkpoint_sha256']}
    registry.update({int(seed): seedmod.select_seed(frozen, int(seed))['checkpoint_sha256'] for seed in frozen['seeds']})
    require(set(registry) == set(SEEDS), 'Bes egitim seedi icin checkpoint kaydi gerekli')
    return registry, frozen


def read_reference(package, split, seed):
    info = SPLITS[split]
    path = package / info['package'] / 'per_seed_metrics.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['training_seed'] == str(seed)]
    reference = {}
    for row in rows:
        key = row['method'], row['condition']
        require(key not in reference, 'Tekrarlanan referans')
        require(int(row['n']) == info['n'] and int(row['non0_n']) == info['non0_n'], 'Referans ornek sayisi')
        reference[key] = {k: float(row[k]) for k in METRICS}
    require(set(reference) == {(m, c) for m in 'ABCD' for c in AV_CONDITIONS}, '32 referans satiri gerekli')
    return reference


def read_selection(path, split):
    if split == 'valid':
        require(path is None, 'Validation calismasi --selection almaz')
        return None
    require(path is not None, 'Test icin --selection (commit edilmis validation ozeti) gerekli')
    selection = core.load_json(path)
    require(selection.get('protocol') == PROTOCOL and selection.get('split') == 'valid'
            and selection.get('selected_tau') in TAUS and sorted(selection.get('training_seeds', [])) == list(SEEDS),
            'Secim dosyasi protokol v2 validation ozeti degil')
    return selection


def prepare(cli):
    repo = cli.repo_dir.resolve()
    load_stack(repo)
    split, seed = cli.split, cli.seed
    package = (cli.package_dir or repo / 'research/results_package_v1').resolve()
    config = (cli.config or repo / 'research/configs/selfmm_mosi_portable_k0.json').resolve()
    protocol_path = repo / 'research/configs/selfmm_portable_k0_k7_dependencies.json'
    data, bert = cli.data_file.resolve(), cli.bert_dir.resolve()
    checkpoints = dict(baseline=cli.baseline_checkpoint.resolve(), dropout=cli.dropout_checkpoint.resolve())
    output = cli.output_dir.absolute()
    core.check_new_output(output, repo, package, bert, checkpoints)
    selection_path = cli.selection.resolve() if cli.selection else None
    selection = read_selection(selection_path, split)
    registry, frozen = checkpoint_registry(repo)
    for name, expected in frozen['source_sha256_lf'].items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and path.is_file(), f'Kaynak dosyasi eksik: {name}')
        require(source_digest(path) == expected, f'Kaynak referanstan farkli: {name}')
    require(source_digest(protocol_path) == frozen['protocol_sha256_lf'], 'Ortak K0-K7 protokolu degismis')
    protocol = core.load_json(protocol_path)
    protocol_doc = repo / PROTOCOL_DOC
    require(protocol_doc.is_file(), f'Protokol belgesi eksik: {PROTOCOL_DOC}')
    settings = core.effective_config(core.load_json(config))
    require(core.config_digest(settings) == frozen['effective_config_sha256'], 'Model/config referanstan farkli')
    require(core.digest(package / 'manifest.json') == frozen['package_manifest_sha256'], 'Sonuc paketi kimligi farkli')
    verifier_path = repo / 'research/verify_results_package.py'
    load_module(verifier_path, 'diagnostics_package_verifier').verify(package)
    reference = read_reference(package, split, seed)
    print(f'SECIM: egitim seed={seed}; bolme={split}; bozulma seed={CORRUPTION_SEED}; kosullar=K0-K10', flush=True)
    print('ASSET_HASH_KONTROLU: veri ve iki checkpoint bayt olarak okunuyor; model/veri yuklenmez', flush=True)
    require(data.is_file() and core.digest(data) == frozen['data_sha256'], 'Veri eksik veya SHA256 farkli')
    for name, path in checkpoints.items():
        require(path.is_file() and core.digest(path) == registry[seed][name],
                f'Seed {seed} checkpoint eksik/farkli: {name}; dogru seed ve varyant yolunu kullanin')
    bert_files = core.check_bert(bert)
    import transformers
    from MMSA.config import get_config_regression
    from MMSA.data_loader import MMDataset
    from MMSA.models import AMIO
    from MMSA.utils import setup_seed
    from torch.utils.data import DataLoader
    versions = dict(torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__)
    require(versions == frozen['versions'], f'Kutuphane surumleri farkli: {versions}')
    args = get_config_regression('self_mm', 'mosi', str(config))
    imported = dict(args)
    imported.pop('featurePath')
    require(imported == settings, 'MMSA config birlestirmesi farkli')
    args.update(featurePath=str(data), pretrained=str(bert), train_mode='regression', device=torch.device('cuda:0'),
                custom_feature=None, feature_T=None, feature_A=None, feature_V=None, data_missing=False)
    core.MMDataset, core.AMIO, core.setup_seed, core.DataLoader = MMDataset, AMIO, setup_seed, DataLoader
    self_test()
    seedmod.statistics_checks({'modalities': protocol['expected_train_statistics']}, protocol)
    research = repo / 'research'
    own = [Path(__file__).resolve(), protocol_doc, research / 'evaluate_selfmm_portable_seed_test.py',
           research / 'configs/selfmm_portable_k0_dependencies.json']
    sources = [(repo / name).resolve() for name in frozen['source_sha256_lf']]
    protected = set(sources + own + bert_files + [config, protocol_path, data, *checkpoints.values(),
                    research / 'configs/selfmm_portable_seed_dependencies.json',
                    repo / 'src/MMSA/config/config_regression.json'])
    if selection_path:
        protected.add(selection_path)
    protected.update(p for p in package.rglob('*') if p.is_file())
    hashes = {path: core.digest(path) for path in sorted(protected)}
    plan = dict(repo=repo, package=package, output=output.resolve(), data=data, bert=bert, checkpoints=checkpoints,
                split=split, seed=seed, args=args, settings=settings, reference=reference, protocol=protocol,
                frozen=frozen, checkpoint_sha256=registry[seed], selection=selection, hashes=hashes, versions=versions,
                protocol_doc_sha256_lf=source_digest(protocol_doc), script_sha256_lf=source_digest(Path(__file__)),
                cuda_available=torch.cuda.is_available())
    print('BAGIMLILIKLER_DOGRULANDI: config, kaynak, paket, checkpointler, veri, yerel BERT, protokol belgesi')
    print(f'KAPSAM: seed{seed}/{split}; 2 checkpoint x 11 kosul x {len(VARIANTS)} varyant; 128 referans kontrolu')
    print('CUDA:', plan['cuda_available'], 'CIKTI:', plan['output'])
    return plan


# ------------------------------------------------------------------ run

def evaluate(plan):
    require(plan['cuda_available'], 'Gercek calisma icin CUDA gerekli; --check CPU ile yapilabilir')
    output, split, seed = plan['output'], plan['split'], plan['seed']
    info = SPLITS[split]
    n, non0_n = info['n'], info['non0_n']
    core.check_new_output(output, plan['repo'], plan['package'], plan['bert'], plan['checkpoints'])
    require(not core.unchanged(plan['hashes']), 'Kontrolden sonra bir kaynak degisti')
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    manifest = dict(status='running', protocol=PROTOCOL, protocol_doc_sha256_lf=plan['protocol_doc_sha256_lf'],
        script_sha256_lf=plan['script_sha256_lf'], training_seed=seed, corruption_seed=CORRUPTION_SEED, split=split,
        conditions=list(CONDITIONS), text_rates=TEXT_RATES, taus=list(TAUS), variants=list(VARIANTS),
        oracle_zero_columns={k: list(v) for k, v in ORACLE_ZERO.items()}, n=n, non0_n=non0_n, batch_size=4,
        inference_only=True, training_performed=False, offline=True, versions=plan['versions'],
        python_version=platform.python_version(), config_settings=plan['settings'],
        checkpoint_sha256=plan['checkpoint_sha256'], selection=plan['selection'],
        inputs=core.manifest_inputs(plan['hashes'], output),
        input_path_base='directory containing this manifest; different-drive paths may be absolute',
        data_loading_note='Pickle contains all splits. Only TRAIN arrays compute noise scales; only the chosen split is evaluated.',
        timing_scope='total run including statistics, loading and controls; not a latency benchmark',
        created_at_utc=datetime.now(timezone.utc).isoformat())

    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    metrics, checks, controls, predictions, weights, inputs = [], [], [], [], [], []
    fingerprints = {}
    try:
        print('TRAIN_STD_HESAPLANIYOR: yalniz train; gercek zaman adimlari; ddof=0', flush=True)
        stats = seedmod.load_train_statistics(plan['data'])
        stats['source_sha256'] = plan['frozen']['data_sha256']
        stats_rows = seedmod.statistics_checks(stats, plan['protocol'])
        (output / 'train_feature_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        core.write_csv(output / 'train_stats_checks.csv', stats_rows)
        require(all(r['passed'] for r in stats_rows), 'Train istatistik referansi eslesmedi')
        torch.cuda.set_device(0)
        manifest.update(gpu=torch.cuda.get_device_name(0), dtype='float32', autocast=False)
        dataset_args = copy.deepcopy(plan['args'])
        dataset = core.MMDataset(dataset_args, mode=split)
        require(len(dataset) == n and dataset_args.feature_dims == [768, 5, 20], 'Bolme boyutlari uyumsuz')
        require(int(np.count_nonzero(dataset.labels['M'])) == non0_n, 'Non0 sayisi uyumsuz')
        label_sha = seedmod.label_digest(dataset.labels['M'])
        if split == 'test':
            require(label_sha == plan['protocol']['expected_test_labels_float32_sha256'], 'Test etiketi/sirasi uyumsuz')
        manifest['labels_float32_sha256'] = label_sha
        loader = core.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
        for checkpoint in ('baseline', 'dropout'):
            core.setup_seed(seed)
            args = copy.deepcopy(plan['args'])
            model = core.AMIO(args)
            state = torch.load(plan['checkpoints'][checkpoint], map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            layer = model.Model.post_fusion_layer_1
            dims = (args.text_out, args.audio_out, args.video_out)
            for condition in CONDITIONS:
                tick = time.perf_counter()
                totals = {}
                batches, input_sha = prepare_condition(loader, condition, split, stats, totals)
                if checkpoint == 'baseline':
                    fingerprints[condition] = input_sha
                shared = input_sha == fingerprints[condition]
                require(shared, f'Checkpointler ayni bozuk girdiyi almadi: {condition}')
                result = evaluate_condition(model, layer, dims, batches, condition, split, n, args.device)
                rows = condition_rows(result, seed, split, checkpoint, condition, n, non0_n)
                metrics.extend(rows)
                weights.extend(weight_rows(result, seed, split, checkpoint, condition))
                for record in result['records']:
                    predictions.append(dict(training_seed=seed, split=split, checkpoint=checkpoint,
                                            condition=condition, **record))
                if condition in AV_CONDITIONS:
                    for row in rows:
                        method = REFERENCE_METHODS.get((checkpoint, row['variant']))
                        if method:
                            for check in core.metric_checks(method, row, plan['reference'][method, condition], TOLERANCE):
                                checks.append(dict(training_seed=seed, split=split, condition=condition, **check))
                reference_sha = (plan['protocol']['expected_test_input_sha256'].get(condition)
                                 if split == 'test' and condition in AV_CONDITIONS else '')
                controls.append(dict(training_seed=seed, split=split, checkpoint=checkpoint, condition=condition, n=n,
                    max_uniform_error=result['max_identity'], hooks_restored=True, heads_unchanged=True,
                    inputs_unchanged=True, shared_inputs=shared, input_sha256=input_sha,
                    reference_input_sha256=reference_sha or '',
                    reference_input_matched='' if not reference_sha else input_sha == reference_sha,
                    shuffle_permutation_sha256=result['permutation_sha256']))
                if checkpoint == 'baseline':
                    row = dict(training_seed=seed, split=split, condition=condition, n=n, input_sha256=input_sha)
                    for modality in ('audio', 'vision'):
                        total = totals[modality]
                        row[f'{modality}_masked_rate'] = total['masked_frames'] / total['valid_frames']
                        row[f'{modality}_noise_rms_div_train_std'] = (
                            float(np.sqrt(total['z_square'] / total['z_count'])) if total['z_count'] else 0.)
                    row.update(text_eligible_tokens=totals['text']['eligible_tokens'],
                               text_masked_tokens=totals['text']['masked_tokens'],
                               text_masked_rate=totals['text']['masked_tokens'] / totals['text']['eligible_tokens'],
                               target_text_rate=TEXT_RATES.get(condition, 0.))
                    inputs.append(row)
                plain = next(r for r in rows if r['variant'] == 'plain')
                tau1 = next(r for r in rows if r['variant'] == 'agree_tau1')
                print(f'{checkpoint}/{condition}: plain MAE={plain["MAE"]:.4f}; tau1 MAE={tau1["MAE"]:.4f}; '
                      f'{time.perf_counter() - tick:.1f} sn', flush=True)
            del model, layer
            torch.cuda.empty_cache()
        expected_metrics = 2 * (len(CONDITIONS) * (len(VARIANTS) - 1 + 3) + len(ORACLE_ZERO))
        require(len(metrics) == expected_metrics and len(checks) == 128 and len(controls) == 22
                and len(predictions) == 2 * len(CONDITIONS) * n and len(inputs) == len(CONDITIONS),
                'Cikti satir sayisi uyumsuz')
        tables = {'metrics.csv': metrics, 'reference_checks.csv': checks, 'controls.csv': controls,
                  'predictions.csv': predictions, 'weight_summary.csv': weights, 'input_summary.csv': inputs}
        for filename, records in tables.items():
            columns = list(dict.fromkeys(k for r in records for k in r))
            core.write_csv(output / filename, [{c: r.get(c, '') for c in columns} for r in records])
        matched = all(r['passed'] for r in checks)
        inputs_matched = all(r['reference_input_matched'] is not False for r in controls)
        outputs = list(tables) + ['train_feature_stats.json', 'train_stats_checks.csv']
        manifest.update(metric_rows=len(metrics), reference_check_rows=len(checks), prediction_rows=len(predictions),
            reference_metrics_matched=matched, reference_inputs_matched=inputs_matched,
            shared_inputs_verified=True, hooks_restored=True, heads_unchanged=True, inputs_unchanged=True,
            output_hashes={name: core.digest(output / name) for name in outputs})
        require(matched and inputs_matched,
                'Referans metrik/girdi eslesmedi; reference_checks.csv ve controls.csv incelenmeli; toleransi degistirmeyin')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        changed = core.unchanged(plan['hashes'])
        manifest.update(changed_inputs=changed, protected_files_unchanged=not changed,
                        finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        total_run_seconds=time.perf_counter() - start)
        if changed:
            manifest['status'] = 'integrity_failed'
        save_manifest()
    require(manifest['status'] == 'completed', 'Calisma/butunluk kontrolu basarisiz')
    print(f'TANI_V2_TAMAMLANDI: seed{seed}/{split}; 128/128 referans kontrolu gecti:', output)


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true', help='CPU checks only; no assets needed')
    parser.add_argument('--check', action='store_true', help='Verify assets and run self tests; no model/data loading')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--split', choices=tuple(SPLITS), default='valid')
    parser.add_argument('--selection', type=Path, help='Validation summary JSON with selected_tau; test split only')
    parser.add_argument('--repo-dir', type=Path, default=here.parent)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--package-dir', type=Path)
    for name in ('data-file', 'baseline-checkpoint', 'dropout-checkpoint', 'bert-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path)
    cli = parser.parse_args(argv)
    if not cli.self_test:
        missing = [name for name in ('seed', 'data_file', 'baseline_checkpoint', 'dropout_checkpoint',
                                     'bert_dir', 'output_dir') if getattr(cli, name) is None]
        if missing:
            parser.error('eksik secenekler: ' + ', '.join('--' + m.replace('_', '-') for m in missing))
    return cli


def main(argv=None):
    cli = parse_args(argv)
    if cli.self_test:
        load_stack(cli.repo_dir.resolve())
        self_test()
        print('Ozdenetim dosya yazmadi; model, veri veya checkpoint acilmadi.')
        return
    plan = prepare(cli)
    if cli.check:
        require(not core.unchanged(plan['hashes']), 'Kontrol sirasinda kaynak degisti')
        print(f'KONTROL_BASARILI: seed{plan["seed"]}/{plan["split"]}; gercek model/veri acilmadi, dosya yazilmadi')
        return
    evaluate(plan)


if __name__ == '__main__':
    main()
