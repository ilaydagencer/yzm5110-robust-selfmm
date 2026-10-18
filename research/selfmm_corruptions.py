"""Deterministic feature corruptions shared by all Self-MM variants.

Only leading frames covered by the original sequence length are modified.
No model name, batch number or evaluation order enters the random seed.
"""
import hashlib
import numpy as np

PARTIAL_RATES = {'K4': 0.30, 'K5': 0.50, 'K6': 0.70}
CONDITIONS = ('K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7')
NOISE_FACTOR = 0.5


def rng_for(seed, split, index, modality, family):
    key = f'selfmm-corruption-v1|{seed}|{split}|{index}|{modality}|{family}'
    value = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')
    return np.random.default_rng(value)


def corrupt_modality(features, length, modality, condition, std, index,
                     split='valid', seed=1111):
    if condition not in CONDITIONS or modality not in ('audio', 'vision'):
        raise ValueError('Unknown modality/condition')
    x = np.asarray(features)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError('Expected floating [time, feature] array')
    if int(length) != length or not 1 <= int(length) <= len(x):
        raise ValueError('Invalid sequence length')
    length = int(length)
    if not np.isfinite(x[:length]).all():
        raise ValueError('Nonfinite valid input frames')
    out = x.copy()
    mask = np.zeros(len(x), dtype=bool)
    full_drop = (modality == 'audio' and condition in ('K1', 'K3')) or (
                 modality == 'vision' and condition in ('K2', 'K3'))
    if full_drop:
        mask[:length] = True
    elif condition in PARTIAL_RATES:
        # Shared uniform draws make K4 masks a subset of K5 and K6.
        u = rng_for(seed, split, index, modality, 'partial').random(length)
        mask[:length] = u < PARTIAL_RATES[condition]
    out[mask] = 0
    if condition == 'K7':
        scale = np.asarray(std, dtype=np.float64)
        if scale.shape != (x.shape[1],) or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError('Invalid training standard deviations')
        z = rng_for(seed, split, index, modality, 'gaussian').standard_normal((length, x.shape[1]))
        out[:length] = (x[:length].astype(np.float64) + NOISE_FACTOR * scale * z).astype(x.dtype)
    if not np.isfinite(out[:length]).all():
        raise ValueError('Nonfinite corrupted frames')
    return out, mask


def training_statistics(features, lengths):
    """Population mean/std per feature, pooled across valid TRAIN frames."""
    count = 0
    mean = np.zeros(features.shape[2], dtype=np.float64)
    m2 = mean.copy()
    for x, raw_length in zip(features, lengths):
        length = int(raw_length)
        if length != raw_length or not 1 <= length <= len(x):
            raise ValueError('Invalid training length')
        batch = np.asarray(x[:length], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError('Nonfinite training frames')
        n = len(batch)
        bmean = batch.mean(axis=0)
        bm2 = ((batch-bmean)**2).sum(axis=0)
        delta = bmean-mean
        total = count+n
        m2 += bm2 + delta**2 * count*n/total
        mean += delta*n/total
        count = total
    if count == 0:
        raise ValueError('No training frames')
    return dict(valid_frames=count, mean=mean.tolist(),
                std=np.sqrt(np.maximum(m2/count, 0)).tolist())
