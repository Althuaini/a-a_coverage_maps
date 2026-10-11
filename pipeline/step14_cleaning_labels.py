# -*- coding: utf-8 -*-
"""Clean the normalized arrays and generate the per-pixel propagation-regime labels.

Stages (called ph1, ph2 and ph2.5 in the code; stages 1, 2 and 3 in the article):
  1: keep samples with 0 <= mean(R1B1 - R0B1) < 20 dB
  2: remove samples with more than 50 pixels where R1B1 - R0B1 < -25 dB
  3: remove samples where R0B1 has no signal except at the TX pixel
Writes X_, Y_, L_{split}_ph2.npy, the metadata and the removal audit.
Exported from Google Colab; the exploratory plots are kept as run.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.saving import register_keras_serializable
from google.colab import drive
import time
import gc


drive.mount('/content/drive')
print(f'TensorFlow: {tf.__version__}')
print(f'GPUs: {tf.config.list_physical_devices("GPU")}')

# ── Paths ────────────────────────────────────────────────────────────────────
SAVE_DIR    = '<DRIVE_DIR>/preprocessed'

# ── Normalisation (must match step13_preprocessing_split.ipynb) ──────────────────────────────
PHYS_MIN  = -120.0
PHYS_MAX  =    0.0
RANGE_DBM =  120.0

GRID_H, GRID_W = 100, 100
N_CH_BASE      = 7
RANDOM_SEED   = 42

print('Loading preprocessed data...')
t0 = time.time()

X_tr_norm   = np.load(f'{SAVE_DIR}/X_tr_norm.npy')
Y_tr_norm   = np.load(f'{SAVE_DIR}/Y_tr_norm.npy')
X_val_norm  = np.load(f'{SAVE_DIR}/X_val_norm.npy')
Y_val_norm  = np.load(f'{SAVE_DIR}/Y_val_norm.npy')
X_test_norm = np.load(f'{SAVE_DIR}/X_test_norm.npy')
Y_test_norm = np.load(f'{SAVE_DIR}/Y_test_norm.npy')

print(f'Loaded in {(time.time()-t0)/60:.1f} min')

meta_df = pd.read_csv(f'{SAVE_DIR}/split_metadata_roi_split.csv')

with open(f'{SAVE_DIR}/config.json') as f:
    cfg = json.load(f)

assert cfg['random_seed'] == RANDOM_SEED, f'Seed mismatch: {cfg["random_seed"]}'
assert cfg['phys_min']   == PHYS_MIN,     'Norm bounds mismatch — rerun step13_preprocessing_split.ipynb.'
assert cfg['n_channels'] == N_CH_BASE,    f'Channel mismatch: {cfg["n_channels"]}'

print(f'Train : {X_tr_norm.shape}')
print(f'Val   : {X_val_norm.shape}')
print(f'Test  : {X_test_norm.shape}')
print(f'Target: {Y_tr_norm.shape}')
print()
print('Channel map:')
print('  ch0: TX location   ch1: BLD height  ch2: R0B0')
print('  ch3: R0B1 (GT)     ch4: R1B0        ch5: Cos   ch6: Sin')

# City split summary
summary = meta_df.groupby(['city', 'split'])['roi_id'].nunique().unstack(fill_value=0)
summary['total'] = summary.sum(axis=1)
print()
print(summary.to_string())

print(meta_df.columns)

print(meta_df.head())

print(meta_df['split'].value_counts())
print()
print(meta_df['split'].unique())  # should show order: train → val → test

def to_dbm(arr_norm):
    # Normalised [0,1] → dBm using fixed physical bounds [-120, 0]
    return arr_norm * RANGE_DBM + PHYS_MIN


# ── (R1B1 − R0B1) in dBm ────────────────────────────────────────────────────
# R1B1 = Y_norm * 120 − 120   (target)
# R0B1 = X_norm[..., 3] * 120 − 120   (ch3)

diff_tr   = to_dbm(Y_tr_norm[..., 0])   - to_dbm(X_tr_norm[..., 3]) # (N_tr,  100, 100)
diff_val  = to_dbm(Y_val_norm[..., 0])  - to_dbm(X_val_norm[..., 3]) # (N_val, 100, 100)
diff_test = to_dbm(Y_test_norm[..., 0]) - to_dbm(X_test_norm[..., 3])  # (N_te,  100, 100)

diff_all  = np.concatenate([diff_tr, diff_val, diff_test], axis=0) # (72500, 100, 100)

print(f'diff_tr   : {diff_tr.shape}    range [{diff_tr.min():.1f}, {diff_tr.max():.1f}] dBm')
print(f'diff_val  : {diff_val.shape}')
print(f'diff_test : {diff_test.shape}')
print(f'diff_all  : {diff_all.shape}    range [{diff_all.min():.1f}, {diff_all.max():.1f}] dBm')

# Within-split position for each row
meta_df['_split_idx'] = meta_df.groupby('split').cumcount()

# Rebuild diff_all aligned to meta_df row order
diff_arrays = {'train': diff_tr, 'val': diff_val, 'test': diff_test}

diff_all = np.empty((len(meta_df), 100, 100), dtype=np.float32)
for split_name, arr in diff_arrays.items():
    mask = (meta_df['split'] == split_name).values
    idxs = meta_df.loc[meta_df['split'] == split_name, '_split_idx'].values
    diff_all[mask] = arr[idxs]

# Verify
meta_df['diff_idx'] = meta_df.index
print(f'diff_all shape: {diff_all.shape}')

# Spot check
row = meta_df.iloc[0]
print(f'Row 0: {row[["city","roi_id","tx_idx","split"]].to_dict()}  → diff_all[0] min/max: {diff_all[0].min():.1f}/{diff_all[0].max():.1f}')

def plot_sample(meta_idx):
    row   = meta_df.iloc[meta_idx]
    sidx  = int(row['_split_idx'])
    split = row['split']

    X = {'train': X_tr_norm, 'val': X_val_norm, 'test': X_test_norm}[split]
    Y = {'train': Y_tr_norm, 'val': Y_val_norm, 'test': Y_test_norm}[split]

    tx   = X[sidx, :, :, 0]
    bld  = X[sidx, :, :, 1] * 60.0          # denorm to metres
    r0b1 = to_dbm(X[sidx, :, :, 3])         # dBm
    r1b1 = to_dbm(Y[sidx, :, :, 0])         # dBm
    diff = diff_all[meta_idx]                # dBm

    # --- locate TX pixel ---
    ty, tx_col = np.unravel_index(np.argmax(tx), tx.shape)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"{row['city']}  |  ROI {row['roi_id']}  |  TX {row['tx_idx']}  |  {split}", fontsize=12)

    im0 = axes[0].imshow(tx,   cmap='Reds');   axes[0].set_title('TX location');         plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(bld,  cmap='gray');   axes[1].set_title('Building height (m)'); plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(r0b1, cmap='jet', vmin=-120, vmax=0); axes[2].set_title('R0B1 (dBm)'); plt.colorbar(im2, ax=axes[2])
    im3 = axes[3].imshow(r1b1, cmap='jet', vmin=-120, vmax=0); axes[3].set_title('R1B1 (dBm)'); plt.colorbar(im3, ax=axes[3])
    im4 = axes[4].imshow(diff, cmap='RdBu_r', vmin=-30, vmax=30); axes[4].set_title('R1B1 − R0B1 (dBm)'); plt.colorbar(im4, ax=axes[4])

    # --- TX marker on building height plot ---
    axes[1].plot(tx_col, ty, marker='*', color='red', markersize=12,
                 markeredgecolor='white', markeredgewidth=0.8, label='TX')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Plot row 0
plot_sample(11600)

# Per-sample mean of diff (one value per sample, not per pixel)
diff_mean_per_sample = diff_all.mean(axis=(1, 2))   # (72500,)

print(f'Shape : {diff_mean_per_sample.shape}')
print(f'Mean  : {diff_mean_per_sample.mean():.2f} dBm')
print(f'Std   : {diff_mean_per_sample.std():.2f} dBm')
print(f'Min   : {diff_mean_per_sample.min():.2f} dBm')
print(f'Max   : {diff_mean_per_sample.max():.2f} dBm')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, log_scale in zip(axes, [False, True]):
    ax.hist(diff_mean_per_sample, bins=200, color='steelblue', log=log_scale)
    ax.axvline(0, color='red', lw=1, linestyle='--', label='0 dBm')
    ax.axvline(diff_mean_per_sample.mean(), color='orange', lw=1,
               linestyle='--', label=f'mean={diff_mean_per_sample.mean():.2f} dBm')
    ax.set_xlabel('mean diff (dBm)')
    ax.set_ylabel('sample count (log)' if log_scale else 'sample count')
    ax.set_title('Log scale' if log_scale else 'Linear scale')
    ax.legend(loc='upper left')

fig.suptitle('Distribution of per-sample mean(R1B1 − R0B1)  (one point per map)', fontsize=13)
plt.tight_layout()
plt.show()



# ── 1: Histogram — negative and positive tails ────────────────────────────────
neg_mean = diff_mean_per_sample[diff_mean_per_sample < 0]
pos_mean = diff_mean_per_sample[diff_mean_per_sample > 0]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for ax, log_scale in zip(axes[0], [False, True]):
    ax.hist(neg_mean, bins=100, color='darkorange', log=log_scale)
    ax.set_xlabel('mean diff (dBm)')
    ax.set_ylabel('sample count (log)' if log_scale else 'sample count')
    ax.set_title(f'Log scale  (n={len(neg_mean):,})' if log_scale else f'Linear scale  (n={len(neg_mean):,})')

for ax, log_scale in zip(axes[1], [False, True]):
    ax.hist(pos_mean, bins=100, color='steelblue', log=log_scale)
    ax.set_xlabel('mean diff (dBm)')
    ax.set_ylabel('sample count (log)' if log_scale else 'sample count')
    ax.set_title(f'Log scale  (n={len(pos_mean):,})' if log_scale else f'Linear scale  (n={len(pos_mean):,})')

axes[0][0].set_title('Negative — Linear scale')
axes[0][1].set_title('Negative — Log scale')
axes[1][0].set_title('Positive — Linear scale')
axes[1][1].set_title('Positive — Log scale')

fig.suptitle('Per-sample mean(R1B1 − R0B1): negative vs positive tails', fontsize=13)
plt.tight_layout()
plt.show()

# ── Isolate extreme  counts — negative and positive ───────────────────────────────
print('=== Negative tail ===')
for threshold in [0, -5, -10, -20, -30, -40, -50]:
    count = (diff_mean_per_sample < threshold).sum()
    print(f'  mean diff < {threshold:4d} dBm : {count:5d} samples  ({count/len(diff_mean_per_sample)*100:.2f}%)')

print('\n=== Positive tail ===')
for threshold in [0, 5, 10, 20, 30, 40, 50]:
    count = (diff_mean_per_sample > threshold).sum()
    print(f'  mean diff > {threshold:4d} dBm : {count:5d} samples  ({count/len(diff_mean_per_sample)*100:.2f}%)')

#  Band plots — negative and positive ─────────────────────────────────────
rng = np.random.default_rng(42)

neg_bands = [
    ('0 to -5 dBm',    (diff_mean_per_sample <  0)  & (diff_mean_per_sample >= -5)),
    ('-5 to -10 dBm',  (diff_mean_per_sample < -5)  & (diff_mean_per_sample >= -10)),
    ('-10 to -20 dBm', (diff_mean_per_sample < -10) & (diff_mean_per_sample >= -20)),
    ('-20 to -30 dBm', (diff_mean_per_sample < -20) & (diff_mean_per_sample >= -30)),
    ('-30 to -40 dBm', (diff_mean_per_sample < -30) & (diff_mean_per_sample >= -40)),
    ('-40 to -50 dBm', (diff_mean_per_sample < -40) & (diff_mean_per_sample >= -50)),
    ('< -50 dBm',      (diff_mean_per_sample < -50)),
]

pos_bands = [
    ('0 to 5 dBm',    (diff_mean_per_sample >  0) & (diff_mean_per_sample <= 5)),
    ('5 to 10 dBm',   (diff_mean_per_sample >  5) & (diff_mean_per_sample <= 10)),
    ('10 to 20 dBm',  (diff_mean_per_sample > 10) & (diff_mean_per_sample <= 20)),
    ('20 to 30 dBm',  (diff_mean_per_sample > 20) & (diff_mean_per_sample <= 30)),
    ('30 to 40 dBm',  (diff_mean_per_sample > 30) & (diff_mean_per_sample <= 40)),
    ('40 to 50 dBm',  (diff_mean_per_sample > 40) & (diff_mean_per_sample <= 50)),
    ('> 50 dBm',      (diff_mean_per_sample > 50)),
]

for side_name, bands in [('NEGATIVE', neg_bands), ('POSITIVE', pos_bands)]:
    print(f'\n{"#"*60}')
    print(f'  {side_name} BANDS')
    print(f'{"#"*60}')
    for band_name, band_mask in bands:
        band_idx = np.where(band_mask)[0]
        if len(band_idx) == 0:
            continue
        n = min(3, len(band_idx))
        sample_idx = rng.choice(band_idx, size=n, replace=False)
        print(f'\n{"="*60}')
        print(f'  {band_name}  ({len(band_idx):,} samples)')
        print(f'{"="*60}')
        for meta_idx in sample_idx:
            row = meta_df.iloc[meta_idx]
            print(f'meta_idx={meta_idx}  mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
                  f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
            plot_sample(meta_idx)

#  Band plots — negative and positive ─────────────────────────────────────
rng = np.random.default_rng(42)

neg_bands = [
    ('0 to -5 dBm',    (diff_mean_per_sample <  0)  & (diff_mean_per_sample >= -5)),
    ('-5 to -10 dBm',  (diff_mean_per_sample < -5)  & (diff_mean_per_sample >= -10)),
]

pos_bands = [
    ('10 to 20 dBm',  (diff_mean_per_sample > 10) & (diff_mean_per_sample <= 20)),
    ('20 to 30 dBm',  (diff_mean_per_sample > 20) & (diff_mean_per_sample <= 30)),
    ('30 to 40 dBm',  (diff_mean_per_sample > 30) & (diff_mean_per_sample <= 40)),
    ('40 to 50 dBm',  (diff_mean_per_sample > 40) & (diff_mean_per_sample <= 50)),
    ('> 50 dBm',      (diff_mean_per_sample > 50)),
]

for side_name, bands in [('NEGATIVE', neg_bands), ('POSITIVE', pos_bands)]:
    print(f'\n{"#"*60}')
    print(f'  {side_name} BANDS')
    print(f'{"#"*60}')
    for band_name, band_mask in bands:
        band_idx = np.where(band_mask)[0]
        if len(band_idx) == 0:
            continue
        n = min(5, len(band_idx))
        sample_idx = rng.choice(band_idx, size=n, replace=False)
        print(f'\n{"="*60}')
        print(f'  {band_name}  ({len(band_idx):,} samples)')
        print(f'{"="*60}')
        for meta_idx in sample_idx:
            row = meta_df.iloc[meta_idx]
            print(f'meta_idx={meta_idx}  mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
                  f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
            plot_sample(meta_idx)

#  Band plots — negative and positive ─────────────────────────────────────
rng = np.random.default_rng(42)

neg_bands = [

]

pos_bands = [
    ('10 to 20 dBm',  (diff_mean_per_sample > 10) & (diff_mean_per_sample <= 20)),
]

for side_name, bands in [('NEGATIVE', neg_bands), ('POSITIVE', pos_bands)]:
    print(f'\n{"#"*60}')
    print(f'  {side_name} BANDS')
    print(f'{"#"*60}')
    for band_name, band_mask in bands:
        band_idx = np.where(band_mask)[0]
        if len(band_idx) == 0:
            continue
        n = min(20, len(band_idx))
        sample_idx = rng.choice(band_idx, size=n, replace=False)
        print(f'\n{"="*60}')
        print(f'  {band_name}  ({len(band_idx):,} samples)')
        print(f'{"="*60}')
        for meta_idx in sample_idx:
            row = meta_df.iloc[meta_idx]
            print(f'meta_idx={meta_idx}  mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
                  f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
            plot_sample(meta_idx)

#  City table — negative and positive bands ───────────────────────────────
neg_bands_def = {
    '  0 to -5' : (diff_mean_per_sample <  0)  & (diff_mean_per_sample >= -5),
    ' -5 to -10': (diff_mean_per_sample < -5)  & (diff_mean_per_sample >= -10),
    '-10 to -20': (diff_mean_per_sample < -10) & (diff_mean_per_sample >= -20),
    '-20 to -30': (diff_mean_per_sample < -20) & (diff_mean_per_sample >= -30),
    '-30 to -40': (diff_mean_per_sample < -30) & (diff_mean_per_sample >= -40),
    '-40 to -50': (diff_mean_per_sample < -40) & (diff_mean_per_sample >= -50),
    '< -50'     : (diff_mean_per_sample < -50),
}

pos_bands_def = {
    ' 0 to  5' : (diff_mean_per_sample >  0) & (diff_mean_per_sample <= 5),
    ' 5 to 10' : (diff_mean_per_sample >  5) & (diff_mean_per_sample <= 10),
    '10 to 20' : (diff_mean_per_sample > 10) & (diff_mean_per_sample <= 20),
    '20 to 30' : (diff_mean_per_sample > 20) & (diff_mean_per_sample <= 30),
    '30 to 40' : (diff_mean_per_sample > 30) & (diff_mean_per_sample <= 40),
    '40 to 50' : (diff_mean_per_sample > 40) & (diff_mean_per_sample <= 50),
    '> 50'     : (diff_mean_per_sample > 50),
}

for side_name, bands_def in [('NEGATIVE bands', neg_bands_def), ('POSITIVE bands', pos_bands_def)]:
    for band_name, band_mask in bands_def.items():
        meta_df[band_name] = band_mask

    band_cols = list(bands_def.keys())
    table = meta_df.groupby('city')[band_cols].sum().astype(int)
    table['total_flagged'] = table[band_cols].sum(axis=1)
    table['total_samples'] = meta_df.groupby('city')['roi_id'].count()
    table['flagged_%']     = (table['total_flagged'] / table['total_samples'] * 100).round(2)

    total_row = table.sum(numeric_only=True)
    total_row['flagged_%'] = (total_row['total_flagged'] / total_row['total_samples'] * 100).round(2)
    total_row.name = 'TOTAL'
    table = pd.concat([table, total_row.to_frame().T])

    print(f'\n=== City breakdown — {side_name} ===')
    print(table.to_string())

# ── ph1_clean: keep only samples where 0 <= mean_diff < 20 dBm ───────────────
#
#  Removal criteria confirmed by visual band inspection:
#
#  mean diff < 0 dBm  (5.82% of samples, 4,219 total):
#    Visual inspection revealed two plausible causes:
#    1. MATLAB ray tracer creates blockage zones where no buildings exist in the
#       OpenStreetMap footprint — R1B1 shows a shadow (−120 dBm) at pixels that
#       R0B1 (computed in Python) correctly treats as open space. This produces
#       diff = R1B1 − R0B1 = −120 − (−60) = −60 dBm across a spatial cluster.
#    2. R0B1 (Python) misses some tall buildings and assigns real signal to pixels
#       that MATLAB correctly blocks in R1B1. R1B1 is the more physically complete
#       simulation (uses 3D siteviewer), so when R0B1 misses a building, the diff
#       becomes negative at those pixels.
#    Both causes reflect a pipeline inconsistency between the Python R0B1 and the
#    MATLAB R1B1, not a real physical phenomenon. Samples dominated by this
#    inconsistency (negative per-map mean) are excluded.
#
#  mean diff >= 20 dBm  (0.76% of samples, 548 total):
#  These samples are excluded as outliers (mean gain of 20 dB or more over R0B1).
# ─────────────────────────────────────────────────────────────────────────────

ph1_mask    = (diff_mean_per_sample >= 0) & (diff_mean_per_sample < 20)
ph1_idx     = np.where(ph1_mask)[0]
ph1_meta_df = meta_df.iloc[ph1_idx].reset_index(drop=True)
ph1_mean    = diff_mean_per_sample[ph1_idx]

# ── Track removed samples ─────────────────────────────────────────────────────
neg_removed_mask = diff_mean_per_sample < 0
pos_removed_mask = diff_mean_per_sample >= 20

ph1_removed_idx  = np.where(~ph1_mask)[0]   # everything NOT kept
ph1_removed_df   = meta_df.iloc[ph1_removed_idx].copy()
ph1_removed_df['meta_idx']       = ph1_removed_idx
ph1_removed_df['mean_diff_dbm']  = diff_mean_per_sample[ph1_removed_idx]
ph1_removed_df['removal_reason'] = np.where(
    diff_mean_per_sample[ph1_removed_idx] < 0,
    'mean_diff<0 (R0B1/R1B1 inconsistency)',
    'mean_diff>=20 (unrealistic gain)'
)
ph1_removed_df = ph1_removed_df.reset_index(drop=True)

print(f'Original samples : {len(meta_df):,}')
print(f'Removed (< 0)    : {neg_removed_mask.sum():,}  ({neg_removed_mask.mean()*100:.2f}%)')
print(f'Removed (>= 20)  : {pos_removed_mask.sum():,}  ({pos_removed_mask.mean()*100:.2f}%)')
print(f'ph1_clean        : {len(ph1_idx):,}  ({len(ph1_idx)/len(meta_df)*100:.2f}%)')
print(f'Total removed    : {len(ph1_removed_df):,}')

print(f'\nRemoval reason breakdown:')
print(ph1_removed_df['removal_reason'].value_counts().to_string())

print(f'\nMean diff stats (ph1_clean):')
print(f'  Mean : {ph1_mean.mean():.2f} dBm')
print(f'  Std  : {ph1_mean.std():.2f} dBm')
print(f'  Min  : {ph1_mean.min():.2f} dBm')
print(f'  Max  : {ph1_mean.max():.2f} dBm')

print(f'\nSplit breakdown:')
print(f'{"Split":<8} {"Original":>12} {"ph1_clean":>12} {"Removed":>12} {"Removed %":>12}')
print('-' * 58)
for split_name in ['train', 'val', 'test']:
    orig  = (meta_df['split'] == split_name).sum()
    clean = (ph1_meta_df['split'] == split_name).sum()
    print(f'{split_name:<8} {orig:>12,} {clean:>12,} {orig-clean:>12,} {(orig-clean)/orig*100:>11.1f}%')
print('-' * 58)
print(f'{"TOTAL":<8} {len(meta_df):>12,} {len(ph1_idx):>12,} {len(meta_df)-len(ph1_idx):>12,} {(len(meta_df)-len(ph1_idx))/len(meta_df)*100:>11.1f}%')

# ── City breakdown of removed samples ─────────────────────────────────────────
print(f'\nRemoved samples by city × reason:')
print(ph1_removed_df.groupby(['city', 'removal_reason']).size().unstack(fill_value=0).to_string())

# ── Pixel-wise diff distribution on ph1_clean ─────────────────────────────────
diff_ph1  = diff_all[ph1_idx]          # (67733, 100, 100)
flat_ph1  = diff_ph1.ravel()           # (677,330,000 pixels)

print(f'ph1_clean pixel stats:')
print(f'  Total pixels : {len(flat_ph1):,}')
print(f'  Mean         : {flat_ph1.mean():.2f} dBm')
print(f'  Std          : {flat_ph1.std():.2f} dBm')
print(f'  Min / Max    : {flat_ph1.min():.1f} / {flat_ph1.max():.1f} dBm')

# ── Class breakdown ────────────────────────────────────────────────────────────
print(f'\n{"Class":<45} {"Pixels":>14} {"% of total":>12}')
print('-' * 74)
class_defs = {
    'A  Severe loss    (diff < -25)      ': flat_ph1 < -25,
    'B  Normal loss    (-25 <= diff < 0) ': (flat_ph1 < 0)   & (flat_ph1 >= -25),
    'Z  No change      (diff == 0)       ': flat_ph1 == 0,
    'C  Normal gain    (0 < diff <= 25)  ': (flat_ph1 > 0)   & (flat_ph1 <= 25),
    'D  Multipath      (25 < diff <= 60) ': (flat_ph1 > 25)  & (flat_ph1 <= 60),
    'E  Severe gain    (diff > 60)       ': flat_ph1 > 60,
}
for name, mask in class_defs.items():
    count = mask.sum()
    print(f'{name:<45} {count:>14,} {count/len(flat_ph1)*100:>11.1f}%')
print('-' * 74)
print(f'{"TOTAL":<45} {len(flat_ph1):>14,} {"100.0%":>12}')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

neg_ph1 = flat_ph1[flat_ph1 < 0]
pos_ph1 = flat_ph1[flat_ph1 > 0]

boundaries = [
    (-25, 'red',    'A/B  −25 dBm'),
    ( 25, 'green',  'C/D  +25 dBm'),
    ( 60, 'purple', 'D/E  +60 dBm'),
    (  0, 'black',  '0 dBm'),
]

# Full distribution
for ax, log_scale in zip(axes[0], [False, True]):
    ax.hist(flat_ph1, bins=200, color='steelblue', log=log_scale)
    for x, color, label in boundaries:
        ax.axvline(x, color=color, lw=1.5, linestyle='--', label=label)
    ax.set_xlabel('diff (dBm)')
    ax.set_ylabel('count (log)' if log_scale else 'count')
    ax.set_title(f'Full distribution — {"Log" if log_scale else "Linear"}  (n={len(flat_ph1):,})')
    ax.legend(loc='upper left', fontsize=8)

# Negative vs positive
axes[1, 0].hist(neg_ph1, bins=100, color='darkorange', log=True)
axes[1, 0].set_title(f'Negative pixels — log  (n={len(neg_ph1):,},  {len(neg_ph1)/len(flat_ph1)*100:.1f}%)')
axes[1, 0].set_xlabel('diff (dBm)')
axes[1, 0].axvline(-25, color='red', lw=1.5, linestyle='--', label='−25 dBm')
axes[1, 0].axvline(neg_ph1.mean(), color='black', lw=1, linestyle='--',
                   label=f'mean={neg_ph1.mean():.1f} dBm')
axes[1, 0].legend(loc='upper left', fontsize=8)

axes[1, 1].hist(pos_ph1, bins=100, color='steelblue', log=True)
axes[1, 1].set_title(f'Positive pixels — log  (n={len(pos_ph1):,},  {len(pos_ph1)/len(flat_ph1)*100:.1f}%)')
axes[1, 1].set_xlabel('diff (dBm)')
axes[1, 1].axvline(25, color='green',  lw=1.5, linestyle='--', label='+25 dBm')
axes[1, 1].axvline(60, color='purple', lw=1.5, linestyle='--', label='+60 dBm')
axes[1, 1].axvline(pos_ph1.mean(), color='black', lw=1, linestyle='--',
                   label=f'mean={pos_ph1.mean():.1f} dBm')
axes[1, 1].legend(loc='upper left', fontsize=8)

fig.suptitle(f'Pixel-wise diff distribution — ph1_clean  ({len(ph1_idx):,} samples)', fontsize=13)
plt.tight_layout()
plt.show()

rng = np.random.default_rng(42)

class_defs_ph1 = {
    'A  Normal loss    (-25 <= diff < 0)': (diff_ph1 < 0)  & (diff_ph1 >= -25),
    'B  Severe loss    (diff < -25)     ': diff_ph1 < -25,
    'Z  No change      (diff == 0)      ': diff_ph1 == 0,
    'C  Normal gain    (0 < diff <= 25) ': (diff_ph1 > 0)  & (diff_ph1 <= 25),
    'D  Larg gian      (25 < diff <= 60)': (diff_ph1 > 25) & (diff_ph1 <= 60),
    'E  Severe gain    (diff > 60)      ': diff_ph1 > 60,
}

for class_name, class_mask in class_defs_ph1.items():
    per_sample_count = class_mask.sum(axis=(1, 2))
    eligible_ph1     = np.where(per_sample_count > 0)[0]

    print(f'\n{"="*60}')
    print(f'  {class_name}')
    print(f'  Eligible samples : {len(eligible_ph1):,}')
    print(f'{"="*60}')

    chosen = rng.choice(eligible_ph1, size=min(2, len(eligible_ph1)), replace=False)
    for pos in chosen:
        meta_idx = ph1_idx[pos]
        row = meta_df.iloc[meta_idx]
        print(f'meta_idx={meta_idx}  '
              f'class_px={per_sample_count[pos]}  '
              f'mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
              f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
        plot_sample(meta_idx)

# ── Per-sample pixel count distribution for each class in ph1_clean ───────────
#
#  For each class, shows how many samples have AT LEAST t pixels in that class.
#  The distribution shape reveals whether high-px samples are rare outliers
#  (sharp drop-off) or a gradual tail — which determines where to draw the
#  ph2_clean cutoff.
# ─────────────────────────────────────────────────────────────────────────────

thresholds = [1, 5, 10, 50, 100, 500, 1000, 5000]

class_defs_ph1 = {
    'A  Normal loss    (-25 <= diff < 0)': (diff_ph1 < 0)  & (diff_ph1 >= -25),
    'B  Severe loss    (diff < -25)     ': diff_ph1 < -25,
    'Z  No change      (diff == 0)      ': diff_ph1 == 0,
    'C  Normal gain    (0 < diff <= 25) ': (diff_ph1 > 0)  & (diff_ph1 <= 25),
    'D  Multipath      (25 < diff <= 60)': (diff_ph1 > 25) & (diff_ph1 <= 60),
    'E  Severe gain    (diff > 60)      ': diff_ph1 > 60,
}

for class_name, class_mask in class_defs_ph1.items():
    per_sample_count = class_mask.sum(axis=(1, 2))     # (67733,) per ph1 sample
    affected         = per_sample_count[per_sample_count > 0]

    print(f'\n=== {class_name} ===')
    print(f'  Affected samples : {len(affected):,}  ({len(affected)/len(ph1_idx)*100:.1f}%)')
    print(f'  Mean px          : {affected.mean():.1f}')
    print(f'  Median px        : {np.median(affected):.1f}')
    print(f'  Max px           : {affected.max()}')
    print(f'  {"Threshold":<12} {"Samples":>10} {"% of affected":>15}')
    print(f'  {"-"*40}')
    for t in thresholds:
        count = (affected >= t).sum()
        print(f'  >= {t:5d} px  : {count:>10,}  ({count/len(affected)*100:>13.1f}%)')

rng = np.random.default_rng(42)

for class_name, class_mask in [
    ('B  Severe loss  (diff < -25)',  diff_ph1 < -25),
]:
    per_sample_count = class_mask.sum(axis=(1, 2))   # (67733,)

    px_bands = [
        ('1 to 10 px',     (per_sample_count >= 1)   & (per_sample_count < 10)),
        ('10 to 50 px',    (per_sample_count >= 10)  & (per_sample_count < 50)),
        ('50 to 100 px',   (per_sample_count >= 50)  & (per_sample_count < 100)),
        ('100 to 500 px',  (per_sample_count >= 100) & (per_sample_count < 500)),
        ('500 to 1000 px', (per_sample_count >= 500) & (per_sample_count < 1000)),
        ('>= 1000 px',     per_sample_count >= 1000),
    ]

    print(f'\n{"#"*65}')
    print(f'  {class_name}')
    print(f'{"#"*65}')

    for band_name, band_mask in px_bands:
        eligible = np.where(band_mask)[0]   # positions within ph1_idx
        if len(eligible) == 0:
            continue

        chosen = rng.choice(eligible, size=min(5, len(eligible)), replace=False)

        print(f'\n{"="*65}')
        print(f'  {band_name}  ({len(eligible):,} samples)')
        print(f'{"="*65}')
        for pos in chosen:
            meta_idx = ph1_idx[pos]
            row = meta_df.iloc[meta_idx]
            print(f'meta_idx={meta_idx}  '
                  f'class_px={per_sample_count[pos]}  '
                  f'mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
                  f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
            plot_sample(meta_idx)

# ── ph2_clean_draft: remove samples with > 50 class B pixels ─────────────────
#
#  Class B (diff < −25 dBm) visual inspection showed that samples with > 50 px
#  exhibit systematic R0B1/R1B1 inconsistency — blockage zones appearing where
#  no buildings exist, or buildings missed by the Python R0B1 pipeline.
#  Samples with 1−50 class B pixels are isolated boundary artifacts with
#  negligible impact and are retained.
# ─────────────────────────────────────────────────────────────────────────────

# Class B pixel count per ph1_clean sample
b_count_ph1 = (diff_ph1 < -25).sum(axis=(1, 2))   # (67733,)

# Keep samples with <= 50 class B pixels
ph2_mask_in_ph1   = b_count_ph1 <= 50
ph2_idx_in_ph1    = np.where(ph2_mask_in_ph1)[0]          # positions within ph1_idx
ph2_idx           = ph1_idx[ph2_idx_in_ph1]               # positions in original meta_df
ph2_meta_df       = meta_df.iloc[ph2_idx].reset_index(drop=True)
ph2_mean          = diff_mean_per_sample[ph2_idx]

# ── Track removed samples (ph1 → ph2 stage) ───────────────────────────────────
ph2_removed_idx_in_ph1 = np.where(~ph2_mask_in_ph1)[0]        # positions within ph1_idx
ph2_removed_idx         = ph1_idx[ph2_removed_idx_in_ph1]      # positions in original meta_df

ph2_removed_df = meta_df.iloc[ph2_removed_idx].copy()
ph2_removed_df['meta_idx']       = ph2_removed_idx
ph2_removed_df['mean_diff_dbm']  = diff_mean_per_sample[ph2_removed_idx]
ph2_removed_df['class_b_px']     = b_count_ph1[ph2_removed_idx_in_ph1]
ph2_removed_df['removal_reason'] = 'class_b_px>50 (R0B1/R1B1 inconsistency)'
ph2_removed_df = ph2_removed_df.reset_index(drop=True)

print(f'ph1_clean samples     : {len(ph1_idx):,}')
print(f'Removed (B > 50 px)   : {(~ph2_mask_in_ph1).sum():,}  ({(~ph2_mask_in_ph1).mean()*100:.2f}%)')
print(f'ph2_clean_draft       : {len(ph2_idx):,}  ({len(ph2_idx)/len(meta_df)*100:.2f}% of original)')

print(f'\nClass B px stats (removed samples):')
print(f'  Mean : {ph2_removed_df["class_b_px"].mean():.1f} px')
print(f'  Median : {ph2_removed_df["class_b_px"].median():.1f} px')
print(f'  Max  : {ph2_removed_df["class_b_px"].max():,} px')

print(f'\nMean diff stats (ph2_clean_draft):')
print(f'  Mean : {ph2_mean.mean():.2f} dBm')
print(f'  Std  : {ph2_mean.std():.2f} dBm')
print(f'  Min  : {ph2_mean.min():.2f} dBm')
print(f'  Max  : {ph2_mean.max():.2f} dBm')

print(f'\nSplit breakdown:')
print(f'{"Split":<8} {"Original":>12} {"ph1_clean":>12} {"ph2_draft":>12} {"Removed":>12}')
print('-' * 52)
for split_name in ['train', 'val', 'test']:
    orig  = (meta_df['split'] == split_name).sum()
    ph1   = (ph1_meta_df['split'] == split_name).sum()
    ph2   = (ph2_meta_df['split'] == split_name).sum()
    print(f'{split_name:<8} {orig:>12,} {ph1:>12,} {ph2:>12,} {ph1-ph2:>12,}')
print('-' * 52)
print(f'{"TOTAL":<8} {len(meta_df):>12,} {len(ph1_idx):>12,} {len(ph2_idx):>12,} {len(ph1_idx)-len(ph2_idx):>12,}')

# ── City breakdown of ph2 removals ────────────────────────────────────────────
print(f'\nph2 removed samples by city:')
print(ph2_removed_df.groupby('city').size().sort_values(ascending=False).to_string())

# ── Combine ph1 + ph2 removal logs into one unified audit trail ──────────────
#  Requires ph1_removed_df to already exist (from the ph1_clean cell).
#  Columns aligned across both stages: meta_idx, city, roi_id, tx_idx, split,
#  mean_diff_dbm, removal_reason. ph2-specific 'class_b_px' is NaN for ph1 rows.
common_cols = ['meta_idx', 'city', 'roi_id', 'tx_idx', 'split',
               'mean_diff_dbm', 'removal_reason']

all_removed_df = pd.concat(
    [ph1_removed_df[common_cols].assign(stage='ph1'),
     ph2_removed_df[common_cols + ['class_b_px']].assign(stage='ph2')],
    ignore_index=True
)

print(f'\n=== Combined removal audit ===')
print(f'Total removed (ph1 + ph2) : {len(all_removed_df):,}')
print(f'  ph1 : {(all_removed_df["stage"]=="ph1").sum():,}')
print(f'  ph2 : {(all_removed_df["stage"]=="ph2").sum():,}')
print(f'\nBy stage × reason:')
print(all_removed_df.groupby(['stage', 'removal_reason']).size().to_string())

# ── "R0B1 blocked everywhere but TX" ──────
#  Samples where R0B1 (direct-path channel) is at the sentinel (-120 dBm)
#     everywhere except the TX pixel itself
# ─────────────────────────────────────────────────────────────────────────────

SENTINEL_NORM = 0.0
EPS           = 1e-6


# --- Diagnostic : R0B1 blocked everywhere except TX pixel ---
r0b1_nontx_real_count = np.empty(len(ph2_idx), dtype=np.int32)

for pos, meta_idx in enumerate(ph2_idx):
    row   = meta_df.iloc[meta_idx]
    sidx  = int(row['_split_idx'])
    split = row['split']

    X = {'train': X_tr_norm, 'val': X_val_norm, 'test': X_test_norm}[split]

    tx   = X[sidx, :, :, 0]
    r0b1 = X[sidx, :, :, 3]

    ty, tx_col = np.unravel_index(np.argmax(tx), tx.shape)

    real_mask = r0b1 > (SENTINEL_NORM + EPS)   # True = has direct-path signal
    real_mask[ty, tx_col] = False              # exclude the TX pixel itself

    r0b1_nontx_real_count[pos] = real_mask.sum()

tx_locked = np.where(r0b1_nontx_real_count == 0)[0]   # positions within ph2_idx

print(f'Samples with R0B1 real ONLY at TX   : {len(tx_locked):,}  '
      f'({len(tx_locked)/len(ph2_idx)*100:.2f}% of ph2_clean_draft)')



# ── Visual inspection: R0B1 TX-locked cases ──────────────────────────────────
rng = np.random.default_rng(42)
chosen_tx = rng.choice(tx_locked, size=min(10, len(tx_locked)), replace=False)

print(f'\n{"="*65}')
print(f'  R0B1 blocked everywhere except TX  ({len(tx_locked):,} samples)')
print(f'{"="*65}')
for pos in chosen_tx:
    meta_idx = ph2_idx[pos]
    row = meta_df.iloc[meta_idx]
    print(f'\nmeta_idx={meta_idx}  '
          f'mean_diff={diff_mean_per_sample[meta_idx]:+.2f} dBm  '
          f'{row["city"]}  ROI {row["roi_id"]}  TX {row["tx_idx"]}  {row["split"]}')
    plot_sample(meta_idx)

# ── Log the samples where R0B1 has no signal except at the TX pixel ─────────
#  (removed in the next block)
# ─────────────────────────────────────────────────────────────────────────────

tx_locked_meta_idx = ph2_idx[tx_locked]   # map ph2-positions → original meta_df index

tx_locked_df = meta_df.iloc[tx_locked_meta_idx].copy()
tx_locked_df['meta_idx']            = tx_locked_meta_idx
tx_locked_df['mean_diff_dbm']       = diff_mean_per_sample[tx_locked_meta_idx]
tx_locked_df['r0b1_nontx_real_px']  = r0b1_nontx_real_count[tx_locked]   # will be 0 by definition
tx_locked_df['removal_reason']      = 'r0b1_blocked_except_tx'
tx_locked_df = tx_locked_df.reset_index(drop=True)

print(f'R0B1-locked samples tracked : {len(tx_locked_df):,}')
print(f'\nBy city:')
print(tx_locked_df.groupby('city').size().sort_values(ascending=False).to_string())
print(f'\nBy split:')
print(tx_locked_df['split'].value_counts().to_string())

print(f'\nFull list:')
print(tx_locked_df[['meta_idx', 'city', 'roi_id', 'tx_idx', 'split',
                     'mean_diff_dbm']].to_string(index=False))

# ── Remove "R0B1 blocked except TX" samples from ph2_clean ──────────────────
#
#  These 23 samples pass the Class B <= 50 px filter but exhibit a separate
#  pathology: R0B1 has no direct-path signal anywhere in the ROI except the
#  TX pixel itself. Treated as a pipeline/geometry artifact (e.g. TX boxed in
#  by buildings, or an edge case in the Python direct-path computation) and
#  excluded from the final dataset.
#
#  This block redefines ph2_idx and ph2_meta_df; all later blocks use the
#  final set.
# ─────────────────────────────────────────────────────────────────────────────

ph2_pre_txfilter_idx     = ph2_idx.copy()            # snapshot before this filter
ph2_pre_txfilter_meta_df = ph2_meta_df.copy()

tx_locked_mask = np.isin(ph2_idx, tx_locked_meta_idx)   # True where sample should be dropped

ph2_idx     = ph2_idx[~tx_locked_mask]
ph2_meta_df = meta_df.iloc[ph2_idx].reset_index(drop=True)
ph2_mean    = diff_mean_per_sample[ph2_idx]

print(f'ph2_clean_draft (before TX-lock filter) : {len(ph2_pre_txfilter_idx):,}')
print(f'Removed (R0B1 blocked except TX)        : {tx_locked_mask.sum():,}')
print(f'ph2_clean (final)                       : {len(ph2_idx):,}')

print(f'\nSplit breakdown:')
print(f'{"Split":<8} {"Pre-TX-filter":>14} {"Final ph2":>12} {"Removed":>10}')
print('-' * 48)
for split_name in ['train', 'val', 'test']:
    pre   = (ph2_pre_txfilter_meta_df['split'] == split_name).sum()
    final = (ph2_meta_df['split'] == split_name).sum()
    print(f'{split_name:<8} {pre:>14,} {final:>12,} {pre-final:>10,}')

# ── Fold into unified removal audit trail ────────────────────────────────────
tx_locked_df['stage'] = 'ph2.5'

all_removed_df = pd.concat(
    [all_removed_df, tx_locked_df[common_cols + ['stage']]],
    ignore_index=True
)

print(f'\n=== Updated combined removal audit ===')
print(f'Total removed (ph1 + ph2 + ph2.5) : {len(all_removed_df):,}')
print(all_removed_df.groupby(['stage', 'removal_reason']).size().to_string())

# ── Per-sample pixel count distribution for each class in ph2_clean_draft ─────
diff_ph2 = diff_all[ph2_idx]   # (N_ph2, 100, 100)

thresholds = [1, 5, 10, 50, 100, 500, 1000, 5000]

class_defs_ph2 = {
    'A  Normal loss    (-25 <= diff < 0)': (diff_ph2 < 0)  & (diff_ph2 >= -25),
    'B  Severe loss    (diff < -25)     ': diff_ph2 < -25,
    'Z  No change      (diff == 0)      ': diff_ph2 == 0,
    'C  Normal gain    (0 < diff <= 25) ': (diff_ph2 > 0)  & (diff_ph2 <= 25),
    'D  Multipath      (25 < diff <= 60)': (diff_ph2 > 25) & (diff_ph2 <= 60),
    'E  Severe gain    (diff > 60)      ': diff_ph2 > 60,
}

for class_name, class_mask in class_defs_ph2.items():
    per_sample_count = class_mask.sum(axis=(1, 2))
    affected         = per_sample_count[per_sample_count > 0]

    print(f'\n=== {class_name} ===')
    print(f'  Affected samples : {len(affected):,}  ({len(affected)/len(ph2_idx)*100:.1f}%)')
    print(f'  Mean px          : {affected.mean():.1f}')
    print(f'  Median px        : {np.median(affected):.1f}')
    print(f'  Max px           : {affected.max()}')
    print(f'  {"Threshold":<12} {"Samples":>10} {"% of affected":>15}')
    print(f'  {"-"*40}')
    for t in thresholds:
        count = (affected >= t).sum()
        print(f'  >= {t:5d} px  : {count:>10,}  ({count/len(affected)*100:>13.1f}%)')

# ── Propagation Regime Label Generation (ph2_clean only) ─────────────────────
#
#  Labels are assigned per pixel based on whether R0B1 and R1B1 are at the
#  sentinel value (−120 dBm → normalised 0.0), which indicates no signal.
#
#  Physical basis:
#    R0B1 = buildings-only channel (no reflections, Python-computed)
#    R1B1 = 1-reflection + buildings channel (MATLAB ray tracer) — the target
#
#    Since R1B1 includes all propagation paths of R0B1 plus reflected paths,
#    it is a strict superset: any pixel real in R0B1 must be real in R1B1.
#
#  Three regime classes:
#    Class 0 — Deep shadow : R0B1 sentinel AND R1B1 sentinel
#               Both simulations agree the pixel has no signal.
#               No direct path, no reflected path reaches this pixel.
#
#    Class 1 — Multipath   : R0B1 sentinel AND R1B1 real
#               The direct path is blocked by a building, but a single
#               reflection reaches the pixel.
#
#    Class 2 — LOS         : R0B1 real (R1B1 anything)
#               A direct path exists. The reflection may add or subtract
#               a small amount via interference but the pixel is fundamentally
#               line-of-sight dominated.
#
#  Labels are computed ONLY over the ph2_clean subset (not the full dataset)
#  to avoid unnecessary memory allocation for samples that will not be used.
# ─────────────────────────────────────────────────────────────────────────────

SENTINEL_NORM = 0.0
EPS           = 1e-6   # float32 tolerance — sentinel is exactly 0.0 under fixed norm

# ── Step 1: Extract R0B1 and R1B1 normalised for ph2_clean samples only ───────
#  ph2_meta_df contains the metadata of the samples kept after all three stages.
#  _split_idx gives the position of each row within its split array,
#  enabling correct mapping from meta_df order to the split numpy arrays.
meta_df['_split_idx'] = meta_df.groupby('split').cumcount()

n_ph2 = len(ph2_idx)
r0b1_ph2 = np.empty((n_ph2, 100, 100), dtype=np.float32)
r1b1_ph2 = np.empty((n_ph2, 100, 100), dtype=np.float32)

for split_name, X, Y in [('train', X_tr_norm, Y_tr_norm),
                          ('val',   X_val_norm, Y_val_norm),
                          ('test',  X_test_norm, Y_test_norm)]:
    # Boolean mask selecting ph2_clean rows belonging to this split
    split_in_ph2 = (ph2_meta_df['split'] == split_name).values
    # Within-split position for each selected row
    idxs         = ph2_meta_df.loc[split_in_ph2, '_split_idx'].values

    r0b1_ph2[split_in_ph2] = X[idxs, :, :, 3]   # ch3 = R0B1
    r1b1_ph2[split_in_ph2] = Y[idxs, :, :, 0]   # target = R1B1

print(f'r0b1_ph2 : {r0b1_ph2.shape}')
print(f'r1b1_ph2 : {r1b1_ph2.shape}')

# ── Step 2: Identify sentinel pixels ─────────────────────────────────────────
#  Under fixed normalisation [−120, 0] dBm → [0, 1]:
#  −120 dBm maps exactly to 0.0. EPS handles float32 rounding.
r0b1_sentinel_ph2 = r0b1_ph2 <= SENTINEL_NORM + EPS   # True = no signal in R0B1
r1b1_sentinel_ph2 = r1b1_ph2 <= SENTINEL_NORM + EPS   # True = no signal in R1B1

# ── Step 3: Assign per-pixel regime labels ────────────────────────────────────
#  Initialise all pixels to 0 (deep shadow), then overwrite with higher classes.
#  LOS assignment (~r0b1_sentinel) comes last and takes priority — any pixel
#  with a real R0B1 value is LOS regardless of R1B1.
labels_ph2 = np.zeros((n_ph2, 100, 100), dtype=np.uint8)
labels_ph2[ r0b1_sentinel_ph2 &  r1b1_sentinel_ph2] = 0   # deep shadow
labels_ph2[ r0b1_sentinel_ph2 & ~r1b1_sentinel_ph2] = 1   # multipath
labels_ph2[~r0b1_sentinel_ph2                      ] = 2   # LOS

# ── Step 4: Class distribution ────────────────────────────────────────────────
total_px_ph2 = labels_ph2.size
print(f'\n=== ph2_clean label distribution ({n_ph2:,} samples) ===')
for cls, name in zip([0, 1, 2], ['Deep shadow', 'Multipath  ', 'LOS        ']):
    count = (labels_ph2 == cls).sum()
    print(f'  Class {cls} ({name}) : {count:>12,}  ({count/total_px_ph2*100:.1f}%)')
print(f'  Total pixels : {total_px_ph2:,}')

# ── Step 5: Inverse-frequency class weights per regime ────────────────────────
#  The class imbalance (LOS >> Deep shadow >> Multipath) requires weighted loss.
#  Weight for class c = total_pixels / (n_classes * count_c)
#  Higher weight → model pays more attention to that class during training.
counts  = [(labels_ph2 == c).sum() for c in range(3)]
weights = [total_px_ph2 / (3 * c) for c in counts]

print(f'\n=== Inverse-frequency class weights ===')
for cls, name, w in zip([0, 1, 2], ['Deep shadow', 'Multipath  ', 'LOS        '], weights):
    print(f'  Class {cls} ({name}) : weight = {w:.4f}')
print(f'\n  Imbalance ratio  LOS : Deep shadow : Multipath')
print(f'  = {counts[2]/counts[1]:.1f} : {counts[0]/counts[1]:.1f} : 1')

# ── Build a lookup: meta_df index → position in ph2_idx ──────────────────────
#  labels_ph2 is indexed 0..n_ph2-1, not by meta_df index.
#  This dict maps any meta_idx (original meta_df row) to its ph2 position.
ph2_meta_pos = {int(meta_idx): pos for pos, meta_idx in enumerate(ph2_idx)}

def plot_sample_with_labels(meta_idx, diff_vmin=-30, diff_vmax=30):
    row   = meta_df.iloc[meta_idx]
    sidx  = int(row['_split_idx'])
    split = row['split']

    X = {'train': X_tr_norm, 'val': X_val_norm, 'test': X_test_norm}[split]
    Y = {'train': Y_tr_norm, 'val': Y_val_norm, 'test': Y_test_norm}[split]

    tx    = X[sidx, :, :, 0]
    bld   = X[sidx, :, :, 1] * 60.0
    r0b1  = to_dbm(X[sidx, :, :, 3])
    r1b1  = to_dbm(Y[sidx, :, :, 0])
    diff  = diff_all[meta_idx]

    # Look up label map from ph2_clean labels using the position mapping
    ph2_pos = ph2_meta_pos.get(meta_idx, None)
    if ph2_pos is None:
        print(f'Warning: meta_idx={meta_idx} is not in ph2_clean — label unavailable.')
        return
    label = labels_ph2[ph2_pos]   # (100, 100) uint8: 0=shadow, 1=MP, 2=LOS

    fig, axes = plt.subplots(1, 6, figsize=(26, 4))
    fig.suptitle(f"{row['city']}  |  ROI {row['roi_id']}  |  TX {row['tx_idx']}  |  {split}", fontsize=12)

    im0 = axes[0].imshow(tx,    cmap='Reds')
    im1 = axes[1].imshow(bld,   cmap='gray')
    im2 = axes[2].imshow(r0b1,  cmap='jet',    vmin=-120, vmax=0)
    im3 = axes[3].imshow(r1b1,  cmap='jet',    vmin=-120, vmax=0)
    im4 = axes[4].imshow(diff,  cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)

    axes[0].set_title('TX location')
    axes[1].set_title('Building height (m)')
    axes[2].set_title('R0B1 (dBm)')
    axes[3].set_title('R1B1 (dBm)')
    axes[4].set_title(f'diff [{diff_vmin}, {diff_vmax}] dBm')

    # Label map: 0=deep shadow (black), 1=multipath (red), 2=LOS (white/ignored)
    cmap_label = plt.cm.colors.ListedColormap(['black', 'red', 'white'])
    im5 = axes[5].imshow(label, cmap=cmap_label, vmin=0, vmax=2)
    axes[5].set_title('Regime label')
    cbar = plt.colorbar(im5, ax=axes[5], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Shadow', 'Multipath', 'OpenSpace'])

    for ax, im in zip(axes[:5], [im0, im1, im2, im3, im4]):
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()


# ── Plot random samples from ph2_clean ───────────────────────────────────────
rng = np.random.default_rng(42)
sample_idxs = rng.choice(ph2_idx, size=5, replace=False)

for meta_idx in sample_idxs:
    plot_sample_with_labels(meta_idx)

# ── Save ph2_clean dataset ────────────────────────────────────────────────────
#
#  Three-stage filtering applied:
#    ph1  : samples with mean diff < 0 or >= 20 dBm removed        (sample-level)
#    ph2  : samples with more than 50 class-B pixels (diff < -25) removed
#           (sample-level, based on a pixel count)
#    ph2.5: samples where R0B1 has no direct-path signal anywhere except
#           the TX pixel itself removed (geometry/pipeline artifact)
#
#  Saves per split:
#    X_{split}_ph2.npy  — 7-channel input  (N, 100, 100, 7)
#    Y_{split}_ph2.npy  — R1B1 target      (N, 100, 100, 1)
#    L_{split}_ph2.npy  — regime labels    (N, 100, 100) uint8
#
#  Also saves the full removal audit trail (all samples dropped at any stage,
#  with reason and stage) as removed_samples_audit.csv.
# ─────────────────────────────────────────────────────────────────────────────

PH2_DIR = f'{SAVE_DIR}/ph2_clean'
os.makedirs(PH2_DIR, exist_ok=True)

for split_name, X_norm, Y_norm in [('train', X_tr_norm,   Y_tr_norm),
                                    ('val',   X_val_norm,  Y_val_norm),
                                    ('test',  X_test_norm, Y_test_norm)]:

    # Boolean mask over ph2_meta_df rows for this split
    split_mask = (ph2_meta_df['split'] == split_name).values
    # Within-split position for each ph2 row
    idxs       = ph2_meta_df.loc[split_mask, '_split_idx'].values

    X_save = X_norm[idxs]            # (N, 100, 100, 7)
    Y_save = Y_norm[idxs]            # (N, 100, 100, 1)
    L_save = labels_ph2[split_mask]  # (N, 100, 100) uint8 — ph2-indexed directly

    np.save(f'{PH2_DIR}/X_{split_name}_ph2.npy', X_save)
    np.save(f'{PH2_DIR}/Y_{split_name}_ph2.npy', Y_save)
    np.save(f'{PH2_DIR}/L_{split_name}_ph2.npy', L_save)

    print(f'{split_name:5s}  X: {X_save.shape}  Y: {Y_save.shape}  L: {L_save.shape}')

# ── Metadata, config, and removal audit ──────────────────────────────────────
ph2_meta_df.to_csv(f'{PH2_DIR}/split_metadata_ph2.csv', index=False)

all_removed_df.to_csv(f'{PH2_DIR}/removed_samples_audit.csv', index=False)
print(f'\nSaved removal audit : {len(all_removed_df):,} rows → '
      f'{PH2_DIR}/removed_samples_audit.csv')
print(all_removed_df.groupby(['stage', 'removal_reason']).size().to_string())

ph2_cfg = {
    'phase'                    : 'ph2_clean',
    'original_samples'         : int(len(meta_df)),
    'ph1_samples'              : int(len(ph1_idx)),
    'ph2_draft_samples'        : int(len(ph2_pre_txfilter_idx)),   # after Class B filter, before TX-lock
    'ph2_samples'              : int(len(ph2_idx)),                # final, after all filters
    'total_removed'            : int(len(meta_df) - len(ph2_idx)),
    'filter_ph1_mean_min'      : 0.0,
    'filter_ph1_mean_max'      : 20.0,
    'filter_ph2_class_b_max'   : 50,
    'filter_ph2_5_tx_locked'   : 'r0b1 real nowhere except TX pixel',
    'filter_ph2_5_removed'     : int(tx_locked_mask.sum()),
    'n_channels'               : int(X_tr_norm.shape[-1]),
    'n_classes'                : 3,
    'class_map'                : {'0': 'deep_shadow', '1': 'multipath', '2': 'LOS'},
    'phys_min'                 : PHYS_MIN,
    'phys_max'                 : 0.0,
    'random_seed'              : RANDOM_SEED,
    'removal_audit_file'       : 'removed_samples_audit.csv',
}

with open(f'{PH2_DIR}/config_ph2.json', 'w') as f:
    json.dump(ph2_cfg, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"Split":<8} {"Original":>12} {"ph1_clean":>12} {"ph2_clean":>12} {"Total removed":>14}')
print('-' * 62)
for split_name in ['train', 'val', 'test']:
    orig = (meta_df['split']     == split_name).sum()
    ph1  = (ph1_meta_df['split'] == split_name).sum()
    ph2  = (ph2_meta_df['split'] == split_name).sum()
    print(f'{split_name:<8} {orig:>12,} {ph1:>12,} {ph2:>12,} {orig-ph2:>14,}')
print('-' * 62)
print(f'{"TOTAL":<8} {len(meta_df):>12,} {len(ph1_idx):>12,} {len(ph2_idx):>12,} {len(meta_df)-len(ph2_idx):>14,}')
print(f'\nSaved to : {PH2_DIR}')
print(json.dumps(ph2_cfg, indent=2))