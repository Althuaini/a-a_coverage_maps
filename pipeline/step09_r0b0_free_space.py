#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate R0B0 (free-space received power) maps for the ROIs listed in
df_20city_rois.csv, on an azimuthal equidistant (AEQD) projection centred on
each ROI.

GRID_SIZE sets the grid (100 for the data) and the output folder name. ROIs
already written are skipped. Maps are stored as np.fliplr(ss.T). Setting
COMPARE = True compares the maps with the MATLAB output.
"""

import os
import numpy as np
import pandas as pd
import h5py
import pyproj
from tqdm import tqdm

# ============================================================
# CONFIGURATION — change here only
# ============================================================

GRID_SIZE    = 100       # single source of truth: 50 or 100
HALF_SPAN_M  = 500.0     # ROI half-width in metres (fixed — 1km × 1km)
FREQ_HZ      = 1.9e9     # carrier frequency Hz
TX_POWER_DBM = 40.0      # transmit power dBm

BASE_TX_FOLDER = (
    "<DATA_ROOT>/"
    "01_Process/01.2_txrx_generation/Tx_Data200feet/"
)
BBOX_CSV = (
    "<MATLAB_DRIVE>/Test/"
    "All_Cities_ROI_Grid1.csv"
)
DF_WHITELIST = (
    "<DATA_ROOT>/"
    "01_Process/01.3_Coverage_Maps/df_20city_rois.csv"
)
OUTPUT_FOLDER = (
    f"<DATA_ROOT>/"
    f"01_Process/01.3_Coverage_Maps/20City_R0B0_Res{GRID_SIZE}/"
)

# Optional test filter — set to None for full run
# Example: TEST_ONLY = {"Hefei": ["231983"]}
TEST_ONLY = None

# ============================================================
# RF HELPER
# ============================================================

def fspl_db(dist_m: np.ndarray, freq_hz: float) -> np.ndarray:
    """
    Free-space path loss (dB):
      FSPL = 20*log10(d) + 20*log10(f) - 147.55
    d in metres, f in Hz. Distance clamped to >= 1m to avoid log(0).
    """
    dist_m = np.maximum(dist_m, 1.0)
    return 20.0 * np.log10(dist_m) + 20.0 * np.log10(freq_hz) - 147.55

# ============================================================
# AEQD GEOMETRY HELPER
# ============================================================

def make_aeqd_projector(center_lat: float, center_lon: float):
    """
    WGS84 (lon, lat) → local AEQD metres (x, y) centred on ROI.
    always_xy=True: transformer.transform(lon, lat) → (x, y)
    """
    aeqd  = pyproj.Proj(proj="aeqd", lat_0=center_lat, lon_0=center_lon,
                        datum="WGS84", units="m")
    wgs84 = pyproj.Proj("EPSG:4326")
    return pyproj.Transformer.from_proj(wgs84, aeqd, always_xy=True)

# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

bbox_df = pd.read_csv(BBOX_CSV)

# Load df whitelist — only process ROIs in this list
df_wl = pd.read_csv(DF_WHITELIST, dtype=str)
df_wl['roi_id'] = df_wl['roi_id'].str.strip()
df_wl['city']   = df_wl['city'].str.strip()
whitelist = set(zip(df_wl['city'], df_wl['roi_id']))
print(f"Loaded df whitelist: {len(whitelist)} ROIs across "
      f"{df_wl['city'].nunique()} cities.")

# Locate all TX CSV files
all_tx_files = []
for root, _, files in os.walk(BASE_TX_FOLDER):
    for f in files:
        if f.endswith("_tx_rx_pairs.csv"):
            all_tx_files.append(os.path.join(root, f))

print(f"Found {len(all_tx_files)} TX files. Filtering against whitelist...")

# Pre-filter TX files to whitelist
filtered_tx_files = []
for tx_csv in all_tx_files:
    fname     = os.path.basename(tx_csv)
    parts     = fname.split("_roi_")
    if len(parts) < 2:
        continue
    city_name = parts[0]
    roi_id    = parts[1].split("_")[0]
    if TEST_ONLY is not None:
        if city_name not in TEST_ONLY or roi_id not in TEST_ONLY[city_name]:
            continue
    if (city_name, roi_id) in whitelist:
        filtered_tx_files.append((tx_csv, city_name, roi_id))

print(f"Processing {len(filtered_tx_files)} whitelisted ROIs at "
      f"{GRID_SIZE}×{GRID_SIZE} resolution.")

# ============================================================
# Build RX grid — same for every ROI
# ============================================================
x  = np.linspace(-HALF_SPAN_M, HALF_SPAN_M, GRID_SIZE)  # columns
y  = np.linspace(-HALF_SPAN_M, HALF_SPAN_M, GRID_SIZE)  # rows
Xs, Ys = np.meshgrid(x, y, indexing="xy")               # (GRID_SIZE, GRID_SIZE)

# ============================================================
# Process each ROI
# ============================================================
for tx_csv, city_name, roi_id in tqdm(filtered_tx_files, desc="R0B0"):
    try:
        roi_idx = int(roi_id)
        if roi_idx < 0 or roi_idx >= len(bbox_df):
            print(f"⚠️ ROI index {roi_idx} out of range — skipping.")
            continue

        roi_row    = bbox_df.iloc[roi_idx]
        center_lat = (float(roi_row["Min_Latitude"])  + float(roi_row["Max_Latitude"]))  / 2.0
        center_lon = (float(roi_row["Min_Longitude"]) + float(roi_row["Max_Longitude"])) / 2.0

        city_out_dir = os.path.join(OUTPUT_FOLDER, city_name)
        os.makedirs(city_out_dir, exist_ok=True)

        output_h5 = os.path.join(
            city_out_dir,
            f"Output_{city_name}_roi_{roi_id}_R0B0.h5"
        )
        if os.path.exists(output_h5):
            continue

        transformer = make_aeqd_projector(center_lat, center_lon)

        tx_df = pd.read_csv(tx_csv)
        if "txLat" not in tx_df.columns or "txLon" not in tx_df.columns:
            print(f"⚠️ Missing txLat/txLon in {tx_csv}")
            continue

        with h5py.File(output_h5, "w") as h5f:
            for i, row in tx_df.iterrows():
                tx_x, tx_y = transformer.transform(
                    float(row["txLon"]), float(row["txLat"])
                )
                dists = np.sqrt((Xs - tx_x)**2 + (Ys - tx_y)**2)
                ss    = TX_POWER_DBM - fspl_db(dists, FREQ_HZ)

                # np.fliplr(ss.T) aligns the AEQD grid to match
                # MATLAB's flipud(reshape(...)) orientation convention.
                # Verified against MATLAB R0B0 output at 50×50.
                ss = np.fliplr(ss.T)

                h5f.create_dataset(
                    f"Tx_{i+1}",
                    data=ss.astype(np.float32),
                    compression="gzip"
                )

    except Exception as e:
        print(f"❌ Error {city_name} ROI {roi_id}: {e}")
        continue

print("\n=== ALL R0B0 MAPS COMPLETED ===")

# ============================================================
# OPTIONAL COMPARISON — gate with COMPARE = True
# ============================================================
COMPARE = False

if COMPARE:
    import matplotlib.pyplot as plt

    cmp_city  = "Hefei"
    cmp_roi   = "231983"
    num_tx    = 10
    DB_FLOOR  = -120.0

    mat_h5 = (
        "<DATA_ROOT>/"
        "01_Process/01.3_Coverage_Maps/20City/Hefei/"
        f"Output_Hefei_roi_{cmp_roi}_R0B0.h5"
    )
    py_h5 = os.path.join(
        OUTPUT_FOLDER, cmp_city,
        f"Output_{cmp_city}_roi_{cmp_roi}_R0B0.h5"
    )

    if not os.path.exists(mat_h5) or not os.path.exists(py_h5):
        print("⚠️ Comparison files not found — skipping.")
    else:
        for tx_index in range(1, num_tx + 1):
            with h5py.File(py_h5,  "r") as f_py, \
                 h5py.File(mat_h5, "r") as f_mat:
                ss_py  = f_py[f"Tx_{tx_index}"][:]
                ss_mat = f_mat[f"Tx_{tx_index}"][:]

            los_mask  = (ss_mat > DB_FLOOR + 1e-6)
            los_mask &= (ss_mat != np.max(ss_mat))
            diff      = ss_py[los_mask] - ss_mat[los_mask]

            print(f"\nTx_{tx_index} | LOS px: {los_mask.sum()} | "
                  f"Mean diff: {np.mean(diff):.4f} dB | "
                  f"Std: {np.std(diff):.4f} dB | "
                  f"Max|diff|: {np.max(np.abs(diff)):.4f} dB")

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(ss_mat, cmap="plasma", vmin=-120, vmax=0)
            axes[0].set_title(f"MATLAB R0B0 Tx_{tx_index}")
            axes[1].imshow(ss_py,  cmap="plasma", vmin=-120, vmax=0)
            axes[1].set_title(f"Python R0B0 Tx_{tx_index}")
            im = axes[2].imshow(ss_py - ss_mat, cmap="bwr", vmin=-3, vmax=3)
            axes[2].set_title("Difference (Python − MATLAB) dB")
            for ax in axes:
                ax.axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)
            plt.tight_layout()
            plt.show()

    print("\n✅ R0B0 COMPARISON COMPLETED")
