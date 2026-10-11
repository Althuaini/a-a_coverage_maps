#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute R0B1 maps: R0B0 with the direct path set to the no-signal value
(-120 dBm) where a building taller than the UAV altitude lies on the
Bresenham line from the TX cell.

Inputs per ROI:
  {city}_roi_{id}_allBuildings_heights.h5   building_matrix (heights in metres)
  {city}_roi_{id}_TX_locations.h5           Tx_1 ... Tx_50 (1.0 at the TX cell)
  the R0B0 file of the ROI
The blocking mask is computed once per ROI.
"""

import os
import h5py
import numpy as np
from skimage.draw import line
from tqdm import tqdm
import pandas as pd

# ============================================================
# CONFIGURATION — change here only
# ============================================================

GRID_SIZE    = 100          # single source of truth: 50 or 100
DB_FLOOR     = -120.0       # noise floor dBm
TX_HEIGHT_M  = 60.96        # 200 feet in metres — only buildings above
                             # this height block the ray

# v0.4 File 1 — building height matrices (one per ROI, no TX info)
BLD_MAT_DIR = (
    "<DATA_ROOT>/"
    "01_Process/01.4_Building_Matrix/20City/All_buildings_heights_res100/"
)

# v0.4 File 2 — TX location matrices (50 per ROI, binary 0/1)
TX_LOC_DIR = (
    "<DATA_ROOT>/"
    "02_Complete/20city_input_tensor/TX_locations/All_buildings_heights"
)

R0B0_DIR = (
    f"<DATA_ROOT>/"
    f"01_Process/01.3_Coverage_Maps/20City_R0B0_Res{GRID_SIZE}"
)
OUT_DIR = (
    f"<DATA_ROOT>/"
    f"01_Process/01.3_Coverage_Maps/20City_R0B1_Res{GRID_SIZE}"
)
DF_WHITELIST = (
    "<DATA_ROOT>/"
    "01_Process/01.3_Coverage_Maps/df_20city_rois.csv"
)

# Optional test filter — e.g. {"Hefei": ["231983"]}
TEST_ROI_IDS = None

# ============================================================
# MAIN
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

# Load df whitelist
df_wl = pd.read_csv(DF_WHITELIST, dtype=str)
df_wl['roi_id'] = df_wl['roi_id'].str.strip()
df_wl['city']   = df_wl['city'].str.strip()
whitelist = set(zip(df_wl['city'], df_wl['roi_id']))
print(f"Loaded df whitelist: {len(whitelist)} ROIs across "
      f"{df_wl['city'].nunique()} cities.")
print(f"TX blocking height threshold: {TX_HEIGHT_M} m "
      f"({TX_HEIGHT_M/0.3048:.0f} ft)")
print(f"Grid size: {GRID_SIZE}×{GRID_SIZE}")

# Discover city folders from the buildings heights directory
city_folders = sorted([
    d for d in os.listdir(BLD_MAT_DIR)
    if os.path.isdir(os.path.join(BLD_MAT_DIR, d))
])
print(f"Found {len(city_folders)} city folders in BLD_MAT_DIR.")

for city_folder in city_folders:

    # Folder name: <City>_allBuildings -> <City>
    base_city = city_folder.split("_ROI_")[0]

    if TEST_ROI_IDS is not None and base_city not in TEST_ROI_IDS:
        continue

    city_bld_dir  = os.path.join(BLD_MAT_DIR, city_folder)
    city_tx_dir   = os.path.join(TX_LOC_DIR,
                                 f"{base_city}_ROI_ROI_TX_locations")
    city_r0b0_dir = os.path.join(R0B0_DIR, base_city)
    city_out_dir  = os.path.join(OUT_DIR,  base_city)

    if not os.path.isdir(city_r0b0_dir):
        print(f"⚠️ Missing R0B0 dir for {base_city} — skipping.")
        continue
    if not os.path.isdir(city_tx_dir):
        print(f"⚠️ Missing TX_locations dir for {base_city} — skipping.")
        continue

    os.makedirs(city_out_dir, exist_ok=True)

    # All building height files for this city
    bld_files = sorted([
        f for f in os.listdir(city_bld_dir) if f.endswith(".h5")
    ])
    if not bld_files:
        print(f"⚠️ No height H5 files in {city_bld_dir}")
        continue

    print(f"\n=== {base_city} — {len(bld_files)} ROIs ===")

    for bld_name in tqdm(bld_files, desc=base_city):

        # Parse ROI ID from filename
        # e.g. Bangkok_roi_379429_allBuildings_heights.h5
        try:
            roi_id = bld_name.split("_roi_")[1].split("_")[0]
        except (IndexError, Exception):
            print(f"⚠️ Could not parse ROI from {bld_name}")
            continue

        # Whitelist check
        if (base_city, roi_id) not in whitelist:
            continue

        if TEST_ROI_IDS is not None:
            if roi_id not in TEST_ROI_IDS.get(base_city, []):
                continue

        # Build file paths
        bld_path  = os.path.join(city_bld_dir, bld_name)
        tx_path   = os.path.join(
            city_tx_dir,
            f"{base_city}_roi_{roi_id}_TX_locations.h5"
        )
        r0b0_path = os.path.join(
            city_r0b0_dir,
            f"Output_{base_city}_roi_{roi_id}_R0B0.h5"
        )
        out_path  = os.path.join(
            city_out_dir,
            f"Output_{base_city}_roi_{roi_id}_R0B1.h5"
        )

        if not os.path.exists(r0b0_path):
            print(f"⚠️ Missing R0B0: {r0b0_path}")
            continue
        if not os.path.exists(tx_path):
            print(f"⚠️ Missing TX_locations: {tx_path}")
            continue
        if os.path.exists(out_path):
            continue

        try:
            # ── Load building height matrix (once per ROI) ──────────
            with h5py.File(bld_path, "r") as f_bld:
                # shape (GRID_SIZE, GRID_SIZE), float32
                # 0.0 = free space, h > 0 = height in metres
                height_mat = f_bld["building_matrix"][:].T  # .T at load (needed for the storage layout this was run on; see README note 1)

            # ── Blocking mask (once per ROI) ─────────────────────────
            # True where building height exceeds the UAV altitude (TX_HEIGHT_M).
            # Same for all 50 TX — building geometry does not change.
            blocking_mask = (height_mat > TX_HEIGHT_M)

            # ── Process each TX ──────────────────────────────────────
            with h5py.File(tx_path,   "r") as f_tx, \
                 h5py.File(r0b0_path, "r") as f_r0b0, \
                 h5py.File(out_path,  "w") as f_out:

                tx_keys = sorted(
                    f_tx.keys(),
                    key=lambda k: int(k.split("_")[-1])
                )

                for tx_key in tx_keys:

                    # TX location matrix: 0.0 everywhere, 1.0 at TX cell
                    tx_mat = f_tx[tx_key][:].T   # .T at load (needed for the storage layout this was run on; see README note 1)

                    tx_pos = np.argwhere(tx_mat == 1.0)
                    if len(tx_pos) != 1:
                        print(f"⚠️ Invalid TX marker in {tx_key} "
                              f"of ROI {roi_id}")
                        continue

                    ty, tx = tx_pos[0]

                    if tx_key not in f_r0b0:
                        print(f"⚠️ Missing R0B0 dataset {tx_key}")
                        continue

                    r0b0 = f_r0b0[tx_key][:]
                    r0b1 = r0b0.copy()

                    # Bresenham LOS blockage
                    for ry in range(GRID_SIZE):
                        for rx in range(GRID_SIZE):
                            if ry == ty and rx == tx:
                                continue
                            rr, cc = line(ty, tx, ry, rx)
                            # rr[0], cc[0] is the TX cell — skip it
                            if np.any(blocking_mask[rr[1:], cc[1:]]):
                                r0b1[ry, rx] = DB_FLOOR

                    # Preserve TX cell power from R0B0
                    r0b1[ty, tx] = r0b0[ty, tx]

                    f_out.create_dataset(
                        tx_key,
                        data=r0b1.astype(np.float32),
                        compression="gzip"
                    )

        except Exception as e:
            print(f"❌ Exception for {base_city} ROI {roi_id}: {e}")
            continue

print("\n=== ALL R0B1 MAPS COMPLETED ===")

# ============================================================
# OPTIONAL COMPARISON — gate with COMPARE = True
# ============================================================
COMPARE = False

if COMPARE:
    import matplotlib.pyplot as plt

    cmp_city = "Hefei"
    cmp_roi  = "231983"
    num_tx   = 10

    py_h5  = os.path.join(OUT_DIR, cmp_city,
                          f"Output_{cmp_city}_roi_{cmp_roi}_R0B1.h5")
    mat_h5 = (
        "<DATA_ROOT>/"
        "01_Process/01.3_Coverage_Maps/20City/Hefei/"
        f"Output_Hefei_roi_{cmp_roi}_R0B1.h5"
    )
    bld_h5 = os.path.join(
        BLD_MAT_DIR,
        f"Hefei_ROI_Height_matrix_allBuildings",
        f"Hefei_roi_{cmp_roi}_allBuildings_heights.h5"
    )
    tx_h5 = os.path.join(
        TX_LOC_DIR,
        f"Hefei_ROI_TX_locations",
        f"Hefei_roi_{cmp_roi}_TX_locations.h5"
    )

    if not os.path.exists(py_h5) or not os.path.exists(mat_h5):
        print("⚠️ Comparison files not found — skipping.")
    else:
        # Load blocking buildings for overlay
        bld_mat = None
        if os.path.exists(bld_h5):
            with h5py.File(bld_h5, "r") as hf:
                bld_mat = hf["building_matrix"][:].T

        for tx_index in range(1, num_tx + 1):
            with h5py.File(py_h5,  "r") as f_py, \
                 h5py.File(mat_h5, "r") as f_mat:
                ss_py  = f_py[f"Tx_{tx_index}"][:]
                ss_mat = f_mat[f"Tx_{tx_index}"][:]

            los_mask  = (ss_mat > DB_FLOOR + 1e-6)
            los_mask &= (ss_mat != np.max(ss_mat))
            diff = ss_py[los_mask] - ss_mat[los_mask]

            print(f"\nTx_{tx_index} | "
                  f"Blocked(MATLAB): {(ss_mat <= DB_FLOOR).sum()} | "
                  f"Blocked(Python): {(ss_py  <= DB_FLOOR).sum()} | "
                  f"Mean diff: {np.mean(diff):.4f} dB")

            # TX position for overlay
            tx_pos_arr = None
            if os.path.exists(tx_h5):
                with h5py.File(tx_h5, "r") as hf:
                    key = f"Tx_{tx_index}"
                    if key in hf:
                        tx_mat_cmp = hf[key][:].T
                        tx_pos_arr = np.argwhere(tx_mat_cmp == 1.0)

            n_cols = 4 if bld_mat is not None else 3
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

            axes[0].imshow(ss_mat, cmap="plasma", vmin=-120, vmax=0)
            axes[0].set_title(f"MATLAB R0B1 Tx_{tx_index}")
            axes[0].axis('off')

            axes[1].imshow(ss_py, cmap="plasma", vmin=-120, vmax=0)
            axes[1].set_title(f"Python R0B1 Tx_{tx_index}")
            axes[1].axis('off')

            im = axes[2].imshow(ss_py - ss_mat, cmap="bwr", vmin=-5, vmax=5)
            axes[2].set_title("Difference dB")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            if bld_mat is not None:
                axes[3].imshow(ss_py, cmap="plasma", vmin=-120, vmax=0)
                bmask   = (bld_mat > TX_HEIGHT_M)
                overlay = np.zeros((*bmask.shape, 4), dtype=float)
                overlay[bmask] = [0, 1, 1, 0.7]
                axes[3].imshow(overlay)
                if tx_pos_arr is not None and len(tx_pos_arr) > 0:
                    axes[3].scatter(
                        [tx_pos_arr[0, 1]], [tx_pos_arr[0, 0]],
                        s=120, c='red', marker='x', linewidths=2.5
                    )
                axes[3].set_title("R0B1 + blocking buildings overlay")
                axes[3].axis('off')

            plt.suptitle(f"{cmp_city} ROI {cmp_roi} Tx_{tx_index}",
                         fontsize=11)
            plt.tight_layout()
            plt.show()

    print("\n✅ R0B1 COMPARISON COMPLETED")