"""
step11_direction_cos_sin.py

Compute the cosine and sine of the direction angle from the TX cell to every
cell, for every TX of every ROI (100 x 100).

Reads the TX maps from the _TX_locations.h5 files
(datasets Tx_1 ... Tx_N, float32, 1.0 at the TX cell, 0 elsewhere).

Input folder structure (TX_LOCATIONS_ROOT):
    <city>_TX_locations/
        <city>_roi_<roi_id>_TX_locations.h5
            Tx_1 … Tx_N   (100×100, float32)

Output folder structure (OUTPUT_ROOT):
    <city>_TX_locations_CosSin/
        <city>_TX_locations_roi_<roi_id>_CosSinTheta.h5
    (the city name keeps the _TX_locations suffix of the input folder)
            Cos_1 … Cos_N  (100×100, float32)
            Sin_1 … Sin_N  (100×100, float32)
"""

import os
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

TX_LOCATIONS_ROOT = (
    "<DATA_ROOT>/"
    "02_Complete/20city_input_tensor/TX_locations"
)

OUTPUT_ROOT = (
    "<DATA_ROOT>/"
    "02_Complete/20city_input_tensor/cos_sin_res100/"
)

MATRIX_RESOLUTION = 100

# Optional: restrict to specific cities/ROIs for testing
# Example: TEST_ONLY = {"Hefei": ["231983"]}
TEST_ONLY = None

# ============================================================
# Cos/Sin computation, vectorized over all cells
# ============================================================

def compute_cos_sin(tx_row: int, tx_col: int, grid_size: int):
    """
    Compute cos(θ) and sin(θ) for every grid cell relative to the TX pixel.

    Convention:
        dx = col - tx_col   (rightward positive)
        dy = tx_row - row   (upward positive — row index increases downward)
        θ  = atan2(dy, dx)

    Returns:
        cos_theta, sin_theta — both shape (grid_size, grid_size), float32
    """
    rows = np.arange(grid_size, dtype=np.float32)
    cols = np.arange(grid_size, dtype=np.float32)
    Col, Row = np.meshgrid(cols, rows)          # shape (G, G)

    dx = Col - tx_col                           # rightward
    dy = tx_row - Row                           # upward (flip row axis)

    theta = np.arctan2(dy, dx)                  # (G, G), float64
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Discover city folders named <city>_TX_locations
city_folders = sorted([
    d for d in os.listdir(TX_LOCATIONS_ROOT)
    if os.path.isdir(os.path.join(TX_LOCATIONS_ROOT, d))
    and d.endswith("_TX_locations")
])

print(f"Found {len(city_folders)} city TX-location folders.")

for city_folder in city_folders:

    city_name = city_folder.replace("_ROI_TX_locations", "")

    if TEST_ONLY is not None and city_name not in TEST_ONLY:
        continue

    city_tx_dir  = os.path.join(TX_LOCATIONS_ROOT, city_folder)
    city_out_dir = os.path.join(OUTPUT_ROOT, f"{city_name}_CosSin")
    os.makedirs(city_out_dir, exist_ok=True)

    tx_files = sorted([
        f for f in os.listdir(city_tx_dir)
        if f.endswith("_TX_locations.h5")
    ])

    print(f"\n=== {city_name}: {len(tx_files)} ROI TX files ===")

    for tx_file in tqdm(tx_files, desc=city_name):

        # Parse ROI id from filename
        try:
            roi_id = tx_file.split("_roi_")[1].split("_")[0]
        except IndexError:
            print(f"  ⚠️ Cannot parse ROI id from: {tx_file}")
            continue

        if TEST_ONLY is not None and roi_id not in TEST_ONLY.get(city_name, []):
            continue

        tx_path  = os.path.join(city_tx_dir, tx_file)
        out_path = os.path.join(city_out_dir, f"{city_name}_roi_{roi_id}_CosSinTheta.h5")

        if os.path.exists(out_path):
            continue   # safe resume

        try:
            with h5py.File(tx_path, "r") as h5tx:
                # Sort Tx keys numerically: Tx_1, Tx_2, …
                tx_keys = sorted(h5tx.keys(), key=lambda k: int(k.split("_")[-1]))
                n_tx = len(tx_keys)

                with h5py.File(out_path, "w") as h5out:
                    for tx_key in tx_keys:
                        tx_idx = int(tx_key.split("_")[-1])   # 1-based

                        tx_matrix = h5tx[tx_key][:]            # (100, 100), float32

                        # Locate TX pixel (value == 1.0)
                        tx_pos = np.argwhere(tx_matrix == 1.0)

                        if len(tx_pos) != 1:
                            print(f"  ⚠️ {city_name} ROI {roi_id} {tx_key}: "
                                  f"expected 1 TX pixel, found {len(tx_pos)} — skipping")
                            continue

                        tx_row, tx_col = tx_pos[0]

                        cos_theta, sin_theta = compute_cos_sin(
                            int(tx_row), int(tx_col), MATRIX_RESOLUTION
                        )

                        h5out.create_dataset(
                            f"Cos_{tx_idx}",
                            data=cos_theta,
                            compression="gzip"
                        )
                        h5out.create_dataset(
                            f"Sin_{tx_idx}",
                            data=sin_theta,
                            compression="gzip"
                        )

            print(f"  ✅ {city_name} ROI {roi_id} — {n_tx} Tx pairs saved.")

        except Exception as e:
            print(f"  ❌ Error | {city_name} ROI {roi_id}: {e}")
            # Remove partial output to allow clean retry
            if os.path.exists(out_path):
                os.remove(out_path)

print("\n=== ALL Cos/Sin MAPS (100×100) COMPLETE ===")

# ============================================================
# OPTIONAL VISUALIZATION  (set VISUALIZE = True to enable)
# ============================================================

VISUALIZE = False

if VISUALIZE:

    VIZ_CITY  = "Hefei"
    VIZ_ROI   = "231983"
    VIZ_TX    = [1, 5]    # 1-based Tx indices to plot

    viz_path = os.path.join(
        OUTPUT_ROOT,
        f"{VIZ_CITY}_CosSin",
        f"{VIZ_CITY}_roi_{VIZ_ROI}_CosSinTheta.h5"
    )

    if not os.path.exists(viz_path):
        print(f"⚠️ Visualization file not found: {viz_path}")
    else:
        with h5py.File(viz_path, "r") as h5file:
            for idx in VIZ_TX:
                cos = h5file[f"Cos_{idx}"][:]
                sin = h5file[f"Sin_{idx}"][:]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                fig.suptitle(f"{VIZ_CITY} – ROI {VIZ_ROI} – Tx #{idx}")

                im0 = axes[0].imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
                axes[0].set_title("Cos(θ)")
                plt.colorbar(im0, ax=axes[0], fraction=0.046)

                im1 = axes[1].imshow(sin, cmap="coolwarm", vmin=-1, vmax=1)
                axes[1].set_title("Sin(θ)")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                # Quiver vector field
                step = 5
                G = MATRIX_RESOLUTION
                X, Y = np.meshgrid(np.arange(G), np.arange(G))
                axes[2].quiver(
                    X[::step, ::step], Y[::step, ::step],
                    cos[::step, ::step], sin[::step, ::step],
                    scale=25, color="black"
                )
                axes[2].invert_yaxis()
                axes[2].set_title("Direction field")
                axes[2].set_aspect("equal")

                plt.tight_layout()
                plt.show()
