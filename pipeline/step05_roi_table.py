#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the ROI grid (All_Cities_ROI_Grid1.csv) with the ROI building summary
(ROISummary.csv) and write ROI_df.csv: one row per ROI with at least one
building of 60 m or more, with its bounding box and building counts.
"""

import pandas as pd

GRID_CSV = "<DATA_ROOT>/01_Process/01.0_bbox/All_Cities_ROI_Grid1.csv"
SUMMARY_CSV = "<DATA_ROOT>/01_Process/0.1_Buildings_summary_and_footprint/ROI_Summary_50City_TallBuildings_v02/ROISummary.csv"
OUT_CSV = "<DATA_ROOT>/02_Complete/ROI_data/ROI_df.csv"

# ----------------------------
# Load
# ----------------------------
grid = pd.read_csv(GRID_CSV)       # All ROIs (grid), bbox columns
summ = pd.read_csv(SUMMARY_CSV)    # ROIs listed in ROISummary.csv, with building counts

# ----------------------------
# Validate columns
# ----------------------------
required_grid_cols = {"City","Country","Max_Latitude","Min_Longitude","Min_Latitude","Max_Longitude"}
required_summ_cols = {"City","ROI","BuildingsCount","Above60mCount"}

missing_grid = required_grid_cols - set(grid.columns)
missing_summ = required_summ_cols - set(summ.columns)

if missing_grid:
    raise ValueError(f"All_Cities_ROI_Grid1 missing columns: {sorted(missing_grid)}")
if missing_summ:
    raise ValueError(f"ROISummary missing columns: {sorted(missing_summ)}")

# ROI is the raw index in grid
summ["ROI"] = pd.to_numeric(summ["ROI"], errors="raise").astype(int)

# Bounds check
bad = summ[(summ["ROI"] < 0) | (summ["ROI"] >= len(grid))]
if len(bad) > 0:
    raise ValueError(
        f"{len(bad)} ROI ids are out of bounds for grid (0..{len(grid)-1}). "
        f"Example: {bad['ROI'].head(10).tolist()}"
    )

# ----------------------------
# Merge bbox from grid using ROI as index
# ----------------------------
grid_with_roi = grid.reset_index().rename(columns={"index": "ROI"})

ROI_df = summ.merge(grid_with_roi, on="ROI", how="left", suffixes=("_summary", ""))

# Keep clean City from grid (ROISummary city might be like 'Abu Dhabi_ROI')
if "City_summary" in ROI_df.columns:
    ROI_df = ROI_df.drop(columns=["City_summary"])

# Reorder columns
ROI_df = ROI_df[
    ["City","Country","ROI",
     "Min_Latitude","Max_Latitude","Min_Longitude","Max_Longitude",
     "BuildingsCount","Above60mCount"]
].sort_values(["City","ROI"], ignore_index=True)

# Save
ROI_df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Shape:", ROI_df.shape)
print(ROI_df.head(10))

print(len(summ), len(ROI_df))                 # should match
print(ROI_df["ROI"].duplicated().sum())       # should be 0 unless ROISummary itself has duplicate ROI ids
print(ROI_df[["Min_Latitude","Max_Latitude","Min_Longitude","Max_Longitude"]].isna().any())
