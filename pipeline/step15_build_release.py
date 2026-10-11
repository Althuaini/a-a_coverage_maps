# -*- coding: utf-8 -*-
"""
step15_build_release.py
=======================
Builds the files of the public data record (Zenodo) for the paper
"A ray-traced air-to-air signal strength dataset for drone-to-drone links
over 17 cities, with per-pixel propagation-regime labels".

The record contains only the cleaned (released) data: 63,270 samples in the
1,444 ROIs that keep at least one sample after cleaning.

HOW TO USE
----------
- Each block starting with "# %%" is a cell (for example in Spyder).
- Run the cells one at a time, in order, and read the printout.
- Every cell ends with checks (assert). If a check fails, stop and look at
  the message. Do not continue to the next cell.
- Edit only Cell 1 (paths and settings).

SAFETY
------
- This script only reads the source folders set in Cell 1.
- It only writes inside OUT_DIR.
- It never changes or deletes any source file.
- The coverage maps are processed one ROI at a time, so memory stays low
  (about 1 to 2 GB).

PARTS OF THE RECORD (as described in the paper)
-----------------------------------------------
A  raw_maps/<city>.h5   per ROI with released samples: coverage maps in dBm,
                        building heights and labels of the released samples
B  osm/<city>.zip       OSM building files of the same ROIs
C  arrays/              X_, Y_, L_{train,val,test}_c.npy (hard links, no copy)
D  samples.csv          one row per released sample, with its TX position
E  rois.csv             one row per ROI with released samples
F  removed_samples.csv  one row per sample removed by cleaning (no maps)
G  README.md, config.json, LICENSE
   SHA256SUMS.txt       checksums of every file
"""

# %% Cell 1 — SETTINGS (edit this cell only)
import os
import re
import glob
import json
import time
import shutil
import hashlib
import zipfile

import numpy as np
import pandas as pd
import h5py

# --- Project and output -------------------------------------------------
PROJECT_DIR = "<PROJECT_DIR>"
OUT_DIR     = os.path.join(PROJECT_DIR, "release", "data")

# --- Released arrays and their metadata (inputs/data) -------------------
PH2_DIR     = os.path.join(PROJECT_DIR, "inputs", "data", "ph2_clean")
PREPROC_DIR = os.path.join(PROJECT_DIR, "inputs", "data", "preprocessed")

# --- External drive -----------------------------------------------------
DATA_ROOT = "<DATA_ROOT>"

# ROI table (city, country, ROI id, bounding box, building counts)
ROI_DF_CSV = os.path.join(DATA_ROOT, "02_Complete", "ROI_data", "ROI_df.csv")

# TX files used for the released data.
# File names look like "<City>_roi_<ID>_tx_rx_pairs.csv".
TX_DIR = os.path.join(DATA_ROOT, "01_Process", "01.2_txrx_generation", "Tx_Data200feet")

# OSM files: <OSM_DIR>/<City>_ROI/<City>_roi_<ID>.osm
OSM_DIR = os.path.join(DATA_ROOT, "01_Process", "01.1_OverpassResponses", "Responses")

# The JSON that step13_preprocessing_split.ipynb used.
# It lists, for every ROI, the 7 HDF5 files used to build the released arrays.
META_JSON = os.path.join(PROJECT_DIR, "inputs", "data", "dataset_meta_res100.json")

# The JSON stores the paths used in Google Colab. The same files are in the
# folders below. The script keeps the last folder and the file name
# from the JSON path (e.g. "Abu Dhabi_TX_locations/Abu Dhabi_roi_383452_TX_locations.h5")
# and puts it under the folder below. Cell 11 later checks that these files
# reproduce the released arrays exactly.
CM = os.path.join(DATA_ROOT, "01_Process", "01.3_Coverage_Maps")
ROOTS = {
    "buildings_h5": os.path.join(DATA_ROOT, "01_Process", "01.4_Building_Matrix", "20City",
                                 "All_buildings_heights_res100"),
    "tx_loc_h5":    os.path.join(DATA_ROOT, "02_Complete", "20city_input_tensor", "TX_locations"),
    "sincos_h5":    os.path.join(DATA_ROOT, "02_Complete", "20city_input_tensor", "cos_sin_res100"),
    "r0b0_h5":      os.path.join(CM, "20City_R0B0_Res100"),
    "r0b1_h5":      os.path.join(CM, "20City_R0B1_Res100"),
    "r1b0_h5":      os.path.join(CM, "20City_Res100"),
    "r1b1_h5":      os.path.join(CM, "20City_Res100"),
}

# --- Settings -----------------------------------------------------------
PHYS_MIN, PHYS_MAX, RANGE_DBM = -120.0, 0.0, 120.0   # coverage normalization
MAX_HEIGHT = 60.0                                    # building-height normalization
LABEL_EPS = 1e-6                                     # no-signal test after normalization
GZIP_LEVEL = 4                                       # compression of raw_maps/
TEST_LIMIT_ROIS = None      # Cell 7 first builds raw_maps/ for only this many ROIs.
                         # Set to None for the full run after the test passes.
N_VERIFY = 1000          # number of random released samples checked in Cell 11

# Expected numbers (from the paper). Every cell checks against these.
EXP = dict(cities=17, rois_simulated=1450, rois=1444, tx_per_roi=50,
           samples_simulated=72500, samples=63270,
           samples_per_split={"train": 44143, "val": 9596, "test": 9531},
           removed=9230, removed_per_stage={1: 4767, 2: 4440, 3: 23})

SPLITS = ["train", "val", "test"]
os.makedirs(OUT_DIR, exist_ok=True)
print("Settings loaded. Output folder:", OUT_DIR)


# %% Cell 2 — LOAD METADATA AND CHECK THAT ALL SOURCE FILES EXIST
H5_KEYS = ["buildings_h5", "tx_loc_h5", "sincos_h5",
           "r0b0_h5", "r0b1_h5", "r1b0_h5", "r1b1_h5"]


def local_path(key, p):
    """Turn a Google Drive path from the JSON into the path on the external drive:
    keep the last folder and the file name, put them under ROOTS[key]."""
    folder, name = os.path.split(p)
    return os.path.join(ROOTS[key], os.path.basename(folder), name)


# Pre-cleaning table: one row per simulated sample (72,500), with its split.
meta_all = pd.read_csv(os.path.join(PREPROC_DIR, "split_metadata_roi_split.csv"))
meta_all["roi_id"] = meta_all["roi_id"].astype(int)
# Released table: one row per released sample (63,270), in the row order of the arrays.
meta_ph2 = pd.read_csv(os.path.join(PH2_DIR, "split_metadata_ph2.csv"))
meta_ph2["roi_id"] = meta_ph2["roi_id"].astype(int)
roi_df = pd.read_csv(ROI_DF_CSV)

assert len(meta_all) == EXP["samples_simulated"], len(meta_all)
assert len(meta_ph2) == EXP["samples"], len(meta_ph2)

with open(META_JSON) as f:
    meta_json = json.load(f)["cities"]

sim_rois = meta_all[["city", "roi_id"]].drop_duplicates()
# ROIs with released samples: the only ROIs in the record.
rel_rois = (meta_ph2[["city", "roi_id"]].drop_duplicates()
            .sort_values(["city", "roi_id"]).reset_index(drop=True))

# Map (city, roi_id) -> dict of local HDF5 paths, for the released ROIs only.
roi_paths, missing = {}, []
for city, roi_id in rel_rois.itertuples(index=False):
    entry = meta_json[city]["rois"][str(roi_id)]
    paths = {k: local_path(k, entry[k]) for k in H5_KEYS}
    roi_paths[(city, roi_id)] = paths
    missing += [p for p in paths.values() if not os.path.exists(p)]

print("Example paths for the first ROI:")
for k, v in roi_paths[tuple(rel_rois.iloc[0])].items():
    print(f"  {k:13s} {v}")
print(f"\nSimulated ROIs: {len(sim_rois)}   ROIs with released samples: {len(rel_rois)}   "
      f"cities: {rel_rois.city.nunique()}")
print(f"Missing HDF5 files: {len(missing)}")
for p in missing[:10]:
    print("  MISSING:", p)

assert len(sim_rois) == EXP["rois_simulated"]
assert len(rel_rois) == EXP["rois"]
assert rel_rois.city.nunique() == EXP["cities"]
assert len(missing) == 0, "Some source files are missing. Check ROOTS in Cell 1."
print("\nCell 2 OK.")


# %% Cell 3 — TX POSITIONS of the released ROIs (merged into samples.csv in Cell 4)
tx_files = glob.glob(os.path.join(TX_DIR, "**", "*_tx_rx_pairs.csv"), recursive=True)
name_re = re.compile(r"^(?P<city>.+?)_roi_(?P<roi>\d+)_tx_rx_pairs\.csv$")
tx_lookup = {}
for p in tx_files:
    m = name_re.match(os.path.basename(p))
    if m:
        tx_lookup[(m["city"], int(m["roi"]))] = p

rows = []
for city, roi_id in rel_rois.itertuples(index=False):
    p = tx_lookup.get((city, roi_id))
    assert p is not None, f"No TX file for {city} ROI {roi_id}"
    t = pd.read_csv(p)
    # find the TX columns whatever their exact capitalization
    col = {c.lower(): c for c in t.columns}
    lat, lon, hgt = col["txlat"], col["txlon"], col["txheight"]
    for i, r in enumerate(t.itertuples(index=False), start=1):   # row i = Tx_i
        rows.append((city, roi_id, i,
                     getattr(r, lat), getattr(r, lon), getattr(r, hgt)))

tx_pos = pd.DataFrame(rows, columns=["city", "roi_id", "tx_idx",
                                     "txLat", "txLon", "txHeight"])
per_roi = tx_pos.groupby(["city", "roi_id"]).size()
print(tx_pos.head())
print(f"\nTX rows: {len(tx_pos):,}   TX per ROI: min {per_roi.min()}, max {per_roi.max()}")
print(f"txHeight: min {tx_pos.txHeight.min():.2f} m, max {tx_pos.txHeight.max():.2f} m")
assert len(tx_pos) == EXP["rois"] * EXP["tx_per_roi"]
assert per_roi.min() == per_roi.max() == EXP["tx_per_roi"]
print("TX positions OK.")


# %% Cell 4 — PART D: samples.csv  (row i of a set = row i of that set's arrays)
samples = meta_ph2[["city", "roi_id", "tx_idx", "split"]].copy()
# The arrays of each set keep the order of split_metadata_ph2.csv.
samples["row"] = samples.groupby("split").cumcount()
samples = samples.merge(tx_pos, on=["city", "roi_id", "tx_idx"], how="left", validate="one_to_one")
samples = samples[["city", "roi_id", "tx_idx", "txLat", "txLon", "txHeight", "split", "row"]]
samples.to_csv(os.path.join(OUT_DIR, "samples.csv"), index=False)

counts = samples.split.value_counts().to_dict()
print(samples.head())
print("Released samples per set:", counts)
for s in SPLITS:
    n_arr = np.load(os.path.join(PH2_DIR, f"X_{s}_ph2.npy"), mmap_mode="r").shape[0]
    assert counts[s] == n_arr == EXP["samples_per_split"][s], (s, counts[s], n_arr)
    rr = samples.loc[samples.split == s, "row"].values
    assert (rr == np.arange(len(rr))).all(), f"rows of {s} are not 0..N-1 in file order"
assert not samples.duplicated(["city", "roi_id", "tx_idx"]).any()
assert samples[["txLat", "txLon", "txHeight"]].notna().all().all(), "released sample without TX position"
print("Part D OK.")


# %% Cell 5 — PART F: removed_samples.csv  (deduplicated, stages 1 / 2 / 3)
audit = pd.read_csv(os.path.join(PH2_DIR, "removed_samples_audit.csv"))
audit["roi_id"] = audit["roi_id"].astype(int)
n_before = len(audit)
audit = audit.drop_duplicates(["city", "roi_id", "tx_idx", "stage"])   # ph2.5 rows were logged twice
audit["stage"] = audit["stage"].map({"ph1": 1, "ph2": 2, "ph2.5": 3})
assert audit["stage"].notna().all(), "Unknown stage name in the audit file."
cols = ["city", "roi_id", "tx_idx", "split", "stage",
        "removal_reason", "mean_diff_dbm", "class_b_px"]
removed = audit.reindex(columns=cols).sort_values(["stage", "city", "roi_id", "tx_idx"])
removed.to_csv(os.path.join(OUT_DIR, "removed_samples.csv"), index=False)

print(f"Audit rows: {n_before:,} -> {len(removed):,} after removing duplicates")
print(removed.groupby("stage").size())
assert len(removed) == EXP["removed"]
assert removed.groupby("stage").size().to_dict() == EXP["removed_per_stage"]
# released + removed must be exactly all simulated samples, with no overlap
key = lambda d: set(zip(d.city, d.roi_id, d.tx_idx))
assert not (key(removed) & key(samples))
assert len(key(removed) | key(samples)) == EXP["samples_simulated"]
# simulated ROIs = ROIs in samples.csv plus ROIs that appear only in removed_samples.csv
assert len(set(zip(removed.city, removed.roi_id)) | set(zip(samples.city, samples.roi_id))) == EXP["rois_simulated"]
print("Part F OK.")


# %% Cell 6 — PART E: rois.csv  (ROIs with released samples only)
roi_df["ROI"] = roi_df["ROI"].astype(int)
split_of_roi = meta_ph2.groupby(["city", "roi_id"])["split"].first()
released_per_roi = samples.groupby(["city", "roi_id"]).size()

rois = rel_rois.merge(roi_df.rename(columns={"City": "city", "ROI": "roi_id"}),
                      on=["city", "roi_id"], how="left")
rois["split"] = [split_of_roi[(c, r)] for c, r in zip(rois.city, rois.roi_id)]
rois["n_released"] = [int(released_per_roi[(c, r)]) for c, r in zip(rois.city, rois.roi_id)]
rois = rois.rename(columns={"Country": "country",
                            "Min_Latitude": "min_lat", "Max_Latitude": "max_lat",
                            "Min_Longitude": "min_lon", "Max_Longitude": "max_lon",
                            "BuildingsCount": "n_buildings",
                            "Above60mCount": "n_buildings_ge60m"})
rois = rois[["city", "country", "roi_id", "min_lat", "max_lat", "min_lon", "max_lon",
             "n_buildings", "n_buildings_ge60m", "split", "n_released"]]
rois = rois.sort_values(["city", "roi_id"])
rois.to_csv(os.path.join(OUT_DIR, "rois.csv"), index=False)

print(rois.head())
print("ROIs per set:", rois.split.value_counts().to_dict())
print(f"Buildings: {rois.n_buildings.sum():,}   at least 60 m: {rois.n_buildings_ge60m.sum():,}")
assert len(rois) == EXP["rois"]
assert rois[["country", "min_lat", "n_buildings"]].notna().all().all(), "ROI missing from ROI_df.csv"
assert rois.n_released.sum() == EXP["samples"]
assert (rois.n_released > 0).all()
assert (rois.n_buildings_ge60m >= 1).all()
print("Part E OK.")


# %% Cell 7 — PART A: raw_maps/<city>.h5  (run with TEST_LIMIT_ROIS first)
# Each ROI is read with the SAME loading code as step13_preprocessing_split.ipynb
# (no transpose), so the maps have the same orientation as the released arrays.
# Only the released TX of each ROI are written, in increasing tx_idx order.
# The labels are copied from the released label arrays (L_<set>).
def load_tx_keyed(path):
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))   # Tx_1 ... Tx_N
        return np.stack([f[k][:] for k in keys]).astype(np.float32)


RAW_DIR = os.path.join(OUT_DIR, "raw_maps")
L_arr = {s: np.load(os.path.join(PH2_DIR, f"L_{s}_ph2.npy"), mmap_mode="r") for s in SPLITS}
comp = dict(compression="gzip", compression_opts=GZIP_LEVEL, shuffle=True)

todo = rel_rois if TEST_LIMIT_ROIS is None else rel_rois.head(TEST_LIMIT_ROIS)
if TEST_LIMIT_ROIS is not None:
    RAW_DIR = os.path.join(OUT_DIR, "raw_maps_TEST")
    print(f"TEST RUN on {TEST_LIMIT_ROIS} ROIs, writing to {RAW_DIR}")
os.makedirs(RAW_DIR, exist_ok=True)

by_roi = {k: g.sort_values("tx_idx") for k, g in samples.groupby(["city", "roi_id"])}
t0, n_written, n_groups = time.time(), 0, 0
for city, grp in todo.groupby("city", sort=True):
    out_path = os.path.join(RAW_DIR, f"{city.replace(' ', '_')}.h5")
    with h5py.File(out_path, "w") as h5:
        h5.attrs["city"] = city
        h5.attrs["description"] = ("Coverage maps in dBm (not clipped), building heights and "
                                   "propagation-regime labels of the released samples, "
                                   "one group per ROI with released samples.")
        h5.attrs["axes"] = "first axis west to east, second axis north to south"
        for roi_id in grp.roi_id:
            p = roi_paths[(city, roi_id)]
            s_roi = by_roi[(city, roi_id)]
            tx_idx = s_roi.tx_idx.values.astype(np.int16)                 # released TX only
            split = s_roi.split.iloc[0]
            assert (s_roi.split == split).all()
            maps = {name: load_tx_keyed(p[k]) for name, k in
                    [("R0B0", "r0b0_h5"), ("R0B1", "r0b1_h5"),
                     ("R1B0", "r1b0_h5"), ("R1B1", "r1b1_h5")]}
            with h5py.File(p["buildings_h5"], "r") as f:
                bld = f["building_matrix"][:].astype(np.float32)
            assert all(m.shape == (EXP["tx_per_roi"], 100, 100) for m in maps.values())
            assert bld.shape == (100, 100)
            labels = np.asarray(L_arr[split][s_roi.row.values]).astype(np.uint8)

            g = h5.create_group(f"roi_{roi_id}")
            g.attrs.update(city=city, roi_id=int(roi_id), split=str(split))
            g.create_dataset("tx_idx", data=tx_idx).attrs["description"] = \
                "TX index (1 to 50) of each released sample; entry k of every dataset below belongs to tx_idx[k]"
            for name, arr in maps.items():
                d = g.create_dataset(name, data=arr[tx_idx - 1], chunks=(1, 100, 100), **comp)
                d.attrs["units"] = "dBm"
            d = g.create_dataset("BLD", data=bld, **comp)
            d.attrs["units"] = "m"
            d = g.create_dataset("labels", data=labels, chunks=(1, 100, 100), **comp)
            d.attrs["description"] = "0 deep shadow, 1 multipath, 2 line of sight (same as L_<set>_c.npy)"
            n_written += len(tx_idx)
            n_groups += 1
    print(f"  {city:12s} {len(grp):4d} ROIs  "
          f"{os.path.getsize(out_path)/1e9:6.2f} GB  ({(time.time()-t0)/60:.1f} min)")

print(f"\nROI groups written: {n_groups:,}   released samples written: {n_written:,}")
if TEST_LIMIT_ROIS is None:
    assert n_groups == EXP["rois"]
    assert n_written == EXP["samples"]
    print("Part A OK.")
else:
    print("Test run finished. Run Cell 11 (verification) on these ROIs only if you like,\n"
          "then set TEST_LIMIT_ROIS = None in Cell 1, re-run Cell 1, Cell 2 and this cell,\n"
          "and delete the raw_maps_TEST folder.")


# %% Cell 8 — PART B: osm/<city>.zip  (ROIs with released samples only)
OSM_OUT = os.path.join(OUT_DIR, "osm")
os.makedirs(OSM_OUT, exist_ok=True)
n_files = 0
for city, grp in rel_rois.groupby("city"):
    zpath = os.path.join(OSM_OUT, f"{city.replace(' ', '_')}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for roi_id in grp.roi_id:
            src = os.path.join(OSM_DIR, f"{city}_ROI", f"{city}_roi_{roi_id}.osm")
            assert os.path.exists(src), f"Missing OSM file: {src}"
            z.write(src, arcname=os.path.basename(src))
            n_files += 1
    print(f"  {city:12s} {len(grp):4d} files  {os.path.getsize(zpath)/1e6:7.1f} MB")
assert n_files == EXP["rois"]
print("Part B OK.")


# %% Cell 9 — PART C: arrays/X_, Y_, L_<set>_c.npy (hard links to the _ph2 files, no copy)
ARR_OUT = os.path.join(OUT_DIR, "arrays")
os.makedirs(ARR_OUT, exist_ok=True)
old = sorted(glob.glob(os.path.join(ARR_OUT, "*_ph2.npy")))
assert not old, f"Delete the old array names first: {old}"
for s in SPLITS:
    for pre in ["X", "Y", "L"]:
        src = os.path.join(PH2_DIR, f"{pre}_{s}_ph2.npy")
        dst = os.path.join(ARR_OUT, f"{pre}_{s}_c.npy")
        if not os.path.exists(dst):
            os.link(src, dst)          # hard link: same file, no copy
        assert os.path.samefile(src, dst)
        arr = np.load(dst, mmap_mode="r")
        print(f"  {os.path.basename(dst):16s} {str(arr.shape):22s} {arr.dtype}")
        assert arr.shape[0] == EXP["samples_per_split"][s]
print("Part C OK.")


# %% Cell 10 — PART G (README, config)
config = {
    "n_cities": EXP["cities"], "n_rois": EXP["rois"],
    "n_samples_released": EXP["samples"],
    "n_samples_per_set": EXP["samples_per_split"],
    "n_samples_removed": EXP["removed"],
    "n_rois_simulated": EXP["rois_simulated"],
    "n_samples_simulated": EXP["samples_simulated"],
    "grid": [100, 100],
    "array_files": "arrays/X_<set>_c.npy, arrays/Y_<set>_c.npy, arrays/L_<set>_c.npy (c = cleaned)",
    "input_channels": ["TX", "building_height", "R0B0", "R0B1", "R1B0", "cos", "sin"],
    "target": "R1B1",
    "label_classes": {"0": "deep shadow", "1": "multipath", "2": "line of sight"},
    "coverage_normalization": "stored = (dBm + 120) / 120 after clipping to [-120, 0] dBm",
    "coverage_denormalization": "dBm = 120 * stored - 120",
    "building_height_normalization": "stored = min(height_m, 60) / 60",
    "no_signal_value_dbm": PHYS_MIN,
    "array_axes": "first axis west to east, second axis north to south",
    "split": "by ROI, seed 42, 70/15/15 of ROIs",
    "cleaning_stages": {
        "1": "keep 0 <= mean(R1B1 - R0B1) < 20 dB (called ph1 in the code)",
        "2": "keep <= 50 pixels with R1B1 - R0B1 < -25 dB (called ph2 in the code)",
        "3": "keep if R0B1 has signal beyond the TX pixel (called ph2.5 in the code)"},
    "carrier_frequency_ghz": 1.9, "tx_power_dbm": 40.0, "altitude_m": 60,
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

readme = f"""# Ray-traced air-to-air coverage maps and propagation-regime labels for 17 cities

Data record for the article "A ray-traced air-to-air signal strength coverage map
dataset over 17 cities, with per-pixel propagation-regime labels" (Data in Brief).
Code: [GITHUB_URL]. Licence: ODbL 1.0 (data), building data
(c) OpenStreetMap contributors.

The record contains the cleaned data only: {EXP['samples']:,} samples in {EXP['rois']:,} ROIs
of {EXP['cities']} cities. Most users need only arrays/.

## After download
Zenodo stores the files without folders. In the folder that holds the
downloaded files, put them into the folders used below and check them:

    mkdir arrays raw_maps osm
    mv *_c.npy arrays/
    mv *.h5 raw_maps/
    mv *.zip osm/
    sha256sum -c SHA256SUMS.txt

## Files
| Part | File(s) | Content |
|---|---|---|
| A | raw_maps/<city>.h5 | one group per ROI (roi_<id>): tx_idx (K) int16; R0B0, R0B1, R1B0, R1B1 (K, 100, 100) float32 dBm, not clipped; BLD (100, 100) m; labels (K, 100, 100) uint8. Entry k of each dataset belongs to tx_idx[k]. |
| B | osm/<city>.zip | OSM building files of the same ROIs |
| C | arrays/X_<set>_c.npy | (N, 100, 100, 7) float32, normalized inputs: TX, building height, R0B0, R0B1, R1B0, cos, sin |
| C | arrays/Y_<set>_c.npy | (N, 100, 100, 1) float32, normalized target R1B1 |
| C | arrays/L_<set>_c.npy | (N, 100, 100) uint8, labels 0 deep shadow, 1 multipath, 2 line of sight |
| D | samples.csv | city, roi_id, tx_idx, txLat, txLon (WGS-84 degrees), txHeight (m, antenna height above the surface beneath it, chosen so that the TX is 60 m above ground), split, row (row index in the arrays of that set) |
| E | rois.csv | city, country, roi_id, min_lat, max_lat, min_lon, max_lon, n_buildings, n_buildings_ge60m, split, n_released |
| F | removed_samples.csv | city, roi_id, tx_idx, split, stage (1, 2, 3), removal_reason, mean_diff_dbm, class_b_px |
| G | README.md, config.json, LICENSE | documentation and the ODbL 1.0 licence text |

Spaces in city names are written as underscores in file names.

## Conventions
- Arrays: coverage maps are clipped to [-120, 0] dBm, then stored as (dBm + 120) / 120; dBm = 120 * stored - 120. Stored 0 = no signal (-120 dBm).
- Building height in the arrays: stored = min(height, 60 m) / 60.
- Arrays and maps: first axis west to east, second axis north to south.
- Row i of X, Y and L of a set is row i of that set in samples.csv.
- The split is by ROI: all samples of an ROI are in the same set. Do not move samples between sets.
- Cleaning stage 3 is called "ph2.5" in the cleaning code; stages 1 and 2 are "ph1" and "ph2".

## Counts
{EXP['cities']} cities, {EXP['rois']:,} ROIs and {EXP['samples']:,} samples in the record
({EXP['rois_simulated']:,} ROIs and {EXP['samples_simulated']:,} samples were simulated; {EXP['removed']:,} samples were removed by cleaning).

## Checksums
SHA256SUMS.txt lists the SHA-256 checksum of every file, with its folder (see After download).
"""
with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
    f.write(readme)
print("README.md and config.json written.")

# Licence: download the ODbL 1.0 text yourself from
#   https://opendatacommons.org/licenses/odbl/1-0/
# and save it in OUT_DIR as "LICENSE" (no extension). The MIT licence belongs
# with the code on GitHub, not in the data record.
print("LICENSE:", "found" if os.path.exists(os.path.join(OUT_DIR, "LICENSE")) else "MISSING, add it")

print("Part G OK (add LICENSE if it is missing).")


# %% Cell 11 — VERIFICATION: raw_maps/ reproduces the released arrays and labels
# (1) For N_VERIFY random released samples, normalizes the raw_maps/ maps exactly
#     as in the paper and compares them with the released X and Y; also re-derives
#     the labels from the raw_maps/ R0B0 ... R1B1 with the label rule.
# (2) For every ROI group, checks that tx_idx and labels equal samples.csv and
#     the released L arrays. Reads little data.
def norm_cov(a):
    a = a.astype(np.float32).copy()
    a[~np.isfinite(a)] = PHYS_MIN
    return (np.clip(a, PHYS_MIN, PHYS_MAX) - PHYS_MIN) / RANGE_DBM


raw_dir = os.path.join(OUT_DIR, "raw_maps" if TEST_LIMIT_ROIS is None else "raw_maps_TEST")
in_test = samples if TEST_LIMIT_ROIS is None else \
    samples.merge(rel_rois.head(TEST_LIMIT_ROIS), on=["city", "roi_id"])
X = {s: np.load(os.path.join(OUT_DIR, "arrays", f"X_{s}_c.npy"), mmap_mode="r") for s in SPLITS}
Y = {s: np.load(os.path.join(OUT_DIR, "arrays", f"Y_{s}_c.npy"), mmap_mode="r") for s in SPLITS}
L = {s: np.load(os.path.join(OUT_DIR, "arrays", f"L_{s}_c.npy"), mmap_mode="r") for s in SPLITS}

# (1) maps and label rule on random samples
check = in_test.sample(min(N_VERIFY, len(in_test)), random_state=0)
worst, n_ok = 0.0, 0
for city, grp in check.groupby("city"):
    with h5py.File(os.path.join(raw_dir, f"{city.replace(' ', '_')}.h5"), "r") as h5:
        for r in grp.itertuples(index=False):
            g = h5[f"roi_{r.roi_id}"]
            k = int(np.where(g["tx_idx"][:] == r.tx_idx)[0][0])
            x = X[r.split][r.row]
            r0b1, r1b1 = norm_cov(g["R0B1"][k]), norm_cov(g["R1B1"][k])
            diffs = [
                np.abs(norm_cov(g["R0B0"][k]) - x[..., 2]).max(),
                np.abs(r0b1 - x[..., 3]).max(),
                np.abs(norm_cov(g["R1B0"][k]) - x[..., 4]).max(),
                np.abs(r1b1 - Y[r.split][r.row][..., 0]).max(),
                np.abs(np.clip(g["BLD"][:], 0, MAX_HEIGHT) / MAX_HEIGHT - x[..., 1]).max(),
            ]
            s0, s1 = r0b1 <= LABEL_EPS, r1b1 <= LABEL_EPS
            rule = np.full((100, 100), 2, np.uint8)
            rule[s0 & s1] = 0
            rule[s0 & ~s1] = 1
            assert (rule == g["labels"][k]).all(), f"label rule mismatch {r}"
            assert x[..., 0].sum() == 1, f"TX map does not have one TX pixel {r}"
            worst = max(worst, max(diffs))
            n_ok += 1
print(f"(1) Checked {n_ok} released samples. Largest map difference: {worst:.2e}")
assert worst < 1e-5, "raw_maps/ does not reproduce the released arrays. Do not publish."

# (2) tx_idx and labels of every group
n_groups, n_lab = 0, 0
for city, grp in in_test.groupby("city"):
    with h5py.File(os.path.join(raw_dir, f"{city.replace(' ', '_')}.h5"), "r") as h5:
        for roi_id, s_roi in grp.groupby("roi_id"):
            s_roi = s_roi.sort_values("tx_idx")
            g = h5[f"roi_{roi_id}"]
            assert (g["tx_idx"][:] == s_roi.tx_idx.values).all(), f"tx_idx mismatch {city} {roi_id}"
            assert g["R1B1"].shape == (len(s_roi), 100, 100)
            ref = np.asarray(L[s_roi.split.iloc[0]][s_roi.row.values])
            assert (g["labels"][:] == ref).all(), f"labels differ from L arrays {city} {roi_id}"
            n_groups += 1
            n_lab += len(s_roi)
        if TEST_LIMIT_ROIS is None:
            assert len(h5.keys()) == grp.roi_id.nunique(), f"extra groups in {city}"
print(f"(2) Checked {n_groups:,} ROI groups and {n_lab:,} label maps.")
if TEST_LIMIT_ROIS is None:
    assert n_groups == EXP["rois"] and n_lab == EXP["samples"]
print("Verification OK.")


# %% Cell 12 — CHECKSUMS (SHA256SUMS.txt) and final file list
def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


assert not os.path.isdir(os.path.join(OUT_DIR, "raw_maps_TEST")), \
    "Delete raw_maps_TEST before the final checksums."
for stale in ["tx_positions.csv", "figure_data"]:
    assert not os.path.exists(os.path.join(OUT_DIR, stale)), f"Delete the old {stale} first."
assert not glob.glob(os.path.join(OUT_DIR, "arrays", "*_ph2.npy")), "Delete the old *_ph2.npy names first."
files = sorted(os.path.relpath(os.path.join(d, n), OUT_DIR)
               for d, _, names in os.walk(OUT_DIR) for n in names
               if n != "SHA256SUMS.txt")
t0 = time.time()
with open(os.path.join(OUT_DIR, "SHA256SUMS.txt"), "w") as out:
    for rel in files:
        out.write(f"{sha256(os.path.join(OUT_DIR, rel))}  {rel}\n")
        print(f"  {rel:40s} {os.path.getsize(os.path.join(OUT_DIR, rel))/1e9:8.3f} GB")
total = sum(os.path.getsize(os.path.join(OUT_DIR, r)) for r in files)
print(f"\n{len(files) + 1} files, {total/1e9:.1f} GB  ({(time.time()-t0)/60:.1f} min)")
assert len(files) + 1 <= 100, "Zenodo allows at most 100 files per record."
assert total < 50e9, "Zenodo allows at most 50 GB per record."
print("Release folder ready for upload.")
