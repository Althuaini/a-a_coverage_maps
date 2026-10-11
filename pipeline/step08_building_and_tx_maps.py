"""
Build the building-height map and the TX maps for every ROI (100 x 100 cells).

Heights come from OSM tags, then from the MATLAB height table, then a default
of 10 m (DEFAULT_HEIGHT). If several buildings overlap a cell, the tallest wins.

Writes two files per ROI:
  {city}_roi_{roi_id}_allBuildings_heights.h5
      building_matrix (GRID_SIZE x GRID_SIZE, float32);
      0 = no building, otherwise the building height in metres
  {city}_roi_{roi_id}_TX_locations.h5
      Tx_1 ... Tx_50 (GRID_SIZE x GRID_SIZE, float32);
      1.0 at the TX cell, 0 elsewhere
All TX share one fixed altitude, so the TX maps are binary.
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import transform
from shapely.prepared import prep
import numpy as np
import h5py
import pyproj
import matplotlib.pyplot as plt

# ========================= Config ========================= #

txrx_base_path = (
    "<DATA_ROOT>/"
    "01_Process/01.2_txrx_generation/Tx_Data200feet"
)
csv_base_path = (
    "<DATA_ROOT>/"
    "01_Process/01.4_Building_Matrix/20City/Buildings_Height_ref_20City"
)
osm_base_path = (
    "<DATA_ROOT>/"
    "01_Process/01.1_OverpassResponses/"
)
bbox_csv = (
    "<MATLAB_DRIVE>/"
    "AIAA/bbox/All_Cities_ROI_Grid1.csv"
)

# Output directories
buildings_mat_dir = (
    "<DATA_ROOT>/"
    "01_Process/01.4_Building_Matrix/20City/All_buildings_heights_res100/"
)
tx_loc_mat_dir = (
    "<DATA_ROOT>/"
    "02_Complete/20city_input_tensor/TX_locations"
)

matrix_resolution = 100
DEFAULT_HEIGHT     = 10.0   # metres — fallback when no height available

bbox_data = pd.read_csv(bbox_csv)

# =================== Height Helper =================== #

def parse_height_from_tags(tags):
    """Extract building height in metres from OSM tags.
    Sources: 'height' tag (explicit metres) or 'building:levels' × 3m/floor.
    Returns float or None if no height info available.
    """
    if 'height' in tags:
        try:
            return float(tags['height'].replace('m', '').strip())
        except (ValueError, AttributeError):
            pass
    if 'building:levels' in tags:
        try:
            return float(tags['building:levels']) * 3.0
        except (ValueError, TypeError):
            pass
    return None

# =================== Discover Cities =================== #

txrx_cities = [
    d for d in os.listdir(txrx_base_path)
    if os.path.isdir(os.path.join(txrx_base_path, d)) and d.endswith("_ROI_txrx_Gen")
]

print(f"Found {len(txrx_cities)} city folders in Tx_Data200feet")

# ===================== Main Process ===================== #

for city_folder in txrx_cities:

    clean_city_name = city_folder.replace("_ROI_txrx_Gen", "")
    print(f"\n=== Processing city: {clean_city_name} ===")

    city_txrx_path = os.path.join(txrx_base_path, city_folder)
    city_osm_path  = os.path.join(osm_base_path, f"{clean_city_name}_ROI")
    city_csv_path  = os.path.join(csv_base_path, f"{clean_city_name}_ROI")

    # Output dirs
    city_building_out = os.path.join(
        buildings_mat_dir, f"{clean_city_name}_allBuildings"
    )
    city_tx_loc_out = os.path.join(
        tx_loc_mat_dir, f"{clean_city_name}_TX_locations"
    )
    os.makedirs(city_building_out, exist_ok=True)
    os.makedirs(city_tx_loc_out,   exist_ok=True)

    if not os.path.exists(city_osm_path):
        print(f" ❌ Missing OSM folder for {clean_city_name}")
        continue

    # Extract ROIs
    txrx_csvs = [f for f in os.listdir(city_txrx_path) if f.endswith("_tx_rx_pairs.csv")]
    roi_ids = sorted({
        f.split("_")[f.split("_").index("roi") + 1]
        for f in txrx_csvs if "roi" in f.split("_")
    })
    print(f"  Found {len(roi_ids)} ROI IDs")

    # ------------------- Process Each ROI ------------------- #
    for roi_id in roi_ids:

        csv_clean_city_name = clean_city_name.replace("_ROI", "")

        osm_file = os.path.join(
            city_osm_path, f"{csv_clean_city_name}_roi_{roi_id}.osm"
        )
        txrx_csv = os.path.join(
            city_txrx_path, f"{csv_clean_city_name}_roi_{roi_id}_tx_rx_pairs.csv"
        )
        csv_file = os.path.join(
            city_csv_path, f"{csv_clean_city_name}_roi_{roi_id}_building_heights.csv"
        )

        # Validate
        if not os.path.exists(osm_file):
            print(f"   ⚠ Missing OSM file for ROI {roi_id}")
            continue
        if not os.path.exists(txrx_csv):
            print(f"   ⚠ Missing TX CSV for ROI {roi_id}")
            continue

        # Output file paths
        building_outfile = os.path.join(
            city_building_out,
            f"{csv_clean_city_name}_roi_{roi_id}_allBuildings_heights.h5"
        )
        tx_loc_outfile = os.path.join(
            city_tx_loc_out,
            f"{csv_clean_city_name}_roi_{roi_id}_TX_locations.h5"
        )

        # Skip if both outputs already exist
        if os.path.exists(building_outfile) and os.path.exists(tx_loc_outfile):
            print(f"   ⏭️  Both outputs exist for ROI {roi_id}, skipping.")
            continue

        # Load CSV height lookup if available (fallback for missing OSM heights)
        height_lookup = {}
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            try:
                csv_data      = pd.read_csv(csv_file)
                height_lookup = dict(zip(
                    csv_data['BuildingID'].astype(str),
                    csv_data['MaxHeight']
                ))
            except Exception as e:
                print(f"   ⚠ Could not load CSV for ROI {roi_id}: {e}")

        # ---------- Parse OSM ----------
        try:
            tree = ET.parse(osm_file)
        except ET.ParseError as e:
            print(f"   ❌ Failed to parse OSM for ROI {roi_id}: {e}")
            continue

        root  = tree.getroot()
        nodes = {
            node.get('id'): (float(node.get('lon')), float(node.get('lat')))
            for node in root.findall(".//node")
        }

        processed_way_ids = set()
        building_data     = []   # list of (id_str, geom, height_m)

        # Relations
        for relation in root.findall(".//relation"):
            tags = {t.get('k'): t.get('v') for t in relation.findall("tag")}
            if not any(k.startswith('building') for k in tags):
                continue

            building_id    = str(relation.get('id'))
            outer_polygons = []
            member_heights = []

            for member in relation.findall("member[@type='way']"):
                way_ref = member.get('ref')
                way     = root.find(f".//way[@id='{way_ref}']")
                if way is None:
                    continue

                fp = [
                    nodes[n.get('ref')]
                    for n in way.findall("nd")
                    if n.get('ref') in nodes
                ]
                if len(fp) > 2 and member.get('role') == 'outer':
                    outer_polygons.append(Polygon(fp))

                processed_way_ids.add(way_ref)
                h = parse_height_from_tags(
                    {t.get('k'): t.get('v') for t in way.findall("tag")}
                )
                if h is not None:
                    member_heights.append(h)

            if outer_polygons:
                geom = (
                    outer_polygons[0]
                    if len(outer_polygons) == 1
                    else MultiPolygon(outer_polygons)
                )
                # Height priority: relation tags → member avg → CSV → default
                h = (parse_height_from_tags(tags)
                     or (float(np.mean(member_heights)) if member_heights else None)
                     or height_lookup.get(building_id)
                     or DEFAULT_HEIGHT)
                building_data.append((building_id, geom, float(h)))

        # Ways
        for way in root.findall(".//way"):
            way_id = way.get('id')
            if way_id in processed_way_ids:
                continue
            tags = {t.get('k'): t.get('v') for t in way.findall("tag")}
            if not any(k.startswith('building') for k in tags):
                continue
            fp = [
                nodes[n.get('ref')]
                for n in way.findall("nd")
                if n.get('ref') in nodes
            ]
            if len(fp) > 2:
                # Height priority: OSM tags → CSV → default
                h = (parse_height_from_tags(tags)
                     or height_lookup.get(way_id)
                     or DEFAULT_HEIGHT)
                building_data.append((way_id, Polygon(fp), float(h)))

        if not building_data:
            print(f"   ⚠ No buildings found in OSM for ROI {roi_id}")

        # ---------- Projection ----------
        row_info  = bbox_data.iloc[int(roi_id)]
        min_lon, max_lon = row_info['Min_Longitude'], row_info['Max_Longitude']
        min_lat, max_lat = row_info['Min_Latitude'],  row_info['Max_Latitude']

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        aeqd        = pyproj.Proj(proj='aeqd', lat_0=center_lat, lon_0=center_lon)
        wgs84       = pyproj.Proj("EPSG:4326")
        transformer = pyproj.Transformer.from_proj(wgs84, aeqd, always_xy=True)
        project     = lambda lon, lat: transformer.transform(lon, lat)

        proj_polygons    = [prep(transform(project, p)) for _, p, _ in building_data]
        building_heights = [h for _, _, h in building_data]

        # === XY grid — same as MATLAB ===
        xy_min, xy_max = -500.0, 500.0
        x_edges = np.linspace(xy_min, xy_max, matrix_resolution + 1)
        y_edges = np.linspace(xy_min, xy_max, matrix_resolution + 1)

        # ======= RAW HEIGHT MATRIX — tallest building wins per cell =======
        raw_heights = np.zeros(
            (matrix_resolution, matrix_resolution), dtype=np.float32
        )
        for i in range(matrix_resolution):
            for j in range(matrix_resolution):
                cell = box(x_edges[j], y_edges[i], x_edges[j+1], y_edges[i+1])
                for pp, h in zip(proj_polygons, building_heights):
                    if pp.intersects(cell):
                        if h > raw_heights[i, j]:
                            raw_heights[i, j] = h

        # ======= FILE 1: Building height matrix (no TX) =======
        if not os.path.exists(building_outfile):
            with h5py.File(building_outfile, "w") as h5f:
                ds = h5f.create_dataset(
                    "building_matrix",
                    data=np.flipud(raw_heights).T,
                    compression="gzip"
                )
                ds.attrs["description"] = (
                    "Building height matrix (metres). "
                    "0.0 = free space. h > 0 = building height in metres."
                )
                ds.attrs["default_height_m"] = DEFAULT_HEIGHT

        # ======= FILE 2: TX location matrices (binary, one per TX) =======
        txdf = pd.read_csv(txrx_csv)

        if not os.path.exists(tx_loc_outfile):
            with h5py.File(tx_loc_outfile, "w") as h5out:
                h5out.attrs["description"] = (
                    "TX location matrices. "
                    "0.0 = not TX position. 1.0 = TX position."
                )

                for tx_idx, row_tx in txdf.iterrows():
                    tx_lon = row_tx["txLon"]
                    tx_lat = row_tx["txLat"]

                    tx_x, tx_y = project(tx_lon, tx_lat)

                    j = np.searchsorted(x_edges, tx_x) - 1
                    i = np.searchsorted(y_edges, tx_y) - 1

                    i = np.clip(i, 0, matrix_resolution - 1)
                    j = np.clip(j, 0, matrix_resolution - 1)

                    # Binary TX matrix — 1.0 at TX position, 0.0 everywhere else
                    tx_matrix        = np.zeros(
                        (matrix_resolution, matrix_resolution), dtype=np.float32
                    )
                    tx_matrix[i, j]  = 1.0

                    h5out.create_dataset(
                        f"Tx_{tx_idx + 1}",
                        data=np.flipud(tx_matrix).T,
                        compression="gzip"
                    )

        # Height source stats
        n_total   = len(building_data)
        n_default = sum(1 for _, _, h in building_data if h == DEFAULT_HEIGHT)
        n_csv     = sum(
            1 for bid, _, h in building_data
            if h != DEFAULT_HEIGHT and bid in height_lookup
            and abs(float(height_lookup[bid]) - h) < 0.01
        )
        n_osm     = n_total - n_default - n_csv

        print(f"   ✅ {csv_clean_city_name} ROI {roi_id} | "
              f"{n_total} buildings | "
              f"OSM: {n_osm} | CSV: {n_csv} | Default(10m): {n_default}")

print("\n=== ALL v0.4 MATRICES COMPLETE ===")

# ============================================================
# OPTIONAL VISUALIZATION (gated — set VISUALIZE = True)
# ============================================================

VISUALIZE = False

if VISUALIZE:
    r1b1_root = (
        "<DATA_ROOT>/"
        "01_Process/01.3_Coverage_Maps/20City_Res100"
    )
    target_rois = ["379429", "345603"]
    tx_index    = 0   # 0 → Tx_1

    # Collect matching files
    roi_files = []
    for dirpath, _, files in os.walk(tx_loc_mat_dir):
        for f in files:
            if f.endswith("_TX_locations.h5"):
                for roi in target_rois:
                    if f"_roi_{roi}_" in f:
                        city = os.path.basename(dirpath).split("_ROI_")[0]
                        roi_files.append((roi, city, os.path.join(dirpath, f)))

    for roi, city_name, tx_path in roi_files:
        dset_name = f"Tx_{tx_index + 1}"

        # Build matching building heights path
        bld_path = os.path.join(
            buildings_mat_dir,
            f"{city_name}_ROI_Height_matrix_allBuildings",
            f"{city_name}_roi_{roi}_allBuildings_heights.h5"
        )
        r1b1_path = os.path.join(
            r1b1_root, city_name,
            f"Output_{city_name}_roi_{roi}_R1B1.h5"
        )

        print(f"\nROI {roi} | {city_name}")

        # Load TX location — transpose at load
        with h5py.File(tx_path, "r") as hf:
            if dset_name not in hf:
                print(f"  ⚠️ {dset_name} not found in TX locations file")
                continue
            tx_mat = hf[dset_name][:].T   # .T at load

        # Load building heights — no transpose needed (building_matrix is a
        # single dataset, not per-TX)
        bld_mat = None
        if os.path.exists(bld_path):
            with h5py.File(bld_path, "r") as hf:
                bld_mat = hf["building_matrix"][:].T   # .T at load

        # Load R1B1
        r1b1_mat    = None
        r1b1_exists = os.path.exists(r1b1_path)
        if r1b1_exists:
            with h5py.File(r1b1_path, "r") as hf:
                if dset_name in hf:
                    r1b1_mat = hf[dset_name][:]

        n_cols = 3 if (r1b1_mat is not None and bld_mat is not None) else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        # Panel 1 — building height matrix
        if bld_mat is not None:
            im0 = axes[0].imshow(bld_mat, cmap="plasma", vmin=0, vmax=200)
            axes[0].set_title(f"Building heights (m) | ROI {roi}")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, label='Height (m)')
        else:
            axes[0].text(0.5, 0.5, "No building file", ha='center',
                         transform=axes[0].transAxes)
        axes[0].axis('off')

        # Panel 2 — TX location
        axes[1].imshow(tx_mat, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"TX location | {dset_name}")
        tx_pos = np.argwhere(tx_mat == 1.0)
        if len(tx_pos) > 0:
            axes[1].scatter([tx_pos[0,1]], [tx_pos[0,0]], s=120,
                            c='red', marker='x', linewidths=2.5, zorder=5)
        axes[1].axis('off')

        # Panel 3 — R1B1 + building overlay + TX marker
        if r1b1_mat is not None and bld_mat is not None:
            axes[2].imshow(r1b1_mat, cmap="plasma", vmin=-120, vmax=0)
            # Cyan overlay for all buildings
            bld_mask = (bld_mat > 0)
            overlay  = np.zeros((*bld_mask.shape, 4), dtype=float)
            overlay[bld_mask] = [0, 1, 1, 0.7]
            axes[2].imshow(overlay)
            # Red X for TX
            if len(tx_pos) > 0:
                axes[2].scatter([tx_pos[0,1]], [tx_pos[0,0]], s=120,
                                c='red', marker='x', linewidths=2.5, zorder=5)
            axes[2].set_title(f"R1B1 + buildings | {dset_name}")
            axes[2].axis('off')

        plt.suptitle(f"{city_name} | ROI {roi} | {dset_name}", fontsize=11)
        plt.tight_layout()
        plt.show()