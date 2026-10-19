# Code for the air-to-air coverage map dataset

Code for the data article "A ray-traced air-to-air signal strength coverage map dataset, with per-pixel propagation-regime labels” (Data in Brief).

- Data: https://doi.org/10.5281/zenodo.23303511, under the Open Database License 1.0 (© OpenStreetMap contributors).
- Code: this repository, under the MIT licence (see `LICENSE`).

Authors: Abdullah Althuwaini and Adan E. Vela, Department of Industrial Engineering and Management Systems, University of Central Florida. Contact: abdullah.althuwaini@ucf.edu

## Before you run anything

The scripts in `pipeline/` are released **exactly as they were run** to produce the data.
- **Paths:** the hard-coded input and output paths are replaced by placeholders such as `<DATA_ROOT>` (see Paths below). Replace them with your own folders before reuse.
- **Building data:** the OpenStreetMap building files of the released ROIs are in the data record (part B). OpenStreetMap changes over time, so a new download would return current data.
- **TX positions:** the script that drew them is not included; the positions of the released samples are in `samples.csv` of the data record (`txLat`, `txLon`, `txHeight`).
- **Altitude:** the scripts use a TX and RX altitude of 200 ft (60.96 m); the article gives this as 60 m.
- **Names:** folder and file names containing 20City or 50City are historical; the data cover 17 cities.

## Candidate cities

The candidate cities were the top 50 cities by number of tall buildings in the Skyscraper Center ranking (accessed in 2024). Because of the available computing power, only the first 20 in alphabetical order were processed; in three of them (Chengdu, Chongqing, Foshan) no ROI qualified, so the data cover 17 cities.

| # | City | Processed | In the data |
|---|---|---|---|
| 1 | Abu Dhabi | yes | yes |
| 2 | Bangkok | yes | yes |
| 3 | Beijing | yes | yes |
| 4 | Busan | yes | yes |
| 5 | Changsha | yes | yes |
| 6 | Chengdu | yes | no |
| 7 | Chicago | yes | yes |
| 8 | Chongqing | yes | no |
| 9 | Dalian | yes | yes |
| 10 | Doha | yes | yes |
| 11 | Dubai | yes | yes |
| 12 | Foshan | yes | no |
| 13 | Guangzhou | yes | yes |
| 14 | Hangzhou | yes | yes |
| 15 | Hefei | yes | yes |
| 16 | Hong Kong | yes | yes |
| 17 | Houston | yes | yes |
| 18 | Incheon | yes | yes |
| 19 | Istanbul | yes | yes |
| 20 | Jakarta | yes | yes |
| 21 | Jinan | no | no |
| 22 | Johor Bahru | no | no |
| 23 | Kuala Lumpur | no | no |
| 24 | London | no | no |
| 25 | Makati | no | no |
| 26 | Melbourne | no | no |
| 27 | Miami | no | no |
| 28 | Moscow | no | no |
| 29 | Mumbai | no | no |
| 30 | Nanchang | no | no |
| 31 | Nanjing | no | no |
| 32 | Nanning | no | no |
| 33 | New York City | no | no |
| 34 | Osaka | no | no |
| 35 | Panama City | no | no |
| 36 | Penang | no | no |
| 37 | Qingdao | no | no |
| 38 | Seoul | no | no |
| 39 | Shanghai | no | no |
| 40 | Shenyang | no | no |
| 41 | Shenzhen | no | no |
| 42 | Singapore | no | no |
| 43 | Suzhou | no | no |
| 44 | Sydney | no | no |
| 45 | Tianjin | no | no |
| 46 | Tokyo | no | no |
| 47 | Toronto | no | no |
| 48 | Wuhan | no | no |
| 49 | Xiamen | no | no |
| 50 | Zhuhai | no | no |

## Pipeline order (`pipeline/`)

All scripts are named `stepNN_name`, where NN is the step number; steps 03 and 06 have no script in this repository. `xy2ll.m` keeps its name because step 12 calls it by name.

| Step | Script | Inputs | Outputs |
|---|---|---|---|
| 1 | `step01_city_bbox_nominatim.ipynb` | city list (Skyscraper Center ranking) | city bounding boxes (Nominatim) |
| 2 | `step02_roi_grid.ipynb` | city bounding boxes | grid of regions of interest (ROIs) of about 1 km × 1 km |
| 3 | OSM building download via an Overpass API query (not included) | ROI grid (bounding boxes) | OSM building files: an Overpass QL query, the query language used by Overpass turbo, returning all OpenStreetMap ways and relations tagged as buildings in each ROI bounding box; the files of the released ROIs are part B of the data record |
| 4 | `step04_tall_building_rois.m` | OSM building files, ROI grid | table of ROIs with at least one building of 60 m or more |
| 5 | `step05_roi_table.py` | ROI grid, ROI table | ROI table with bounding boxes |
| 6 | TX placement (not included); the TX positions are released as data in `samples.csv` | OSM building files, ROI grid | 50 TX positions per ROI (CSV), drawn uniformly at random outside buildings of 60 m or more |
| 7 | `step07_matlab_building_heights.m` | OSM building files | building heights reported by MATLAB (CSV per ROI) |
| 8 | `step08_building_and_tx_maps.py` | OSM files, TX CSVs, MATLAB height CSVs, ROI grid | building-height maps and TX maps (HDF5) |
| 9 | `step09_r0b0_free_space.py` | TX CSVs, ROI grid | R0B0, simple coverage map: free-space received signal strength (HDF5) |
| 10 | `step10_r0b1_direct_path_blockage.py` | building-height maps, TX maps, R0B0 | R0B1, simple coverage map: R0B0 with direct-path blockage (HDF5); see note 1 |
| 11 | `step11_direction_cos_sin.py` | TX maps | cosine and sine direction channels (HDF5) |
| 12 | `step12_ray_tracing_r1b0_r1b1.m` (uses `xy2ll.m`) | TX CSVs, OSM files, ROI grid, `df_20city_rois.csv`: list of the 1,450 simulated ROIs (the 1,444 with released samples are in `rois.csv` of the data record) | R1B0 and R1B1, ray-traced coverage maps with one reflection: received signal strength over terrain only and over buildings and terrain (HDF5) |
| 13 | `step13_preprocessing_split.ipynb` | the HDF5 files of steps 8 to 12 | normalized arrays, split by ROI (70/15/15) |
| 14 | `step14_cleaning_labels.py` | normalized arrays | cleaned arrays, propagation-regime labels, removal record; only the filter, label and save parts define the data |
| 15 | `step15_build_release.py` | steps 4 to 14 and the OSM files | the files of the data record (parts A to G) |

Notes:
1. `step11_direction_cos_sin.py` takes the city name from the input folder name and keeps its `_TX_locations` suffix, so its output names are `<City>_TX_locations_CosSin/<City>_TX_locations_roi_<ID>_CosSinTheta.h5`. Later steps use these names.
2. The cleaning code calls the three cleaning stages `ph1`, `ph2` and `ph2.5`; the article and the data record call them stages 1, 2 and 3.

## Paths

The scripts are kept as run, except that the personal paths are replaced by the placeholders below. Replace each placeholder with the matching folder on your system.

| Placeholder | Folder it stands for | Used in |
|---|---|---|
| `<CITY_LIST_DIR>` | folder with the city list (`cities_list.xlsx`) and the bounding boxes written by step 1 | 01 |
| `<ROI_GRID_DIR>` | folder with the city bounding boxes read by step 2 and the ROI grid it writes | 02 |
| `<DATA_ROOT>` | main working folder of the pipeline, with the subfolders `01_Process/` (bounding boxes, OSM files, TX files, building matrices, coverage maps) and `02_Complete/` (ROI table, input tensors) | 04, 05, 07 to 12, 15 |
| `<MATLAB_DRIVE>` | MATLAB Drive folder that holds copies of the ROI grid (`Test/` and `AIAA/bbox/All_Cities_ROI_Grid1.csv`) | 04, 08, 09 |
| `<DRIVE_DIR>` | Google Drive folder used in Colab, with `dataset_meta_res100.json` and the `preprocessed/` arrays | 13, 14 |
| `<PROJECT_DIR>` | folder that holds `inputs/data/` (the metadata files written by steps 13 and 14, and `dataset_meta_res100.json`) and `release/data/` (the data record written by step 15) | 15 |

Steps 13 and 14 also mount Google Drive at `/content/drive`, the standard Colab mount point.

## Software

- MATLAB R2025a: steps 04, 07 and 12, with `xy2ll.m`.
- Python, with the packages in `requirements.txt`: all other steps. Step 13 is a Jupyter notebook that can run locally or in Google Colab.

## How to cite

- Article: A. Althuwaini, A. E. Vela, A ray-traced air-to-air signal strength coverage map dataset over 17 cities, with per-pixel propagation-regime labels, Data in Brief, [DOI].
- Dataset: A. Althuwaini, A. E. Vela, Ray-traced air-to-air coverage maps and propagation-regime labels for 17 cities, Zenodo, [ZENODO_DOI].

## Licence

MIT, see `LICENSE`. The data are under the Open Database License 1.0.
