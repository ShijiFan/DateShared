# Date-Shared Shift Priors for Cross-Season Sentinel-1 Crop Mapping With Short or Borrowed Archives

Code, frozen model archives and per-transfer result tables accompanying the research manuscript of the same title (prepared for submission to IEEE TGRS, 2026).

Repository: [ShijiFan/DateShared](https://github.com/ShijiFan/DateShared).
Earlier working title: "A Water-Cloud Shift Prior for Unsupervised Cross-Season Adaptation of Sentinel-1 Crop Classifiers".

The method adapts a crop classifier trained on one labelled Sentinel-1 season to a new, unlabelled season. It estimates the
class-conditional backscatter/coherence shift between the two seasons by MAP-EM under a Gaussian prior whose covariance combines a
**date-shared shift basis** (one canopy-attenuated moisture offset per acquisition date, shared by all crops, as in the water-cloud
model, plus a phenology-timing column per crop and a coherence column per slot) with a full-rank residual learned from an archive of past season pairs. The
estimated shifts are used either directly (Gaussian classifier) or to resynthesise the source season before training any classifier
(HGB, random forest, TempCNN, DANN), including the seasons of a labelled archive before pooled training. No weather data and no target
labels are used at run time.

> **Status.** The code is released exactly as it was run for the paper (research scripts, not a packaged library). Absolute
> paths were left in place on purpose (see [Paths](#paths-and-configuration)). Sentinel-1 products, parcel labels and weather data
> are not redistributed; all of them are public and the scripts that download and process them are included.

---

## Repository layout

The layout mirrors the original working tree, because the scripts import each other through relative paths
(`os.path.join(HERE, '..', 'multiyear_20261017')`, etc.). Keep the folders side by side.

```
.
├── shiftinfer_20261105/          main project of the paper
│   ├── frozen/                   FROZEN METHOD: hybridD_lw.py (self-contained), archive_NL_*.npz ("weights"), SHA256_FROZEN.tsv, VERIFY.json
│   ├── p1_tasks.py ... p10_lw.py Gaussian-framework experiments on the Dutch sites (P1-P10), shared helpers p9_lib.py, p3_joint.py
│   ├── p11_c2.py, p11d_deep.py   classifier plug-in: resynthesis + HGB / RF (P11) and + TempCNN / DANN (P11d)
│   ├── s3a_deep.py               TempCNN no adaptation / AdaBN / DANN baselines (S3a)
│   ├── s3b_recent.py             early re-implementations of SFUDA / RAINCOAT (S3b; only feeds unused columns of BASELINES.json)
│   ├── s3c_official.py           RAINCOAT (official code) and SFUDA with uniform learning-rate tuning (S3c)
│   ├── s3d_q1.py, s3d_table.py   RAINCOAT-SRS (4 checkpoint rules) and SFUDA incl. France (S3d); main comparison table
│   ├── s3d_timing.py             wall-clock cost per fold (Sec. "Computational Cost")
│   ├── s3h_*.py                  pooled training vs prior-shifted pooled training, ablations (date shuffle, no date, no tau),
│   │                             concurrent-season exclusion (S3h); s3h_summary.py -> s3h_summary.csv, s3h_medians.csv
│   ├── s3i_*.py                  archive-size curve, K = 1, 2, 4 labelled archive years (S3i); s3i_summary.py -> s3i_summary.csv
│   ├── s3j_*.py                  Dutch single-source tables recomputed with the archive WITHOUT the transfer's years (S3j)
│   ├── diag_frozen.py            label-free date offsets of the frozen method vs antecedent precipitation (Fig. 4, NL)
│   ├── results/                  excl_tables.py -> EXCL.json (read by the Dutch tables, S3j); consolidate.py -> CONSOLIDATED.json,
│   │                             compare_baselines.py -> BASELINES.json (earlier full-archive versions, kept for traceability)
│   ├── climid/                   ClimID-UDA port (climid_sar.py) and ERA5 download (fetch_era5land.py)
│   ├── newregion/                independent confirmation in Pays de Caux, France
│   │   ├── confirm_fr.py, confirm_fr_summary.py, fr_deep.py, fr_deep_summary.py
│   │   ├── pipeline/             FR data preparation: RPG labels, HyP3 download/alignment, features, 400 m exclusion
│   │   └── plan_region.py, submit_fr.py, count_fr.py, FR_MAPPING_FROZEN.csv, FR-A_PaysDeCaux/plan_pairs.csv
│   ├── toy_B/                    discovery of the shift equations (toy B: build_table.py, evalb.py, r1-r3.py)
│   │   └── full/                 BA-scored equation search (evalba.py, cands_*.py), archive-length curve (curve_b.py)
│   │       └── frozen_j2/        frozen rain-driven prior (j2_wc.py, archive_j2_NL_*.npz, SHA256_J2.tsv, VERIFY_J2.json)
│   ├── paper_tgrs/               table and figure scripts (tables/*.py, figs/*.py), generated tables/*.tex as reference output
│   └── *.csv, *.json             per-transfer results read by the table / figure scripts
├── multiyear_20261017/           NL 2017-2023 data preparation + shared helpers
│   ├── xyexcl.py                 cross-year 400 m exclusion helper (imported by every NL experiment)
│   ├── deep_baseline_2023/db_run.py   TempCNN / AdaBN / DANN implementation (imported by all deep experiments)
│   └── build_*.py, brp_*.py, align_my.py, download_*.py, submit_plan_*.py, fetch_knmi_my.py, plan_pairs*.csv
├── shift_20261017/               p3f_em.py, p3k_target_side.py, p3n_baselines.py (Gaussian fit / EM helpers exec-ed by p1_tasks.py);
│                                 p2_dump.py (NL 2024/2025 feature dump, see "Data preparation")
├── A1_observation_budget_20260923/, site2_20260930/   NL grid specification, block->fold map and BRP crop-code mapping read by build_labels_my.py
├── docs/                         protocols and pre-registrations (Chinese, unchanged), see "Protocol documents"
├── paths.py                      list of every absolute path + checker of the pinned SHA256 manifests
├── requirements.txt, requirements-gpu.txt
├── CHANGES_FOR_RELEASE.md        what was (not) changed relative to the original files
├── MANIFEST_SHA256.tsv           SHA256 of every file in this release
└── LICENSE
```

Site / group codes: **A37**, **A88** = Dutch site A, relative orbits 37 and 88; **B88** = Dutch site B, orbit 88; **FR** = Pays de
Caux (France), orbit 59. A transfer is an ordered pair of seasons (source → target) within one group. "testB88_priorA" = test on
site B with the archive of site A; "testA_priorB" = test on site A with the archive of site B.

## Using the frozen method on your own data

`shiftinfer_20261105/frozen/hybridD_lw.py` depends only on numpy, scipy and scikit-learn.

```python
import hybridD_lw as F
arch = F.Archive.load('frozen/archive_NL_all.npz')        # archive learned on all Dutch season pairs
labels, delta = F.adapt(Es, ys, Et, tk, doy, arch)       # Es/ys: labelled source season, Et: unlabelled target season
Es_shifted = F.resynthesize(Es, ys, delta)               # then train any classifier on (Es_shifted, ys)
```

The feature layout (`E = [VV(slot, 2 acquisitions)] + [VH(...)] + [COH(slot)]`), task keys `tk` and day-of-year vector `doy` are
documented in the module docstring; `p11d_deep.tk_doy` and `newregion/confirm_fr.py` show how they are built from the feature files.
`archive_NL_A.npz` / `archive_NL_B.npz` are the leave-site-out archives used on the Dutch sites, `archive_NL_all.npz` the archive used
in France, `archive_NL_T88.npz` the orbit-88-only archive (France secondary analysis). `frozen/build_and_verify.py` rebuilds them and
checks on the first transfers of every group that the module reproduces `p10_main_<grp>.csv` exactly (`VERIFY.json`: 4 transfers per group, max |ΔBA| = 0).

## Verify the downloaded release

Run `python verify_release.py` from the repository root. This uses only the Python standard library and checks every file against `MANIFEST_SHA256.tsv`. Git preserves the original file bytes through `.gitattributes`, including the SHA-pinned research scripts.

## Environment

Two conda environments were used (Windows 11, Python 3.11 / 3.10):

| Environment | Used for | File |
|---|---|---|
| `sar` (CPU) | Gaussian framework, HGB / RF, France confirmation, summaries, tables, figures, data preparation (geopandas, rasterio, HyP3) | `requirements.txt` |
| `yolo11marin` (GPU, NVIDIA RTX 5080, CUDA 12.8) | TempCNN / AdaBN / DANN, RAINCOAT, RAINCOAT-SRS, SFUDA, ClimID-TempCNN, timing | `requirements-gpu.txt` |

The versions in the two files are those installed when the experiments were run. The GPU environment used a PyTorch nightly build
(2.11.0.dev, cu128) because of the RTX 50-series GPU; any recent PyTorch with CUDA should work, but deep-learning results are not
bit-reproducible across PyTorch/CUDA versions. Most scripts were run with `OMP_NUM_THREADS=4` (deep) or `6` (France confirmation).

**Third-party code (not included).** `s3c_official.py` and `s3d_q1.py` import the official RAINCOAT implementation (He et al., ICML
2023): clone <https://github.com/mims-harvard/Raincoat> and point the constant `RC` in both scripts to the clone (see `paths.py`).
RAINCOAT is distributed under the MIT License (its `LICENSE` file reads "Copyright (c) 2022 Emadeldeen Eldele", inherited from AdaTime).
Only `models/models.py`, `models/loss.py` and `algorithms/RAINCOAT.py` are used; the scripts apply two in-memory patches without
touching the clone (the hard-coded `irfft(..., n=128)` is set to the sequence length, and the unused `pytorch_metric_learning`
import is stubbed). SHA256 of the files used for the paper:

| File in the RAINCOAT clone | SHA256 |
|---|---|
| `algorithms/RAINCOAT.py` | `520DFCDAAB11F232E7D49F5DB6F32EB4F5E43EE8E211A1D7FDDD2359F2DDBAF9` |
| `models/models.py` | `BB1661EC2BC24E3D63080D59F3A390EE42C4409CC91A9E7A79ACD59FB5BCDA0A` |
| `models/loss.py` | `9E5F358633A839D40B7613FFFD13AABDFEE90C4CD0FB08D81EAB4C9F450D5239` |

SFUDA (Mohammadi et al., RSE 2024), RAINCOAT-SRS (Gackstetter et al., ISPRS JPRS 2025) and ClimID-UDA (Wang et al., 2025) are
re-implemented in `s3c_official.py`, `s3d_q1.py` and `climid/climid_sar.py` from the published descriptions.

## Data sources

No data are redistributed. Everything below is public.

| Data | Source | Used by |
|---|---|---|
| Sentinel-1 RTC γ⁰ VV/VH (20 m, power) and 12-day InSAR coherence (10×2 looks) | ASF HyP3 on-demand processing (`RTC_GAMMA`, `INSAR_GAMMA`), S1A IW SLC, March–October; scene pairs listed in `multiyear_20261017/plan_pairs*.csv` (NL 2017–2023) and `shiftinfer_20261105/newregion/FR-A_PaysDeCaux/plan_pairs.csv` (FR 2017–2024). A free NASA Earthdata account is needed; credentials are read by `hyp3_sdk` from your own `~/.netrc` and are not stored in this repository | `multiyear_20261017/build_plan.py`, `make_plan_B.py`, `submit_plan_B/C.py`, `download_par.py`, `align_my.py`; `newregion/plan_region.py`, `submit_fr.py`, `pipeline/fr_download_align.py` |
| Dutch parcels and crops: Basisregistratie Gewaspercelen (BRP), 2017–2025 | PDOK (<https://www.pdok.nl>), CC0 | `multiyear_20261017/brp_get.py`, `brp_gdb.py`, `brp_one.py`, `build_labels_my.py` |
| French parcels and crops: Registre Parcellaire Graphique (RPG), 2017–2024 | IGN Géoplateforme WFS `RPG.<year>:parcelles_graphiques` (<https://data.geopf.fr/wfs/ows>), also on data.gouv.fr; Licence Ouverte 2.0. Crop-code tables `REF_CULTURES_2024.csv`, `REF_CULTURES_GROUPES_CULTURES_2024.csv` (IGN) | `newregion/pipeline/fr_fetch_labels.py`, `fr_build_labels.py`, `newregion/count_fr.py`; frozen 7-class mapping `newregion/FR_MAPPING_FROZEN.csv` |
| KNMI daily station data (RH, EV24), stations 283 (site A) and 280 (site B), 2016-12-01 to 2025-10-31 | KNMI daggegevens API (<https://www.daggegevens.knmi.nl/klimatologie/daggegevens>) | `multiyear_20261017/fetch_knmi_my.py` → `multiyear_20261017/knmi_daily_my.csv` |
| Météo-France daily climatological data, département 76 (`Q_76_previous-1950-2024_RR-T-Vent.csv.gz`), station 76679001 | meteo.data.gouv.fr ("Données climatologiques de base – quotidiennes"), Licence Ouverte 2.0 | `toy_B/full/frozen_j2/fr_station.py` → `fr_rr_daily.csv`, `fr_station.json` (station selection rule in `docs/toy_B/full/frozen_j2/N6_ADDENDUM_ZH.md`) |
| ERA5 / ERA5-Land daily reanalysis (6 variables, 3×3 points per region) | Open-Meteo archive API (<https://archive-api.open-meteo.com>), no account | `climid/fetch_era5land.py` → `climid/era5land_<A|B|FR>.csv` (ClimID-UDA only) |

Regions (lon/lat bounding boxes): site A (6.00, 51.98, 6.54, 52.38), site B (6.55, 52.75, 7.05, 53.10), Pays de Caux
(0.55, 49.55, 1.10, 49.90). Seven classes: permanent grass, temporary grass, maize, potato, beet, winter cereal, spring cereal.

### Data preparation

| Step | Netherlands (2017–2023) | France (2017–2024) |
|---|---|---|
| Plan and order S1 pairs | `build_plan.py`, `make_plan_B.py`, `submit_plan_B.py`, `submit_plan_C.py` | `newregion/plan_region.py`, `newregion/submit_fr.py` |
| Download + align to the 40 m grid | `download_par.py` (or `download_B.py`), `align_my.py`, `completeness_my.py` | `pipeline/fr_download_align.py` |
| Labels, 40 m parcel raster, 6 spatial folds | `brp_get.py` / `brp_gdb.py` / `brp_one.py`, `build_labels_my.py` (config in `A1_observation_budget_20260923/`, `site2_20260930/`) | `pipeline/fr_fetch_labels.py`, `pipeline/fr_build_labels.py` |
| Parcel features `features_<site>.npz` | `build_features_my.py <A|B>` | `pipeline/fr_features.py <years>` |
| Cross-year 400 m exclusion `xyexcl_<grp>.npz` | `build_xyexcl.py` | `pipeline/fr_xyexcl.py` |
| Weather | `fetch_knmi_my.py` | `toy_B/full/frozen_j2/fr_station.py` |

**Not included / not reproducible from this repository alone:** the Dutch 2024 and 2025 seasons were processed earlier by the
pipeline of a previous study (sites set up in `A1_observation_budget_20260923` and `site2_20260930`); their parcel features enter
through `shift_20261017/dump_{A,B}.npz`, written by `shift_20261017/p2_dump.py`, which imports that earlier pipeline
(`a1_s6_experiments.Year`, not included). `build_features_my.py` copies 2024/2025 from these dumps, and `p1_tasks.py` loads them at
import time (through `p3f_em.py`), so every Dutch experiment needs them. `build_xyexcl.py` also reads the 2024/2025 BRP files and
parcel tables of that pipeline. The HyP3 job lists (`SUBMITTED_JOBS*.csv`) are account-specific and not included; the scene pairs
are in `plan_pairs*.csv`.

### Checksums of the data used for the paper

Recreated files can be compared with these values (`python paths.py` also checks the manifests pinned by the frozen method).

| File | Size (bytes) | SHA256 |
|---|---|---|
| `D:\sar_data\multiyear\features_A.npz` | 59840246 | `BEE368DA47BBCC40691F9849EC15FDD2D807A93AD9917850D1D8EB71581AE7B9` |
| `D:\sar_data\multiyear\features_A_meta.json` | 13161 | `20AE7E7C927103111D20685574E52C53B768EA2F9C8B10195CA74B4DE5D9C80E` |
| `D:\sar_data\multiyear\features_B.npz` | 16618185 | `70D8A3779A88E5C140A205AF207751C047CE560A9BFB7E340CA8D14D01D15CE8` |
| `D:\sar_data\multiyear\features_B_meta.json` | 6582 | `89820A33807822729545CD12D0D87F588B2CBEC094F50DE81EE3E814FC2362E2` |
| `D:\sar_data\multiyear\xyexcl_A37.npz` | 115659 | `4A0D6862717AD8193A6CAAF5F21D5490CF0EEDA41E02EBE539A1420E887E1487` |
| `D:\sar_data\multiyear\xyexcl_A88.npz` | 115688 | `118DA85A3D18CCEC5D9D30050F7244FB005DAC123909C1014C4595375D19DE63` |
| `D:\sar_data\multiyear\xyexcl_B88.npz` | 107733 | `45F4E6F122329DB751B42AD8E94F2FE9AAC2EE3EBB33C934E815F75490E6CB58` |
| `D:\sar_data\fr_a\features_FR.npz` | 15875987 | `0824B73DC00A46133D50755644646455E2794F92F233C5231B386BBA1C03670C` |
| `D:\sar_data\fr_a\features_FR_meta.json` | 5843 | `C7DB5279C46D8770E6FD5561DCED413CD34232FCE44C45D9621F268A3F4985C3` |
| `D:\sar_data\fr_a\xyexcl_FR.npz` | 81522 | `74F91E6B5B60B54BF9E54D88D511D1ACA0B90851E258B44D602C6411E68FA2C7` |
| `shift_20261017/dump_A.npz` | 15671056 | `5F7DAF50054B019404AA0D9A045A4DF338E988512E3C2DEC125A883555569DDC` |
| `shift_20261017/dump_B.npz` | 8462145 | `622CDC536FE70B345BFD98B2306B4D3056ED4E10C2C3D40CEFE62975B6AA2082` |
| `shift_20261017/dump_A_meta.json` | 3649 | `7AF38DE3DAE3B2A3D1836FB4B97D84E380118DBB74913F062234351ACE7A09FF` |
| `shift_20261017/dump_B_meta.json` | 3633 | `8F472B2579265DEB503873D128F26E90551B7A884D6434464B43173AFC9D0582` |
| `multiyear_20261017/knmi_daily_my.csv` | 164828 | `D7A641CEAA0838E89E7B0991AE9C473F87729F0E9810A64100DB9D7B2AF0D1BA` |
| `toy_B/full/frozen_j2/weather_fr/Q_76_previous-1950-2024_RR-T-Vent.csv.gz` | 9254901 | `64620095CDD1FE3F73574E96C399354938986F6E4792A70917DF17AF61D4FD2E` |
| `toy_B/full/frozen_j2/fr_rr_daily.csv` | 48009 | `341D22215E8605C2B08694D16E6A4F57A1CB68610815CC4C673A19C7B85A5F50` |
| `climid/era5land_A.csv` | 1863688 | `592401020BEBECF4AFD3B9A835414EF189C2A472B5867C529F78D632BDC68345` |
| `climid/era5land_B.csv` | 1849383 | `543E2AFD47B4D71855C550B4FD49C122E4B82F3134C7F00A063207B3B04F2F99` |
| `climid/era5land_FR.csv` | 1835467 | `31C20FB925A764C6EFDA42D1F75CFE7EB1EB2C956F7A244A981F99347706F66D` |

Public sources are revised from time to time (KNMI, Météo-France and Open-Meteo in particular), so a fresh download may not match
byte for byte.

## Paths and configuration

The scripts contain absolute Windows paths (`D:\sar_data\multiyear`, `D:\sar_data\fr_a`, the RAINCOAT clone, and a few
`E:\research\SAR\...` paths in the preparation scripts). They were **not** edited, because the frozen method and the France
pre-registration pin the SHA256 of the code (see `CHANGES_FOR_RELEASE.md`). `paths.py` lists every absolute path, the files and line
numbers that use it and what it must contain; `python paths.py` prints which of them exist on your machine and verifies the pinned
manifests. Either recreate the same locations (e.g. a `D:` drive via `subst`, or a directory junction) or edit the listed constants in
your working copy.

| Path in the code | Must contain |
|---|---|
| `D:\sar_data\multiyear` | `features_{A,B}.npz` (+ `_meta.json`), `xyexcl_{A37,A88,B88}.npz`; working folders of the NL preparation |
| `D:\sar_data\fr_a` | `features_FR.npz` (+ `_meta.json`), `xyexcl_FR.npz`; working folders of the FR pipeline |
| `E:\research\_ext\s3b_src\raincoat\Raincoat-main` | clone of the RAINCOAT repository |
| `E:\research\SAR` (preparation scripts only) | the root of this repository |
| `E:\research\SAR\shift_20261017` (`build_features_my.py`) | `dump_{A,B}.npz`, `dump_{A,B}_meta.json` |
| `E:\research\SAR\multiyear_20261017\knmi_daily_my.csv` (`toy_B/full/p9_lib_snap.py`) | KNMI daily data |
| `E:\...\references.bib` (`paper_tgrs/make_bib.py`) | bibliography source; not needed for any result |

## Reproducing the paper

Run every script from its own folder (`cd` into it); the scripts resolve their inputs relative to their own location. `<grp>` is one
of `A37 A88 B88`. Most experiment scripts are resumable (they skip transfers already present in their output CSV), so **delete or
move the shipped result CSV first** if you want to recompute it. All tables and figures can be regenerated from the shipped results
without any data (except Fig. 1 and Fig. 2, which plot the site-B features).

### 1. Experiments (need the data)

Run in this order (folder `shiftinfer_20261105/` unless stated):

| Step | Command | Output |
|---|---|---|
| Task rows | `python p1_tasks.py <grp>` | `p1_rows_<grp>.parquet`, `p1_base_<grp>.csv` |
| Archive cache | `python p9_archive.py <grp>` | `p9_arch_<grp>.pkl` |
| Shift table for the equation search | `cd toy_B && python build_table.py` | `toy_B/shift_table.csv` |
| Structure ablation, baselines EM / MLLR / empirical prior | `python p9_basis.py <grp>` | `p9_basis_<grp>.csv`, `p9_delta_<grp>.csv`, `p9_coef_<grp>.csv` |
| Archive-length curve (P9) / covariance update (P5) | `python p9_curve.py <grp>`, `python p5_cov.py <grp>` | `p9_curve_<grp>.csv`, `p5_cov_<grp>.csv` |
| **Proposed prior** (main and archive length) | `python p10_lw.py <grp> main`, `python p10_lw.py <grp> curve` | `p10_main_<grp>.csv`, `p10_curve_<grp>.csv` |
| Freeze + verify the method | `cd frozen && python build_and_verify.py` | `archive_NL_*.npz`, `VERIFY.json` |
| Label-free date offsets | `python diag_frozen.py <grp>` | `diag_frozen_<grp>.csv`, `diag_frozen_coef_<grp>.csv` |
| Plug-in: HGB / RF | `python p11_c2.py <grp> 1`, `python p11_c2.py <grp> 2` | `p11_c2_<grp>.csv`, `p11_c2b_<grp>.csv` |
| Plug-in: TempCNN / DANN (GPU) | `python p11d_deep.py <grp>` | `p11d_deep_<grp>.csv` |
| Deep baselines (GPU) | `python s3a_deep.py <grp>` | `s3a_deep_<grp>.csv` |
| Early SFUDA / RAINCOAT runs (GPU) | `python s3b_recent.py <grp>` | `s3b_recent_<grp>.csv` |
| RAINCOAT (official) + SFUDA (GPU) | `python s3c_official.py tune`; pick per method the learning rate with the best mean BA over the 4 tuning transfers (`s3c_selected.txt`: RAINCOAT 5e-4, SFUDA 1e-3); `python s3c_official.py run <grp> <lr_raincoat> <lr_sfuda>` | `s3c_tune.csv`, `s3c_official_<grp>.csv` |
| RAINCOAT-SRS + SFUDA incl. France (GPU) | `python s3d_q1.py tune` (or `tune1` cells + `select`), then `python s3d_q1.py run <B88|A37|A88|FR> <lr>` with `<lr>` from `s3d_selected.txt` | `s3d_tune*.csv`, `s3d_selected.txt`, `s3d_<grp>.csv`, `s3d_trace_<grp>.csv` |
| Timing (GPU) | `python s3d_timing.py` | `s3d_timing.json` |
| ClimID-UDA | `cd climid && python fetch_era5land.py && python climid_sar.py <A37|A88|B88|FR>` | `climid/climid_<grp>.csv` |
| Equation search (toy B) | `cd toy_B && python r1.py && python r2.py && python r3.py` | `toy_B/toy_results.jsonl` |
| Equation search (BA-scored) | `cd toy_B/full && python evalba.py cands_r1 search > runs/r1.log` (same for `cands_r2`, `cands_r3` → `runs/r3.log`), `python evalba.py cands_final confirm > runs/confirm.log`, `python score.py <tag> [search|confirm]` | `runs/ba_*.csv`, `results.jsonl` |
| Rain-driven prior, archive length + freeze | `cd toy_B/full && python curve_b.py <grp>`; `cd frozen_j2 && python fr_station.py && python build_and_verify_j2.py` | `runs/curve_<grp>.csv`, `archive_j2_NL_*.npz`, `VERIFY_J2.json` |
| France confirmation | `cd newregion && python confirm_fr.py && python confirm_fr_summary.py` | `CONFIRM_FR_rows.csv`, `CONFIRM_FR_coef.csv`, `CONFIRM_FR_N5_subsets.json`, `CONFIRM_FR_SUMMARY.json` |
| France deep baselines (GPU) | `cd newregion && python fr_deep.py && python fr_deep_summary.py` | `FR_DEEP_rows.csv`, `FR_DEEP_SUMMARY.json` |
| Pooled training, ablations, concurrent-season exclusion (S3h; deep part GPU) | `python s3h_cpu.py <grp|FR> [part n]`, `python s3h_deep.py <grp|FR> [part n]`, `python s3h_poolx.py <cpu|deep> <grp|FR> [part n]` (`s3h_supervise*.py`, `s3h_finish.py`: the runners used) | `s3h_cpu_*.csv`, `s3h_deep_*.csv`, `s3h_poolx_*.csv` |
| Archive-size curve (S3i; deep part GPU) | `python s3i_curve.py <grp|FR> [part n]`, `python s3i_deep.py <grp|FR> [part n]` (runners `s3i_supervise*.py`) | `s3i_curve_*.csv`, `s3i_deep_*.csv` |
| Dutch tables, archive without the transfer's years (S3j; deep part GPU) | `python s3j_cpu.py <grp>`, `python s3j_deep.py <grp>` (runner `s3j_run.py`; `--check` reproduces the full-archive rows, outputs in `s3j_check/`) | `s3j_cpu_<grp>.csv`, `s3j_deep_<grp>.csv` |

### 2. Summaries (no data needed)

| Command (folder `shiftinfer_20261105/`) | Output |
|---|---|
| `python p9_summary.py`, `python p10_summary.py` | `P9_SUMMARY.json`, `P10_SUMMARY.json` (pre-registered criteria; not read by the tables) |
| `python p11_summary.py` | `P11_SUMMARY.json` |
| `cd results && python consolidate.py` | `results/CONSOLIDATED.json` (+ two diagnostic PNGs) |
| `cd results && python compare_baselines.py` | `results/BASELINES.json` |
| `python s3d_table.py` | `s3d_main_table.csv` (proposed NL columns from `s3j_deep_*.csv`) |
| `python s3h_summary.py`, `python s3i_summary.py` | `s3h_summary.csv`, `s3h_medians.csv`, `s3i_summary.csv`, `s3i_medians.csv` |
| `cd results && python excl_tables.py` | `results/EXCL.json` (Dutch tables and severity split, archive without the transfer's years) |
| `cd toy_B/full && python curve_summary.py` | `runs/curve_summary.csv` |

`P11D_SUMMARY.json` (medians and paired differences of `p11d_deep_*.csv` and `s3a_deep_*.csv`, incl. the severity split) and
`S3C_SUMMARY.json` (medians of `s3c_official_*.csv`) were computed interactively; no script for them is in the repository. Both are
shipped together with their per-transfer inputs.

### 3. Tables and figures (folder `shiftinfer_20261105/paper_tgrs/`)

| Paper item | Script | Reads |
|---|---|---|
| Table: comparison with deep and recent methods (`tab_sota`) | `tables/make_tab_sota.py` | `s3d_main_table.csv` |
| Table: Dutch sites, Gaussian framework (`tab_nl`) | `tables/make_tables.py` | `results/EXCL.json` |
| Table: classifier plug-in (`tab_plugin`) | `tables/make_tables.py` | `results/EXCL.json` |
| Table: France confirmation (`tab_fr`) | `tables/make_tables.py` | `newregion/CONFIRM_FR_SUMMARY.json` |
| Table: rain-driven prior (`tab_rain`) + fitted parameters quoted in the text | `tables/make_tab_rain.py` | `toy_B/full/runs/curve_*.csv`, `ba_cands_final_confirm_*.csv`, `r3.log`, `p10_curve_*.csv`, `p10_main_*.csv`, `newregion/CONFIRM_FR_rows.csv`, `frozen_j2/VERIFY_J2.json` |
| Supplementary tables S-ablation, S-France | `tables/make_supp.py` | `p9_basis_*.csv`, `p10_main_*.csv`, `newregion/CONFIRM_FR_SUMMARY.json` (also writes an older table from `S3C_SUMMARY.json` to `tab_s_recent_s3c_old.tex`, not used) |
| Supplementary table S-recent (RAINCOAT-SRS rules, ClimID-UDA) | `tables/make_tab_s_q1.py` | `s3d_*.csv`, `s3d_tune*.csv`, `climid/climid_*.csv`, `s3j_deep_*.csv`, `newregion/FR_DEEP_rows.csv` |
| Fig. 1 (shift phenomenon, site B) | `figs/fig1_shift.py` | `D:\sar_data\multiyear\features_B.npz`, KNMI via `p9_lib.py` |
| Fig. 2 (method diagram with data insets) | `figs/fig2_method.py` | `D:\sar_data\multiyear\features_B.npz` |
| Archive size (main text) | `figs/fig_archsize.py` | `s3i_medians.csv`, `s3i_summary.csv` |
| Archive length against the empirical prior (supplement) | `figs/fig3_archive.py` | `p10_curve_*.csv` |
| Fig. 4 (date offsets vs ΔAPI) | `figs/fig4_coef_api.py` | `diag_frozen_coef_*.csv`, `newregion/CONFIRM_FR_coef.csv` |
| Gain over DANN by shift severity | `figs/fig5_severity.py` | `results/EXCL.json`, `newregion/FR_DEEP_SUMMARY.json` |
| Computational cost (Sec. "Computational Cost") | — | `s3d_timing.json` |
| Supplement S1 "How the basis was found" | — | `toy_B/toy_results.jsonl` (toy B rounds r1–r3) |

The generated `tables/*.tex` are shipped as reference output: rerunning the table scripts on the shipped results must reproduce
them. Figures are written to `figs/` as PDF and PNG.

### 4. Figures and analyses added in the final revision

Figure numbers in the submitted manuscript: 1 `fig1_shift`, 2 `fig_study_area`, 3 `fig_parcels`, 4 `figN_traj`, 5 `fig2_method`,
6 `fig_classmap`, 7 `figN_paired`, 8 `figN_shiftacc`, 9 `figN_perclass`, 10 `fig_archsize`, 11 `fig4_coef_api`, 12 `fig5_severity`,
13 `figN_sens`; supplement: S1 `fig3_archive`, S2 `figS_classmap_api`. The script file names are authoritative (the "Fig. n" labels in the table above refer
to an earlier numbering).

| Item | Command | Reads / writes |
|---|---|---|
| Study areas (map) | `cd paper_tgrs/figs && python fig_study_area.py` (env `sar`) | RTC VV/VH GeoTIFFs of 2019, `spatial_splits_2019.csv`, `FR_BLOCK_FOLDS.json`, Natural Earth 1:10m countries |
| Reference parcels (map + composition) | `python fig_parcels.py` (env `sar`) | `parcel_raster_2019.tif`, `parcel_table_2019.csv`, `features_*.npz` |
| Per-parcel predictions for the maps (GPU) | `python s3g_mappred.py <B88|A88|FR> [src tgt]` | `s3g_pred_<grp>[_src_tgt].csv`, `s3g_pick.json` (without `src tgt`: registered rainfall rule; with them: representative transfer, see S3g in `docs/EXPLORE_S3D_Q1_ZH.md`) |
| Classification maps | `python fig_classmap.py main` / `python fig_classmap.py supp` (env `sar`) | `s3g_pred_*.csv`, `s3g_pick.json`, parcel rasters |
| Shift accuracy, per-class recall, sensitivity (S3f) | `python s3f_core.py ...` (see `s3f_sens_chain.ps1`) | `s3f_shift_<grp>.csv.gz`, `s3f_pred_<grp>.csv.gz`, `s3f_sens_<grp>.csv` |
| Figures from S3f and S3d | `python figN_traj.py`, `figN_shiftacc.py`, `figN_perclass.py`, `figN_paired.py`, `figN_sens.py` | each writes `figN_*_data.csv` next to the figure |
| Submission package | `cd paper_tgrs && python make_submission.py` | manuscript, supplement, cover letter PDFs + LaTeX source zip + SHA256 |

Note on `s3f_sens_<grp>.csv`: the sensitivity rows have five columns but were appended under the seven-column header of the main
run, so the BA of the proposed method sits in the `ba_raw` column for those rows; `figN_sens.py` reads it accordingly.

**Archive without the transfer's years.** The two Dutch sites share orbit 88 and therefore the acquisition dates and weather of a
year. Every Dutch accuracy table and the comparison with recent methods use, for a transfer (ya -> yb), the archive of the other site
without the seasons of ya and yb (S3j, S3h). The analyses of how the prior works (shift errors, per-class recall, maps, physical
coefficients, ablations) use the frozen archive with all eight seasons; the older full-archive files (`CONSOLIDATED.json`,
`BASELINES.json`, `P11*_SUMMARY.json`) are kept for traceability.

## Protocol documents

`docs/` contains the protocols written before each experiment was run, unchanged and in Chinese (working language of the project):

| File | Content |
|---|---|
| `docs/FREEZE_METHOD_ZH.md` | definition of the frozen method (hyper-parameters, archive, feature layout) |
| `docs/newregion/CONFIRM_PREREG_DRAFT_ZH.md` | pre-registration of the France confirmation: criteria N1–N5, run specification, SHA256 of code and data, deviation log (§9) |
| `docs/toy_B/full/frozen_j2/N6_ADDENDUM_ZH.md` | pre-registered addendum N6 (rain-driven prior in France, station rule) |
| `docs/newregion/FR_MAPPING_FROZEN_ZH.md` | RPG → 7-class mapping, frozen before looking at parcel data |
| `docs/newregion/CONFIRM_NEWREGION_ZH.md` | France confirmation: result record |
| `docs/EXPLORE_S3D_Q1_ZH.md` | protocol of the comparison with recent methods (RAINCOAT-SRS checkpoint rules, SFUDA, ClimID-UDA, timing), and of S3f–S3j (shift accuracy, maps, pooled training and ablations, archive-size curve with its amendments, concurrent-season exclusion), each written before the run, with results appended |
| `docs/EXPLORE_P9_ZH.md`, `EXPLORE_P10_ZH.md` | shift-basis prior and the proposed prior (P9, P10) |
| `docs/EXPLORE_P11_C2_ZH.md`, `EXPLORE_P11D_DEEP_ZH.md` | classifier plug-in (HGB/RF, TempCNN/DANN) |
| `docs/EXPLORE_S3A_DEEP_ZH.md`, `EXPLORE_S3C_OFFICIAL_ZH.md` | deep baselines, official RAINCOAT / SFUDA |
| `docs/toy_B/full/PROTOCOL_B_FULL_ZH.md`, `CONFIRM_ZH.md` | BA-scored equation search and its confirmation split |

## Citation

If you use this code, please cite the paper (citation will be added on publication).

## License

Code: MIT (see `LICENSE`). Third-party code and data keep their own licences (RAINCOAT: MIT; BRP: CC0; RPG and Météo-France:
Licence Ouverte 2.0; KNMI and ERA5/Open-Meteo: see the providers' terms).
