# Changes for the release

## Code and result files: none

Every file under `shiftinfer_20261105/`, `multiyear_20261017/`, `shift_20261017/`, `A1_observation_budget_20260923/`,
`site2_20260930/` and `docs/` is a byte-identical copy of the file used for the paper (SHA256 of every file in `MANIFEST_SHA256.tsv`).
No numerics, no logic and no paths were changed.

**Why the absolute paths were not replaced** (e.g. by an environment variable with the same default): the frozen method pins the
SHA256 of its code (`shiftinfer_20261105/frozen/SHA256_FROZEN.tsv`: `p1_tasks.py`, `p9_lib.py`, `p10_lw.py`, `xyexcl.py`, the three
`shift_20261017` helpers, ...), the rain-driven prior pins `j2_wc.py` and its builder (`toy_B/full/frozen_j2/SHA256_J2.tsv`), and the
France pre-registration lists the SHA256 of `confirm_fr.py` and of the FR pipeline (`docs/newregion/CONFIRM_PREREG_DRAFT_ZH.md`
section 8). Editing any of these files, even a path constant, would make the released code differ from the registered code.
`python paths.py` checks the manifests against the release; all shipped code files listed there match.

Instead, `paths.py` (new) lists every absolute path, the files and line numbers that use it, and what it must point to.

## Layout

The relative folder structure of the original working tree is kept (the scripts import each other through relative paths), so the
repository root plays the role of the original `E:\research\SAR` folder. Only the protocol documents were placed under `docs/`
(same relative sub-paths as in `shiftinfer_20261105/`, content unchanged). Only the files needed for the paper were copied; the
other exploration scripts of the same folders (P2-P8 variants, superseded runs, logs, reports) are not part of the release.

## Files added for the release

`README.md`, `LICENSE`, `requirements.txt`, `requirements-gpu.txt`, `.gitignore`, `paths.py`, `CHANGES_FOR_RELEASE.md`,
`MANIFEST_SHA256.tsv`.

## Not included

- Sentinel-1 products, BRP / RPG parcels, parcel feature files (`features_*.npz`, `xyexcl_*.npz`, `dump_*.npz`) and all weather data
  (KNMI, Météo-France, ERA5): public sources and checksums in the README.
- Derived intermediates that are recomputed by the scripts: `p1_rows_<grp>.parquet` (p1_tasks.py), `p9_arch_<grp>.pkl`
  (p9_archive.py), `toy_B/shift_table.csv` (toy_B/build_table.py, 14.5 MB, contains weather features).
- The RAINCOAT repository (third party, MIT): clone it, see README.
- HyP3 job lists and receipts (account-specific).
- The pipeline of the earlier study that produced the Dutch 2024/2025 features (`a1_s6_experiments`, used by `shift_20261017/p2_dump.py`).
- The manuscript sources (only the generated `paper_tgrs/tables/*.tex` are shipped, as reference output of the table scripts).

## Known gaps (documented in the README)

- `P11D_SUMMARY.json` and `S3C_SUMMARY.json` have no generating script; they were computed interactively from the shipped
  per-transfer CSVs.
- `newregion/confirm_fr_summary.py` differs from the SHA256 prefix recorded in the France pre-registration (`A08F217AD5BB32F7`
  registered, `EA40B51CD5857D12` shipped). The shipped file is the last saved version (saved 8 s before `CONFIRM_FR_SUMMARY.json`
  was written). The deviation log (§9) records the change of `confirm_fr.py` (N5 rule) but not a change of the summary script.

## Update 2026-11-12 (Date-Shared revision)

- Added: S3h, S3i and S3j scripts with their per-transfer outputs and summaries; `results/excl_tables.py` and `results/EXCL.json`;
  `paper_tgrs/figs/fig_archsize.py`.
- Re-copied, because the originals changed: table and figure scripts that now read the results with the archive without the
  transfer's years (`make_tables.py`, `make_tab_s_q1.py`, `fig5_severity.py`, `figN_paired.py` docstring, `s3d_table.py`); renamed
  labels (`fig2_method.py`, `make_tab_rain.py`); `fig1_shift.py` (mean-over-crops line, Spearman rho); `make_supp.py` (writes its older
  table to `tab_s_recent_s3c_old.tex` instead of overwriting `tab_s_recent.tex`); regenerated `tables/*.tex`, `s3d_main_table.csv`.
- Still byte-identical copies of the files used for the paper; no SHA256-pinned file changed (`python paths.py`).
- Checked: rerunning `s3h_summary.py`, `s3i_summary.py`, `results/excl_tables.py`, `s3d_table.py` and every table script inside the
  release reproduces the shipped summaries and `tables/*.tex` byte for byte.

Note: `s3h_lib.py:14` (also imported by S3i/S3j helpers) uses the same data roots `D:\sar_data\multiyear` and `D:\sar_data\fr_a` listed by `paths.py`.

## GitHub publication copy (2026-11-12)

- Root README: manuscript status corrected to prepared for submission; repository link and integrity command added.
- Added `.gitattributes` to preserve original file bytes through Git on all platforms.
- Added `verify_release.py`, a standard-library SHA256 manifest checker.
- Rebuilt the release manifest. Research code, frozen archives, results, tables and protocol documents remain byte-identical to the audited release.
