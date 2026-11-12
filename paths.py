"""Path configuration note and checker for this release (no experiment is run).

The research scripts were run on one Windows workstation and contain hard-coded absolute paths. They were NOT edited for the
release, because the frozen method and the France pre-registration pin the SHA256 of the code files (frozen/SHA256_FROZEN.tsv,
toy_B/full/frozen_j2/SHA256_J2.tsv, docs/newregion/CONFIRM_PREREG_DRAFT_ZH.md section 8); editing them would break that audit trail.

To run the scripts, either
  (a) recreate the same locations (on Windows, e.g. `subst D: <folder>` or a directory junction for D:\\sar_data), or
  (b) edit the constants listed in ABSOLUTE_PATHS in your working copy (the line numbers below), knowing that the SHA256 of the
      edited files will then differ from the pinned values.

Usage:  python paths.py            # prints every absolute path, whether it exists here, and checks the pinned SHA256 manifests
"""
import os, hashlib

REPO = os.path.dirname(os.path.abspath(__file__))          # plays the role of E:\research\SAR in the original layout

# (path literal in the code, what it must contain, files and line numbers that use it)
ABSOLUTE_PATHS = [
    (r'D:\sar_data\multiyear',
     'Netherlands data root: features_A.npz, features_B.npz (+ _meta.json) built by multiyear_20261017/build_features_my.py; '
     'xyexcl_A37/A88/B88.npz built by build_xyexcl.py; working folders A/, B/, brp/, downloads/ of the preparation scripts',
     ['shiftinfer_20261105/p1_tasks.py:9 (ROOT)', 'multiyear_20261017/deep_baseline_2023/db_run.py:7 (ROOT)',
      'multiyear_20261017/xyexcl.py:8 (xyexcl_<grp>.npz)', 'shiftinfer_20261105/toy_B/full/evalba.py:81 (features_<site>_meta.json)',
      'shiftinfer_20261105/paper_tgrs/figs/fig1_shift.py:12, fig2_method.py:11 (features_B.npz, features_B_meta.json)',
      'shiftinfer_20261105/newregion/pipeline/smoke_check.py:11',
      'multiyear_20261017/{align_my.py:10, brp_get.py:8, brp_gdb.py:7, brp_one.py:6, build_features_my.py:8, build_labels_my.py:27-28, '
      'build_xyexcl.py:9, completeness_my.py:3, download_B.py:5, download_par.py:7, features_dev_hash.py:6}']),
    (r'D:\sar_data\fr_a',
     'France data root: features_FR.npz (+ _meta.json) built by newregion/pipeline/fr_features.py; xyexcl_FR.npz built by fr_xyexcl.py; '
     'working folders FR/, rpg/, downloads/ of the FR pipeline',
     ['shiftinfer_20261105/newregion/confirm_fr.py:21', 'shiftinfer_20261105/newregion/fr_deep.py:11,18',
      'shiftinfer_20261105/s3d_q1.py:37', 'shiftinfer_20261105/climid/climid_sar.py:25',
      'shiftinfer_20261105/newregion/pipeline/{fr_build_labels.py:18, fr_download_align.py:12, fr_features.py:7, fr_fetch_labels.py:6, '
      'fr_xyexcl.py:10, smoke_check.py:10}']),
    (r'E:\research\_ext\s3b_src\raincoat\Raincoat-main',
     'clone of https://github.com/mims-harvard/Raincoat (third party, MIT; not redistributed here)',
     ['shiftinfer_20261105/s3c_official.py:7 (RC)', 'shiftinfer_20261105/s3d_q1.py:9 (RC)']),
    (r'E:\research\SAR',
     'root of this repository (the folder that contains shiftinfer_20261105/, multiyear_20261017/, ...)',
     ['multiyear_20261017/build_labels_my.py:20 (SAR; reads the A1_observation_budget_20260923/ and site2_20260930/ configs shipped here)',
      'multiyear_20261017/build_xyexcl.py:9 (SAR; also reads sar_v2/labels/nl_brp/ BRP 2024-2025 files and A1/site2 parcel tables: NOT included)',
      'multiyear_20261017/align_my.py:11']),
    (r'E:\research\SAR\shift_20261017',
     'shift_20261017/ of this repository; must contain dump_A.npz, dump_B.npz (+ _meta.json) = NL 2024/2025 parcel features (NOT included)',
     ['multiyear_20261017/build_features_my.py:49-50']),
    (r'E:\research\SAR\multiyear_20261017\knmi_daily_my.csv',
     'KNMI daily station data, written by multiyear_20261017/fetch_knmi_my.py (NOT included)',
     ['shiftinfer_20261105/toy_B/full/p9_lib_snap.py:10 (KNMI)']),
    (r'E:\research\SAR\multiyear_20261017\paper\submission_JAG_20261023\LaTeX_source\references.bib',
     'bibliography of an earlier manuscript; only needed to rebuild paper_tgrs/refs.bib (manuscript files are not part of this release)',
     ['shiftinfer_20261105/paper_tgrs/make_bib.py:5 (SRC)']),
]

# Inputs referenced by RELATIVE paths that are not shipped (data, weather, large intermediates). Paths relative to REPO.
RELATIVE_INPUTS_NOT_SHIPPED = [
    ('shift_20261017/dump_A.npz', 'NL 2024/2025 features; loaded at import time by p3f_em.py (exec-ed by p1_tasks.py) -> needed by every NL script'),
    ('shift_20261017/dump_B.npz', 'idem'),
    ('shift_20261017/knmi_daily.csv', 'older KNMI extract, only compared against by multiyear_20261017/fetch_knmi_my.py'),
    ('multiyear_20261017/knmi_daily_my.csv', 'KNMI stations 283 (site A) / 280 (site B); fetch_knmi_my.py; read by p9_lib.py, toy_B/build_table.py, toy_B/full/evalba.py'),
    ('shiftinfer_20261105/climid/era5land_A.csv', 'ERA5 via Open-Meteo; climid/fetch_era5land.py'),
    ('shiftinfer_20261105/climid/era5land_B.csv', 'idem'),
    ('shiftinfer_20261105/climid/era5land_FR.csv', 'idem'),
    ('shiftinfer_20261105/toy_B/full/frozen_j2/weather_fr/Q_76_previous-1950-2024_RR-T-Vent.csv.gz', 'Meteo-France daily, departement 76 (meteo.data.gouv.fr)'),
    ('shiftinfer_20261105/toy_B/full/frozen_j2/fr_rr_daily.csv', 'station 76679001 daily RR; frozen_j2/fr_station.py'),
    ('shiftinfer_20261105/toy_B/shift_table.csv', 'derived shift table (14.5 MB); toy_B/build_table.py'),
    ('shiftinfer_20261105/p1_rows_A37.parquet', 'derived task rows; p1_tasks.py A37'),
    ('shiftinfer_20261105/p1_rows_A88.parquet', 'p1_tasks.py A88'),
    ('shiftinfer_20261105/p1_rows_B88.parquet', 'p1_tasks.py B88'),
    ('shiftinfer_20261105/p9_arch_A37.pkl', 'derived archive cache; p9_archive.py A37'),
    ('shiftinfer_20261105/p9_arch_A88.pkl', 'p9_archive.py A88'),
    ('shiftinfer_20261105/p9_arch_B88.pkl', 'p9_archive.py B88'),
]

MANIFESTS = [('shiftinfer_20261105/frozen/SHA256_FROZEN.tsv', 'shiftinfer_20261105'),
             ('shiftinfer_20261105/toy_B/full/frozen_j2/SHA256_J2.tsv', 'shiftinfer_20261105/toy_B/full/frozen_j2')]
J2_EXTRA = {'hybridD_lw.py': 'shiftinfer_20261105/frozen/hybridD_lw.py',
            'fr_rr_daily.csv': 'shiftinfer_20261105/toy_B/full/frozen_j2/fr_rr_daily.csv',
            'Q_76_previous-1950-2024_RR-T-Vent.csv.gz': 'shiftinfer_20261105/toy_B/full/frozen_j2/weather_fr/Q_76_previous-1950-2024_RR-T-Vent.csv.gz'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest().upper()


def main():
    print('== absolute paths used by the code')
    for p, what, where in ABSOLUTE_PATHS:
        print(f"[{'ok' if os.path.exists(p) else 'missing'}] {p}\n      {what}\n      used in: " + '; '.join(where))
    print('\n== relative inputs not shipped with the release')
    for p, what in RELATIVE_INPUTS_NOT_SHIPPED:
        print(f"[{'ok' if os.path.exists(os.path.join(REPO, p)) else 'missing'}] {p}  ({what})")
    print('\n== pinned SHA256 manifests')
    for man, base in MANIFESTS:
        for line in open(os.path.join(REPO, man), encoding='utf-8-sig'):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            h, rel = parts[0].upper(), parts[-1].replace('\\', '/')
            if os.path.basename(man) == 'SHA256_J2.tsv' and rel in J2_EXTRA: f = os.path.join(REPO, J2_EXTRA[rel])
            elif rel.startswith('D:') or rel.startswith('E:'): f = rel
            else: f = os.path.normpath(os.path.join(REPO, base, rel))
            if not os.path.exists(f) and os.path.exists(os.path.join(REPO, 'docs', rel)): f = os.path.join(REPO, 'docs', rel)   # protocol docs live in docs/
            st =('match' if sha(f) == h else 'DIFFERENT') if os.path.exists(f) else 'not present (data, not shipped)'
            print(f'  {st:32s} {rel}')


if __name__ == '__main__':
    main()
