"""AMEND_SPATIAL_20261021.md helper: cross-year 400 m exclusion. drop(grp, s, t, k, n) -> boolean mask over the n source-year feature
rows; True = TRAIN parcel within 400 m of a target-year fold-k TEST parcel (built and verified by build_xyexcl.py). s == t -> no drop."""
import numpy as np
_C = {}
def drop(grp, s, t, k, n):
    m = np.zeros(n, bool)
    if int(s) == int(t): return m
    if grp not in _C: _C[grp] = np.load(rf'D:\sar_data\multiyear\xyexcl_{grp}.npz')
    m[_C[grp][f'{int(s)}_{int(t)}_{int(k)}']] = True
    return m
