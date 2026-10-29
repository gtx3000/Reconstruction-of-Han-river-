"""Shared utilities for the reconstruction module."""

import numpy as np


def build_segments(X_all):
    """Group years into segments by identical proxy availability patterns.

    For each year, the set of proxies (columns) with non-NaN values forms its
    availability pattern. All years sharing the same pattern make up one
    reconstruction segment, so segments are DISJOINT (each year belongs to
    exactly one segment) and may be non-contiguous in time (e.g. documentary
    records with gaps). Segments are sorted by their first year.

    Parameters
    ----------
    X_all : np.ndarray
        Full-period proxy matrix, shape (n_years, n_proxies).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        (segment_masks, available_proxies_per_segment)
    """
    n_total = X_all.shape[0]
    patterns = {}
    for i in range(n_total):
        key = tuple(np.flatnonzero(~np.isnan(X_all[i])))
        patterns.setdefault(key, []).append(i)

    if not any(patterns):
        raise ValueError("No valid proxy data!")

    segment_masks = []
    available_proxies_per_segment = []
    for key, years in sorted(patterns.items(), key=lambda kv: kv[1][0]):
        mask = np.zeros(n_total, dtype=bool)
        mask[years] = True
        segment_masks.append(mask)
        available_proxies_per_segment.append(np.asarray(key, dtype=int))

    return segment_masks, available_proxies_per_segment
