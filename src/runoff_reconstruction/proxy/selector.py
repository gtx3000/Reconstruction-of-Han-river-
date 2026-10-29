"""Proxy screening module.

The proxy strategy is fixed to "both": all documentary proxies are kept and
MADA grids fill the remaining candidate slots. Screening steps:
significance test -> MADA within-group collinearity removal (documentary
proxies are skipped) -> candidate truncation by |r|.
Collinearity uses full-period proxy data; significance uses calibration
runoff only.

A MADA pair counts as strongly collinear only when BOTH |r| > corr_thresh and
p < collin_p_thresh (default 0.01) hold for that pair; of the two, the proxy
with the weaker |r| against observed runoff is dropped.
"""

import numpy as np
import pandas as pd
from scipy import stats


def filter_proxies(df, sig_level, corr_thresh, max_candidates, collin_p_thresh=0.01):
    """Full-period proxy screening.

    Significance testing is performed on the calibration period (years with
    runoff); collinearity removal uses full-period proxy data within the MADA
    group only, keeping all documentary proxies. A MADA pair is collinear only
    when |r| > corr_thresh AND p < collin_p_thresh both hold; of that pair the
    proxy with the weaker |r| against observed runoff is dropped.

    Returns
    -------
    tuple[list, pandas.DataFrame]
        (final_proxies, corr_info)
    """
    print(
        f"\n[Step 3] Screening proxies (p<{sig_level}, "
        f"collinearity |r|>{corr_thresh} & p<{collin_p_thresh}, "
        f"max_candidates={max_candidates}, strategy=both)..."
    )
    cal_df = df.dropna(subset=["runoff"])
    Y = np.log(cal_df["runoff"].values)
    X_cols = [col for col in df.columns if col not in ["year", "runoff", "log_runoff"]]
    # Full-period proxy matrix (used for collinearity)
    X_full = df[X_cols].values
    # Calibration-period proxy matrix (used for significance)
    X_cal = cal_df[X_cols].values

    kept_idx = select_proxies_by_correlation(
        X_cal,
        Y,
        X_cols,
        sig_level,
        corr_thresh,
        max_candidates,
        X_full=X_full,
        collin_p_thresh=collin_p_thresh,
    )
    if not kept_idx:
        raise ValueError("No proxy passed the significance test. Relax sig_level!")

    final_proxies = [X_cols[j] for j in kept_idx]
    print(f"  -> Final candidate proxies: {len(final_proxies)}")
    final_corr_info = compute_corr_info(df, final_proxies)
    return final_proxies, final_corr_info


def select_proxies_by_correlation(
    X,
    Y,
    proxy_names,
    sig_level,
    corr_thresh,
    max_candidates,
    verbose=True,
    X_full=None,
    collin_p_thresh=0.01,
):
    """Correlation-based proxy screening.

    Steps: significance test (p < sig_level, sorted by p) -> MADA within-group
    collinearity removal (documentary proxies skipped) -> "both" truncation.

    A MADA pair is treated as strongly collinear only when BOTH
    |r| > corr_thresh AND p < collin_p_thresh hold for that pair; of the two,
    the proxy with the weaker |r| against observed runoff is dropped.

    Parameters
    ----------
    X : np.ndarray
        Calibration/training proxy matrix (n_samples, n_proxies).
    Y : np.ndarray
        Calibration/training target (log runoff).
    proxy_names : list[str]
        Column names of X (used to identify Docu / MADA proxies).
    verbose : bool
        Whether to print screening details.
    X_full : np.ndarray or None
        Full-period proxy matrix used for collinearity.
    collin_p_thresh : float
        Significance level of the pairwise proxy-proxy correlation. A pair is
        collinear only if its p value is below this threshold (default 0.01).

    Returns
    -------
    list[int]
        Kept column indices of X (empty when no proxy passes significance).
    """
    n_cols = X.shape[1]
    corr_rows = []
    for j in range(n_cols):
        x_col = X[:, j]
        mask = ~np.isnan(x_col) & ~np.isnan(Y)
        if np.sum(mask) < 5:
            corr_rows.append({"idx": j, "proxy": proxy_names[j], "r": np.nan, "p": 1.0})
            continue
        r, p = stats.pearsonr(Y[mask], x_col[mask])
        corr_rows.append({"idx": j, "proxy": proxy_names[j], "r": r, "p": p})

    corr_df = pd.DataFrame(corr_rows)
    sig_df = corr_df[corr_df["p"] < sig_level].sort_values("p")
    if verbose:
        print(f"  -> Significance screening kept: {len(sig_df)} / {n_cols}")
    if sig_df.empty:
        return []

    # Collinearity removal: MADA group only, documentary proxies always kept
    sig_idx = sig_df["idx"].tolist()
    docu_mask_sig = sig_df["proxy"].str.startswith("Docu_")
    mada_positions = [i for i, is_docu in enumerate(docu_mask_sig) if not is_docu]

    mada_remove_positions = set()
    if len(mada_positions) > 1:
        mada_sig_idx = [sig_idx[i] for i in mada_positions]
        X_collin = X_full[:, mada_sig_idx] if X_full is not None else X[:, mada_sig_idx]
        n_mada = len(mada_positions)
        # Pairwise-complete Pearson r and p between MADA proxies. A pair is
        # strongly collinear only when |r| > corr_thresh AND p < collin_p_thresh.
        r_pair = np.zeros((n_mada, n_mada))
        p_pair = np.ones((n_mada, n_mada))
        for i in range(n_mada):
            for j in range(i + 1, n_mada):
                xi, xj = X_collin[:, i], X_collin[:, j]
                pair_mask = ~np.isnan(xi) & ~np.isnan(xj)
                if np.sum(pair_mask) < 5:
                    continue
                r_ij, p_ij = stats.pearsonr(xi[pair_mask], xj[pair_mask])
                r_pair[i, j] = r_pair[j, i] = r_ij
                p_pair[i, j] = p_pair[j, i] = p_ij
        # Loser rule: of a collinear pair, drop the one with the weaker |r|
        # against observed runoff (the same r reported in proxy_summary.xlsx).
        abs_r_runoff = sig_df["r"].abs().values
        for i in range(n_mada):
            for j in range(i + 1, n_mada):
                if abs(r_pair[i, j]) > corr_thresh and p_pair[i, j] < collin_p_thresh:
                    pos_i, pos_j = mada_positions[i], mada_positions[j]
                    loser = (
                        pos_i if abs_r_runoff[pos_i] < abs_r_runoff[pos_j] else pos_j
                    )
                    mada_remove_positions.add(loser)

    kept_df = sig_df.iloc[
        [i for i in range(len(sig_df)) if i not in mada_remove_positions]
    ].copy()
    if verbose:
        n_mada_removed = len(mada_remove_positions)
        print(
            f"  [OK] Kept after collinearity removal: {len(kept_df)} "
            f"(MADA removed {n_mada_removed}, documentary kept all)"
        )
    if kept_df.empty:
        return []

    kept_df["abs_r"] = kept_df["r"].abs()
    kept_df = kept_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    docu_mask = kept_df["proxy"].str.startswith("Docu_")
    docu_proxies = kept_df.loc[docu_mask, "proxy"].tolist()
    mada_proxies = kept_df.loc[~docu_mask, "proxy"].tolist()
    final_names = _select_both(
        docu_proxies, mada_proxies, kept_df, max_candidates, verbose
    )

    return kept_df.loc[kept_df["proxy"].isin(final_names), "idx"].tolist()


def _select_both(docu_proxies, mada_proxies, corr_info, max_candidates, verbose=True):
    """ "both" strategy: keep all documentary proxies and fill the rest with MADA."""
    if len(corr_info) > max_candidates:
        remaining_slots = max_candidates // 2 - len(docu_proxies)
        if remaining_slots <= 0:
            final_proxies = (
                docu_proxies[: max_candidates // 2]
                + mada_proxies[: max_candidates // 2]
            )
            if verbose:
                print(
                    f"  [WARN] {len(docu_proxies)} documentary proxies exceed "
                    f"max_candidates//2={max_candidates // 2}; selecting top "
                    f"{max_candidates // 2} from each group by |r|"
                )
        else:
            final_proxies = (
                docu_proxies + mada_proxies[: remaining_slots + max_candidates // 2]
            )
            if verbose:
                print(
                    f"  [WARN] Too many candidates; keeping all {len(docu_proxies)} "
                    f"documentary proxies and top "
                    f"{remaining_slots + max_candidates // 2} MADA grids"
                )
    else:
        final_proxies = corr_info["proxy"].tolist()
    return final_proxies


def compute_corr_info(df, proxies):
    """Compute proxy vs log(runoff) correlation over the calibration period.

    Returns
    -------
    pandas.DataFrame
        Columns proxy, r, p, sorted by |r| descending.
    """
    cal_df = df.dropna(subset=["runoff"])
    Y = np.log(cal_df["runoff"].values)
    rows = []
    for col in proxies:
        x = cal_df[col].values
        m = ~np.isnan(x)
        if np.sum(m) < 5:
            rows.append({"proxy": col, "r": np.nan, "p": 1.0})
            continue
        r, p = stats.pearsonr(Y[m], x[m])
        rows.append({"proxy": col, "r": r, "p": p})
    corr_df = pd.DataFrame(rows)
    corr_df["abs_r"] = corr_df["r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return corr_df[["proxy", "r", "p"]]
