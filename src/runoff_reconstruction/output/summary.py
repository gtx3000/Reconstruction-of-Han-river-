"""Proxy summary table module."""

import os
import pandas as pd


def build_proxy_summary(
    mada_grids,
    stations_selected,
    selected_proxies,
    corr_info,
    segment_stats,
    final_yearly_proxies,
    output_dir,
    all_corr_info=None,
):
    """Build the proxy summary table.

    Parameters
    ----------
    mada_grids : pandas.DataFrame
    stations_selected : DataFrame
    selected_proxies : list[str]
        Proxies entering the candidate pool.
    corr_info : pandas.DataFrame
        Candidate-pool correlation info (backward-compatible).
    segment_stats : list[dict]
    final_yearly_proxies : list
    output_dir : str
    all_corr_info : pandas.DataFrame or None
        Correlation info for all proxies (including non-selected).

    Returns
    -------
    pandas.DataFrame
    """
    print("\n[Step 6] Building proxy summary table...")
    mada_grids = mada_grids.copy()
    mada_grids["proxy_name"] = mada_grids.apply(
        lambda r: f"MADA_{r['lon']}_{r['lat']}", axis=1
    )
    mada_grids["proxy_type"] = "MADA"

    stations_selected = stations_selected.copy()
    stations_selected["proxy_name"] = stations_selected["station_name"].apply(
        lambda x: f"Docu_{x}"
    )
    stations_selected["proxy_type"] = "documentary"

    all_proxies = pd.concat(
        [
            mada_grids[["proxy_name", "proxy_type", "lon", "lat"]],
            stations_selected[["proxy_name", "proxy_type", "lon", "lat"]],
        ],
        ignore_index=True,
    )

    all_proxies["in_candidate_pool"] = all_proxies["proxy_name"].isin(selected_proxies)
    _corr = all_corr_info if all_corr_info is not None else corr_info
    all_proxies = all_proxies.merge(
        _corr[["proxy", "r", "p"]],
        left_on="proxy_name",
        right_on="proxy",
        how="left",
    )
    all_proxies.drop(columns=["proxy"], inplace=True)

    proxy_years = {}
    for yearly_proxies in final_yearly_proxies:
        if yearly_proxies is not None:
            for p in yearly_proxies:
                proxy_years[p] = proxy_years.get(p, 0) + 1

    all_proxies["n_recon_years"] = (
        all_proxies["proxy_name"].map(proxy_years).fillna(0).astype(int)
    )
    all_proxies["in_reconstruction"] = all_proxies["n_recon_years"] > 0

    cols = [
        "proxy_name", "proxy_type", "lon", "lat",
        "r", "p", "n_recon_years", "in_reconstruction", "in_candidate_pool",
    ]
    all_proxies = (
        all_proxies.rename(
            columns={
                "lon": "lon",
                "lat": "lat",
                "r": "r",
                "p": "p",
            }
        )
        .sort_values(["in_reconstruction", "n_recon_years"], ascending=[False, False])
        .reset_index(drop=True)
    )
    cols = [c for c in cols if c in all_proxies.columns]
    all_proxies = all_proxies[cols]

    # Format numeric columns to fixed-digit strings (keeping trailing zeros)
    if "lon" in all_proxies.columns:
        all_proxies["lon"] = all_proxies["lon"].apply(lambda v: f"{v:.2f}")
    if "lat" in all_proxies.columns:
        all_proxies["lat"] = all_proxies["lat"].apply(lambda v: f"{v:.2f}")
    if "r" in all_proxies.columns:
        all_proxies["r"] = all_proxies["r"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else ""
        )
    if "p" in all_proxies.columns:
        all_proxies["p"] = all_proxies["p"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else ""
        )

    summary_path = os.path.join(output_dir, "proxy_summary.xlsx")
    all_proxies.to_excel(summary_path, index=False)
    print(f"  [OK] Proxy summary table saved to: {summary_path}")
    print(
        f"       Total {len(all_proxies)} proxies, "
        f"{all_proxies['in_reconstruction'].sum()} used in reconstruction"
    )
    return all_proxies
