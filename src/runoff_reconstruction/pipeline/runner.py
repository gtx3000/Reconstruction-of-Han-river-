"""Reconstruction pipeline orchestration module.

Provides the unified process_basin entry point that performs basin-boundary
spatial screening and runs the Bayesian hierarchical reconstruction (with a
MADA-only fallback).
"""

import os

import numpy as np
import pandas as pd

from ..spatial import basin_filter
from ..data.loader import load_and_align_data
from ..proxy.selector import filter_proxies, compute_corr_info
from ..reconstruction.bayesian import bayesian_hierarchical_reconstruction
from ..output.exporter import finalize_and_save
from ..output.summary import build_proxy_summary
from ..output.diagnostics import save_model_residual_diagnostics
from ..validation.cv import run_calibration_cv
from ..visualization.figures import (
    save_corr_scatter_plots,
    plot_proxy_distribution_map,
)
from ..utils.logging import redirect_stdout_to_file


def _find_hydro_station(hydro_stations_path, basin_name):
    """Match a hydrometric station by basin name; return (name, lon, lat) or None."""
    if not hydro_stations_path or not os.path.isfile(hydro_stations_path):
        return None
    try:
        hydro_df = pd.read_excel(hydro_stations_path)
    except Exception:
        try:
            hydro_df = pd.read_csv(hydro_stations_path)
        except Exception:
            return None
    target = basin_name.lower()
    basin_col = hydro_df["basin"].astype(str).str.lower()
    name_col = hydro_df["name"].astype(str).str.lower()
    rows = hydro_df[(basin_col == target) | (name_col == target)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return (str(row["name"]), float(row["lon"]), float(row["lat"]))


def _resolve_hydro_stations(task_hydro_stations_path, hydro_stations_path, basin_name):
    """Resolve the hydrometric station list used for map annotation."""
    if task_hydro_stations_path and os.path.isfile(task_hydro_stations_path):
        try:
            df = pd.read_excel(task_hydro_stations_path)
        except Exception:
            df = pd.read_csv(task_hydro_stations_path)
        return [
            (str(r["name"]), float(r["lon"]), float(r["lat"])) for _, r in df.iterrows()
        ]
    matched = _find_hydro_station(hydro_stations_path, basin_name)
    return [matched] if matched else []


def process_basin(
    basin_path,
    runoff_path,
    out_path,
    mada_path,
    docu_stations_meta_path,
    docu_value_path,
    start_year,
    end_year,
    sig_level,
    corr_thresh,
    k_folds,
    max_candidates,
    missing_thresh,
    basin_buffer,
    hydro_stations_path=None,
    map_lon_min=89,
    map_lon_max=125,
    map_lat_min=22,
    map_lat_max=38,
    base_shp_path=None,
    mada_grid_points_path=None,
    grid_res=None,
    task_hydro_stations_path=None,
    cv_mode="contiguous",
):
    """Run the full single-basin reconstruction pipeline.

    Parameters
    ----------
    basin_buffer : float
        Basin buffer distance (degrees).
    map_lon_min, map_lon_max, map_lat_min, map_lat_max : float
        Lon/lat bounds for the proxy distribution map.
    base_shp_path : str or None
        Base boundary shapefile (e.g. province) drawn as a dashed basemap.
    mada_grid_points_path : str or None
        Path to the valid MADA land-grid list (avoids ocean grids).
    grid_res : float or None
        MADA grid resolution (degrees) used for grid-cell boundary screening.
        When None, it is inferred automatically from the grid coordinates.
    task_hydro_stations_path : str or None
        Optional task-specific hydrometric station file for map annotation.
    cv_mode : str
        "contiguous" or "random".
    """
    os.makedirs(out_path, exist_ok=True)
    table_dir = os.path.join(out_path, "table")
    os.makedirs(table_dir, exist_ok=True)
    log_file = os.path.join(out_path, "run_log.txt")

    basin_name = os.path.basename(basin_path)[:-4]
    hydro_stations = _resolve_hydro_stations(
        task_hydro_stations_path, hydro_stations_path, basin_name
    )

    with redirect_stdout_to_file(log_file):
        print("=" * 60)
        print(f" Basin runoff reconstruction " f"(spatial=basin, method=bayesian)")
        print(f" Current basin: {basin_name}")
        print("=" * 60)

        # Step 1: basin-boundary spatial screening (both proxy types)
        basin_gdf = basin_filter.read_basin(basin_path)
        mada_grids = basin_filter.filter_mada_grids(
            mada_path,
            basin_gdf,
            grid_res=grid_res,
            grid_points_path=mada_grid_points_path,
            buffer_deg=basin_buffer,
        )
        stations_selected = basin_filter.filter_stations(
            docu_stations_meta_path, basin_gdf, buffer_deg=basin_buffer
        )

        # Step 2: load and align data
        df = load_and_align_data(
            mada_path,
            mada_grids,
            docu_value_path,
            stations_selected,
            runoff_path,
            start_year,
            end_year,
        )

        # Step 3: proxy screening
        all_proxies = [
            c for c in df.columns if c not in ("year", "runoff", "log_runoff")
        ]
        full_period_proxies, corr_info = filter_proxies(
            df,
            sig_level=sig_level,
            corr_thresh=corr_thresh,
            max_candidates=max_candidates,
        )

        # Step 3.5: correlation scatter plots (proxy vs runoff)
        save_corr_scatter_plots(df, corr_info, out_path)

        # Step 4: Bayesian reconstruction + MADA-only fallback
        (
            recon_log,
            recon_log_std,
            year_r2,
            year_r2a,
            year_n_proxies,
            segment_stats,
            final_yearly_proxies,
            fallback_mask,
            coef_records,
            year_smearing,
            model_diagnostics,
        ) = bayesian_hierarchical_reconstruction(
            df,
            all_proxies,
            final_proxies=full_period_proxies,
        )

        # Step 5: save reconstruction (with uncertainty)
        finalize_and_save(
            df,
            recon_log,
            year_r2,
            year_r2a,
            year_n_proxies,
            table_dir,
            recon_log_std=recon_log_std,
            fallback_mask=fallback_mask,
            smearing=year_smearing,
        )

        # Step 5.5: save segment statistics
        if segment_stats:
            seg_df = pd.DataFrame(segment_stats)
            if "proxies" in seg_df.columns:
                seg_df["n_mada"] = seg_df["proxies"].apply(
                    lambda x: (
                        sum(1 for p in x if p.startswith("MADA_"))
                        if isinstance(x, list)
                        else 0
                    )
                )
                seg_df["n_docu"] = seg_df["proxies"].apply(
                    lambda x: (
                        sum(1 for p in x if p.startswith("Docu_"))
                        if isinstance(x, list)
                        else 0
                    )
                )
                seg_df["docu_stations"] = seg_df["proxies"].apply(
                    lambda x: (
                        ",".join(
                            p[len("Docu_"):] for p in x if p.startswith("Docu_")
                        )
                        if isinstance(x, list)
                        else ""
                    )
                )
            # Per-segment model hyperparameters: intercept, alpha, lambda, n_iter.
            # Model equation: ln(Q) = intercept + sum(coef_j * proxy_j);
            # per-proxy coefficients (coef, posterior sd, t) live in
            # segment_coefficients.xlsx and are not repeated here.
            hyper_by_seg = {}
            for rec in coef_records:
                hyper_by_seg[rec["segment"]] = (
                    rec["intercept"],
                    rec["alpha"],
                    rec["lambda"],
                    rec["n_iter"],
                )
            for k, key in enumerate(("intercept", "alpha", "lambda", "n_iter")):
                seg_df[key] = [
                    hyper_by_seg.get(s, (None, None, None, None))[k]
                    for s in seg_df["segment"]
                ]
            col_order = [
                "segment",
                "year_range",
                "n_recon_years",
                "n_proxies",
                "n_mada",
                "n_docu",
                "docu_stations",
                "n_cal_years",
                "recon_log_mean",
                "recon_log_std",
                "extreme_freq",
                "R2",
                "R2a",
                "var_match_a",
                "var_match_b",
                "rmse",
                "sigma_eps",
                "smearing",
                "intercept",
                "alpha",
                "lambda",
                "n_iter",
            ]
            keep_cols = [c for c in col_order if c in seg_df.columns]
            seg_df = seg_df[keep_cols]
            # Segment ids are already ranked by n_proxies desc, then start
            # year asc (assigned in bayesian.py), so sorting by id gives the
            # final row order.
            seg_df = seg_df.sort_values("segment").reset_index(drop=True)
            seg_path = os.path.join(table_dir, "segment_stats.xlsx")
            seg_df.to_excel(seg_path, index=False)
            print(f"  [OK] Segment statistics saved to: {seg_path}")

        # Step 5.5b: per-segment model coefficients
        if coef_records:
            coef_df = pd.DataFrame(coef_records)
            coef_path = os.path.join(table_dir, "segment_coefficients.xlsx")
            coef_df.to_excel(coef_path, index=False)
            print(f"  [OK] Model coefficients saved to: {coef_path}")

        # Step 5.6: proxy distribution map
        recon_proxies = sorted(
            {p for yearly in final_yearly_proxies if yearly for p in yearly}
        )
        plot_proxy_distribution_map(
            basin_name,
            basin_gdf,
            mada_grids,
            stations_selected,
            out_path,
            map_lon_min=map_lon_min,
            map_lon_max=map_lon_max,
            map_lat_min=map_lat_min,
            map_lat_max=map_lat_max,
            hydro_stations=hydro_stations,
            recon_proxies=recon_proxies,
            base_shp_path=base_shp_path,
        )

        # Step 5.7: diagnose each model on its own calibration predictions.
        save_model_residual_diagnostics(model_diagnostics, out_path, basin_name)

        # Step 6: proxy summary table
        all_corr_info = compute_corr_info(df, all_proxies)
        build_proxy_summary(
            mada_grids,
            stations_selected,
            full_period_proxies,
            corr_info,
            segment_stats,
            final_yearly_proxies,
            table_dir,
            all_corr_info=all_corr_info,
        )

        # Step 7: independent calibration cross-validation
        run_calibration_cv(
            df,
            sig_level,
            corr_thresh,
            max_candidates,
            k_folds,
            output_dir=table_dir,
            cv_mode=cv_mode,
        )

        print("\n" + "=" * 60)
        print(f" Basin {basin_name} finished. Log saved to: {log_file}")
        print("=" * 60)
