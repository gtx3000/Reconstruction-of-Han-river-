"""Visualization module.

Generates standard figures from the reconstruction results and provides
correlation scatter-plot and diagnostic-plot helpers used by the pipeline.

Standard figures:
1. calibration observed vs reconstructed comparison
2. full series + 11-year moving average
3. proxy count and model skill metric
4. calibration-length moving average + anomaly periods
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FormatStrFormatter,
    FixedLocator,
    FuncFormatter,
    MultipleLocator,
)
from pathlib import Path


# ================= Global style =================
def _apply_style():
    """Apply a unified academic plotting style."""
    try:
        import scienceplots

        plt.style.use(["science", "grid", "no-latex", "notebook"])
    except ImportError:
        pass
    plt.rcParams["axes.grid"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "DejaVu Sans",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _num(df, col):
    """Return a column as float values, coercing the exporter's string output."""
    return pd.to_numeric(df[col], errors="coerce").values


# ================= Helpers =================


def bootstrap_ci(
    data,
    stat_func=np.mean,
    n_bootstraps=1000,
    alpha=0.05,
    block_size=10,
    seed=42,
):
    """Percentile CI from a moving-block bootstrap of an ordered series.

    Sample overlapping, non-circular blocks of ``block_size`` consecutive
    observations with replacement. Concatenate blocks and trim each replicate
    to the original series length. Block starts range from 0 to n-block_size.
    ``seed`` fixes a local RNG for reproducible figures without changing the
    global random state. Non-finite values are rejected rather than removed,
    because removing years would create artificial temporal neighbours.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("Block bootstrap requires a one-dimensional time series.")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if not isinstance(n_bootstraps, (int, np.integer)) or n_bootstraps < 1:
        raise ValueError("n_bootstraps must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    if not np.isfinite(data).all():
        raise ValueError("Block bootstrap requires finite, consecutive observations.")
    if block_size > n:
        raise ValueError("block_size cannot exceed the time-series length.")

    rng = np.random.default_rng(seed)
    n_blocks = (n + block_size - 1) // block_size
    offsets = np.arange(block_size)
    boot_stats = []
    for _ in range(n_bootstraps):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = (starts[:, None] + offsets).ravel()[:n]
        boot_stats.append(stat_func(data[indices]))

    return np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def get_optimized_yticks(y_data1, y_data2, target_ticks=8):
    """Compute nice Y-axis ticks adaptively.

    The old fixed step of 50 inflated small-magnitude stations; this version
    adapts the step to the data magnitude.
    """
    y_min = np.nanmin([np.nanmin(y_data1), np.nanmin(y_data2)])
    y_max = np.nanmax([np.nanmax(y_data1), np.nanmax(y_data2)])

    span = max(y_max - y_min, abs(y_max) * 1e-6)
    raw_step = span / (target_ticks - 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw_step)

    y_lower = np.floor(y_min / step) * step
    y_upper = np.ceil(y_max / step) * step
    num_ticks = int(round((y_upper - y_lower) / step)) + 1

    while num_ticks < 5:
        if (y_upper - y_max) <= (y_min - y_lower):
            y_upper += step
        else:
            y_lower -= step
        num_ticks = int(round((y_upper - y_lower) / step)) + 1

    while num_ticks > 9:
        mag = 10 ** np.floor(np.log10(step))
        step = next(s * mag for s in (2, 2.5, 5, 10, 20) if s * mag > step)
        y_lower = np.floor(y_min / step) * step
        y_upper = np.ceil(y_max / step) * step
        num_ticks = int(round((y_upper - y_lower) / step)) + 1

    ticks = np.arange(y_lower, y_upper + step * 0.1, step)
    return y_lower, y_upper, ticks


def _format_tick(v, step):
    """Choose the number of decimals from the tick step."""
    if step >= 1:
        return f"{v:.0f}"
    return f"{v:.{max(0, int(-np.floor(np.log10(step))))}f}"


# ================= Single-basin figures =================


def generate_basin_figures(csv_path, save_dir, folder_name):
    """Generate the four standard figures for one basin.

    Parameters
    ----------
    csv_path : str or Path
        Path to reconstructed_runoff_final.xlsx.
    save_dir : str or Path
        Output directory.
    folder_name : str
        Basin name (used in titles and filenames).
    """
    _apply_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(csv_path)

    x = df["year"]
    y2 = np.round(_num(df, "reconstructed_runoff"), 2)

    calib_mask = df["observed_runoff"].notna()
    calib_years = df.loc[calib_mask, "year"].values
    y3 = np.round(_num(df, "observed_runoff")[calib_mask.values], 2)

    y2_calib = y2[calib_mask.values]
    calib_len = len(calib_years)

    y2_mean, y2_std = np.nanmean(y2), np.nanstd(y2)
    nPrx = _num(df, "n_proxies")

    # Figure 1: calibration comparison
    _fig1_calibration(calib_years, y3, y2_calib, calib_len, folder_name, save_dir)

    # Conditional predictive interval from the exported BRR bounds.
    plot_reconstruction_with_pi(df, save_dir, folder_name)

    # Combined figure: proxy count + skill metric, full series + 11-yr MA,
    # and calibration-length MA + anomaly periods in a single 3-panel plot.
    _fig_combined_series(
        x,
        y2,
        calib_years,
        y3,
        y2_mean,
        y2_std,
        nPrx,
        df,
        calib_len,
        folder_name,
        save_dir,
    )

    print(f"{folder_name}: figures saved to {save_dir}")


def _fig_combined_series(
    x,
    y2,
    calib_years,
    y3,
    y2_mean,
    y2_std,
    nPrx,
    df,
    calib_len,
    folder_name,
    save_dir,
):
    """Combined three-panel figure sharing a common x-axis.

    Panel A: number of proxies + skill metric (R2a).
    Panel B: full reconstruction series + 11-year moving average.
    Panel C: calibration-length moving average + anomaly periods;
    reference CI uses a 10-year moving-block bootstrap of instrumental runoff.
    """
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        nrows=3, sharex=True, figsize=(14, 12), dpi=600
    )

    for ax in [ax_a, ax_b, ax_c]:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))

    # ---- Panel A: proxy count + skill metric ----
    proxy_color = "#4C72B0"  # steel blue for the proxy count
    r2a_color = "r"  # muted red for the skill metric

    ax_a.plot(
        x, nPrx, label="Number of proxies", linewidth=1.8, alpha=1, color=proxy_color
    )
    ax_a.set_ylabel("Number of proxies", fontsize=14)

    max_nPrx = int(np.nanmax(nPrx))
    min_nPrx = int(np.nanmin(nPrx))
    lower_n = max(0, min_nPrx - 1)
    upper_n = max(max_nPrx + 1, int(np.ceil(max_nPrx * 1.1)))
    ax_a.set_ylim(lower_n, upper_n)
    count_step = 1
    ax_a.set_yticks(np.arange(lower_n, max_nPrx + 2, count_step))
    ax_a.tick_params(axis="x", top=False, bottom=True)
    ax_a.tick_params(axis="y", left=True, right=False, labelsize=14)

    r2a = _num(df, "R2a")
    ax2 = ax_a.twinx()
    ax2.plot(x, r2a, label="$R^2_a$", linewidth=1.8, alpha=0.9, color=r2a_color)

    ax2.set_ylabel("$R^2_a$", rotation=270, labelpad=20, fontsize=14)
    min_y2 = np.nanmin(r2a)
    max_y2 = np.nanmax(r2a)
    min_y2 = np.floor((min_y2 - 0.1) * 10) / 10
    max_y2 = np.ceil((max_y2 + 0.1) * 10) / 10
    y_range = max_y2 - min_y2
    step = 0.2 if y_range > 0.8 else 0.1
    ticks = np.round(np.arange(min_y2, max_y2 + 1e-5, step), 1)
    ax2.set_ylim(min_y2, max_y2)
    ax2.set_yticks(ticks)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.tick_params(axis="y", left=False, right=True, labelsize=14, pad=15)

    lines_1, labels_1 = ax_a.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        ncol=2,
        fontsize=12,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
    )

    # ---- Panel B: full series + 11-year moving average ----
    (Recon,) = ax_b.plot(x, y2, label="Recon", color="gray", linewidth=1, alpha=0.6)
    (Obs,) = ax_b.plot(
        calib_years, y3, label="Obs", color="deepskyblue", linewidth=1, alpha=0.6
    )
    Recon_mean = ax_b.axhline(
        y2_mean, label="Recon Mean", color="r", linewidth=1, alpha=0.7
    )
    Recon_std = ax_b.axhline(
        y2_mean + y2_std,
        label="Recon Mean±Std",
        color="r",
        linewidth=1,
        alpha=0.7,
        linestyle="--",
    )
    ax_b.axhline(y2_mean - y2_std, color="r", linewidth=1, alpha=0.7, linestyle="--")
    (Recon_11_yr,) = ax_b.plot(
        x[10:],
        pd.Series(y2).rolling(11, center=False, min_periods=11).mean()[10:],
        label="Recon 11-yr MA",
        color="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax_b.set_ylabel(r"Runoff ($10^8\ \mathrm{m}^3$)")
    y_min2 = float(min(np.nanmin(y2), np.nanmin(y3))) * 0.99
    y_max2 = float(max(np.nanmax(y2), np.nanmax(y3))) * 1.01
    ax_b.set_ylim(y_min2, y_max2)
    raw_step2 = (y_max2 - y_min2) / 8
    mag2 = 10 ** np.floor(np.log10(raw_step2))
    step2 = next(s * mag2 for s in (1, 2, 2.5, 5, 10) if s * mag2 >= raw_step2)
    ticks2 = np.arange(np.ceil(y_min2 / step2) * step2, y_max2 + step2 * 0.01, step2)
    ax_b.set_yticks(ticks2)
    ax_b.set_yticklabels([_format_tick(v, step2) for v in ticks2])
    ax_b.tick_params(
        which="both",
        direction="in",
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelbottom=False,
    )
    handles = [Recon, Obs, Recon_11_yr, Recon_mean, Recon_std]
    labels = ["Recon", "Obs", "Recon 11-yr MA", "Recon Mean", "Recon Mean±Std"]
    ax_b.legend(handles, labels, ncol=5, fontsize=12, loc="upper left", frameon=False)

    # ---- Panel C: calibration-length moving average + anomaly periods ----
    window = calib_len
    y2_series = pd.Series(y2)
    y2_roll = y2_series.rolling(window=window, center=False, min_periods=window).mean()
    # Reference CI for the instrumental mean: moving blocks of 10 years.
    # Keep the existing 1,000 replicates and fix the seed for reproducibility.
    if not np.all(np.diff(calib_years) == 1):
        raise ValueError("Block-bootstrap calibration years must be consecutive.")
    low, high = bootstrap_ci(
        y3,
        stat_func=np.mean,
        n_bootstraps=1000,
        alpha=0.05,
        block_size=10,
        seed=42,
    )
    print(
        f"{folder_name}: instrumental mean 95% CI (moving-block bootstrap, "
        f"block=10 years, replicates=1000, seed=42): [{low:.3f}, {high:.3f}]"
    )
    valid_mask = y2_roll.notna()
    x_plot = x[valid_mask].values
    y2_roll_plot = y2_roll[valid_mask].values
    outside_ci_mask = (y2_roll_plot < low) | (y2_roll_plot > high)
    outside_ci_years = x_plot[outside_ci_mask]
    if len(outside_ci_years) > 0:
        ax_c.bar(
            outside_ci_years,
            9999,
            bottom=0,
            width=1,
            alpha=0.4,
            color="pink",
            label="Abnormal periods",
            edgecolor="none",
        )
    ax_c.plot(x_plot, y2_roll_plot, label=f"{window}-yr MA", color="black", linewidth=1)
    ax_c.plot(
        x_plot,
        [low] * len(x_plot),
        linestyle="--",
        color="red",
        label="95% CI",
        linewidth=1,
    )
    ax_c.plot(x_plot, [high] * len(x_plot), linestyle="--", color="red", linewidth=1)
    y_lower3, y_upper3, y_ticks3 = get_optimized_yticks(
        y2_roll_plot, np.concatenate([y2_roll_plot, [low, high]])
    )
    y_range3 = y_upper3 - y_lower3
    y_lower3 = y_lower3 - y_range3 * 0.10
    y_upper3 = y_upper3 + y_range3 * 0.10
    y_lower3, y_upper3, y_ticks3 = get_optimized_yticks(
        y2_roll_plot, np.concatenate([y2_roll_plot, [y_lower3, y_upper3]])
    )
    ax_c.set_ylim(y_lower3, y_upper3)
    ax_c.set_yticks(y_ticks3)
    ax_c.set_yticklabels([_format_tick(v, y_ticks3[1] - y_ticks3[0]) for v in y_ticks3])
    ax_c.set_ylabel(r"Runoff ($10^8\ \mathrm{m}^3$)", rotation=270, labelpad=20)
    ax_c.yaxis.tick_right()
    ax_c.yaxis.set_label_position("right")
    ax_c.tick_params(
        which="both",
        direction="in",
        top=False,
        bottom=True,
        left=False,
        right=True,
        labelbottom=True,
    )
    # Separate the bottom-right y tick from the final year label.
    ax_c.tick_params(axis="y", which="both", pad=10)
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    by_label = dict(zip(labels_c, handles_c))
    ax_c.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        fontsize=12,
        ncol=3,
        frameon=False,
    )

    ax_c.set_xlim(960, 2010)
    ax_c.set_xticks(np.arange(1000, 2001, 100))
    ax_c.set_xlabel("Year (CE)")

    plt.subplots_adjust(hspace=0)
    plt.savefig(save_dir / f"{folder_name}_combined.png", dpi=600, bbox_inches=None)
    plt.close(fig)


def _fig1_calibration(calib_years, y3, y2_calib, calib_len, folder_name, save_dir):
    """Figure 1: observed vs reconstructed runoff over the instrumental period."""
    fig, ax = plt.subplots(figsize=(14, 4), dpi=600)
    (Obs,) = ax.plot(calib_years, y3, label="Obs", color="b", linewidth=2, alpha=0.7)
    (Recon,) = ax.plot(
        calib_years, y2_calib, label="Recon", color="r", linewidth=2, alpha=0.8
    )
    ax.set_xlabel("Year (CE)", fontsize=14)
    ax.set_ylabel(r"Runoff ($10^8\ \mathrm{m}^3$)", fontsize=14)

    start_year, end_year = 1950, calib_years[-1]
    step = 5 if calib_len <= 30 else (10 if calib_len <= 80 else 20)
    ax.set_xlim(start_year, end_year)
    xticks = np.arange(np.ceil(start_year / step) * step, end_year + 1, step)
    ax.set_xticks(xticks.astype(int))

    y_lower, y_upper, y_ticks = get_optimized_yticks(y2_calib, y3)
    y_range = y_upper - y_lower
    y_lower = y_lower - y_range * 0.10
    y_upper = y_upper + y_range * 0.10
    y_lower, y_upper, y_ticks = get_optimized_yticks([y_lower, y_upper], y3)

    ax.set_ylim(y_lower, y_upper)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_format_tick(v, y_ticks[1] - y_ticks[0]) for v in y_ticks])

    ax.legend(
        handles=[Recon, Obs],
        fontsize=12,
        ncol=2,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        borderpad=0.5,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Separate the bottom-left y tick from the first year label.
    ax.tick_params(axis="y", which="both", pad=15)
    plt.savefig(save_dir / f"{folder_name}_obs_runoff.png", dpi=600, bbox_inches=None)
    plt.close(fig)


# ================= Correlation scatter plots =================


def save_corr_scatter_plots(df, corr_info, out_path):
    """Plot proxy vs ln(runoff) scatter plots into out_path/scatter_plots/.

    The two variables are the proxy value (x) and ln(observed runoff) (y),
    using calibration samples only. Each candidate proxy gets one scatter plot
    with an OLS fit line.

    Parameters
    ----------
    df : pandas.DataFrame
        Aligned annual data table (year, runoff, proxy columns).
    corr_info : pandas.DataFrame
        Proxy correlation table (proxy, r, p columns, sorted by |r| desc).
    out_path : str or Path
        Output folder.
    """
    print("\n[Step 3.5] Plotting correlation scatter plots (proxy vs ln runoff)...")
    _apply_style()

    out_path = Path(out_path)
    scatter_dir = out_path / "figure" / "scatter_plots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    cal_df = df.dropna(subset=["runoff"])
    Y = np.log(cal_df["runoff"].values)

    plot_items = []
    for _, row in corr_info.iterrows():
        col = row["proxy"]
        if col not in cal_df.columns:
            continue
        x = cal_df[col].values
        mask = ~np.isnan(x)
        if np.sum(mask) < 5:
            continue
        plot_items.append((col, x[mask], Y[mask]))

    def _draw_panel(ax, xs, ys):
        ax.scatter(xs, ys, s=18, alpha=0.7, edgecolor="k", linewidth=0.3)
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(np.min(xs), np.max(xs), 100)
        ax.plot(x_line, slope * x_line + intercept, color="r", linewidth=1.2)
        ax.minorticks_off()
        ax.tick_params(top=False, right=False)

    for col, xs, ys in plot_items:
        fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
        _draw_panel(ax, xs, ys)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("ln(runoff)", fontsize=11)
        if col.startswith("Docu_"):
            tick_vals = np.unique(xs)
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"{v:.0f}" for v in tick_vals])
        else:
            ax.xaxis.set_major_locator(
                FixedLocator(np.linspace(np.min(xs), np.max(xs), 5))
            )
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(scatter_dir / f"{col}_vs_runoff.png", dpi=300, bbox_inches=None)
        plt.close(fig)

    print(
        f"  [OK] Plotted {len(plot_items)} proxy-runoff scatter plots to: {scatter_dir}"
    )


# ================= Proxy distribution map =================


def plot_proxy_distribution_map(
    basin_name,
    basin_gdf,
    mada_grids,
    stations_selected,
    out_path,
    map_lon_min=89,
    map_lon_max=125,
    map_lat_min=22,
    map_lat_max=38,
    hydro_stations=None,
    recon_proxies=None,
    base_shp_path=None,
):
    """Plot the proxy spatial distribution map.

    Colors proxies by whether they participated in reconstruction:
    - red = used proxies (MADA "+" / documentary dots)
    - blue = unused proxies
    Basemap: base boundary (dashed) and hydrometric stations (stars).

    Parameters
    ----------
    basin_name : str
        Station/basin name (title and filename).
    basin_gdf : geopandas.GeoDataFrame
        Controlling basin boundary.
    mada_grids : pandas.DataFrame
        MADA grids (lon, lat columns).
    stations_selected : pandas.DataFrame
        Documentary stations (station_name, lon, lat columns).
    out_path : str or Path
        Output folder.
    map_lon_min, map_lon_max, map_lat_min, map_lat_max : float
        Map lon/lat bounds.
    hydro_stations : list[tuple] or None
        Hydrometric stations [(name, lon, lat), ...] marked with stars.
    recon_proxies : list[str] or None
        Proxies used in reconstruction (red); the rest are blue.
    base_shp_path : str or None
        Base boundary shapefile drawn as a dashed basemap.
    """
    import geopandas as gpd

    print("\n[Step 5.5] Plotting proxy distribution map...")
    _apply_style()

    out_path = Path(out_path)
    figure_dir = out_path / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    C_SIG = "#C0392B"
    C_NS = "#2E86C1"
    C_STAR = "#F1C40F"

    recon_set = set(recon_proxies) if recon_proxies else set()

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=300)

    if base_shp_path and Path(base_shp_path).exists():
        base_gdf = gpd.read_file(base_shp_path)
        if base_gdf.crs is None:
            base_gdf = base_gdf.set_crs("EPSG:4326")
        elif base_gdf.crs.to_string() != "EPSG:4326":
            base_gdf = base_gdf.to_crs("EPSG:4326")
        base_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#95A5A6",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )

    basin_gdf.plot(
        ax=ax,
        facecolor="#58D68D",
        edgecolor="#145A32",
        linewidth=2.2,
        alpha=0.6,
        zorder=3,
    )

    mada_df = mada_grids.copy()
    mada_df["proxy_name"] = mada_df.apply(
        lambda r: f"MADA_{r['lon']}_{r['lat']}", axis=1
    )
    mada_recon = mada_df[mada_df["proxy_name"].isin(recon_set)]
    mada_ns = mada_df[~mada_df["proxy_name"].isin(recon_set)]
    ax.scatter(
        mada_ns["lon"],
        mada_ns["lat"],
        marker="+",
        s=70,
        linewidths=1.4,
        color=C_NS,
        zorder=5,
        label="MADA grid (not used)",
    )
    ax.scatter(
        mada_recon["lon"],
        mada_recon["lat"],
        marker="+",
        s=110,
        linewidths=2.4,
        color=C_SIG,
        zorder=6,
        label="MADA grid (used)",
    )

    docu_df = stations_selected.copy()
    docu_df["proxy_name"] = "Docu_" + docu_df["station_name"].astype(str)
    docu_recon = docu_df[docu_df["proxy_name"].isin(recon_set)]
    docu_ns = docu_df[~docu_df["proxy_name"].isin(recon_set)]
    ax.scatter(
        docu_ns["lon"],
        docu_ns["lat"],
        marker="o",
        s=45,
        facecolor=C_NS,
        edgecolor="white",
        linewidths=0.8,
        zorder=5,
        label="Documentary station (not used)",
    )
    ax.scatter(
        docu_recon["lon"],
        docu_recon["lat"],
        marker="o",
        s=60,
        facecolor=C_SIG,
        edgecolor="white",
        linewidths=0.8,
        zorder=6,
        label="Documentary station (used)",
    )

    if hydro_stations:
        hnames = [h[0] for h in hydro_stations]
        hlons = [h[1] for h in hydro_stations]
        hlats = [h[2] for h in hydro_stations]
        ax.scatter(
            hlons,
            hlats,
            marker="*",
            s=320,
            facecolor=C_STAR,
            edgecolor="black",
            linewidths=1.0,
            zorder=7,
            label=f"Gauge ({', '.join(hnames)})",
        )
        for hname, hlon, hlat in hydro_stations:
            ax.annotate(
                hname,
                (hlon, hlat),
                textcoords="offset points",
                xytext=(7, -4),
                fontsize=11,
                fontweight="bold",
                zorder=8,
            )

    ax.set_xlim(map_lon_min, map_lon_max)
    ax.set_ylim(map_lat_min, map_lat_max)

    # Choose the lower-left/right corner with less occlusion
    from shapely.geometry import box as shp_box, Point as shp_Point

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx, fy = 0.20, 0.32
    corner_boxes = {
        "lower left": shp_box(x0, y0, x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy),
        "lower right": shp_box(x1 - (x1 - x0) * fx, y0, x1, y0 + (y1 - y0) * fy),
    }

    pts_lon = (
        mada_df["lon"].tolist()
        + docu_df["lon"].tolist()
        + ([h[1] for h in hydro_stations] if hydro_stations else [])
    )
    pts_lat = (
        mada_df["lat"].tolist()
        + docu_df["lat"].tolist()
        + ([h[2] for h in hydro_stations] if hydro_stations else [])
    )
    basin_union = (
        basin_gdf.union_all()
        if hasattr(basin_gdf, "union_all")
        else basin_gdf.unary_union
    )

    def _corner_score(b):
        n_pts = sum(
            1 for lo, la in zip(pts_lon, pts_lat) if b.contains(shp_Point(lo, la))
        )
        score = float(n_pts)
        if basin_union.intersects(b):
            score += 50
        return score

    scores = {loc: _corner_score(b) for loc, b in corner_boxes.items()}
    legend_loc = min(scores, key=scores.get)
    print(
        f"     Legend location: {legend_loc} "
        f"(occlusion score lower-left {scores['lower left']:.0f} / "
        f"lower-right {scores['lower right']:.0f})"
    )

    ax.legend(
        loc=legend_loc,
        fontsize=8,
        framealpha=0.9,
        edgecolor="gray",
        handletextpad=0.6,
        borderpad=0.8,
    )

    save_file = figure_dir / f"{basin_name}_proxy_distribution.png"
    plt.savefig(save_file, dpi=300, bbox_inches=None)
    plt.close(fig)
    print(
        f"  [OK] Proxy distribution map saved to: {save_file}\n"
        f"       MADA {len(mada_df)} (used {len(mada_recon)}), "
        f"documentary {len(docu_df)} (used {len(docu_recon)})"
    )


# ================= Residual diagnostics and uncertainty =================


def run_residual_tests(residuals, fitted_values=None):
    """Residual tests: autocorrelation (DW / Ljung-Box) + heteroscedasticity (BP).

    Returns
    -------
    dict
        Contains DW, LB_stat, LB_p, BP_stat, BP_p, etc.
    """
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
    from statsmodels.tools.tools import add_constant

    result = {}
    n = len(residuals)

    dw = durbin_watson(residuals)
    result["DW"] = round(dw, 4)

    max_lag = min(10, n // 3)
    if max_lag >= 1:
        lags_to_test = [l for l in [1, 5, 10] if l <= max_lag]
        lb_res = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)
        last_lag = lags_to_test[-1]
        result["LB_lag"] = last_lag
        result["LB_stat"] = round(float(lb_res.loc[last_lag, "lb_stat"]), 4)
        result["LB_p"] = float(lb_res.loc[last_lag, "lb_pvalue"])
    else:
        result["LB_lag"] = None
        result["LB_stat"] = np.nan
        result["LB_p"] = np.nan

    if fitted_values is not None and len(fitted_values) == n:
        try:
            fitted = np.asarray(fitted_values, dtype=float).reshape(-1)
            if not np.isfinite(fitted).all() or np.ptp(fitted) == 0:
                raise ValueError("BP test requires finite, nonconstant fitted values.")
            # The auxiliary regression needs an intercept and a varying regressor.
            # robust=True explicitly uses the default Koenker/studentized variant.
            bp_stat, bp_p, _, _ = het_breuschpagan(
                residuals, add_constant(fitted, has_constant="add"), robust=True
            )
            result["BP_stat"] = round(bp_stat, 4)
            result["BP_p"] = float(bp_p)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  [WARN] Breusch-Pagan test unavailable: {exc}")
            result["BP_stat"] = np.nan
            result["BP_p"] = np.nan
    else:
        result["BP_stat"] = np.nan
        result["BP_p"] = np.nan

    return result


def plot_residual_diagnostics(residuals, out_path, folder_name, fitted_values=None):
    """Four-panel residual diagnostics: Q-Q / time series / ACF / resid vs fitted.

    Parameters
    ----------
    residuals : np.ndarray
        Calibration residuals in log-runoff space (NaNs removed).
    out_path : str or Path
        Output folder.
    folder_name : str
        Station name.
    fitted_values : np.ndarray or None
        Calibration fitted values.
    """
    from scipy.stats import probplot
    from statsmodels.graphics.tsaplots import plot_acf

    _apply_style()
    out_path = Path(out_path)
    (out_path / "figure").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    # (a) Q-Q plot
    ax_qq = axes[0, 0]
    probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set(
        marker="o",
        markersize=3,
        linestyle="none",
        markerfacecolor="#2E86C1",
        markeredgecolor="k",
        markeredgewidth=0.3,
        alpha=0.7,
    )
    ax_qq.get_lines()[1].set(color="#C0392B", linewidth=1.5)
    ax_qq.set_xlabel("Theoretical quantiles", fontsize=11)
    ax_qq.set_ylabel("Sample quantiles", fontsize=11)
    ax_qq.set_title("(a) Q-Q Plot", fontsize=12, pad=6)
    ax_qq.minorticks_off()
    ax_qq.tick_params(top=False, right=False)

    # (b) residual time series
    ax_ts = axes[0, 1]
    ax_ts.plot(residuals, color="#2E86C1", linewidth=0.8, alpha=0.8)
    ax_ts.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax_ts.set_xlabel("Sample index", fontsize=11)
    ax_ts.set_ylabel("Residuals", fontsize=11)
    ax_ts.set_title("(b) Residuals over Time", fontsize=12, pad=6)
    ax_ts.minorticks_off()
    ax_ts.tick_params(top=False, right=False)

    # (c) ACF
    ax_acf = axes[1, 0]
    max_lag = min(20, len(residuals) // 3)
    if max_lag >= 1:
        plot_acf(
            residuals,
            lags=max_lag,
            ax=ax_acf,
            alpha=0.05,
            color="#2E86C1",
            markerfacecolor="#2E86C1",
            vlines_kwargs={"colors": "#2E86C1"},
        )
    ax_acf.set_title("(c) Autocorrelation (ACF)", fontsize=12, pad=6)
    ax_acf.set_xlabel("Lag", fontsize=11)
    ax_acf.set_ylabel("ACF", fontsize=11)
    ax_acf.minorticks_off()
    ax_acf.tick_params(top=False, right=False)

    # (d) residuals vs fitted
    ax_sc = axes[1, 1]
    if fitted_values is not None and len(fitted_values) == len(residuals):
        ax_sc.scatter(
            fitted_values,
            residuals,
            s=12,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.2,
            color="#2E86C1",
        )
        ax_sc.axhline(0, color="k", linewidth=0.6, linestyle="--")
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            sm = lowess(np.abs(residuals), fitted_values, frac=0.6)
            ax_sc.plot(
                sm[:, 0],
                sm[:, 1],
                color="#C0392B",
                linewidth=1.5,
                linestyle="--",
                label="|resid| trend",
            )
            ax_sc.legend(fontsize=9, loc="upper right", frameon=False)
        except ImportError:
            pass
        ax_sc.set_xlabel("Fitted values", fontsize=11)
        ax_sc.set_ylabel("Residuals", fontsize=11)
    else:
        ax_sc.text(
            0.5,
            0.5,
            "No fitted values",
            transform=ax_sc.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
    ax_sc.set_title("(d) Residuals vs Fitted", fontsize=12, pad=6)
    ax_sc.minorticks_off()
    ax_sc.tick_params(top=False, right=False)

    plt.savefig(
        out_path / "figure" / f"{folder_name}_residual_diagnostics.png",
        dpi=300,
        bbox_inches=None,
    )
    plt.close(fig)

    print(f"  [OK] Residual diagnostics saved: {folder_name}_residual_diagnostics.png")


def plot_reconstruction_with_pi(df, save_dir, folder_name):
    """Plot runoff estimates and conditional 95% BRR predictive bounds.

    Uses the exporter's point estimates (including smearing) and predictive
    quantiles directly. Legacy *_95ci column names contain PI bounds; no
    additional smearing correction is applied to these quantiles.
    """
    _apply_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    years = _num(df, "year")
    recon = _num(df, "reconstructed_runoff")
    lower = _num(df, "recon_lower_95ci")
    upper = _num(df, "recon_upper_95ci")
    observed = _num(df, "observed_runoff")

    fig, ax = plt.subplots(figsize=(15, 5), dpi=600)
    ax.fill_between(
        years, lower, upper, color="#85C1E9", alpha=0.5,
        label="95% conditional PI",
    )
    ax.plot(years, recon, color="r", linewidth=0.9, label="Recon")
    cal = np.isfinite(observed)
    ax.plot(years[cal], observed[cal], color="b", linewidth=0.8,
            alpha=0.7, label="Obs")
    ax.set_xlim(960, 2010)
    ax.set_xticks(np.arange(1000, 2010, 100))
    ax.set_xlabel("Year (CE)", fontsize=14)
    ax.set_ylabel(r"Runoff ($10^8\ \mathrm{m}^3$)", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12, loc="upper left", frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="gray")
    output = save_dir / f"{folder_name}_recon_pi.png"
    fig.savefig(output, dpi=600, bbox_inches=None)
    plt.close(fig)
    print(f"  [OK] Conditional predictive interval plot saved: {output.name}")


# ================= Station summary table =================


def build_station_summary(csv_path):
    """Build the evaluation summary record for one station.

    Parameters
    ----------
    csv_path : str or Path
        Path to <station>/table/reconstructed_runoff_final.xlsx.

    Returns
    -------
    dict
        Summary record for the station.
    """
    csv_path = Path(csv_path)
    station_dir = csv_path.parent.parent
    df = pd.read_excel(csv_path)

    observed = _num(df, "observed_runoff")
    calib_mask = ~np.isnan(observed)
    r2a = _num(df, "R2a")
    r2a_calib = np.nanmean(r2a[calib_mask]) if calib_mask.any() else np.nan
    record = {
        "basin": station_dir.name,
        "R2a_calibration": (round(r2a_calib, 3) if not np.isnan(r2a_calib) else np.nan),
    }

    seg_path = csv_path.parent / "segment_stats.xlsx"
    if seg_path.exists():
        seg_df = pd.read_excel(seg_path)
        record["n_segments"] = len(seg_df)

    record["R2a_mean"] = np.round(np.nanmean(r2a), 3)

    cv_path = csv_path.parent / "cv_results.xlsx"
    if cv_path.exists():
        cv_df = pd.read_excel(cv_path)
        mean_mask = cv_df["fold"].astype(str) == "mean"
        if "repeat" in cv_df.columns:
            mean_mask = mean_mask | (cv_df["repeat"].astype(str) == "mean")
        mean_row = cv_df[mean_mask]
        if not mean_row.empty:
            for col in ["R2a", "RE", "CE"]:
                if col in mean_row.columns:
                    record[f"CV_{col}"] = np.round(float(mean_row[col].iloc[0]), 3)
        else:
            for col in ["R2a", "RE", "CE"]:
                if col in cv_df.columns:
                    record[f"CV_{col}"] = np.round(cv_df[col].mean(), 3)

    diag_path = csv_path.parent / "residual_diagnostics.xlsx"
    if diag_path.exists():
        diag_df = pd.read_excel(diag_path)
        if "segment" in diag_df.columns:
            record["residual_models"] = len(diag_df)
            for col in ["SW_p", "LB_p", "BP_p"]:
                if col not in diag_df.columns:
                    continue
                values = pd.to_numeric(diag_df[col], errors="coerce").dropna()
                record[f"{col}_n_tested"] = len(values)
                record[f"{col}_min"] = round(float(values.min()), 4)
                record[f"{col}_max"] = round(float(values.max()), 4)
                record[f"{col}_n_below_0_05"] = int((values < 0.05).sum())
        elif not diag_df.empty:
            row = diag_df.iloc[0]
            for col in [
                "SW_W",
                "SW_p",
                "DW",
                "LB_lag",
                "LB_stat",
                "LB_p",
                "BP_stat",
                "BP_p",
            ]:
                if col in row.index and not pd.isna(row[col]):
                    record[col] = (
                        round(float(row[col]), 4) if col != "LB_lag" else int(row[col])
                    )
    else:
        log_path = station_dir / "run_log.txt"
        if log_path.exists():
            try:
                log_text = log_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                log_text = log_path.read_text(encoding="gbk")
            sw_match = re.search(
                r"\[SW normality\] n=(\d+), W=([\d.]+), p=([\d.eE+-]+)",
                log_text,
            )
            if sw_match:
                record["SW_n"] = int(sw_match.group(1))
                record["SW_W"] = round(float(sw_match.group(2)), 3)
                record["SW_p"] = round(float(sw_match.group(3)), 3)

    return record


def generate_station_summary(csv_path, out_dir=None):
    """Write the evaluation summary table (averages_summary.xlsx) for one station.

    Parameters
    ----------
    csv_path : str or Path
        Path to <station>/table/reconstructed_runoff_final.xlsx.
    out_dir : str or Path or None
        Directory to write averages_summary.xlsx (default: the table directory).
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    record = build_station_summary(csv_path)
    summary_df = pd.DataFrame([record])
    save_path = out_dir / "averages_summary.xlsx"
    summary_df.to_excel(save_path, index=False)
    print(f"{csv_path.parent.parent.name}: summary saved to {save_path}")
    print(summary_df)


# ================= Runoff distribution figure =================

# src/runoff_reconstruction/visualization/figures.py -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def plot_runoff_distribution(runoff_dir=None, out_path=None):
    """Plot runoff distributions (histogram + KDE) for all hydrometric stations.

    Used to check whether the observed runoff series are normal or skewed.

    Parameters
    ----------
    runoff_dir : str or Path or None
        Directory containing the per-station runoff *.xlsx files. When None,
        ``paths.runoff_dir`` from ``config/default.yaml`` is used.
    out_path : str or Path or None
        Output PNG path (default: <project>/results/all_stations_runoff_distribution.png).
    """
    from scipy import stats as _stats

    if runoff_dir is None:
        import yaml

        config_path = _PROJECT_ROOT / "config" / "default.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        runoff_dir = _PROJECT_ROOT / config["paths"]["runoff_dir"]
    runoff_dir = Path(runoff_dir)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    files = []
    if runoff_dir.exists():
        files = sorted(runoff_dir.glob("*.xlsx"))

    print(f"Found {len(files)} runoff files")

    ncols = 4
    nrows = int(np.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), dpi=150)
    axes = np.atleast_2d(axes)

    for idx, fpath in enumerate(files):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        try:
            df = pd.read_excel(fpath, sheet_name=0)
            runoff_col = "runoff" if "runoff" in df.columns else df.columns[1]
            runoff = df[runoff_col].values.astype(float)
            runoff = runoff[~np.isnan(runoff)]

            if len(runoff) < 5:
                ax.set_title(
                    f"{fpath.stem}\n(insufficient data)", fontsize=9, color="red"
                )
                continue

            ax.hist(
                runoff,
                bins=min(30, max(10, len(runoff) // 3)),
                density=True,
                alpha=0.6,
                color="steelblue",
                edgecolor="white",
                label="Observed distribution",
            )

            xmin, xmax = runoff.min(), runoff.max()
            pad = (xmax - xmin) * 0.1
            x_grid = np.linspace(xmin - pad, xmax + pad, 200)
            kde = _stats.gaussian_kde(runoff)
            ax.plot(x_grid, kde(x_grid), linewidth=2, color="darkblue", label="KDE")

            mu, sigma = np.mean(runoff), np.std(runoff, ddof=1)
            norm_pdf = _stats.norm.pdf(x_grid, mu, sigma)
            ax.plot(
                x_grid,
                norm_pdf,
                linewidth=1.5,
                linestyle="--",
                color="red",
                alpha=0.8,
                label="Normal fit",
            )

            skewness = _stats.skew(runoff)
            if len(runoff) >= 8:
                _, sw_p = _stats.shapiro(runoff)
                sw_text = f"SW p={sw_p:.3f}"
            else:
                sw_text = "SW: n<8"

            info = (
                f"n={len(runoff)}\n"
                f"$\\mu$={mu:.1f}, $\\sigma$={sigma:.1f}\n"
                f"skew={skewness:.2f}\n"
                f"{sw_text}"
            )
            ax.text(
                0.97,
                0.95,
                info,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

            ax.set_title(fpath.stem, fontsize=10, fontweight="bold")
            ax.set_xlabel("Runoff", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
            ax.tick_params(labelsize=7)

        except Exception as e:
            ax.set_title(f"{fpath.stem}\n(read failed)", fontsize=9, color="red")
            print(f"  [WARN] {fpath.name}: {e}")

    for idx in range(len(files), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.suptitle(
        "Runoff distribution by station (histogram + KDE vs normal fit)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if out_path is None:
        out_path = _PROJECT_ROOT / "results" / "all_stations_runoff_distribution.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Saved to: {out_path}")


if __name__ == "__main__":
    plot_runoff_distribution()
