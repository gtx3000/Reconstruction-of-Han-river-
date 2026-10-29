"""Compare self-reconstructed Hanjiang runoff against two published tree-ring
reconstructions and quantify their correlation.

Data sources
------------
- self_runoff : `results/Hanjiang/table/reconstructed_runoff_final.xlsx`
                 column ``year`` / ``reconstructed_runoff`` (unit: 10^8 m^3)
- tree_1000   : Chen Feng (Qinling), 100 tree-ring sites, ``C1-6 RD/mm`` (mm)
- tree_8      : Chen Feng, 8 tree-ring sites, ``runoff`` (10^8 m^3)

Outputs
-------
- ``correlation_summary.xlsx`` / ``.csv`` (in ``results/table``) : Pearson /
  Spearman correlation of the raw series, with AR(1)-adjusted significance
  and a paired trend-difference test.
- Comparison line plots with 11-year LOWESS curves for visual comparison.

Methods: Santer et al. (2008), Section 4.1, Eqs. 4-6: calculate lag-1
autocorrelation from linear-trend residuals, n_eff = n(1-r1)/(1+r1), and
replace n-2 by n_eff-2 in the trend standard error. Correlation p-values use
the two-series AR(1) extension n_eff = n(1-r1_x*r1_y)/(1+r1_x*r1_y).
These p-values assume AR(1) residuals. When n_eff <= 2, adjusted p-values
are undefined (NaN). No correlation test is run on smoothed curves.

Pearson/Spearman correlation is invariant to positive rescaling. The trend
difference is scale-dependent and uses the existing plot scale factor for
tree_1000; interpret it only if that conversion has physical meaning.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr, rankdata, spearmanr, t
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELF_RUNOFF_PATH = (
    PROJECT_ROOT / "results" / "Hanjiang" / "table" / "reconstructed_runoff_final.xlsx"
)
TREE_1000_PATH = Path(r"D:\0study\长江项目\汉江800年\0汉江\陈峰26年秦岭文章\Data.xlsx")
TREE_8_PATH = Path(r"D:\0study\长江项目\汉江800年\0汉江\汉江径流重建_陈峰.xlsx")

OUTPUT_DIR = PROJECT_ROOT / "results" / "table"

LOWESS_WINDOW = 11  # years, as in the original comparison


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_plot_style() -> None:
    """Original matplotlib style (unchanged)."""
    try:
        plt.style.use(["science", "grid", "no-latex", "notebook"])
    except Exception:
        pass
    plt.rcParams["axes.grid"] = False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three runoff series."""
    self_runoff = pd.read_excel(SELF_RUNOFF_PATH)
    tree_1000 = pd.read_excel(TREE_1000_PATH)
    tree_8 = pd.read_excel(TREE_8_PATH)
    return self_runoff, tree_1000, tree_8


def align_on_year(
    self_runoff: pd.DataFrame,
    other: pd.DataFrame,
    year_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Inner-join ``other`` to ``self_runoff`` on year, dropping NaNs."""
    merged = pd.merge(
        self_runoff[["year", "reconstructed_runoff"]],
        other[[year_col, value_col]].rename(
            columns={year_col: "year", value_col: "tree_runoff"}
        ),
        on="year",
        how="inner",
    ).dropna().sort_values("year").reset_index(drop=True)
    if merged["year"].duplicated().any():
        raise ValueError("Each runoff series must have at most one value per year")
    return merged


def lag1_residual_correlation(years: np.ndarray, values: np.ndarray) -> float:
    """AR(1) estimate from linearly detrended, consecutive-year residuals.

    Santer et al. (2008), Eq. 6, use the lag-1 correlation of trend residuals.
    Missing years are not treated as adjacent observations.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(years, values, 1)
    residuals = values - (intercept + slope * years)
    adjacent = np.isclose(np.diff(years), 1)
    if adjacent.sum() < 3:
        return float("nan")
    left, right = residuals[:-1][adjacent], residuals[1:][adjacent]
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def effective_n(n: int, r1: float) -> float:
    """Santer et al. (2008), Eq. 6: n_eff = n * (1-r1)/(1+r1)."""
    if not np.isfinite(r1) or r1 <= -1 or r1 >= 1:
        return float("nan")
    return min(float(n), n * (1 - r1) / (1 + r1))


def correlation_p_with_effective_n(r: float, n_eff: float) -> float:
    """Two-sided correlation p-value using effective degrees of freedom."""
    if not np.isfinite(r) or not np.isfinite(n_eff) or n_eff <= 2:
        return float("nan")
    if abs(r) >= 1:
        return 0.0
    statistic = abs(r) * np.sqrt((n_eff - 2) / (1 - r**2))
    return float(2 * t.sf(statistic, df=n_eff - 2))


def corr_stats(years: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Correlation plus AR(1)-adjusted significance for two aligned series.

    For a *pair* of series, the AR(1) correlation effective N uses the product
    of their residual lag-1 correlations. This is an extension of Santer's
    single-series trend formula, not a formula stated in that paper.
    """
    r, p = pearsonr(x, y)
    rho, sp = spearmanr(x, y)
    r1_x = lag1_residual_correlation(years, x)
    r1_y = lag1_residual_correlation(years, y)
    ne_corr = effective_n(len(x), r1_x * r1_y)
    # Spearman's test applies the same AR(1) calculation to ranks.
    rank_r1_x = lag1_residual_correlation(years, rankdata(x))
    rank_r1_y = lag1_residual_correlation(years, rankdata(y))
    ne_rank = effective_n(len(x), rank_r1_x * rank_r1_y)
    return {
        "pearson_r": r,
        "pearson_p": p,
        "pearson_p_ar1": correlation_p_with_effective_n(r, ne_corr),
        "r2": r**2,
        "spearman_rho": rho,
        "spearman_p": sp,
        "spearman_p_ar1": correlation_p_with_effective_n(rho, ne_rank),
        "r1_self_residual": r1_x,
        "r1_tree_residual": r1_y,
        "n_eff_pearson": ne_corr,
        "n_eff_spearman": ne_rank,
    }


def paired_trend_stats(years: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Test the slope of tree minus self using Santer's AR(1)-adjusted SE.

    Fitting the paired difference accounts for shared year-to-year variation
    between reconstructions; the paper's two-independent-trends denominator
    is not appropriate when the reconstructions are correlated.
    """
    years = np.asarray(years, dtype=float)
    difference = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    slope, intercept = np.polyfit(years, difference, 1)
    residuals = difference - (intercept + slope * years)
    r1 = lag1_residual_correlation(years, difference)
    ne = effective_n(len(years), r1)
    se = float("nan")
    p = float("nan")
    if np.isfinite(ne) and ne > 2:
        se = float(np.sqrt(np.sum(residuals**2) /
                           ((ne - 2) * np.sum((years - years.mean())**2))))
        if se > 0:
            p = float(2 * norm.sf(abs(slope / se)))
        elif slope != 0:
            p = 0.0
    return {
        "trend_difference_per_year": float(slope),
        "trend_difference_se_ar1": se,
        "trend_difference_p_ar1": p,
        "r1_difference_residual": r1,
        "n_eff_trend_difference": ne,
    }


def lowess_smooth(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """11-year (fractional) LOWESS smoothing, returned in input order."""
    frac = window / len(x)
    return lowess(y, x, frac=frac, return_sorted=False)


def build_row(name: str, smoothing: str, df: pd.DataFrame, stats_: dict) -> dict:
    """Assemble one row of the correlation summary table."""
    return {
        "comparison": f"self_runoff vs {name}",
        "smoothing": smoothing,
        "start_year": int(df["year"].min()),
        "end_year": int(df["year"].max()),
        "n": int(len(df)),
        **stats_,
    }


def plot_comparison(
    cfg: dict,
    df: pd.DataFrame,
    self_sm: np.ndarray,
    tree_sm: np.ndarray,
    r: float,
    p: float,
) -> None:
    """Draw the comparison line plot, keeping the original figure style."""
    x = df["year"].values
    y_true = df["reconstructed_runoff"].values
    y_pred = df["tree_runoff"].values  # already scaled (mm -> 10^8 m^3)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=600)

    ax.plot(x, y_true, color="brown", linewidth=0.7, alpha=0.5)
    ax.plot(x, y_pred, color="slateblue", linewidth=0.7, alpha=0.5)
    (Obs,) = ax.plot(
        x, self_sm, label="self_recon", color="brown", linewidth=1.5, alpha=1
    )
    (Recon,) = ax.plot(
        x, tree_sm, label="tree_recon", color="slateblue", linewidth=1.5, alpha=1
    )

    ax.set_xlabel("Year (CE)", fontsize=12)
    ax.set_ylabel(r"Runoff  ($10^8\ \mathrm{m}^3$)", fontsize=12)
    ax.set_title(cfg["title"], fontsize=14)

    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.set_xticks(cfg["xticks"])
    ax.set_yticks(cfg["yticks"])

    handles = [Recon, Obs]
    labels = ["Recon", "Obs"]
    ax.legend(handles, labels, fontsize=9)

    if cfg.get("ylabel_right") is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(cfg["ylabel_right"], fontsize=12)
        ax2.set_ylim(0, 225)
        ax2.set_yticks(np.arange(0, 225, 50))
        ax2.tick_params(axis="y", labelsize=12)

    ax.legend(ncol=2, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    tx1, ty1 = cfg["text1"]
    ax.text(
        tx1, ty1, "r = {:.3f}\n$P_{{AR1}}$ = {:.3g}".format(r, p), fontsize=12, color="black"
    )

    fig.savefig(cfg["save_path"], bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"saved figure: {cfg['save_path']}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style()

    self_runoff, tree_1000, tree_8 = load_data()

    sources = [
        {
            "name": "tree_1000",
            "data": tree_1000,
            "year_col": "Year",
            "value_col": "C1-6 RD/mm",
            "scale": 4,  # mm -> 10^8 m^3, as in the original script
            "title": "self_recon vs tree_100_recon",
            "save_path": OUTPUT_DIR / "compare_tree_1000.png",
            "xlim": (1590, 2010),
            "xticks": np.arange(1600, 2001, 50),
            "ylim": (0, 900),
            "yticks": np.arange(0, 900, 200),
            "text1": (1600, 750),
            "ylabel_right": "Runoff depth （mm）",
        },
        {
            "name": "tree_8",
            "data": tree_8,
            "year_col": "year",
            "value_col": "runoff",
            "scale": 1,  # already in 10^8 m^3
            "title": "self_recon vs tree_8_recon",
            "save_path": OUTPUT_DIR / "compare_tree_8.png",
            "xlim": (1575, 2005),
            "xticks": np.arange(1600, 2001, 50),
            "ylim": (0, 900),
            "yticks": np.arange(0, 900, 200),
            "text1": (1585, 750),
            "ylabel_right": None,
        },
    ]

    rows = []
    for cfg in sources:
        df = align_on_year(self_runoff, cfg["data"], cfg["year_col"], cfg["value_col"])
        df["tree_runoff"] = df["tree_runoff"] * cfg["scale"]

        x = df["year"].values
        y_self = df["reconstructed_runoff"].values
        y_tree = df["tree_runoff"].values

        # raw series
        stats_raw = corr_stats(x, y_self, y_tree)
        rows.append(build_row(
            cfg["name"], "raw", df,
            {**stats_raw, **paired_trend_stats(x, y_self, y_tree)},
        ))

        # 11-year LOWESS curves are for visual comparison only.
        self_sm = lowess_smooth(x, y_self, LOWESS_WINDOW)
        tree_sm = lowess_smooth(x, y_tree, LOWESS_WINDOW)

        plot_comparison(
            cfg,
            df,
            self_sm,
            tree_sm,
            stats_raw["pearson_r"],
            stats_raw["pearson_p_ar1"],
        )

    summary = pd.DataFrame(rows)
    summary.to_excel(OUTPUT_DIR / "correlation_summary.xlsx", index=False)
    summary.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)
    print("correlation summary saved to:", OUTPUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
