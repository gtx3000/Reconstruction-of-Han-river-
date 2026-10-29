"""Residual diagnostics for every fitted reconstruction segment model."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro

from ..visualization.figures import plot_residual_diagnostics, run_residual_tests


def save_model_residual_diagnostics(model_diagnostics, out_path, basin_name):
    """Save one summary row and figure per model, using matched log residuals.

    Calibration samples are sorted by year. Historical segment ranges identify
    where a model is applied, not the years used for residual diagnostics.
    """
    out_path = Path(out_path)
    table_dir = out_path / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in model_diagnostics:
        years = np.asarray(model["years"])
        fitted = np.asarray(model["fitted"], dtype=float)
        residuals = np.asarray(model["residuals"], dtype=float)
        if not (years.shape == fitted.shape == residuals.shape):
            raise ValueError("Calibration years, fitted values and residuals must align.")
        valid = np.isfinite(years) & np.isfinite(fitted) & np.isfinite(residuals)
        order = np.argsort(years[valid], kind="stable")
        years, fitted, residuals = (
            values[valid][order] for values in (years, fitted, residuals)
        )
        row = {
            "segment": model["segment"],
            "year_range": model["year_range"],
            "model_type": model["model_type"],
            "n_proxies": model["n_proxies"],
            "proxies": ",".join(model["proxies"]),
            "cal_start": int(years[0]) if len(years) else None,
            "cal_end": int(years[-1]) if len(years) else None,
            "n": len(residuals),
        }
        if len(residuals) < 3:
            print(f"  [WARN] Segment {model['segment']}: insufficient calibration samples")
        else:
            sw_stat, sw_p = shapiro(residuals)
            row.update({"SW_W": float(sw_stat), "SW_p": float(sw_p)})
            row.update(run_residual_tests(residuals, fitted_values=fitted))
            plot_residual_diagnostics(
                residuals, out_path,
                f"{basin_name}_segment_{model['segment']:02d}",
                fitted_values=fitted,
            )
            print(
                f"  [Residual diagnostics] Segment {model['segment']}: "
                f"n={len(residuals)}, SW p={sw_p:.4g}, "
                f"LB p={row['LB_p']:.4g}, BP p={row['BP_p']:.4g}"
            )
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_excel(table_dir / "residual_diagnostics.xlsx", index=False)
    print(f"  [OK] Residual diagnostics saved for {len(rows)} models")
    return diagnostics
