"""Reconstruction result output module (Bayesian, with MADA fallback flag)."""

import os
import numpy as np
import pandas as pd
from scipy.special import erf


def finalize_and_save(
    df,
    recon_log,
    year_r2,
    year_r2a,
    year_n_proxies,
    output_dir,
    recon_log_std,
    fallback_mask=None,
    smearing=None,
):
    """Back-transform and save reconstruction results (with conditional 95% PI).

    ``reconstructed_runoff`` is the mean estimate, exp(recon_log) * smearing,
    where ``smearing`` is the Duan (1983) smearing factor that corrects the
    log-space median back-transform to the mean in runoff units.

    Bounds are exp(recon_log +/- 1.96 * recon_log_std), where the sd
    includes BRR coefficient-posterior uncertainty and model noise after
    standardization and the fixed affine variance-matching transform. The
    interval is conditional on selected proxies, fitted hyperparameters,
    scalers and matching parameters. It excludes proxy measurement and
    selection uncertainty. Smearing corrects the point estimate only; it is
    not applied to predictive quantiles. Existing *_95ci column names are
    retained for compatibility, but their values now represent a 95% PI.

    Parameters
    ----------
    df : pandas.DataFrame
    recon_log : np.ndarray
        Reconstructed values in log space.
    year_r2, year_r2a, year_n_proxies : np.ndarray
    output_dir : str
    recon_log_std : np.ndarray
        Per-year conditional predictive standard deviation in log space.
    fallback_mask : np.ndarray or None
        Boolean mask of years filled by the MADA-only fallback model.
    smearing : np.ndarray or None
        Per-year Duan (1983) smearing factor; when None, no smearing
        correction is applied (median output).
    """
    print("\n[Step 5] Saving results (BRR conditional 95% predictive interval)...")
    if fallback_mask is None:
        fallback_mask = np.zeros(len(recon_log), dtype=bool)

    if smearing is None:
        smearing = np.ones(len(recon_log))
    smearing = np.asarray(smearing, dtype=float)

    recon_runoff = np.exp(recon_log) * smearing
    recon_lower = np.exp(recon_log - 1.96 * recon_log_std)
    recon_upper = np.exp(recon_log + 1.96 * recon_log_std)

    # Credibility: probability that the true log runoff lies within
    # +/-delta_log of the log-space predictive location, using the conditional
    # predictive sd (delta_log = half the sd of observed log runoff).
    log_obs = np.log(df.loc[df["runoff"].notna(), "runoff"].values.astype(float))
    delta_log = 0.5 * float(log_obs.std(ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        credibility = erf(delta_log / (recon_log_std * np.sqrt(2.0)))
    credibility = np.where(np.isnan(recon_log_std), np.nan, credibility)

    out_dict = {
        "year": df["year"].values.astype(int),
        "observed_runoff": _fmt(df["runoff"].values.astype(float), ".1f"),
        "reconstructed_runoff": _fmt(recon_runoff, ".1f"),
        "recon_lower_95ci": _fmt(recon_lower, ".1f"),
        "recon_upper_95ci": _fmt(recon_upper, ".1f"),
        "credibility": _fmt(credibility, ".3f"),
        "R2": _fmt(year_r2, ".3f"),
        "R2a": _fmt(year_r2a, ".3f"),
    }
    out_dict["n_proxies"] = pd.array(year_n_proxies, dtype="Int64")
    out_dict["mada_fallback"] = [
        "yes" if fb else "" for fb in np.asarray(fallback_mask, dtype=bool)
    ]

    out_df = pd.DataFrame(out_dict)
    out_path = os.path.join(output_dir, "reconstructed_runoff_final.xlsx")
    out_df.to_excel(out_path, index=False)
    n_fb = int(np.sum(np.asarray(fallback_mask, dtype=bool)))
    if n_fb:
        print(f"  [WARN] {n_fb} years were filled by the MADA-only fallback model "
              f"(see the 'mada_fallback' column)")
    print(f"  [OK] Reconstruction results saved to: {out_path}")


def _fmt(arr, fmt):
    """Format a numpy array into fixed-digit strings; NaN becomes an empty string."""
    result = []
    for v in arr:
        if np.isnan(v):
            result.append("")
        else:
            result.append(format(v, fmt))
    return result
