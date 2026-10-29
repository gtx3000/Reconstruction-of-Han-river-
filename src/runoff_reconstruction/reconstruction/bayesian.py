"""Bayesian hierarchical reconstruction module.

Uses Bayesian Ridge regression to reconstruct runoff. Years are grouped into
disjoint segments by identical proxy availability patterns; each segment fits
a model on final_proxies directly, without cross-validation or CE
competition. After the main loop, a MADA-only fallback fills any remaining
missing years.
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from ._common import build_segments


def _fit_bayesian_segment(
    X_all,
    Y_all,
    fin_indices,
    min_train=10,
):
    """Fit a BayesianRidge on the given proxy columns and predict all years.

    Shared by the main segment loop and the MADA-only fallback so that both
    use identical training-mask, scaling, fitting and variance-matching logic.

    The log-space predictions are variance-matched against the
    calibration-period target y (log runoff): pred' = a * pred + b, with
    a = sd(y)/sd(pred) and b = mean(y) - a*mean(pred). R2/R2a are computed
    BEFORE matching (on the raw calibration predictions); sigma_eps, smearing,
    rmse and the coefficients are computed AFTER matching.

    Returns
    -------
    dict or None
        {"pred_log", "pred_log_std", "r2", "r2a", "a", "b", "valid_hist_mask",
         "N_train", "rmse_orig", "sigma_eps", "smearing", "coef", "coef_sd",
         "intercept", "alpha", "lambda", "calibration_mask",
         "calibration_fitted", "calibration_residuals"}; None when training samples are
        insufficient.
        pred_log: variance-matched historical predictions in log space.
        pred_log_std: conditional predictive sd in log space, including
        coefficient-posterior uncertainty and model noise. Standardization and
        the deterministic variance-matching transform are propagated to this sd;
        scaler, hyperparameter and matching-parameter uncertainty are excluded.
        a, b: variance-matching parameters (pred' = a * pred + b).
        sigma_eps: sd of the matched calibration residuals (log space),
        retained as a diagnostic only; it does not set the predictive band.
        smearing: Duan (1983) smearing factor mean(exp(residual)), applied on
        output to convert the log-space median to the mean in runoff units.
        rmse_orig: calibration RMSE in original runoff units (matched).
        coef / intercept: coefficients of the variance-matched model per unit
        of original proxy value, so that ln(Q) = intercept + sum(coef_j * proxy_j).
        coef_sd: posterior standard deviation of coef (same units).
        alpha / lambda: posterior noise and weight precisions.
    """
    X_hist = X_all[:, fin_indices]
    valid_years_mask = ~np.isnan(X_hist).any(axis=1) & ~np.isnan(Y_all)
    N_train = int(np.sum(valid_years_mask))
    if N_train < min_train or len(fin_indices) == 0:
        return None

    X_train, Y_train = X_hist[valid_years_mask], Y_all[valid_years_mask]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_std = scaler_X.fit_transform(X_train)
    Y_train_std = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()

    model = BayesianRidge(max_iter=1000, tol=1e-6, compute_score=True)
    model.fit(X_train_std, Y_train_std)

    # Calibration R2 / R2a
    pred_cal_std = model.predict(X_train_std)
    pred_cal = scaler_Y.inverse_transform(pred_cal_std.reshape(-1, 1)).ravel()
    ss_res_cal = np.sum((Y_train - pred_cal) ** 2)
    ss_tot_cal = np.sum((Y_train - np.mean(Y_train)) ** 2)
    r2 = 1.0 - (ss_res_cal / ss_tot_cal) if ss_tot_cal > 0 else 0.0
    p = len(fin_indices)
    r2a = (
        1.0 - (1.0 - r2) * (N_train - 1) / (N_train - p - 1)
        if N_train - p - 1 > 0
        else r2
    )

    # Full-period historical prediction (pre-variance-matching).
    valid_hist_mask = ~np.isnan(X_hist).any(axis=1)
    X_hist_std = scaler_X.transform(X_hist[valid_hist_mask])
    # BRR predictive variance = x @ posterior_cov @ x.T + 1 / alpha.
    # return_std already includes model noise; do not add residual variance.
    pred_hist_std, pred_hist_std_dev = model.predict(X_hist_std, return_std=True)
    pred_hist_log_std = pred_hist_std_dev * float(scaler_Y.scale_[0])
    pred_hist_log = scaler_Y.inverse_transform(pred_hist_std.reshape(-1, 1)).ravel()

    # Per-segment variance matching against the calibration-period target y
    # (log runoff): pred' = a * pred + b, with a = sd(y)/sd(pred) and
    # b = mean(y) - a * mean(pred). This rescales each segment's prediction to
    # the mean and variance of its calibration series, reducing artificial
    # level / spread discontinuities between segments.
    mu_m, sd_m = pred_cal.mean(), pred_cal.std(ddof=1)
    mu_t, sd_t = Y_train.mean(), Y_train.std(ddof=1)
    if sd_m > 0:
        a = sd_t / sd_m
        b = mu_t - a * mu_m
    else:
        a, b = 1.0, 0.0
    pred_hist_log = a * pred_hist_log + b
    # Treat variance matching as a fixed affine transform of the predictive
    # distribution: variance scales by a**2, while sd scales by abs(a).
    pred_hist_log_std = abs(a) * pred_hist_log_std

    # Calibration residuals of the variance-matched predictor, in log space.
    pred_cal_matched = a * pred_cal + b
    resid_matched = Y_train - pred_cal_matched

    # Retain empirical residual sd as a fit diagnostic, not as interval width.
    sigma_eps = float(resid_matched.std(ddof=1))
    # Duan (1983) smearing factor: median -> mean back-transform correction,
    # non-parametric (no normality assumption on the residuals).
    smearing = float(np.mean(np.exp(resid_matched)))

    # Segment RMSE in original runoff units, from the matched predictor.
    rmse_orig = float(
        np.sqrt(np.mean((np.exp(Y_train) - np.exp(pred_cal_matched)) ** 2))
    )

    # Coefficients of the variance-matched model, expressed per unit of
    # original proxy value (the equation predicts log runoff):
    #   ln(Q) = intercept + sum(coef_j * proxy_j)
    sy = float(scaler_Y.scale_[0])
    sx = scaler_X.scale_
    w = model.coef_
    coef_eff = a * sy * w / sx
    intercept_eff = a * (
        float(scaler_Y.mean_[0])
        + sy * float(model.intercept_)
        - sy * float(np.sum(w * scaler_X.mean_ / sx))
    ) + b
    coef_sd_eff = a * sy * np.sqrt(np.diag(model.sigma_)) / sx

    return {
        "calibration_mask": valid_years_mask,
        "calibration_fitted": pred_cal_matched,
        "calibration_residuals": resid_matched,
        "pred_log": pred_hist_log,
        "pred_log_std": pred_hist_log_std,
        "r2": r2,
        "r2a": r2a,
        "a": a,
        "b": b,
        "valid_hist_mask": valid_hist_mask,
        "N_train": N_train,
        "rmse_orig": rmse_orig,
        "sigma_eps": sigma_eps,
        "smearing": smearing,
        "coef": coef_eff,
        "coef_sd": coef_sd_eff,
        "intercept": intercept_eff,
        "alpha": float(model.alpha_),
        "lambda": float(model.lambda_),
        "n_iter": int(model.n_iter_),
    }


def bayesian_hierarchical_reconstruction(
    df,
    selected_proxies,
    final_proxies=None,
):
    """Bayesian hierarchical reconstruction.

    Years are grouped into disjoint segments by identical proxy availability
    patterns (see ``_common.build_segments``): all years sharing the same set
    of available proxies form one segment and are reconstructed by one
    BayesianRidge fitted on the final_proxies available in that segment.
    A segment with no final proxy or too few calibration years is skipped;
    after the main loop, a MADA-only fallback fills any remaining missing
    years.

    Returns
    -------
    tuple
        (recon_log, recon_log_std, year_r2, year_r2a,
         year_n_proxies, segment_stats, final_yearly_proxies, fallback_mask,
         coef_records, year_smearing, model_diagnostics)
        model_diagnostics: per-model calibration years, matched fitted values
        and residuals, including an applied MADA-only fallback.
        recon_log: reconstructed log runoff.
        recon_log_std: per-year conditional predictive sd in log space
        (coefficient-posterior uncertainty plus model noise, transformed by
        the fitted scalers and fixed variance matching); drives the 95% PI.
        year_smearing: per-year Duan (1983) smearing factor, applied on output
        to convert the log-space median to the mean in runoff units.
        fallback_mask: boolean mask of years filled by the MADA-only fallback.
        coef_records: long-format per-segment model coefficients (one row per
        segment-proxy pair); ln(Q) = intercept + sum(coef_j * proxy_j).
    """
    print("\n[Step 4] Running Bayesian hierarchical reconstruction...")

    df["log_runoff"] = np.log(df["runoff"])

    X_all = df[selected_proxies].values
    Y_all = df["log_runoff"].values
    n_total = len(Y_all)
    all_years = df["year"].values

    final_set = set(final_proxies) if final_proxies else set()
    final_positions = [j for j, p in enumerate(selected_proxies) if p in final_set]

    # Group years by the availability pattern of the FINAL proxies only:
    # proxies outside the final set never enter a model, so their NaN
    # pattern must not split segments.
    segment_masks, avail_per_segment = build_segments(X_all[:, final_positions])

    # Segment numbering: rank groups by n_proxies descending, then by the
    # group's start year ascending; groups are processed in that order.
    group_order = sorted(
        range(len(segment_masks)),
        key=lambda i: (
            -len(avail_per_segment[i]),
            int(all_years[segment_masks[i]][0]),
        ),
    )

    recon_log = np.full(n_total, np.nan)
    recon_log_std = np.full(n_total, np.nan)
    year_smearing = np.full(n_total, np.nan)
    final_yearly_proxies = [None] * n_total
    year_r2 = np.full(n_total, np.nan)
    year_r2a = np.full(n_total, np.nan)
    year_n_proxies = np.full(n_total, np.nan)
    segment_stats = []
    coef_records = []
    model_diagnostics = []

    for seg_num, seg_idx in enumerate(group_order, start=1):
        mask = segment_masks[seg_idx]
        fin_indices = [final_positions[j] for j in avail_per_segment[seg_idx]]

        seg = _fit_bayesian_segment(X_all, Y_all, fin_indices)
        if seg is None:
            continue

        seg_years = all_years[mask]
        print(
            f"  -> Segment {seg_num} ({seg_years[0]}-{seg_years[-1]}, "
            f"{int(np.sum(mask))} years): "
            f"final proxies {len(fin_indices)}, "
            f"calibration {seg['N_train']} years"
        )

        model_diagnostics.append({
            "segment": seg_num,
            "year_range": f"{seg_years[0]}-{seg_years[-1]}",
            "model_type": "segment",
            "n_proxies": len(fin_indices),
            "proxies": [selected_proxies[j] for j in fin_indices],
            "years": all_years[seg["calibration_mask"]],
            "fitted": seg["calibration_fitted"],
            "residuals": seg["calibration_residuals"],
        })

        r2, r2a, a, b = seg["r2"], seg["r2a"], seg["a"], seg["b"]
        valid_hist_mask = seg["valid_hist_mask"]

        pred_hist_full = np.full(n_total, np.nan)
        pred_hist_full[valid_hist_mask] = seg["pred_log"]
        pred_hist_std_full = np.full(n_total, np.nan)
        pred_hist_std_full[valid_hist_mask] = seg["pred_log_std"]

        # Segments are disjoint proxy-combination groups; each year is claimed
        # by exactly one segment (the isnan guard is only defensive).
        update_mask = mask & np.isnan(recon_log)
        recon_log[update_mask] = pred_hist_full[update_mask]
        recon_log_std[update_mask] = pred_hist_std_full[update_mask]
        year_smearing[update_mask] = seg["smearing"]

        current_proxies = [selected_proxies[j] for j in fin_indices]
        update_indices = np.where(update_mask)[0]
        for idx in update_indices:
            final_yearly_proxies[idx] = current_proxies

        year_r2[update_mask] = r2
        year_r2a[update_mask] = r2a
        year_n_proxies[update_mask] = len(fin_indices)

        segment_stats.append(
            {
                "segment": seg_num,
                "n_proxies": len(fin_indices),
                "n_cal_years": seg["N_train"],
                "n_recon_years": int(np.sum(mask)),
                "R2": round(r2, 3),
                "R2a": round(r2a, 3),
                "var_match_a": round(a, 4),
                "var_match_b": round(b, 4),
                "rmse": round(seg["rmse_orig"], 3),
                "sigma_eps": round(seg["sigma_eps"], 4),
                "smearing": round(seg["smearing"], 4),
                "proxies": current_proxies,
                "year_range": f"{seg_years[0]}-{seg_years[-1]}",
                "year_idx": update_indices,
            }
        )

        for j, col_idx in enumerate(fin_indices):
            coef_sd_j = float(seg["coef_sd"][j])
            coef_records.append(
                {
                    "segment": seg_num,
                    "year_range": f"{seg_years[0]}-{seg_years[-1]}",
                    "proxy": selected_proxies[col_idx],
                    "coef": round(float(seg["coef"][j]), 6),
                    "coef_post_sd": round(coef_sd_j, 6),
                    "t_ratio": (
                        round(float(seg["coef"][j]) / coef_sd_j, 3)
                        if coef_sd_j > 0
                        else np.nan
                    ),
                    "intercept": round(seg["intercept"], 6),
                    "alpha": round(seg["alpha"], 6),
                    "lambda": round(seg["lambda"], 6),
                    "n_iter": seg["n_iter"],
                }
            )

        print(
            f"    [OK] Bayesian fit done, R2a={r2a:.3f}, "
            f"variance matching a={a:.4f} b={b:.4f}"
        )

    # ---- MADA-only fallback: fill remaining missing years ----
    fallback_mask = np.zeros(n_total, dtype=bool)
    remain_mask = np.isnan(recon_log)
    if remain_mask.any():
        mada_cols = [j for j, p in enumerate(selected_proxies) if p.startswith("MADA_")]
        # Prefer final-selected MADA proxies, otherwise fall back to all MADA
        fb_cols = [j for j in mada_cols if selected_proxies[j] in final_set]
        if not fb_cols:
            fb_cols = list(mada_cols)

        if not fb_cols:
            n_remain = int(np.sum(remain_mask))
            print(
                f"\n  [WARN] {n_remain} years remain unreconstructed and no MADA "
                f"proxy is available for fallback."
            )
        else:
            print(
                f"\n  [MADA fallback] Filling the remaining "
                f"{int(np.sum(remain_mask))} missing years with a MADA-only model "
                f"({len(fb_cols)} candidate MADA proxies)..."
            )
            fb = _fit_bayesian_segment(
                X_all,
                Y_all,
                fb_cols,
            )
            if fb is None:
                print(
                    "  [WARN] MADA fallback model has insufficient training samples, skipped."
                )
            else:
                update_mask = fb["valid_hist_mask"] & remain_mask
                if not update_mask.any():
                    print(
                        "  [WARN] MADA fallback has no available proxy in the remaining years, skipped."
                    )
                else:
                    pred_fb_full = np.full(n_total, np.nan)
                    pred_fb_std_full = np.full(n_total, np.nan)
                    pred_fb_full[fb["valid_hist_mask"]] = fb["pred_log"]
                    pred_fb_std_full[fb["valid_hist_mask"]] = fb["pred_log_std"]
                    recon_log[update_mask] = pred_fb_full[update_mask]
                    recon_log_std[update_mask] = pred_fb_std_full[update_mask]
                    year_smearing[update_mask] = fb["smearing"]
                    fallback_mask[update_mask] = True

                    fb_proxies = [selected_proxies[j] for j in fb_cols]
                    update_indices = np.where(update_mask)[0]
                    for idx in update_indices:
                        final_yearly_proxies[idx] = fb_proxies

                    year_r2[update_mask] = fb["r2"]
                    year_r2a[update_mask] = fb["r2a"]
                    year_n_proxies[update_mask] = len(fb_cols)

                    fb_years = all_years[update_mask]
                    # Main-loop segment ids follow the group index and may
                    # have gaps (skipped groups), so the fallback id must not
                    # reuse len(segment_stats)+1.
                    fb_seg_id = (
                        max((s["segment"] for s in segment_stats), default=0) + 1
                    )
                    model_diagnostics.append({
                        "segment": fb_seg_id,
                        "year_range": f"{fb_years[0]}-{fb_years[-1]}",
                        "model_type": "mada_fallback",
                        "n_proxies": len(fb_cols),
                        "proxies": fb_proxies,
                        "years": all_years[fb["calibration_mask"]],
                        "fitted": fb["calibration_fitted"],
                        "residuals": fb["calibration_residuals"],
                    })
                    segment_stats.append(
                        {
                            "segment": fb_seg_id,
                            "n_proxies": len(fb_cols),
                            "n_cal_years": fb["N_train"],
                            "n_recon_years": int(np.sum(update_mask)),
                            "R2": round(fb["r2"], 3),
                            "R2a": round(fb["r2a"], 3),
                            "var_match_a": round(fb["a"], 4),
                            "var_match_b": round(fb["b"], 4),
                            "rmse": round(fb["rmse_orig"], 3),
                            "sigma_eps": round(fb["sigma_eps"], 4),
                            "smearing": round(fb["smearing"], 4),
                            "proxies": fb_proxies,
                            "year_range": f"{fb_years[0]}-{fb_years[-1]} (MADA fallback)",
                            "year_idx": update_indices,
                        }
                    )
                    for j, col_idx in enumerate(fb_cols):
                        coef_sd_j = float(fb["coef_sd"][j])
                        coef_records.append(
                            {
                                "segment": fb_seg_id,
                                "year_range": (
                                    f"{fb_years[0]}-{fb_years[-1]} (MADA fallback)"
                                ),
                                "proxy": selected_proxies[col_idx],
                                "coef": round(float(fb["coef"][j]), 6),
                                "coef_post_sd": round(coef_sd_j, 6),
                                "t_ratio": (
                                    round(float(fb["coef"][j]) / coef_sd_j, 3)
                                    if coef_sd_j > 0
                                    else np.nan
                                ),
                                "intercept": round(fb["intercept"], 6),
                                "alpha": round(fb["alpha"], 6),
                                "lambda": round(fb["lambda"], 6),
                                "n_iter": fb["n_iter"],
                            }
                        )
                    print(
                        f"    [OK] MADA fallback done, R2a={fb['r2a']:.3f}, "
                        f"variance matching a={fb['a']:.4f} b={fb['b']:.4f}, "
                        f"filled {np.sum(update_mask)} years "
                        f"({fb_years[0]}-{fb_years[-1]})"
                    )

    nan_count = np.sum(np.isnan(recon_log))
    if nan_count > 0:
        print(
            f"\n  [WARN] {nan_count} years remain unreconstructed "
            f"(all available proxies are missing in those years)."
        )

    # ---- Segment-level reconstruction statistics (log space) ----
    # Mean / standard deviation / extreme-event frequency of the reconstructed
    # log runoff in each segment. Computed in log space because the model is
    # fitted there and segment-to-segment discontinuities (intercept /
    # coefficient differences) manifest as level / spread shifts in log space.
    # Extreme events use a GLOBAL threshold (log-mean +/- 1 log-sd of the full
    # reconstruction) so that any artificial shift is visible in these
    # diagnostics (used to assess segment-transition artifacts).
    valid_full = ~np.isnan(recon_log)
    if valid_full.any():
        global_mean = float(np.nanmean(recon_log))
        global_std = float(np.nanstd(recon_log))
    else:
        global_mean = global_std = np.nan
    low_thresh = global_mean - global_std
    high_thresh = global_mean + global_std
    for seg in segment_stats:
        idx = np.asarray(seg.get("year_idx"), dtype=int)
        vals = recon_log[idx]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            seg["recon_log_mean"] = round(float(np.mean(vals)), 3)
            seg["recon_log_std"] = round(float(np.std(vals, ddof=1)), 3)
            n_extreme = int(np.sum((vals < low_thresh) | (vals > high_thresh)))
            seg["extreme_freq"] = round(n_extreme / len(vals), 4)
        else:
            seg["recon_log_mean"] = np.nan
            seg["recon_log_std"] = np.nan
            seg["extreme_freq"] = np.nan
        seg.pop("year_idx", None)

    _print_summary(segment_stats)

    return (
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
    )


def _print_summary(segment_stats):
    """Print the Bayesian reconstruction summary."""
    print(f"\n{'=' * 90}\n Bayesian hierarchical reconstruction summary\n{'=' * 90}")
    print(
        f"{'Seg':<6}{'Years':<22}{'n_prox':<8}{'n_cal':<8}"
        f"{'R2a':<10}{'a':<8}{'b':<8}{'RMSE':<10}{'sig_eps':<9}"
    )
    print("-" * 90)
    for s in segment_stats:
        print(
            f"{s['segment']:<6}{s['year_range']:<22}{s['n_proxies']:<8}"
            f"{s['n_cal_years']:<8}{s['R2a']:<10.3f}"
            f"{s.get('var_match_a', 1.0):<8.4f}{s.get('var_match_b', 0.0):<8.4f}"
            f"{s.get('rmse', np.nan):<10.3f}{s.get('sigma_eps', np.nan):<9.4f}"
        )
