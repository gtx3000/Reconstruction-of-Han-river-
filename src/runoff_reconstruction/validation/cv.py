"""Independent calibration-period cross-validation module.

Runs k-fold cross-validation on the instrumental period (years with runoff).
Proxy screening is recomputed within each fold to avoid data leakage.
Decoupled from the reconstruction pipeline.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from ..proxy.selector import select_proxies_by_correlation


def contiguous_k_fold_split(n_samples, n_splits):
    """Contiguous k-fold split.

    Splits samples into k contiguous blocks in time order, holding out one
    block as the test set each time.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield np.concatenate(
            [np.arange(0, start), np.arange(stop, n_samples)]
        ), np.arange(start, stop)
        current = stop


def random_k_fold_split(n_samples, n_splits, seed=42):
    """Random k-fold split.

    Shuffles samples and splits them into k folds, holding out one fold as
    the test set each time.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, test_idx
        current = stop


def _fold_select_proxies(
    X_tr,
    Y_tr,
    proxy_names,
    sig_level,
    corr_thresh,
    max_candidates,
    X_full=None,
):
    """Re-select proxies within a fold using only training data.

    Uses the same screening steps as the main pipeline (shared
    select_proxies_by_correlation): significance test -> MADA within-group
    collinearity removal (documentary skipped) -> "both" truncation. Falls back
    to the top-|r| ``max_candidates`` proxies when none passes significance.

    Returns
    -------
    list[int]
        Kept column indices (relative to X_tr).
    """
    kept = select_proxies_by_correlation(
        X_tr,
        Y_tr,
        proxy_names,
        sig_level,
        corr_thresh,
        max_candidates,
        verbose=False,
        X_full=X_full,
    )
    if kept:
        return kept

    # Fallback: no significant proxy, take top-|r| max_candidates
    n_cols = X_tr.shape[1]
    r_vals = np.zeros(n_cols)
    for j in range(n_cols):
        m = ~np.isnan(X_tr[:, j]) & ~np.isnan(Y_tr)
        if np.sum(m) < 5:
            continue
        r_vals[j] = stats.pearsonr(Y_tr[m], X_tr[m, j])[0]
    return list(np.argsort(-np.abs(r_vals))[:max_candidates])


def _run_one_cv_fold(
    train_idx,
    test_idx,
    X_cal,
    Y_cal,
    cal_df,
    df,
    X_cols,
    sig_level,
    corr_thresh,
    max_candidates,
):
    """Run a single CV fold (proxy screening + modeling + evaluation).

    R2a is computed on the training period; RE and CE are computed on the
    test period, with RE referenced to the calibration (training) mean.

    Returns
    -------
    dict or None
        {"R2a": float, "RE": float, "CE": float, "n_proxies": int}
        None when no valid proxy is available.
    """
    X_tr = X_cal[train_idx]
    Y_tr = Y_cal[train_idx]
    X_te = X_cal[test_idx]
    Y_te = Y_cal[test_idx]

    cal_years = cal_df["year"].values
    train_years = cal_years[train_idx]
    df_year_mask = df["year"].isin(train_years)
    X_full_train = df.loc[df_year_mask, X_cols].values

    fold_cols = _fold_select_proxies(
        X_tr,
        Y_tr,
        X_cols,
        sig_level,
        corr_thresh,
        max_candidates,
        X_full=X_full_train,
    )
    if not fold_cols:
        return None

    X_tr_sel = X_tr[:, fold_cols]
    X_te_sel = X_te[:, fold_cols]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_tr_std = scaler_X.fit_transform(X_tr_sel)
    Y_tr_std = scaler_Y.fit_transform(Y_tr.reshape(-1, 1)).ravel()
    model = BayesianRidge(max_iter=1000, tol=1e-6)
    model.fit(X_tr_std, Y_tr_std)
    pred_tr_std = model.predict(X_tr_std)
    pred_tr = scaler_Y.inverse_transform(pred_tr_std.reshape(-1, 1)).ravel()
    X_te_std = scaler_X.transform(X_te_sel)
    pred_te_std = model.predict(X_te_std)
    pred_te = scaler_Y.inverse_transform(pred_te_std.reshape(-1, 1)).ravel()

    ss_res_tr = np.sum((Y_tr - pred_tr) ** 2)
    ss_tot_tr = np.sum((Y_tr - np.mean(Y_tr)) ** 2)
    r2_tr = 1.0 - ss_res_tr / ss_tot_tr if ss_tot_tr > 0 else 0.0
    p = len(fold_cols)
    n_tr = len(train_idx)
    r2a_tr = (
        1.0 - (1.0 - r2_tr) * (n_tr - 1) / (n_tr - p - 1) if n_tr - p - 1 > 0 else r2_tr
    )

    # RE/CE computed on the test predictions (in log space).
    ss_res_te = np.sum((Y_te - pred_te) ** 2)
    y_mean_tr = np.mean(Y_tr)
    ss_re_denom = np.sum((Y_te - y_mean_tr) ** 2)
    re = 1.0 - ss_res_te / ss_re_denom if ss_re_denom > 0 else 0.0

    y_mean_te = np.mean(Y_te)
    ss_ce_denom = np.sum((Y_te - y_mean_te) ** 2)
    ce = 1.0 - ss_res_te / ss_ce_denom if ss_ce_denom > 0 else 0.0

    return {"R2a": r2a_tr, "RE": re, "CE": ce, "n_proxies": len(fold_cols)}


def run_calibration_cv(
    df,
    sig_level,
    corr_thresh,
    max_candidates,
    k_folds,
    output_dir=None,
    cv_mode="contiguous",
    n_random_repeats=100,
):
    """Independent calibration-period cross-validation.

    Runs k-fold CV on the instrumental period only. Proxy screening is
    recomputed in each fold, then a single overall model is fitted on the
    fold's training proxies. R2a is computed on training; RE and CE on test
    (RE referenced to the calibration mean).

    Parameters
    ----------
    df : pandas.DataFrame
        Full data table (full-period proxies + calibration runoff).
    sig_level : float
    corr_thresh : float
    max_candidates : int
    k_folds : int
    output_dir : str or None
        Output directory; None skips saving.
    cv_mode : str
        "contiguous" or "random" (repeated n_random_repeats times).
    n_random_repeats : int
        Number of repeats in random mode (default 100).

    Returns
    -------
    pandas.DataFrame
        CV results with fold/repeat, R2a, RE, CE columns and a mean row.
    """
    print(
        f"\n[Step 7] Running independent cross-validation ({k_folds}-fold, bayesian, mode={cv_mode})..."
    )

    cal_df = df.dropna(subset=["runoff"]).copy()
    cal_df = cal_df.reset_index(drop=True)
    Y_cal = np.log(cal_df["runoff"].values)
    X_cols = [c for c in cal_df.columns if c not in ("year", "runoff", "log_runoff")]
    X_cal = cal_df[X_cols].values

    if k_folds > len(cal_df):
        print(
            f"  [WARN] k_folds={k_folds} exceeds calibration years "
            f"{len(cal_df)}, clamped to {len(cal_df)}."
        )
        k_folds = len(cal_df)

    results = []

    if cv_mode == "random":
        total_folds = 0
        for rep in range(n_random_repeats):
            for fold_idx, (train_idx, test_idx) in enumerate(
                random_k_fold_split(len(cal_df), k_folds, seed=rep)
            ):
                res = _run_one_cv_fold(
                    train_idx,
                    test_idx,
                    X_cal,
                    Y_cal,
                    cal_df,
                    df,
                    X_cols,
                    sig_level,
                    corr_thresh,
                    max_candidates,
                )
                if res is None:
                    continue
                total_folds += 1
                results.append(
                    {
                        "repeat": rep + 1,
                        "fold": fold_idx + 1,
                        "R2a": res["R2a"],
                        "RE": res["RE"],
                        "CE": res["CE"],
                    }
                )
            if (rep + 1) % 20 == 0 or rep == n_random_repeats - 1:
                r2a_vals = [r["R2a"] for r in results]
                print(
                    f"  Random repeat {rep + 1}/{n_random_repeats} done, "
                    f"cumulative {total_folds} folds, mean R2a={np.mean(r2a_vals):.3f}"
                )

        cv_df = pd.DataFrame(results)
        mean_row = {
            "repeat": "mean",
            "fold": "",
            "R2a": round(float(cv_df["R2a"].mean()), 3),
            "RE": round(float(cv_df["RE"].mean()), 3),
            "CE": round(float(cv_df["CE"].mean()), 3),
        }
        cv_df = pd.concat([cv_df, pd.DataFrame([mean_row])], ignore_index=True)

        print(
            f"  [OK] Random CV done ({n_random_repeats}x{k_folds}={total_folds} folds): "
            f"mean R2a={mean_row['R2a']:.3f}, RE={mean_row['RE']:.3f}, CE={mean_row['CE']:.3f}"
        )

    else:
        for fold_idx, (train_idx, test_idx) in enumerate(
            contiguous_k_fold_split(len(cal_df), k_folds)
        ):
            res = _run_one_cv_fold(
                train_idx,
                test_idx,
                X_cal,
                Y_cal,
                cal_df,
                df,
                X_cols,
                sig_level,
                corr_thresh,
                max_candidates,
            )
            if res is None:
                print(f"  Fold {fold_idx + 1}: no valid proxy, skipped")
                results.append(
                    {"fold": fold_idx + 1, "R2a": np.nan, "RE": np.nan, "CE": np.nan}
                )
                continue
            results.append(
                {
                    "fold": fold_idx + 1,
                    "R2a": res["R2a"],
                    "RE": res["RE"],
                    "CE": res["CE"],
                }
            )
            print(
                f"  Fold {fold_idx + 1}: R2a={res['R2a']:.3f}, RE={res['RE']:.3f}, "
                f"CE={res['CE']:.3f} (n_proxies={res['n_proxies']})"
            )

        cv_df = pd.DataFrame(results)
        mean_row = {
            "fold": "mean",
            "R2a": round(float(cv_df["R2a"].mean()), 3),
            "RE": round(float(cv_df["RE"].mean()), 3),
            "CE": round(float(cv_df["CE"].mean()), 3),
        }
        cv_df = pd.concat([cv_df, pd.DataFrame([mean_row])], ignore_index=True)

        print(
            f"  [OK] CV done: mean R2a={mean_row['R2a']:.3f}, "
            f"RE={mean_row['RE']:.3f}, CE={mean_row['CE']:.3f}"
        )

    if output_dir:
        xlsx_path = os.path.join(output_dir, "cv_results.xlsx")
        cv_df.to_excel(xlsx_path, index=False)
        print(f"  [OK] CV results saved to: {xlsx_path}")

    return cv_df
