"""Data loading and alignment module."""

import numpy as np
import pandas as pd
import xarray as xr


def load_and_align_data(
    mada_path,
    mada_grids,
    docu_path,
    stations_selected,
    runoff_path,
    start_year,
    end_year,
):
    """Load and align MADA, documentary and runoff data.

    Parameters
    ----------
    mada_path : str
        Path to the MADA NetCDF file.
    mada_grids : pandas.DataFrame
        MADA grid coordinates (lon, lat).
    docu_path : str
        Path to the documentary value CSV.
    stations_selected : DataFrame
        Selected documentary stations (station_name column).
    runoff_path : str
        Path to the runoff Excel file.
    start_year, end_year : int
        Reconstruction start/end years.

    Returns
    -------
    pandas.DataFrame
        Aligned annual data table.
    """
    print(f"\n[Step 2] Loading and aligning data ({start_year}-{end_year})...")
    years = np.arange(start_year, end_year + 1)
    df_all = pd.DataFrame({"year": years})

    # Load MADA (only when mada_grids is non-empty)
    if len(mada_grids) > 0:
        mada_ds = xr.open_dataset(mada_path)
        mada_years = mada_ds["time"].values.astype(int)
        for _, row in mada_grids.iterrows():
            ts = (
                mada_ds["pdsi"]
                .sel(lon=row["lon"], lat=row["lat"], method="nearest")
                .values
            )
            df_all = df_all.merge(
                pd.DataFrame(
                    {"year": mada_years, f"MADA_{row['lon']}_{row['lat']}": ts}
                ),
                on="year",
                how="left",
            )
        mada_ds.close()
        print(f"  -> Loaded {len(mada_grids)} MADA grid cells")
    else:
        print("  -> Skipped MADA loading (no MADA grids)")

    # Load documentary data (only when stations_selected is non-empty)
    if len(stations_selected) > 0:
        try:
            docu_df = pd.read_csv(docu_path, encoding="utf-8")
        except UnicodeDecodeError:
            docu_df = pd.read_csv(docu_path, encoding="gbk")

        docu_df = docu_df[
            docu_df["station_name"].isin(stations_selected["station_name"].tolist())
        ]

        # Deduplicate (year, station) pairs and average conflicting grades
        docu_df = (
            docu_df.groupby(["year", "station_name"], as_index=False)[
                "inferred_precip"
            ]
            .mean()
        )

        # Documentary proxies use the continuous precipitation inferred from
        # grades, not the discrete drought/flood grade itself.
        docu_pivot = docu_df.pivot(
            index="year", columns="station_name", values="inferred_precip"
        ).reset_index()
        docu_pivot.columns = ["year"] + [
            f"Docu_{col}" for col in docu_pivot.columns[1:]
        ]
        df_all = df_all.merge(docu_pivot, on="year", how="left")
        print(f"  -> Loaded {len(stations_selected)} documentary stations")
    else:
        print("  -> Skipped documentary loading (no documentary stations)")

    # Load runoff data
    runoff_df = pd.read_excel(runoff_path, sheet_name=0)
    df_all = df_all.merge(runoff_df, on="year", how="left")

    proxy_cols = [c for c in df_all.columns if c not in ["year", "runoff"]]
    if len(proxy_cols) == 0:
        raise ValueError("No proxy data loaded. Check the config and data files.")
    print(f"  [OK] Data alignment complete. Total proxies: {len(proxy_cols)}")
    return df_all
