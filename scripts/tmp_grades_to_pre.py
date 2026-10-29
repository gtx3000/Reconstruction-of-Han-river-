"""Data-preparation helper: convert drought/flood grades to inferred precipitation.

1. Read CRU precipitation and extract May-September totals.
2. Read station metadata and extract the nearest grid cell.
3. For each station, compute the representative precipitation of grades 1-5  from the 1901-2000 climatology.
4. Read the grade series and map each grade to its inferred precipitation.

Note: this is a one-off data-preparation script; its input files are not part of the reconstruction pipeline.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

# =========================
# 1. Read CRU data and extract May-September precipitation
# =========================
cru_path = r"D:\code\data\RS_precipitation\CRU\cru_ts4.08.1901.2023.pre.dat.nc"
ds = xr.open_dataset(cru_path)
pre = ds["pre"]

pre = pre.assign_coords(time=pd.DatetimeIndex(pre.time.values))
pre_mam = pre.sel(time=pre.time.dt.month.isin([5, 6, 7, 8, 9]))
pre_annual_59 = pre_mam.resample(time="YE").sum("time")

# =========================
# 2. Read station metadata and extract grid cells
# =========================
site_file = "data/docu/docu_stations_meta.csv"
sites_df = pd.read_csv(site_file)
sites = {
    row["station_name"]: (row["lon"], row["lat"]) for _, row in sites_df.iterrows()
}


def extract_nearest_grid(da, lon, lat):
    abslon = np.abs(da.lon - lon)
    abslat = np.abs(da.lat - lat)
    lon_idx = int(abslon.argmin())
    lat_idx = int(abslat.argmin())
    return da.isel(lon=lon_idx, lat=lat_idx)


# =========================
# 3. Compute representative precipitation per grade for each station
# =========================
site_grade_precip = {}

for site_name, (lon, lat) in sites.items():
    ts = extract_nearest_grid(pre_annual_59, lon, lat)
    # 1901-2000 climatology base period
    ts_base = ts.sel(time=slice("1901-01-01", "2000-12-31"))

    vals = ts_base.values
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        site_grade_precip[site_name] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        continue

    rbar = np.mean(vals)
    sigma = np.std(vals, ddof=1)

    g1 = vals[vals > rbar + 1.17 * sigma]
    g2 = vals[(vals > rbar + 0.33 * sigma) & (vals <= rbar + 1.17 * sigma)]
    g3 = vals[(vals > rbar - 0.33 * sigma) & (vals <= rbar + 0.33 * sigma)]
    g4 = vals[(vals > rbar - 1.17 * sigma) & (vals <= rbar - 0.33 * sigma)]
    g5 = vals[vals <= rbar - 1.17 * sigma]

    p1 = np.mean(g1) if len(g1) > 0 else rbar + 1.50 * sigma
    p2 = np.mean(g2) if len(g2) > 0 else rbar + 0.75 * sigma
    p3 = np.mean(g3) if len(g3) > 0 else rbar
    p4 = np.mean(g4) if len(g4) > 0 else rbar - 0.75 * sigma
    p5 = np.mean(g5) if len(g5) > 0 else rbar - 1.50 * sigma

    site_grade_precip[site_name] = {
        1: np.round(p1, 3),
        2: np.round(p2, 3),
        3: np.round(p3, 3),
        4: np.round(p4, 3),
        5: np.round(p5, 3),
    }

# =========================
# 4. Read the grade series
# =========================
grade_file = "data/docu/docu_stations_value_960-1470.csv"
final_df = pd.read_csv(grade_file)


# =========================
# 5. Map grades to inferred precipitation
# =========================
def map_precip(row):
    site = row["station_name"]
    grade = row["grade"]

    if pd.isna(grade) or grade not in [1, 2, 3, 4, 5]:
        return np.nan

    return site_grade_precip.get(site, {}).get(int(grade), np.nan)


final_df["inferred_precip"] = final_df.apply(map_precip, axis=1)

# =========================
# 6. Save the result
# =========================
out_file = "data/docu/docu_stations_value_960-1470_pre.csv"
final_df.to_csv(out_file, index=False, encoding="utf-8-sig")

print(
    f"[OK] Inferred precipitation computed and saved to:\n{os.path.abspath(out_file)}"
)
