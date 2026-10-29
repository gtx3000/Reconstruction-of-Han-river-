"""Basin-boundary spatial screening module.

Filters MADA grid cells and documentary stations based on a basin shapefile.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


def read_basin(basin_path):
    """Read a basin shapefile and reproject it to WGS84.

    Parameters
    ----------
    basin_path : str
        Path to the basin shapefile.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    print(f"\n[Step 1.1] Reading basin boundary: {basin_path}")
    basin = gpd.read_file(basin_path)
    if basin.crs is None:
        basin.set_crs("EPSG:4326", inplace=True)
    elif basin.crs.to_string() != "EPSG:4326":
        basin = basin.to_crs("EPSG:4326")
    return basin


def _infer_grid_res(grid_lons, grid_lats):
    """Infer the grid resolution (degrees) from the grid coordinates.

    For a regular grid, the spacing of the sorted unique longitudes/latitudes
    equals the resolution. Uses the median to be robust against land-mask gaps
    that may produce a few larger steps.
    """
    diffs = np.concatenate(
        [np.diff(np.unique(grid_lons)), np.diff(np.unique(grid_lats))]
    )
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 1.0


def filter_mada_grids(
    mada_path, basin_gdf, grid_res=None, grid_points_path=None, buffer_deg=0.2
):
    """Select MADA grid cells within the buffered basin boundary.

    The basin boundary is buffer_deg.
    cell is kept when its center point falls inside the buffered boundary.

    Grid coordinates are preferably read from ``grid_points_path`` (a list of
    valid land grid cells with longitude/latitude columns). When not provided,
    all grid cells are generated from the NetCDF coordinate axes (including
    invalid ocean cells, which is not recommended).

    Parameters
    ----------
    mada_path : str
        Path to the MADA NetCDF file.
    basin_gdf : geopandas.GeoDataFrame
        Basin boundary.
    grid_res : float or None
        Grid resolution (degrees). When None, it is inferred automatically
        from the grid coordinates (the valid land-grid list or the NetCDF
        axes).
    grid_points_path : str or None
        Path to the valid MADA grid-cell list (longitude/latitude columns).

    Returns
    -------
    pandas.DataFrame
        DataFrame with lon, lat columns.
    """
    print("[Step 1.2] Screening MADA grid cells...")

    if grid_points_path:
        grids_df = pd.read_csv(grid_points_path)
        grids_df = grids_df.rename(columns={"longitude": "lon", "latitude": "lat"})
        grid_lons = grids_df["lon"].values
        grid_lats = grids_df["lat"].values
        print(f"  -> Grid source: valid land-grid list ({len(grids_df)} cells)")
    else:
        mada_ds = xr.open_dataset(mada_path)
        ax_lons, ax_lats = mada_ds["lon"].values, mada_ds["lat"].values
        mada_ds.close()
        pairs = [(lo, la) for lo in ax_lons for la in ax_lats]
        grid_lons = np.array([p[0] for p in pairs])
        grid_lats = np.array([p[1] for p in pairs])
        print(
            f"  [WARN] No grid list provided; falling back to full coordinate axes "
            f"({len(pairs)} cells, including invalid ocean cells)"
        )

    if grid_res is None:
        grid_res = _infer_grid_res(grid_lons, grid_lats)
        print(f"  -> Inferred grid resolution: {grid_res} degrees")

    # Buffer the basin boundary
    basin_buffered = basin_gdf.copy()
    basin_buffered["geometry"] = basin_gdf.geometry.buffer(buffer_deg)

    mada_gdf = gpd.GeoDataFrame(
        {"lon": grid_lons, "lat": grid_lats},
        geometry=gpd.points_from_xy(grid_lons, grid_lats),
        crs="EPSG:4326",
    )

    try:
        mada_selected = gpd.sjoin(
            mada_gdf, basin_buffered, how="inner", predicate="intersects"
        )
    except TypeError:
        mada_selected = gpd.sjoin(
            mada_gdf, basin_buffered, how="inner", op="intersects"
        )

    mada_selected = mada_selected.drop_duplicates(subset=["lon", "lat"]).reset_index(
        drop=True
    )
    print(f"  [OK] Selected {len(mada_selected)} MADA grid cells")
    return mada_selected[["lon", "lat"]]


def filter_stations(stations_path, search_gdf, buffer_deg=0.2):
    """Select documentary stations within the buffered search area.

    Parameters
    ----------
    stations_path : str
        Path to the documentary station metadata file (.csv / .xlsx).
    search_gdf : geopandas.GeoDataFrame
        Search-area boundary.
    buffer_deg : float
        Buffer distance (degrees) applied outward to the search-area boundary.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    print("[Step 1.3] Screening documentary stations...")
    try:
        if stations_path.endswith(".xlsx"):
            stations_df = pd.read_excel(stations_path)
        else:
            stations_df = pd.read_csv(stations_path, encoding="utf-8")
    except (UnicodeDecodeError, Exception):
        stations_df = pd.read_csv(stations_path, encoding="gbk")

    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat),
        crs="EPSG:4326",
    )

    # Buffer the search-area boundary outward by buffer_deg
    search_buffer_gdf = search_gdf.copy()
    search_buffer_gdf["geometry"] = search_gdf.geometry.buffer(buffer_deg)

    try:
        stations_selected = gpd.sjoin(
            stations_gdf, search_buffer_gdf, how="inner", predicate="intersects"
        )
    except TypeError:
        stations_selected = gpd.sjoin(
            stations_gdf, search_buffer_gdf, how="inner", op="intersects"
        )

    stations_selected = stations_selected.drop_duplicates(
        subset=["station_name"]
    ).reset_index(drop=True)
    print(f"  [OK] Selected {len(stations_selected)} documentary stations")
    return stations_selected
