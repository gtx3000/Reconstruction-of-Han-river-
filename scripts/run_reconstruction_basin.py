"""Basin-boundary reconstruction entry point.

Reads all configuration from config/default.yaml and, for every configured
basin, runs the full Bayesian reconstruction pipeline and then automatically
generates the standard figures and the evaluation summary table.

All paths and parameters are set in default.yaml; no code changes are required.

Usage:
    python scripts/run_reconstruction_basin.py [--config CONFIG_PATH]

    --config      Configuration file path (default config/default.yaml)

"""

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from runoff_reconstruction.pipeline.runner import process_basin
from runoff_reconstruction.visualization.figures import (
    generate_basin_figures,
    generate_station_summary,
)


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_report(out_path, station_name):
    """Generate the standard figures and summary table for a station.

    Reads the reconstructed result from ``out_path/table`` and writes figures
    to ``out_path/figure`` and the summary table to ``out_path/table``.
    """
    table_dir = os.path.join(out_path, "table")
    figure_dir = os.path.join(out_path, "figure")
    xlsx = os.path.join(table_dir, "reconstructed_runoff_final.xlsx")

    if not os.path.exists(xlsx):
        print(f"  [WARN] No reconstruction result at {xlsx}; skipping report figures.")
        return

    generate_basin_figures(xlsx, figure_dir, station_name)
    generate_station_summary(xlsx, table_dir)


def run_reconstruction(config, output_dir=None):
    """Run basin-mode reconstruction and report generation"""
    paths = config["paths"]
    recon = config["reconstruction"]
    map_extent = config.get("map_extent", {})

    mada_path = str(PROJECT_ROOT / paths["mada_nc"])
    mada_grid_points_path = str(PROJECT_ROOT / paths["mada_grid_points"])
    hydro_stations_path = str(PROJECT_ROOT / paths["hydro_stations"])
    docu_stations_meta_path = str(PROJECT_ROOT / paths["docu_stations_meta"])
    docu_value_path = str(PROJECT_ROOT / paths["docu_value"])
    base_shp_path = (
        str(PROJECT_ROOT / paths.get("base_shp", "")) if paths.get("base_shp") else None
    )

    map_lon_min = map_extent.get("lon_min", 89)
    map_lon_max = map_extent.get("lon_max", 125)
    map_lat_min = map_extent.get("lat_min", 22)
    map_lat_max = map_extent.get("lat_max", 38)

    cv_mode = recon.get("cv_mode")
    output_base = str(PROJECT_ROOT / output_dir)

    tasks = config.get("tasks", {})
    stations = tasks.get("stations", [])
    for station in stations:
        station_name = station["name"]
        basin_path = str(PROJECT_ROOT / station["basin_shp"])
        runoff_path = str(PROJECT_ROOT / station["runoff"])
        out_path = os.path.join(output_base, station_name)
        os.makedirs(out_path, exist_ok=True)

        process_basin(
            basin_path=basin_path,
            runoff_path=runoff_path,
            out_path=out_path,
            mada_path=mada_path,
            docu_stations_meta_path=docu_stations_meta_path,
            docu_value_path=docu_value_path,
            start_year=recon["start_year"],
            end_year=recon["end_year"],
            sig_level=recon["sig_level"],
            corr_thresh=recon["corr_thresh"],
            k_folds=recon["k_folds"],
            max_candidates=recon["max_candidates"],
            missing_thresh=recon["missing_thresh"],
            basin_buffer=recon["basin_buffer"],
            hydro_stations_path=hydro_stations_path,
            map_lon_min=map_lon_min,
            map_lon_max=map_lon_max,
            map_lat_min=map_lat_min,
            map_lat_max=map_lat_max,
            base_shp_path=base_shp_path,
            mada_grid_points_path=mada_grid_points_path,
            cv_mode=cv_mode,
        )

        # Generate the standard figures and summary table for this station
        _generate_report(out_path, station_name)

    print("\n" + "=" * 60)
    print(" Basin reconstruction and report generation finished (method=bayesian)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Basin-boundary reconstruction entry point"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file path (default config/default.yaml)",
    )

    args = parser.parse_args()

    config_path = args.config or str(PROJECT_ROOT / "config" / "default.yaml")
    config = load_config(config_path)

    paths = config.get("paths", {})
    output_dir = paths.get("output_dir")
    proxies = config.get("reconstruction", {}).get("max_candidates")

    print(f"Config file: {config_path}")
    print(f"Method: bayesian, proxies: {proxies}")

    run_reconstruction(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
