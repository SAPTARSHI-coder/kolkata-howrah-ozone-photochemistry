"""
Export Raw 15-Minute Station Telemetry
Extracts all 26 station-year continuous monitoring datasets (13 stations x 2 years)
from the compressed conurbation master cache (data/master_cleaned.pkl.gz).

Outputs individual station files to: data/Cleaned_Station_Data/
Supported formats: Excel (.xlsx) and CSV (.csv)
"""

import os
import sys
import gzip
import pickle
import argparse
import pandas as pd

def export_station_datasets(output_format="excel", out_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    pkl_path = os.path.join(base_dir, "data", "master_cleaned.pkl.gz")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Master telemetry cache not found at: {pkl_path}")

    if out_dir is None:
        out_dir = os.path.join(base_dir, "data", "Cleaned_Station_Data")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading master telemetry archive from: {pkl_path} ...")
    with gzip.open(pkl_path, "rb") as f:
        df_master = pickle.load(f)

    stations = sorted(df_master["Station"].unique())
    years = sorted(df_master["Year"].unique())

    print(f"Master dataset contains {len(df_master):,} records across {len(stations)} stations and {len(years)} years.")
    print(f"Exporting 26 station-year datasets to: {out_dir}")
    print(f"Selected format: {output_format.upper()}\n")

    count = 0
    for st in stations:
        for yr in years:
            sub = df_master[(df_master["Station"] == st) & (df_master["Year"] == yr)].copy()
            if len(sub) == 0:
                continue

            # Drop internal metadata partitioning columns
            cols_to_drop = [c for c in ["Station", "Year"] if c in sub.columns]
            sub = sub.drop(columns=cols_to_drop)

            # Export based on format
            if output_format in ["excel", "both", "all"]:
                xlsx_path = os.path.join(out_dir, f"{st}_{yr}_MLL.xlsx")
                sub.to_excel(xlsx_path, index=False)
                print(f"  [+] Saved {xlsx_path} ({len(sub):,} rows)")

            if output_format in ["csv", "both", "all"]:
                csv_path = os.path.join(out_dir, f"{st}_{yr}_MLL.csv")
                sub.to_csv(csv_path, index=False)
                print(f"  [+] Saved {csv_path} ({len(sub):,} rows)")

            count += 1

    print(f"\n[SUCCESS] Successfully exported all {count} station-year datasets into {out_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Export raw 15-minute station telemetry files from master cache.")
    parser.add_argument(
        "--format",
        choices=["excel", "csv", "both"],
        default="excel",
        help="Export format: 'excel' (.xlsx), 'csv' (.csv), or 'both' (default: excel)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Custom output directory (default: data/Cleaned_Station_Data/)"
    )
    args = parser.parse_args()
    export_station_datasets(output_format=args.format, out_dir=args.outdir)

if __name__ == "__main__":
    main()
