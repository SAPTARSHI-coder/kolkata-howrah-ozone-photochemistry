import pandas as pd
import numpy as np

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.abspath(os.path.join(script_dir, "../data/master_cleaned.pkl.gz"))
df = pd.read_pickle(pkl_path)

stations = ["SVSPA", "VC", "RBU", "DN"]
diurnal = df[df["Station"].isin(stations)].groupby(["Station", "Hour"])[["Ozone", "NO", "NO2", "NOx"]].mean().reset_index()

for st in stations:
    st_d = diurnal[diurnal["Station"] == st]
    print(f"\n=== Station: {st} ===")
    o3_max_idx = st_d["Ozone"].idxmax()
    o3_min_idx = st_d["Ozone"].idxmin()
    no_max_idx = st_d["NO"].idxmax()
    no_min_idx = st_d["NO"].idxmin()
    no2_max_idx = st_d["NO2"].idxmax()
    
    h_o3_max, val_o3_max = int(st_d.loc[o3_max_idx, "Hour"]), st_d.loc[o3_max_idx, "Ozone"]
    h_o3_min, val_o3_min = int(st_d.loc[o3_min_idx, "Hour"]), st_d.loc[o3_min_idx, "Ozone"]
    h_no_max, val_no_max = int(st_d.loc[no_max_idx, "Hour"]), st_d.loc[no_max_idx, "NO"]
    h_no_min, val_no_min = int(st_d.loc[no_min_idx, "Hour"]), st_d.loc[no_min_idx, "NO"]
    h_no2_max, val_no2_max = int(st_d.loc[no2_max_idx, "Hour"]), st_d.loc[no2_max_idx, "NO2"]
    
    print(f"Ozone peak: hour {h_o3_max} = {val_o3_max:.2f} ug/m3, min: hour {h_o3_min} = {val_o3_min:.2f} ug/m3")
    print(f"NO peak: hour {h_no_max} = {val_no_max:.2f} ug/m3, min: hour {h_no_min} = {val_no_min:.2f} ug/m3")
    print(f"NO2 peak: hour {h_no2_max} = {val_no2_max:.2f} ug/m3")
    midday_no2 = st_d[(st_d["Hour"] >= 12) & (st_d["Hour"] <= 16)]["NO2"].mean()
    print(f"Midday NO2 (hours 12-16): {midday_no2:.2f} ug/m3")

print("\n--- Diurnal arrays for each station ---")
for st in stations:
    st_d = diurnal[diurnal["Station"] == st].sort_values("Hour")
    print(f"\nStation {st}:")
    print("O3:", [round(x, 2) for x in st_d["Ozone"].tolist()])
    print("NO:", [round(x, 2) for x in st_d["NO"].tolist()])
    print("NO2:", [round(x, 2) for x in st_d["NO2"].tolist()])
    print("NOx:", [round(x, 2) for x in st_d["NOx"].tolist()])
