import pandas as pd
import numpy as np

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.abspath(os.path.join(script_dir, "../data/master_cleaned.pkl.gz"))
df = pd.read_pickle(pkl_path)
df["Ox"] = df["Ozone"] + df["NO2"]

station_order = [
    ("SVSPA", "Barrackpore", "Industrial/Suburban"),
    ("VC", "Victoria Memorial", "Urban Background"),
    ("RS", "Rabindra Sarovar", "Urban Park"),
    ("PP", "Padmapukur", "Industrial/Residential"),
    ("JD", "Jadavpur", "Mixed Urban"),
    ("DN", "Dasnagar", "Heavy Industrial"),
    ("FW", "Fort William", "Semi-Urban/Military"),
    ("GH", "Ghusuri", "Heavy Industrial"),
    ("RBU", "Rabindra Bharati Univ", "Traffic Hotspot"),
    ("BOG", "Botanical Garden", "Green Background"),
    ("BM", "Belur Math", "Religious/Mixed"),
    ("BG", "Ballygunge", "Residential"),
    ("BN", "Bidhannagar", "Commercial"),
]

midday = df[(df["Hour"] >= 11) & (df["Hour"] <= 15)]

rows = []
for code, name, zone in station_order:
    st_df = df[df["Station"] == code]
    st_mid = midday[midday["Station"] == code]
    
    btex_cols = [c for c in ["Benzene", "Toluene", "Eth-Benzene", "mp-Xylene", "o-Xylene"] if c in st_df.columns]
    btex_mean = st_df[btex_cols].sum(axis=1).mean()
    nox_mean = st_df["NOx"].mean()
    ratio_btex_nox = btex_mean / nox_mean if nox_mean > 0 else np.nan
    
    mid_no2 = st_mid["NO2"].mean()
    mid_no = st_mid["NO"].mean()
    mid_ratio = mid_no2 / mid_no if mid_no > 0 else np.nan
    
    ox_mean = st_df["Ox"].mean()
    ox_std = st_df["Ox"].std()
    
    rows.append({
        "Code": code,
        "Station": name,
        "Zone": zone,
        "BTEX": btex_mean,
        "NOx": nox_mean,
        "BTEX_NOx": ratio_btex_nox,
        "Midday_NO2_NO": mid_ratio,
        "Ox_mean": ox_mean,
        "Ox_std": ox_std
    })

res = pd.DataFrame(rows)
print(res.to_string())

print("\n--- LaTeX Table Rows ---")
for _, r in res.iterrows():
    print(f"{r['Station']} ({r['Code']}) & {r['Zone']} & {r['BTEX']:.1f} & {r['NOx']:.1f} & {r['BTEX_NOx']:.3f} & {r['Midday_NO2_NO']:.2f} & {r['Ox_mean']:.1f} $\\pm$ {r['Ox_std']:.1f} \\\\")
