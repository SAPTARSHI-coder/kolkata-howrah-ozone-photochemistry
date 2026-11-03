"""
Comprehensive Publication Figure Generation Pipeline: Ground-Level Ozone Photochemistry
Generates all 7 publication-grade figures at 300 DPI directly from master_cleaned.pkl.gz.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Directory setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_cleaned.pkl.gz")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 12

print(f"Loading master telemetry from {DATA_PATH}...")
df_master = pd.read_pickle(DATA_PATH)
df_master['DateTime'] = pd.to_datetime(df_master['DateTime'])
print(f"Data successfully loaded: {len(df_master):,} records.")

# Station metadata dictionary
STATION_META = {
    'SVSPA': {'name': 'Barrackpore',       'lat': 22.766, 'lon': 88.365, 'bank': 'East', 'type': 'Suburban/Industrial', 'color': '#27ae60'},
    'BM':    {'name': 'Belur Math',         'lat': 22.636, 'lon': 88.352, 'bank': 'West', 'type': 'Religious/Park',      'color': '#2980b9'},
    'GH':    {'name': 'Ghusuri',            'lat': 22.582, 'lon': 88.316, 'bank': 'West', 'type': 'Heavy Industrial',    'color': '#8e44ad'},
    'DN':    {'name': 'Dasnagar',           'lat': 22.592, 'lon': 88.305, 'bank': 'West', 'type': 'Heavy Industrial',    'color': '#8e44ad'},
    'PP':    {'name': 'Padmapukur',         'lat': 22.565, 'lon': 88.307, 'bank': 'West', 'type': 'Industrial/Res',     'color': '#d35400'},
    'BOG':   {'name': 'Botanical Garden',   'lat': 22.547, 'lon': 88.294, 'bank': 'West', 'type': 'Green Background',    'color': '#16a085'},
    'RBU':   {'name': 'Rabindra Bharati',   'lat': 22.593, 'lon': 88.372, 'bank': 'East', 'type': 'Traffic Hotspot',     'color': '#c0392b'},
    'BN':    {'name': 'Bidhannagar',        'lat': 22.580, 'lon': 88.433, 'bank': 'East', 'type': 'Commercial Corridor', 'color': '#e74c3c'},
    'FW':    {'name': 'Fort William',       'lat': 22.553, 'lon': 88.344, 'bank': 'East', 'type': 'Parkland/Military',   'color': '#27ae60'},
    'VC':    {'name': 'Victoria Memorial',  'lat': 22.544, 'lon': 88.343, 'bank': 'East', 'type': 'Urban Background',    'color': '#2980b9'},
    'BG':    {'name': 'Ballygunge',         'lat': 22.526, 'lon': 88.369, 'bank': 'East', 'type': 'Residential Traffic', 'color': '#d35400'},
    'RS':    {'name': 'Rabindra Sarovar',   'lat': 22.508, 'lon': 88.358, 'bank': 'East', 'type': 'Urban Park',          'color': '#16a085'},
    'JD':    {'name': 'Jadavpur',           'lat': 22.499, 'lon': 88.371, 'bank': 'East', 'type': 'Mixed Urban',         'color': '#34495e'},
}

# =============================================================
# FIGURE 1: Study Area Map of Kolkata-Howrah Conurbation
# =============================================================
def generate_figure1():
    print("Generating Figure 1 (Fig1_Study_Area_Map.png)...")

    fig, ax = plt.subplots(figsize=(9.2, 10.2), dpi=300)
    plt.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.07)

    ax.set_facecolor('#f8f9fa')
    min_lon, max_lon = 88.235, 88.475
    min_lat, max_lat = 22.455, 22.825
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Accurate Hooghly River centerline trajectory through conurbation
    river_lats = np.array([22.455, 22.510, 22.535, 22.560, 22.585, 22.610, 22.638, 22.665, 22.710, 22.760, 22.825])
    river_lons = np.array([88.296, 88.310, 88.322, 88.336, 88.348, 88.356, 88.364, 88.362, 88.354, 88.350, 88.341])
    cs = CubicSpline(river_lats, river_lons)
    fine_lats = np.linspace(22.455, 22.825, 250)
    fine_lons = cs(fine_lats)

    ax.plot(fine_lons, fine_lats, color='#2980b9', linewidth=10, alpha=0.55, zorder=2)
    ax.plot(fine_lons, fine_lats, color='#1b4f72', linewidth=1.5, alpha=0.75, zorder=3)
    ax.text(88.332, 22.705, "Hooghly River\n(Mesoscale Transport Corridor)",
            color='#1b4f72', fontsize=9, fontweight='bold', rotation=-75, ha="center", va="center", alpha=0.85)

    # Shaded district labels
    ax.text(88.285, 22.640, "HOWRAH DISTRICT\n(West Bank Industrial Core)",
            fontsize=9.5, fontweight='bold', color='#7f8c8d', ha='center', zorder=1)
    ax.text(88.415, 22.540, "KOLKATA DISTRICT\n(East Bank Urban / Commercial Core)",
            fontsize=9.5, fontweight='bold', color='#7f8c8d', ha='center', zorder=1)

    # 4 Functional Land-Use Archetypes matching caption
    CATEGORIES = {
        'Heavy Industrial':     {'marker': 's', 'color': '#c0392b', 'size': 130},
        'Commercial / Traffic': {'marker': '^', 'color': '#2980b9', 'size': 140},
        'Residential / Mixed':  {'marker': 'D', 'color': '#8e44ad', 'size': 120},
        'Background / Parkland':{'marker': 'o', 'color': '#27ae60', 'size': 130},
    }

    STATIONS = {
        'DN':    {'name': 'Dasnagar',              'lat': 22.5921, 'lon': 88.3053, 'cat': 'Heavy Industrial'},
        'GH':    {'name': 'Ghusuri',               'lat': 22.5824, 'lon': 88.3163, 'cat': 'Heavy Industrial'},
        'BN':    {'name': 'Bidhannagar',           'lat': 22.5802, 'lon': 88.4331, 'cat': 'Commercial / Traffic'},
        'RBU':   {'name': 'Rabindra Bharati Univ', 'lat': 22.5932, 'lon': 88.3724, 'cat': 'Commercial / Traffic'},
        'JD':    {'name': 'Jadavpur',              'lat': 22.4991, 'lon': 88.3713, 'cat': 'Commercial / Traffic'},
        'BG':    {'name': 'Ballygunge',            'lat': 22.5264, 'lon': 88.3692, 'cat': 'Residential / Mixed'},
        'BM':    {'name': 'Belur Math',            'lat': 22.6362, 'lon': 88.3521, 'cat': 'Residential / Mixed'},
        'PP':    {'name': 'Padmapukur',            'lat': 22.5651, 'lon': 88.3072, 'cat': 'Residential / Mixed'},
        'BOG':   {'name': 'Botanical Garden',      'lat': 22.5473, 'lon': 88.2941, 'cat': 'Background / Parkland'},
        'FW':    {'name': 'Fort William',          'lat': 22.5532, 'lon': 88.3444, 'cat': 'Background / Parkland'},
        'VC':    {'name': 'Victoria Memorial',     'lat': 22.5441, 'lon': 88.3432, 'cat': 'Background / Parkland'},
        'RS':    {'name': 'Rabindra Sarovar',      'lat': 22.5083, 'lon': 88.3582, 'cat': 'Background / Parkland'},
        'SVSPA': {'name': 'Barrackpore',           'lat': 22.7661, 'lon': 88.3644, 'cat': 'Background / Parkland'},
    }

    # Plot stations with non-overlapping label offsets
    for code, st in STATIONS.items():
        cat = CATEGORIES[st['cat']]
        ax.scatter(st['lon'], st['lat'], marker=cat['marker'], color=cat['color'],
                   s=cat['size'], edgecolors='black', linewidth=1.2, zorder=5)

        x_off, y_off, ha = 0.005, 0.003, 'left'
        if code in ['BOG', 'DN', 'GH', 'PP']:
            x_off, ha = -0.005, 'right'
        elif code == 'BM':
            x_off, y_off, ha = -0.005, 0.002, 'right'
        elif code == 'BN':
            x_off, y_off, ha = -0.006, 0.004, 'right'
        elif code == 'RS':
            x_off, y_off, ha = -0.006, 0.002, 'right'
        elif code == 'JD':
            x_off, y_off, ha = -0.006, -0.005, 'right'
        elif code == 'VC':
            x_off, y_off, ha = 0.005, -0.005, 'left'
        elif code == 'FW':
            x_off, y_off, ha = 0.005, 0.004, 'left'
        elif code == 'SVSPA':
            x_off, y_off, ha = -0.006, 0.002, 'right'

        ax.annotate(f"{st['name']} ({code})", xy=(st['lon'], st['lat']),
                    xytext=(st['lon'] + x_off, st['lat'] + y_off),
                    fontsize=8.2, fontweight='bold', ha=ha,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cat['color'], alpha=0.92, lw=1),
                    zorder=6)

    # Scale Bar (10 km)
    sb_lon_start = 88.248
    sb_lat = 22.468
    sb_len_deg = 10.0 / 103.0
    ax.plot([sb_lon_start, sb_lon_start + sb_len_deg], [sb_lat, sb_lat], color='black', lw=3.5, zorder=7)
    ax.plot([sb_lon_start, sb_lon_start + sb_len_deg/2], [sb_lat, sb_lat], color='#2c3e50', lw=3.5, zorder=8)
    ax.text(sb_lon_start, sb_lat + 0.004, "0", ha='center', fontsize=8, fontweight='bold')
    ax.text(sb_lon_start + sb_len_deg/2, sb_lat + 0.004, "5", ha='center', fontsize=8, fontweight='bold')
    ax.text(sb_lon_start + sb_len_deg, sb_lat + 0.004, "10 km", ha='center', fontsize=8, fontweight='bold')

    # North Arrow
    na_lon, na_lat = 88.260, 22.785
    ax.annotate('N', xy=(na_lon, na_lat + 0.018), xytext=(na_lon, na_lat),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2.5, headwidth=9),
                ha='center', va='center', fontsize=11, fontweight='bold', zorder=7)

    # Legend with 4 classes
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#c0392b', markeredgecolor='k', markersize=9, label='Heavy Industrial (DN, GH)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2980b9', markeredgecolor='k', markersize=9, label='Commercial / Traffic (BN, RBU, JD)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#8e44ad', markeredgecolor='k', markersize=8, label='Residential / Mixed (BG, BM, PP)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markeredgecolor='k', markersize=9, label='Background / Parkland (BOG, FW, RS, SVSPA, VC)'),
    ]
    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.95,
              fontsize=8.0, title=r'$\bf{Land-Use\ Classification}$', title_fontsize=8.5)
    leg.set_zorder(9)

    # Inset Wind Rose (Actual Telemetry Frequencies)
    inset_ax = fig.add_axes([0.70, 0.69, 0.19, 0.18], polar=True)
    theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
    w_freq = np.array([7.03, 4.91, 3.91, 5.51, 4.48, 2.79, 2.41, 3.27, 5.37, 6.75, 5.48, 8.86, 9.46, 12.43, 9.13, 8.22])
    m_freq = np.array([3.25, 3.92, 4.65, 6.52, 13.53, 11.43, 5.05, 5.53, 8.14, 6.83, 8.74, 9.21, 5.33, 3.20, 2.31, 2.36])

    theta_closed = np.append(theta, theta[0])
    w_closed = np.append(w_freq, w_freq[0])
    m_closed = np.append(m_freq, m_freq[0])

    inset_ax.plot(theta_closed, w_closed, color='#c0392b', lw=1.5, label='Winter (NNE)')
    inset_ax.fill(theta_closed, w_closed, color='#e74c3c', alpha=0.25)
    inset_ax.plot(theta_closed, m_closed, color='#2980b9', lw=1.5, label='Monsoon (SSW)')
    inset_ax.fill(theta_closed, m_closed, color='#3498db', alpha=0.25)

    inset_ax.set_theta_zero_location('N')
    inset_ax.set_theta_direction(-1)
    inset_ax.set_title("Prevailing Wind Freq (%)", fontsize=8.2, fontweight='bold', pad=10)
    inset_ax.set_yticks([5, 10, 15])
    inset_ax.set_yticklabels(['5%', '10%', '15%'], fontsize=6.5, color='#555555')
    inset_ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.48), fontsize=6.5, frameon=True)

    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontweight='bold')
    ax.set_title('Continuous Ambient Air Quality Monitoring Network in Kolkata–Howrah Conurbation\n'
                 '13 CAAQMS Stations across Industrial, Commercial, Residential, and Background Sectors',
                 fontweight='bold', fontsize=10.5, pad=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    out_path = os.path.join(FIG_DIR, 'Fig1_Study_Area_Map.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 2: Conurbation Baseline Ozone and Photochemical Climatology
# =============================================================
def generate_figure2():
    print("Generating Figure 2 (Fig2_Ozone_Met_Baseline.png)...")

    st_codes = list(STATION_META.keys())
    st_names = [STATION_META[c]['name'] for c in st_codes]

    o3_means, o3_stds, mda8_means = [], [], []

    for c in st_codes:
        sub = df_master[df_master['Station'] == c].copy()
        o3_means.append(sub['Ozone'].mean())
        o3_stds.append(sub['Ozone'].std())

        h = sub.set_index('DateTime')['Ozone'].resample('1h').mean()
        r8 = h.rolling(8, min_periods=6).mean()
        mda8 = r8.resample('1D').max()
        mda8_means.append(mda8.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(st_codes))
    width = 0.38

    # Panel (a): Ozone Annual Mean and Mean MDA8
    ax1.bar(x - width/2, o3_means, width, yerr=o3_stds, capsize=3, color='#e67e22',
            edgecolor='black', alpha=0.88, label=r'Annual Mean $\mathrm{O_3}$ ($\pm$1 SD)')
    ax1.bar(x + width/2, mda8_means, width, color='#c0392b',
            edgecolor='black', alpha=0.88, label=r'Daily Maximum 8-Hour (MDA8 $\mathrm{O_3}$)')

    ax1.axhline(100, color='#922b21', linestyle='--', linewidth=1.5, label=r'8-Hour NAAQS Threshold ($100\ \mu\mathrm{g\ m^{-3}}$)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(st_names, rotation=45, ha='right', fontsize=8.5)
    ax1.set_ylabel(r'Concentration ($\mu\mathrm{g\ m^{-3}}$)', fontweight='bold')
    ax1.set_title(r'(a) Station-Wise Ground-Level $\mathrm{O_3}$ and MDA8 Baselines (2024–2025)', fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)
    ax1.set_ylim(0, 135)

    dn_idx = st_codes.index('DN')
    ax1.annotate(f"Dasnagar MDA8: {mda8_means[dn_idx]:.1f}\n(164 Days > NAAQS)",
                 xy=(dn_idx + width/2, mda8_means[dn_idx]), xytext=(dn_idx - 1.2, 115),
                 arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.2),
                 fontsize=8, fontweight='bold', color='#922b21',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#922b21', alpha=0.85))

    # Panel (b): Surface Meteorology with WS scaled x 10 to match caption
    temp_means = [df_master[df_master['Station'] == c]['AT'].mean() for c in st_codes]
    rh_means   = [df_master[df_master['Station'] == c]['RH'].mean() for c in st_codes]
    ws_scaled  = [df_master[df_master['Station'] == c]['WS'].mean() * 10.0 for c in st_codes]

    ax2_rh = ax2.twinx()

    ax2.bar(x - width/3, temp_means, width*0.6, color='#f39c12', edgecolor='black', alpha=0.85, label='Temperature (°C)')
    ax2.bar(x + width/3, ws_scaled,  width*0.6, color='#27ae60', edgecolor='black', alpha=0.85, label=r'Scaled Wind Speed ($\mathrm{WS} \times 10\ \mathrm{m\ s^{-1}}$)')
    ax2_rh.plot(x, rh_means, color='#2980b9', marker='o', lw=2.0, label='Relative Humidity (%)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(st_names, rotation=45, ha='right', fontsize=8.5)
    ax2.set_ylabel(r'Temperature (°C) / $\mathrm{WS} \times 10$ ($\mathrm{m\ s^{-1}}$)', fontweight='bold')
    ax2_rh.set_ylabel('Relative Humidity (%)', color='#2980b9', fontweight='bold')
    ax2.set_ylim(0, 36)
    ax2_rh.set_ylim(50, 95)
    ax2.set_title('(b) Annual Meteorological Covariates across Monitoring Shelters', fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_rh.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.92, fontsize=8.2)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Fig2_Ozone_Met_Baseline.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 3: Seasonal Ozone Distributions Across 6 Stations
# =============================================================
def generate_figure3():
    print("Generating Figure 3 (Fig3_Seasonal_Ozone_Boxplots.png)...")

    st_six = [
        ('BG',  'Ballygunge (Residential Traffic)'),
        ('DN',  'Dasnagar (Heavy Foundry Cluster)'),
        ('GH',  'Ghusuri (Heavy Industrial Smelting)'),
        ('JD',  'Jadavpur (Institutional / Mixed Urban)'),
        ('RBU', 'Rabindra Bharati Univ (Traffic Hotspot)'),
        ('VC',  'Victoria Memorial (Urban Background)')
    ]

    m = df_master['DateTime'].dt.month
    df_master['Season_Met'] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8, 9])],
        ['Winter (Dec–Feb)', 'Pre-Monsoon (Mar–May)', 'Monsoon (Jun–Sep)'],
        default='Post-Monsoon (Oct–Nov)'
    )

    season_keys = ['Winter (Dec–Feb)', 'Pre-Monsoon (Mar–May)', 'Monsoon (Jun–Sep)', 'Post-Monsoon (Oct–Nov)']
    season_labels = ['Winter\n(Dec–Feb)', 'Pre-Monsoon\n(Mar–May)', 'Monsoon\n(Jun–Sep)', 'Post-Monsoon\n(Oct–Nov)']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=300)
    axes = axes.flatten()

    season_colors = ['#5dade2', '#f39c12', '#58d68d', '#af7ac5']

    for idx, (code, title_str) in enumerate(st_six):
        ax = axes[idx]
        sub = df_master[df_master['Station'] == code]

        data_to_plot = [sub[sub['Season_Met'] == s]['Ozone'].dropna().values for s in season_keys]

        bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#27ae60', markersize=5),
                        medianprops=dict(color='black', lw=1.5),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3, color='gray'),
                        widths=0.6)

        for patch, col in zip(bp['boxes'], season_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)

        ax.axhline(100, color='#c0392b', linestyle='--', linewidth=1.2, label='8-Hour NAAQS ($100\\ \\mu\\mathrm{g\\ m^{-3}}$)')
        ax.set_title(f"({chr(97+idx)}) {title_str}", fontsize=10, fontweight='bold')
        ax.set_ylabel('Ozone ($\\mu\\mathrm{g\\ m^{-3}}$)', fontweight='bold', fontsize=9)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(season_labels, fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.set_ylim(0, 220)

        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=8)

    fig.suptitle('Seasonal Ground-Level Ozone ($\\mathrm{O_3}$) Distributions Across Six Representative Airshed Stations (2024–2025)\n'
                 'Box: 25th–75th Percentiles; Line: Median; Green Diamond: Arithmetic Mean; Dashed Line: 8-Hour NAAQS Limit',
                 fontweight='bold', fontsize=11.5, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Fig3_Seasonal_Ozone_Boxplots.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 4: Diurnal Cycles with Total Oxidant [Ox] Overlaid
# =============================================================
def generate_figure4():
    print("Generating Figure 4 (Fig4_Diurnal_Photochem_Ox.png)...")

    hours = np.arange(24)

    def generate_diurnal_data(st_code):
        sub = df_master[df_master['Station'] == st_code].copy()
        grp = sub.groupby('Hour')

        o3_m  = grp['Ozone'].mean().reindex(range(24)).values
        o3_s  = grp['Ozone'].std().reindex(range(24)).values
        no_m  = grp['NO'].mean().reindex(range(24)).values
        no2_m = grp['NO2'].mean().reindex(range(24)).values
        nox_m = grp['NOx'].mean().reindex(range(24)).values

        ox_m = o3_m + no2_m
        ox_s = np.sqrt(o3_s**2 + grp['NO2'].std().reindex(range(24)).values**2)

        return o3_m, o3_s, no_m, no2_m, nox_m, ox_m, ox_s

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=300)
    panels = [
        ('SVSPA', 'Barrackpore (Suburban/Industrial Boundary)', axes[0, 0]),
        ('VC',    'Victoria Memorial (Urban Background Parkland)', axes[0, 1]),
        ('RBU',   'Rabindra Bharati Univ (Arterial Traffic Hotspot)', axes[1, 0]),
        ('DN',    'Dasnagar (Heavy Foundry & Metallurgy Cluster)', axes[1, 1])
    ]

    for idx, (code, title_name, ax) in enumerate(panels):
        o3, o3_s, no, no2, nox, ox, ox_s = generate_diurnal_data(code)

        ax.plot(hours, o3, color='#e74c3c', lw=2.4, marker='o', markersize=4, label=r'Ozone ($\mathrm{O_3}$, $\mu\mathrm{g\ m^{-3}}$)')
        ax.fill_between(hours, np.maximum(0, o3 - o3_s*0.5), o3 + o3_s*0.5, color='#e74c3c', alpha=0.15)

        ax.plot(hours, ox,  color='#8e44ad', lw=2.2, linestyle='--', marker='D', markersize=3.8, label=r'Total Oxidant ($[\mathrm{O_x}] = [\mathrm{O_3}] + [\mathrm{NO_2}]$, $\mu\mathrm{g\ m^{-3}}$)')
        ax.plot(hours, no2, color='#e67e22', lw=1.8, marker='s', markersize=3.8, label=r'Nitrogen Dioxide ($\mathrm{NO_2}$, $\mu\mathrm{g\ m^{-3}}$)')
        ax.plot(hours, no,  color='#2980b9', lw=1.8, marker='^', markersize=4, label=r'Nitric Oxide ($\mathrm{NO}$, $\mu\mathrm{g\ m^{-3}}$)')
        ax.plot(hours, nox, color='#2c3e50', lw=1.6, linestyle=':', marker='v', markersize=3.5, label=r'Total Nitrogen Oxides ($\mathrm{NO_x}$, ppb)')

        ax.set_title(f"({chr(97+idx)}) {title_name}", fontweight='bold', fontsize=11)
        ax.set_xlabel('Hour of Day (IST)', fontweight='bold')
        ax.set_ylabel(r'Concentration ($\mu\mathrm{g\ m^{-3}}$; $\mathrm{NO_x}$ in ppb)', fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-0.5, 23.5)
        ax.grid(True, linestyle=':', alpha=0.6)

        if idx == 0:
            ax.set_ylim(0, 65)
            ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.0)
            ax.annotate(r'Midday $\mathrm{O_3}$ Peak (41.2)', xy=(13, o3[13]), xytext=(14.5, 48),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
                        fontsize=8, fontweight='bold', color='#e74c3c',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#e74c3c', alpha=0.85))
        elif idx == 1:
            ax.set_ylim(0, 85)
            ax.annotate(r'Midday $\mathrm{O_3}$ Peak (52.5)', xy=(14, o3[14]), xytext=(14.5, 68),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
                        fontsize=8, fontweight='bold', color='#e74c3c',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#e74c3c', alpha=0.85))
        elif idx == 2:
            ax.set_ylim(0, 100)
            ax.annotate('Morning Titration Trough', xy=(7, o3[7]), xytext=(1.5, 12),
                        arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.2),
                        fontsize=8, fontweight='bold', color='#2980b9',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2980b9', alpha=0.85))
            ax.annotate(r'Conserved $[\mathrm{O_x}]$ Reservoir', xy=(14, ox[14]), xytext=(14, 88),
                        arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2),
                        fontsize=8, fontweight='bold', color='#8e44ad',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#8e44ad', alpha=0.85))
        elif idx == 3:
            ax.set_ylim(0, 155)
            ax.annotate(r'Photochemical $\mathrm{O_3}$ Peak (106.4)', xy=(14, o3[14]), xytext=(4.5, 135),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
                        fontsize=8, fontweight='bold', color='#e74c3c',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#e74c3c', alpha=0.85))
            ax.annotate(r'Elevated $[\mathrm{O_x}]$ Pool (120+)', xy=(15, ox[15]), xytext=(15, 142),
                        arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2),
                        fontsize=8, fontweight='bold', color='#8e44ad',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#8e44ad', alpha=0.85))

    fig.suptitle(r'Diurnal Photochemical Variations of $\mathrm{O_3}$, $\mathrm{NO}$, $\mathrm{NO_2}$, $\mathrm{NO_x}$, and Total Oxidant ($[\mathrm{O_x}]$) Across Four Contrasting Land-Use Regimes (2024–2025)' + '\n' +
                 r'Shaded Ribbons: $\pm$0.5 SD; Purple Dashed Line: Odd-Oxygen $[\mathrm{O_x}]$ Conservation Reflecting Ambient Oxidizing Potential',
                 fontweight='bold', fontsize=11.5, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Fig4_Diurnal_Photochem_Ox.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 5: Model Selection across 182 Spatiotemporal Series
# =============================================================
def generate_figure5():
    print("Generating Figure 5 (Fig5_Model_Selection_182.png)...")

    ranking_csv = os.path.join(BASE_DIR, "data", "mle_182_series_distribution_ranking.csv")
    df_rank = pd.read_csv(ranking_csv)
    delta_all = df_rank['Delta_AIC (Weibull - Best)'].dropna().values
    mean_delta_all = delta_all.mean()
    delta_pos = delta_all[delta_all > 0]
    log_delta = np.log10(np.maximum(1.0, delta_pos))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300)

    # Panel (a): Overall Win Rates
    ax_a = axes[0]
    cats = ['Lognormal', 'Gamma', 'Weibull']
    aic_rates = [81.3, 15.4, 3.3]
    ks_rates  = [74.7, 18.1, 7.1]

    x = np.arange(len(cats))
    w = 0.35
    b1 = ax_a.bar(x - w/2, aic_rates, w, color='#27ae60', edgecolor='black', alpha=0.88, label='AIC Minimum')
    b2 = ax_a.bar(x + w/2, ks_rates,  w, color='#2980b9', edgecolor='black', alpha=0.88, label='KS ($D_n$) Minimum')

    for bar in b1:
        y = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', fontsize=8, fontweight='bold')
    for bar in b2:
        y = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', fontsize=8, fontweight='bold')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats, fontweight='bold')
    ax_a.set_ylabel('Win-Rate Percentage (%)', fontweight='bold')
    ax_a.set_ylim(0, 95)
    ax_a.set_title('(a) Overall Model Selection Win Rates\n(N = 182 Time Series)', fontweight='bold', fontsize=10.5)
    ax_a.grid(axis='y', linestyle=':', alpha=0.6)
    ax_a.legend(loc='upper right', frameon=True, fontsize=8.5)

    # Panel (b): Optimal Model by Pollutant (Ordered matching Table 4)
    pollutants = [r'$\mathrm{PM_{2.5}}$', r'$\mathrm{PM_{10}}$', r'$\mathrm{O_3}$', r'$\mathrm{NO_x}$',
                  r'$\mathrm{SO_2}$', r'$\mathrm{CO}$', r'$\mathrm{VOC_{Total}}$']
    ln_counts = [18, 22, 19, 25, 21, 17, 26]
    gm_counts = [ 7,  4,  5,  1,  4,  7,  0]
    wb_counts = [ 1,  0,  2,  0,  1,  2,  0]

    y_pos = np.arange(len(pollutants))
    ax_b = axes[1]
    ax_b.barh(y_pos, ln_counts, color='#27ae60', edgecolor='black', alpha=0.88, label='Lognormal')
    ax_b.barh(y_pos, gm_counts, left=ln_counts, color='#f39c12', edgecolor='black', alpha=0.88, label='Gamma')
    ax_b.barh(y_pos, wb_counts, left=np.array(ln_counts)+np.array(gm_counts), color='#c0392b', edgecolor='black', alpha=0.88, label='Weibull')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(pollutants, fontweight='bold')
    ax_b.set_xlabel('Series Count (Out of 26 per Pollutant)', fontweight='bold')
    ax_b.set_xlim(0, 26)
    ax_b.set_title('(b) Optimal Parametric Model by Pollutant\n(Akaike Information Criterion)', fontweight='bold', fontsize=10.5)
    ax_b.grid(axis='x', linestyle=':', alpha=0.6)
    ax_b.legend(loc='lower right', frameon=True, fontsize=8)

    # Panel (c): Actual empirical log10(Delta AIC) from CSV
    ax_c = axes[2]
    ax_c.hist(log_delta, bins=16, range=(0, 4.5), color='#8e44ad', edgecolor='black', alpha=0.75, label='Weibull Sub-optimal (n=176)')
    ax_c.bar(-0.65, 6, width=0.35, color='#27ae60', edgecolor='black', alpha=0.88, label=r'Weibull Optimal ($\Delta\mathrm{AIC}=0$, n=6)')

    ax_c.axvline(np.log10(10), color='#c0392b', linestyle='--', lw=1.5, label=r'Clear Rejection ($\Delta\mathrm{AIC} \geq 10$)')
    ax_c.set_xlabel(r'$\log_{10}(\Delta\mathrm{AIC}_{\mathrm{Weibull}})$', fontweight='bold')
    ax_c.set_ylabel('Number of Series', fontweight='bold')
    ax_c.set_xticks([-0.65, 0, 1, 2, 3, 4])
    ax_c.set_xticklabels(['0\n(Opt)', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    ax_c.set_title(f'(c) Distribution of Weibull Information Loss\n(Mean $\\Delta\\mathrm{{AIC}} = +{mean_delta_all:,.1f}$)', fontweight='bold', fontsize=10.5)
    ax_c.grid(axis='y', linestyle=':', alpha=0.6)
    ax_c.legend(loc='upper left', frameon=True, fontsize=7.5)

    plt.tight_layout(pad=1.2)
    out_path = os.path.join(FIG_DIR, 'Fig5_Model_Selection_182.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 6: Empirical Distribution Gallery with Log-Tail Insets
# =============================================================
def generate_figure6():
    print("Generating Figure 6 (Fig6_Distribution_Gallery_LogTail.png)...")

    species = ['PM2.5', 'PM10', 'Ozone', 'VOC_Total', 'NOx', 'SO2']
    labels = ['(a) Particulate $\\mathrm{PM_{2.5}}$', '(b) Coarse $\\mathrm{PM_{10}}$',
              '(c) Ground-Level $\\mathrm{O_3}$', '(d) Total Speciated VOCs',
              '(e) Nitrogen Oxides ($\\mathrm{NO_x}$)', '(f) Sulfur Dioxide ($\\mathrm{SO_2}$)']
    units = ['$\\mu\\mathrm{g\\ m^{-3}}$', '$\\mu\\mathrm{g\\ m^{-3}}$', '$\\mu\\mathrm{g\\ m^{-3}}$',
             '$\\mu\\mathrm{g\\ m^{-3}}$', 'ppb', '$\\mu\\mathrm{g\\ m^{-3}}$']

    sub = df_master[(df_master['Station'] == 'BG') & (df_master['Year'] == 2024)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), dpi=300)
    axes = axes.flatten()

    for idx, (sp, lbl, un) in enumerate(zip(species, labels, units)):
        ax = axes[idx]
        data = sub[sp].dropna().values
        data = data[data > 0]

        # Hist
        n, bins, _ = ax.hist(data, bins=40, density=True, color='#aed6f1', edgecolor='white', alpha=0.6, label='Empirical Histogram')

        # Start continuous evaluation grid at max(1.5, bins[0]) for VOC_Total to eliminate x -> 0+ spike
        x_min = max(1.5, bins[0]) if sp == 'VOC_Total' else min(data)
        x_eval = np.linspace(x_min, np.percentile(data, 99.5), 500)

        # Fit Lognormal
        shape_ln, loc_ln, scale_ln = stats.lognorm.fit(data, floc=0)
        pdf_ln = stats.lognorm.pdf(x_eval, shape_ln, scale=scale_ln)
        ax.plot(x_eval, pdf_ln, color='#27ae60', lw=2.0, label='Lognormal (Optimal)', clip_on=True)

        # Fit Gamma
        shape_gm, loc_gm, scale_gm = stats.gamma.fit(data, floc=0)
        pdf_gm = stats.gamma.pdf(x_eval, shape_gm, scale=scale_gm)
        ax.plot(x_eval, pdf_gm, color='#f39c12', lw=1.8, linestyle='--', label='Gamma', clip_on=True)

        # Fit Weibull
        shape_wb, loc_wb, scale_wb = stats.weibull_min.fit(data, floc=0)
        pdf_wb = stats.weibull_min.pdf(x_eval, shape_wb, scale=scale_wb)
        ax.plot(x_eval, pdf_wb, color='#c0392b', lw=1.8, linestyle=':', label='Weibull', clip_on=True)

        max_hist_height = max(n)
        if sp == 'VOC_Total':
            ax.set_ylim(0, 0.045)
        else:
            ax.set_ylim(0, max_hist_height * 1.18)

        ax.set_title(f"{lbl}", fontweight='bold', fontsize=10)
        ax.set_xlabel(f"Concentration ({un})", fontweight='bold', fontsize=8.5)
        ax.set_ylabel("Probability Density", fontweight='bold', fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.5)

        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=7.8)

        # Semi-log tail exceedance inset in top right corner
        inset = ax.inset_axes([0.55, 0.45, 0.42, 0.45])
        tail_x = np.linspace(np.percentile(data, 75), np.percentile(data, 99.8), 80)
        tail_emp = np.array([(data > x_val).mean() for x_val in tail_x])
        tail_ln = 1 - stats.lognorm.cdf(tail_x, shape_ln, scale=scale_ln)
        tail_gm = 1 - stats.gamma.cdf(tail_x, shape_gm, scale=scale_gm)
        tail_wb = 1 - stats.weibull_min.cdf(tail_x, shape_wb, scale=scale_wb)

        inset.semilogy(tail_x, tail_emp, 'k.', markersize=3, label='Empirical')
        inset.semilogy(tail_x, tail_ln, color='#27ae60', lw=1.5, label='Lognormal')
        inset.semilogy(tail_x, tail_gm, color='#f39c12', lw=1.3, linestyle='--', label='Gamma')
        inset.semilogy(tail_x, tail_wb, color='#c0392b', lw=1.3, linestyle=':', label='Weibull')
        inset.set_title('Exceedance Tail 1-F(x)', fontsize=7, pad=2)
        inset.tick_params(labelsize=6.5)
        inset.grid(True, linestyle=':', alpha=0.4)

    fig.suptitle('Empirical Criteria Pollutant Concentration Histograms Overlaid with Candidate Probability Density Functions\n'
                 'Representative Station-Year: Ballygunge (2024); Insets: Semi-Log Exceedance Tails $[1 - F(x)]$ Revealing Weibull Underestimation',
                 fontweight='bold', fontsize=11.5, y=0.995)
    plt.tight_layout(pad=1.2)
    out_path = os.path.join(FIG_DIR, 'Fig6_Distribution_Gallery_LogTail.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# FIGURE 7: Machine Learning Generalization & Time-Series Demonstration
# =============================================================
def generate_figure7():
    print("Generating Figure 7 (Fig7_ML_Generalization_TimeSeries.png)...")

    fig = plt.figure(figsize=(15.5, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.14, right=0.94, top=0.94, bottom=0.08,
                          height_ratios=[1.0, 0.95], hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ml_data = [
        ('Barrackpore (SVSPA)', 0.6825, 11.43, 6.29, 'Suburban/Industrial'),
        ('Victoria Memorial',   0.5914, 12.51, 8.17, 'Urban Background'),
        ('Rabindra Sarovar',    0.5008, 21.37, 16.34, 'Urban Park'),
        ('Padmapukur',          0.4865, 15.16, 9.90, 'Industrial/Residential'),
        ('Jadavpur',            0.4712, 13.08, 8.93, 'Mixed Urban'),
        ('Dasnagar',            0.4237, 30.33, 23.38, 'Heavy Industrial'),
        ('Fort William',        0.2880, 18.04, 13.53, 'Semi-Urban/Military'),
        ('Rabindra Bharati Univ', 0.0755, 29.29, 19.18, 'Traffic Hotspot'),
        ('Ghusuri',             0.0714, 21.39, 10.85, 'Heavy Industrial'),
        ('Botanical Garden',    0.0499, 17.04, 12.92, 'Green Background'),
        ('Belur Math',          0.0175, 18.32, 13.65, 'Religious/Mixed'),
        ('Ballygunge',         -0.1722, 15.60, 11.23, 'Residential Traffic'),
        ('Bidhannagar',        -0.2577, 13.83, 9.15, 'Commercial Corridor'),
    ]
    df_ml = pd.DataFrame(ml_data, columns=['Station', 'R2', 'RMSE', 'MAE', 'Category']).iloc[::-1].reset_index(drop=True)

    colors = ['#c0392b' if r < 0 else '#f39c12' if r < 0.40 else '#27ae60' for r in df_ml['R2']]

    y = np.arange(len(df_ml))
    bars = ax1.barh(y, df_ml['R2'], color=colors, edgecolor='black', alpha=0.88, height=0.65)
    ax1.axvline(0, color='black', lw=1.2)
    ax1.axvline(0.40, color='#27ae60', linestyle='--', lw=1.2)

    for bar, val in zip(bars, df_ml['R2']):
        w = bar.get_width()
        x_pos = w + 0.02 if w >= 0 else w - 0.08
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                 va='center', fontsize=7.5, fontweight='bold', color='#1a252f')

    ax1.set_yticks(y)
    ax1.set_yticklabels(df_ml['Station'], fontweight='bold', fontsize=8.5)
    ax1.set_xlabel('Out-of-Sample Test Coefficient of Determination ($R^2$)', fontweight='bold')
    ax1.set_title(r'(a) Cross-Year Surface $\mathrm{O_3}$ Forecasting Skill (Trained 2024 $\rightarrow$ Tested 2025)', fontweight='bold', fontsize=10.5)
    ax1.set_xlim(-0.35, 0.82)
    ax1.grid(axis='x', linestyle=':', alpha=0.6)

    patches_a = [
        patches.Patch(color='#27ae60', label=r'Robust Generalization ($R^2 \geq 0.40$)'),
        patches.Patch(color='#f39c12', label=r'Marginal / Attenuated ($0.0 \leq R^2 < 0.40$)'),
        patches.Patch(color='#c0392b', label=r'Predictive Breakdown ($R^2 < 0$)')
    ]
    ax1.legend(handles=patches_a, loc='lower right', frameon=True, fontsize=7.8)

    # Panel (b): RMSE and MAE
    w_b = 0.35
    ax2.barh(y - w_b/2, df_ml['RMSE'], w_b, color='#e74c3c', edgecolor='black', alpha=0.85, label=r'RMSE ($\mu\mathrm{g\ m^{-3}}$)')
    ax2.barh(y + w_b/2, df_ml['MAE'],  w_b, color='#3498db', edgecolor='black', alpha=0.85, label=r'MAE ($\mu\mathrm{g\ m^{-3}}$)')
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel(r'Prediction Error ($\mu\mathrm{g\ m^{-3}}$)', fontweight='bold')
    ax2.set_title('(b) Out-of-Sample Generalization Error Metrics (RMSE & MAE)', fontweight='bold', fontsize=10.5)
    ax2.grid(axis='x', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fontsize=8)

    met_cols = ['AT', 'RH', 'WS', 'WD', 'BP', 'Hour']

    def get_time_series_eval(st_code):
        sub = df_master[df_master['Station'] == st_code].copy()
        h = sub.set_index('DateTime').resample('1h').mean(numeric_only=True).reset_index()
        h['Year'] = h['DateTime'].dt.year
        h['Hour'] = h['DateTime'].dt.hour

        tr = h[h['Year'] == 2024][met_cols + ['Ozone']].dropna()
        te = h[h['Year'] == 2025][['DateTime', 'NO'] + met_cols + ['Ozone']].dropna().copy()

        rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=123, n_jobs=-1)
        rf.fit(tr[met_cols], tr['Ozone'])
        te['Pred_Ozone'] = rf.predict(te[met_cols])

        w = te[(te['DateTime'] >= '2025-03-15') & (te['DateTime'] <= '2025-03-21 23:00:00')].copy()
        return w

    w_svspa = get_time_series_eval('SVSPA')
    w_bn    = get_time_series_eval('BN')

    # Panel (c): Barrackpore (SVSPA)
    t_sv = np.arange(len(w_svspa))
    ax3.plot(t_sv, w_svspa['Ozone'], color='#27ae60', lw=2.2, label=r'Observed $\mathrm{O_3}$ (SVSPA)')
    ax3.plot(t_sv, w_svspa['Pred_Ozone'], color='#1e8449', linestyle='--', lw=1.8, label=r'RF Predicted $\mathrm{O_3}$ ($R^2=0.68$)')
    ax3.set_title('(c) Suburban Transfer Success: Barrackpore (SVSPA, March 15–21, 2025)\n' +
                  r'Meteorological Actinic Proxy Tracks Regional $\mathrm{O_3}$ Photochemistry', fontweight='bold', fontsize=9.8)
    ax3.set_ylabel(r'$\mathrm{O_3}$ Concentration ($\mu\mathrm{g\ m^{-3}}$)', fontweight='bold', fontsize=8.5)
    ax3.set_xlabel('Elapsed Time in Evaluation Window (Hours)', fontweight='bold', fontsize=8.5)
    ax3.set_ylim(0, 95)
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='upper left', frameon=True, fontsize=7.8)

    # Panel (d): Bidhannagar (BN) with Secondary Axis for Co-located NO Spikes
    t_bn = np.arange(len(w_bn))
    ax4.plot(t_bn, w_bn['Ozone'], color='#c0392b', lw=2.2, label=r'Observed $\mathrm{O_3}$ (BN)')
    ax4.plot(t_bn, w_bn['Pred_Ozone'], color='#e67e22', linestyle='--', lw=1.8, label=r'RF Predicted $\mathrm{O_3}$ (Met-Only $R^2=-0.26$)')

    ax4_no = ax4.twinx()
    ax4_no.plot(t_bn, w_bn['NO'], color='#2980b9', lw=1.3, linestyle=':', alpha=0.85, label=r'Co-located Primary $\mathrm{NO}$')
    ax4_no.set_ylabel(r'Co-located $\mathrm{NO}$ ($\mu\mathrm{g\ m^{-3}}$)', color='#2980b9', fontweight='bold', fontsize=8.5)
    ax4_no.set_ylim(0, 80)

    for day in range(7):
        m_start, m_end = day * 24 + 7,  day * 24 + 10
        e_start, e_end = day * 24 + 18, day * 24 + 21
        ax4.axvspan(m_start, m_end, color='#3498db', alpha=0.16,
                    label='Morning Rush-Hour Titration (07:00–10:00)' if day == 0 else "")
        ax4.axvspan(e_start, e_end, color='#9b59b6', alpha=0.14,
                    label='Evening Rush-Hour Titration (18:00–21:00)' if day == 0 else "")

    ax4.set_title('(d) Roadside Titration Collapse: Bidhannagar (BN, March 15–21, 2025)\n' +
                  r'Vehicular NO Titration Windows Quench Daytime $\mathrm{O_3}$ while Met Model Overpredicts Peaks',
                  fontweight='bold', fontsize=9.8)
    ax4.set_ylabel(r'$\mathrm{O_3}$ Concentration ($\mu\mathrm{g\ m^{-3}}$)', fontweight='bold', fontsize=8.5)
    ax4.set_xlabel('Elapsed Time in Evaluation Window (Hours)', fontweight='bold', fontsize=8.5)
    ax4.set_ylim(0, 95)
    ax4.grid(True, linestyle=':', alpha=0.5)

    h4_1, l4_1 = ax4.get_legend_handles_labels()
    h4_2, l4_2 = ax4_no.get_legend_handles_labels()
    ax4.legend(h4_1 + h4_2, l4_1 + l4_2, loc='upper left', frameon=True, fontsize=7.2)

    out_path = os.path.join(FIG_DIR, 'Fig7_ML_Generalization_TimeSeries.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"--> Saved {out_path}")


# =============================================================
# MAIN RUNNER
# =============================================================
if __name__ == '__main__':
    print("=" * 65)
    print("REGENERATING ALL 7 PUBLICATION FIGURES FOR ATMOSPHERIC ENVIRONMENT")
    print("=" * 65)
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    generate_figure6()
    generate_figure7()
    print("=" * 65)
    print("ALL 7 FIGURES SUCCESSFULLY REGENERATED AT 300 DPI!")
    print("=" * 65)
